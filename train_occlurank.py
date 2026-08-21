import os
import random
import math
import sys
from tqdm.auto import tqdm
import argparse
import itertools
import time
import glob
import torch
import torch.nn.functional as F
import numpy as np
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from occlurank.training_data import create_dataloader
from occlurank.conditioning import Resampler, PositionNet
from occlurank.attention import (AttnProcessor2_0 as AttnProcessor,)
from occlurank.attention import (OccluRankAttnProcessor2_0 as OccluRankAttnProcessor,)
logger = get_logger(__name__)


class DualLogger:
    """Mirror stdout to a line-buffered training log."""

    def __init__(self, log_path):
        self.terminal = sys.__stdout__
        self.log = open(log_path, "a", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def log_only(self, message):
        """Write a message to the log without printing it to the terminal."""
        self.log.write(message)


CHECKPOINT_ARG_KEYS = (
    "resolution",
    "train_batch_size",
    "gradient_accumulation_steps",
    "max_train_steps",
    "save_every_n_steps",
    "learning_rate",
    "mixed_precision",
    "seed",
    "max_obj",
    "min_box_size",
    "extract_embedding_layers",
    "global_loss_weight",
    "local_loss_weight",
    "adapter_variant",
    "lrt_pos_encoding",
    "lrt_depth",
    "lrt_num_objects",
)


def checkpoint_args(args):
    """Return reproducibility settings without local filesystem paths."""
    return {key: getattr(args, key) for key in CHECKPOINT_ARG_KEYS}


def check_resume_variant_compatibility(checkpoint, args):
    """Prevent accidental resume across ablation variants. Use --initial_adapter_checkpoint for cross-variant init."""
    if args.allow_variant_resume_mismatch:
        print("Warning: skipping checkpoint variant compatibility checks.")
        return

    ckpt_args = checkpoint.get("args", {})
    if not isinstance(ckpt_args, dict) or not ckpt_args:
        print("Warning: checkpoint has no saved configuration; variant compatibility cannot be verified.")
        return

    keys_to_check = ["adapter_variant", "lrt_pos_encoding", "lrt_depth", "lrt_num_objects"]
    for key in keys_to_check:
        if key in ckpt_args and hasattr(args, key):
            old_value = ckpt_args[key]
            new_value = getattr(args, key)
            if old_value != new_value:
                raise ValueError(
                    f"Cannot resume across ablation settings for {key}: "
                    f"checkpoint={old_value}, current={new_value}. "
                    f"For cross-variant initialization, use --initial_adapter_checkpoint instead of --resume_from_checkpoint."
                )


def apply_condition_dropout(text_embeds, pooled_text_embeds, phrase_embeds, phrase_eot_embeds, all_obj_attention_mask,
                            drop_global_prob=0.3, drop_local_prob=0.15):
    """Apply CFG-style conditional dropout during training.
    Returns possibly zeroed-out embeddings based on random decisions."""
    drop_global = random.random() < drop_global_prob
    drop_local = random.random() < drop_local_prob
    if drop_global:
        text_embeds = torch.zeros_like(text_embeds)
        pooled_text_embeds = torch.zeros_like(pooled_text_embeds)
    if drop_local:
        phrase_embeds = torch.zeros_like(phrase_embeds)
        phrase_eot_embeds = torch.zeros_like(phrase_eot_embeds)
        all_obj_attention_mask = torch.zeros_like(all_obj_attention_mask)
    return text_embeds, pooled_text_embeds, phrase_embeds, phrase_eot_embeds, all_obj_attention_mask


def load_adapter_checkpoint(text_proj_model, adapter_modules, pos_net, ckpt_path: str, skip_pos_net: bool = False):
    import os
    from urllib.parse import urlparse
    import requests
    from tqdm import tqdm
    import torch

    pretrained_models_dir = "./pretrained_models"
    if not os.path.exists(pretrained_models_dir):
        os.makedirs(pretrained_models_dir)
        print(f"Created directory: {pretrained_models_dir}")

    parsed_url = urlparse(ckpt_path)
    is_http_url = parsed_url.scheme in ['http', 'https']
    if is_http_url:
        filename = os.path.basename(urlparse(ckpt_path).path) or "initial_adapter.ckpt"
        local_path = os.path.join(pretrained_models_dir, filename)
        if os.path.exists(local_path):
            print(f"Model file already exists: {local_path}")
            ckpt_path = local_path
        else:
            print(f"Downloading model from {ckpt_path}...")
            try:
                response = requests.get(ckpt_path, stream=True, timeout=60)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                with open(local_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading model") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                print(f"Model download completed: {local_path}")
                ckpt_path = local_path
            except Exception as e:
                raise Exception(f"Failed to download model: {e}")
    else:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model file not found: {ckpt_path}")
        if os.path.isdir(ckpt_path):
            raise IsADirectoryError(
                "--initial_adapter_checkpoint must point to a PyTorch checkpoint file"
            )

    orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in text_proj_model.parameters()]))

    if not skip_pos_net:
        orig_pos_net_sum = torch.sum(torch.stack([torch.sum(p) for p in pos_net.parameters()]))

    state_dict = torch.load(ckpt_path, map_location="cpu")
    print(f"Checkpoint keys: {list(state_dict.keys())}")

    text_proj_model.load_state_dict(state_dict["text_proj_model"], strict=True)
    adapter_load_info = adapter_modules.load_state_dict(state_dict["adapter_modules"], strict=False)
    print("\n" + "=" * 55)
    print("Adapter load_state_dict(strict=False) details:")
    print(f"   missing_keys    : {adapter_load_info.missing_keys}")
    print(f"   unexpected_keys : {adapter_load_info.unexpected_keys}")
    print("=" * 55 + "\n")

    if not skip_pos_net:
        pos_net.load_state_dict(state_dict["pos_net"], strict=True)

    new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in text_proj_model.parameters()]))
    assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of text_proj_model did not change!"

    if not skip_pos_net:
        new_pos_net_sum = torch.sum(torch.stack([torch.sum(p) for p in pos_net.parameters()]))
        assert orig_pos_net_sum != new_pos_net_sum, "Weights of pos_net did not change!"

    print("\n" + "=" * 55)
    print("Checkpoint loading summary:")
    print("   text_proj_model : loaded (strict=True)")
    print("   adapter_modules : loaded (strict=False)")
    print(f"   pos_net         : {'skipped' if skip_pos_net else 'loaded (strict=True)'}")
    print("=" * 55 + "\n")

    print(f"Successfully loaded weights from checkpoint {ckpt_path}")


def encode_phrases(all_obj_ids, all_obj_ids_2, text_encoder, text_encoder_2, all_obj_attention_mask, device,
                   extract_embedding_layers=3):
    eot_pos = (all_obj_attention_mask.sum(axis=-1) - 1).to(device)
    prompt_embeds_list = []
    eot_embeds_list = []
    for te, ids in zip([text_encoder, text_encoder_2], [all_obj_ids, all_obj_ids_2]):
        prompt_embeds = te(ids.to(device), output_hidden_states=True)
        eot_embeddings = prompt_embeds.hidden_states[-2][torch.arange(all_obj_ids.shape[0]), eot_pos, :]
        eot_embeds_list.append(eot_embeddings)
        prompt_embeds = prompt_embeds.hidden_states[-2:-(2 * extract_embedding_layers + 1):-2]
        n, l, dim = prompt_embeds[0].shape
        prompt_embeds = torch.concat(list(prompt_embeds)).reshape(extract_embedding_layers, n, l, dim)
        prompt_embeds = prompt_embeds.permute(1, 0, 2, 3)
        prompt_embeds_list.append(prompt_embeds)
    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    eot_embeds = torch.concat(eot_embeds_list, dim=-1)
    all_obj_attention_mask = all_obj_attention_mask.unsqueeze(1).repeat(1, extract_embedding_layers, 1)
    return prompt_embeds, eot_embeds, all_obj_attention_mask


def prepare_phrase_embeddings(batch, text_encoder, text_encoder_2, device, extract_embedding_layers):
    all_obj_ids = list(itertools.chain(*batch["all_obj_ids"]))
    all_obj_ids = torch.cat(all_obj_ids).to(device)
    all_obj_ids_2 = list(itertools.chain(*batch["all_obj_ids_2"]))
    all_obj_ids_2 = torch.cat(all_obj_ids_2).to(device)
    all_obj_attention_mask = list(itertools.chain(*batch["all_obj_attention_mask"]))
    all_obj_attention_mask = torch.cat(all_obj_attention_mask).to(device)
    phrase_embeds, phrase_eot_embeds, all_obj_attention_mask = encode_phrases(
        all_obj_ids, all_obj_ids_2, text_encoder, text_encoder_2, all_obj_attention_mask, device, extract_embedding_layers
    )
    phrase_num_arr = [len(obj_ids) for obj_ids in batch["all_obj_ids"]]
    return phrase_embeds, phrase_eot_embeds, all_obj_attention_mask, phrase_num_arr


def parse_args():
    parser = argparse.ArgumentParser(description="Train OccluRank on SDXL.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
        help="SDXL base model path or Hugging Face model identifier.",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs/occlurank")
    parser.add_argument(
        "--parquet_path",
        type=str,
        required=True,
        help="Path to one training parquet or a directory containing parquet files.",
    )
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument(
        "--max_parquet_files",
        type=int,
        default=None,
        help="Optional limit when --parquet_path is a directory.",
    )

    # Defaults used for the paper experiments.
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=40)
    parser.add_argument("--max_train_steps", type=int, default=1500)
    parser.add_argument("--save_every_n_steps", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--mixed_precision", type=str, default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_obj", type=int, default=5)
    parser.add_argument("--min_box_size", type=float, default=0.01)
    parser.add_argument("--extract_embedding_layers", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--global_loss_weight", type=float, default=1.0)
    parser.add_argument("--local_loss_weight", type=float, default=2.0)

    # Public aliases map to the internal names used by existing checkpoints.
    parser.add_argument(
        "--initial_adapter_checkpoint",
        type=str,
        default=None,
        help="Optional initialization checkpoint used by the paper training setup.",
    )
    parser.add_argument(
        "--model_variant",
        "--adapter_variant",
        dest="adapter_variant",
        type=str,
        default="lrt",
        choices=["base", "lrt", "lrt_reweight"],
    )
    parser.add_argument(
        "--order_position_encoding",
        "--lrt_pos_encoding",
        dest="lrt_pos_encoding",
        type=str,
        default="learned",
        choices=["learned", "none", "rope"],
    )
    parser.add_argument(
        "--interaction_depth",
        "--lrt_depth",
        dest="lrt_depth",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--rank_embedding_capacity",
        "--lrt_num_objects",
        dest="lrt_num_objects",
        type=int,
        default=10,
        help="Learned ordinal-position embedding capacity. Must match the checkpoint shape.",
    )

    parser.add_argument("--skip_pos_net", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--allow_variant_resume_mismatch", action="store_true")

    args = parser.parse_args()

    if args.max_train_steps <= 0:
        raise ValueError("--max_train_steps must be positive")

    if os.path.isdir(args.parquet_path):
        parquet_files = sorted(glob.glob(os.path.join(args.parquet_path, "*.parquet")))
        if not parquet_files:
            raise ValueError(f"No .parquet files found in directory: {args.parquet_path}")
        if args.max_parquet_files is not None:
            parquet_files = parquet_files[:args.max_parquet_files]
        args.parquet_files = parquet_files
    else:
        if not args.parquet_path.endswith(".parquet"):
            raise ValueError("--parquet_path must be a .parquet file or a directory")
        args.parquet_files = [args.parquet_path]

    return args


def occlurank_forward(unet, text_proj_model, pos_net, noisy_latents, timesteps, text_embeds, pooled_text_embeds,
                      phrase_embeds, all_obj_attention_mask, phrase_eot_embeds, phrase_num_arr, boxes, device):
    batch_size = noisy_latents.shape[0]
    num_objects = phrase_embeds.shape[0]

    ap_tokens = text_proj_model(phrase_embeds, all_obj_attention_mask)  # [num_objects, n_layers, n_q, D]
    _, n_layers, n_q, D = ap_tokens.shape
    ap_tokens = ap_tokens.view(num_objects, n_layers * n_q, D)

    phrase_eot_embeds = phrase_eot_embeds.view(num_objects, 1, D)
    ap_tokens = torch.cat([ap_tokens, phrase_eot_embeds], dim=1)  # [num_objects, n_layers*n_q + 1, D]

    grounding_embeddings = pos_net(boxes).view(num_objects, 1, -1)  # [num_objects, 1, D]
    ap_tokens = ap_tokens + grounding_embeddings

    cross_attention_kwargs = {
        "phrase_num_arr": phrase_num_arr,
        "ap_tokens": ap_tokens,
        "boxes": boxes
    }

    resolution = noisy_latents.shape[-1] * 8
    time_ids = torch.tensor([[resolution, resolution, 0, 0, resolution, resolution]] * batch_size,
                            device=device, dtype=torch.long)
    added_cond_kwargs = {
        "text_embeds": pooled_text_embeds,
        "time_ids": time_ids
    }

    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=text_embeds,
                      added_cond_kwargs=added_cond_kwargs, cross_attention_kwargs=cross_attention_kwargs).sample
    return model_pred

def create_union_mask_from_batch_boxes(batch_all_boxes, latent_h, latent_w, device):
    B = len(batch_all_boxes)
    union_mask = torch.zeros(B, latent_h, latent_w, dtype=torch.float32, device=device)
    for i, boxes_in_sample in enumerate(batch_all_boxes):
        if not boxes_in_sample:
            continue
        boxes = torch.stack(boxes_in_sample).to(device)  # [N_i, 4]
        x0 = (boxes[:, 0] * latent_w).floor().long()
        y0 = (boxes[:, 1] * latent_h).floor().long()
        x1 = (boxes[:, 2] * latent_w).ceil().long()
        y1 = (boxes[:, 3] * latent_h).ceil().long()
        x0 = torch.clamp(x0, 0, latent_w - 1)
        y0 = torch.clamp(y0, 0, latent_h - 1)
        x1 = torch.clamp(x1, 0, latent_w)
        y1 = torch.clamp(y1, 0, latent_h)
        mask_i = torch.zeros(latent_h, latent_w, dtype=torch.bool, device=device)
        for j in range(boxes.size(0)):
            if x1[j] > x0[j] and y1[j] > y0[j]:
                mask_i[y0[j]:y1[j], x0[j]:x1[j]] = True
        union_mask[i] = mask_i.float()
    return union_mask

def main():

    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = DualLogger(os.path.join(args.output_dir, "train1.log"))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=ProjectConfiguration(project_dir=args.output_dir),
        log_with="tensorboard",
    )
    device = accelerator.device

    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path,
                                                                 subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    unet.requires_grad_(False)

    vae.to(accelerator.device)
    vae.eval()
    text_encoder.to(accelerator.device)
    text_encoder.eval()
    text_encoder_2.to(accelerator.device)
    text_encoder_2.eval()

    resampler_dim = 2560
    resampler_depth = 4
    resampler_dim_head = 128
    resampler_num_heads = 20
    resampler_num_queries = 4
    resampler_ff_mult = 4
    fourier_freqs = 64
    text_proj_model = Resampler(
        dim=resampler_dim,
        depth=resampler_depth,
        dim_head=resampler_dim_head,
        heads=resampler_num_heads,
        num_queries=resampler_num_queries,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=resampler_ff_mult,
        phrase_embeddings_dim=text_encoder.config.projection_dim + text_encoder_2.config.projection_dim,
    ).to(device)
    pos_net = PositionNet(unet.config.cross_attention_dim, fourier_freqs=fourier_freqs)

    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            sub_block_id = int(name.split(".")[-3])
            if sub_block_id >= 4:
                cross_attention_dim = None
            hidden_size = unet.config.block_out_channels[-1]
            num_heads = unet.config.attention_head_dim[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            sub_block_id = int(name.split(".")[-3])
            if block_id == 1:
                cross_attention_dim = None
            if sub_block_id >= 4:
                cross_attention_dim = None
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            num_heads = list(reversed(unet.config.attention_head_dim))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
            num_heads = unet.config.attention_head_dim[block_id]
            cross_attention_dim = None
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            attn_procs[name] = OccluRankAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                num_heads=num_heads,
                scale=1,
                num_tokens=resampler_num_queries,
                enable_alpha=True,
                max_obj=args.max_obj,
                adapter_variant=args.adapter_variant,
                lrt_pos_encoding=args.lrt_pos_encoding,
                lrt_depth=args.lrt_depth,
                lrt_num_objects=args.lrt_num_objects,
            )
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())

    if args.initial_adapter_checkpoint is not None:
        load_adapter_checkpoint(
            text_proj_model,
            adapter_modules,
            pos_net,
            ckpt_path=args.initial_adapter_checkpoint,
            skip_pos_net=args.skip_pos_net,
        )
    else:
        logger.info("No initialization checkpoint provided. Training from scratch.")

    params_to_optimize = (
        list(text_proj_model.parameters()) +
        list(pos_net.parameters()) +
        list(adapter_modules.parameters())
    )
    optimizer = torch.optim.AdamW(params_to_optimize, lr=args.learning_rate)

    # Prepare one loader to initialize Accelerate and determine file lengths.
    dummy_dataloader = create_dataloader(
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        parquet_path=args.parquet_files[0],
        image_root=args.image_root,
        img_size=args.resolution,
        batch_size=args.train_batch_size,
        num_workers=0,
        max_obj=args.max_obj,
        min_box_size=args.min_box_size,
    )

    batch_count_cache = {
        os.path.abspath(args.parquet_files[0]): len(dummy_dataloader)
    }

    def get_batches_in_parquet(parquet_path):
        """Return the exact number of micro-batches produced by one parquet file."""
        cache_key = os.path.abspath(parquet_path)
        if cache_key in batch_count_cache:
            return batch_count_cache[cache_key]

        length_loader = create_dataloader(
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            parquet_path=parquet_path,
            image_root=args.image_root,
            img_size=args.resolution,
            batch_size=args.train_batch_size,
            num_workers=0,
            max_obj=args.max_obj,
            min_box_size=args.min_box_size,
        )
        batch_count = len(length_loader)
        batch_count_cache[cache_key] = batch_count
        del length_loader
        return batch_count

    # Repeat complete data passes until max_train_steps can be reached.
    base_parquet_files = list(args.parquet_files)
    optimizer_steps_per_pass = sum(
        math.ceil(get_batches_in_parquet(path) / args.gradient_accumulation_steps)
        for path in base_parquet_files
    )
    if optimizer_steps_per_pass <= 0:
        raise ValueError("The training data produced zero optimizer steps per pass.")
    num_data_passes = max(1, math.ceil(args.max_train_steps / optimizer_steps_per_pass))
    args.parquet_files = base_parquet_files * num_data_passes
    if accelerator.is_main_process:
        print(
            f"Training data: {len(base_parquet_files)} parquet file(s) per pass, "
            f"{optimizer_steps_per_pass} optimizer step(s) per pass; "
            f"using up to {num_data_passes} pass(es) to reach "
            f"max_train_steps={args.max_train_steps}."
        )

    start_file_idx = 0
    global_step = 0
    skip_in_current_file = 0

    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")
        check_resume_variant_compatibility(checkpoint, args)
        accelerator.unwrap_model(text_proj_model).load_state_dict(checkpoint["text_proj_model"])
        accelerator.unwrap_model(pos_net).load_state_dict(checkpoint["pos_net"])
        accelerator.unwrap_model(adapter_modules).load_state_dict(checkpoint["adapter_modules"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint.get("global_step", 0))

        resume_file_idx = int(checkpoint.get("file_idx", 0))
        if resume_file_idx < 0 or resume_file_idx >= len(args.parquet_files):
            raise ValueError(
                f"Checkpoint file_idx={resume_file_idx} is outside the current parquet list "
                f"(length={len(args.parquet_files)})."
            )

        current_parquet = args.parquet_files[resume_file_idx]
        current_file_batches = get_batches_in_parquet(current_parquet)

        # New checkpoints store the next micro-batch exactly. Older checkpoints
        # fall back to reconstructing the position from file-level progress.
        if "next_batch_idx_in_file" in checkpoint:
            skip_in_current_file = int(checkpoint["next_batch_idx_in_file"])
            resume_position_source = "exact checkpoint field"

            saved_parquet = checkpoint.get("parquet_file")
            if saved_parquet is not None:
                saved_name = os.path.basename(saved_parquet)
                current_name = os.path.basename(current_parquet)
                if saved_name != current_name:
                    print(
                        "Warning: checkpoint parquet filename differs from the current file list: "
                        f"checkpoint={saved_name}, current={current_name}. "
                        "The saved file_idx and batch index will still be used."
                    )
        else:
            # Each parquet file has its own DataLoader. Accelerate synchronizes
            # a partial accumulation at the end of each loader, so optimizer
            # steps must be reconstructed file by file.
            optimizer_steps_before_current_file = 0
            previous_file_batches = []

            for previous_idx in range(resume_file_idx):
                previous_batches = get_batches_in_parquet(args.parquet_files[previous_idx])
                previous_steps = math.ceil(
                    previous_batches / args.gradient_accumulation_steps
                )
                optimizer_steps_before_current_file += previous_steps
                previous_file_batches.append(previous_batches)

            optimizer_steps_inside_current_file = (
                global_step - optimizer_steps_before_current_file
            )
            if optimizer_steps_inside_current_file < 0:
                raise ValueError(
                    "Checkpoint global_step is smaller than the optimizer steps implied by "
                    f"the preceding files: global_step={global_step}, "
                    f"steps_before_file={optimizer_steps_before_current_file}, "
                    f"file_idx={resume_file_idx}."
                )

            skip_in_current_file = min(
                optimizer_steps_inside_current_file
                * args.gradient_accumulation_steps,
                current_file_batches,
            )
            resume_position_source = "legacy file-wise fallback"

            print("Warning: legacy checkpoint detected; next_batch_idx_in_file is absent.")
            print(
                "   Resume position was reconstructed file by file, including each "
                "file's incomplete gradient-accumulation tail."
            )
            print(f"   Previous-file batch counts: {previous_file_batches}")
            print(
                f"   Optimizer steps before current file: "
                f"{optimizer_steps_before_current_file}"
            )
            print(
                f"   Optimizer steps inside current file: "
                f"{optimizer_steps_inside_current_file}"
            )

        if skip_in_current_file < 0:
            raise ValueError(
                f"Invalid negative resume batch index: {skip_in_current_file}"
            )
        if skip_in_current_file > current_file_batches:
            print(
                f"Warning: saved batch index {skip_in_current_file} exceeds current file "
                f"length {current_file_batches}; clamping to the file end."
            )
            skip_in_current_file = current_file_batches

        # Skip directly to the next file when the checkpoint is at file end.
        if skip_in_current_file >= current_file_batches:
            start_file_idx = resume_file_idx + 1
            skip_in_current_file = 0
            print(
                f"Checkpoint is at the end of file index {resume_file_idx}; "
                f"continuing from file index {start_file_idx}."
            )
        else:
            start_file_idx = resume_file_idx

        print(
            f"Resumed with {resume_position_source}: "
            f"global_step={global_step}, checkpoint_file_idx={resume_file_idx}"
        )
        print(f"   Current file batches: {current_file_batches}")
        print(f"   Start file index: {start_file_idx}")
        print(f"   Skip in start file: {skip_in_current_file} batches")


    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")


    unet, text_proj_model, pos_net, adapter_modules, optimizer, _ = accelerator.prepare(
        unet, text_proj_model, pos_net, adapter_modules, optimizer, dummy_dataloader
    )
    del dummy_dataloader

    total_files = len(args.parquet_files)

    estimated_total_steps = args.max_train_steps

    for file_idx, parquet_file in enumerate(args.parquet_files[start_file_idx:], start=start_file_idx):
        if global_step >= args.max_train_steps:
            if accelerator.is_main_process:
                print(f"Reached max_train_steps={args.max_train_steps} before file {file_idx + 1}.")
            break

        if accelerator.is_main_process:
            print(f"{'=' * 60}")
            print(f"Processing file [{file_idx + 1}/{total_files}]: {os.path.basename(parquet_file)}")
            if estimated_total_steps:
                pct = min(100.0, global_step / estimated_total_steps * 100)
                print(f"Global progress: {global_step}/{estimated_total_steps} ({pct:.1f}%)")
            print(f"{'=' * 60}")

        train_dataloader = create_dataloader(
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            parquet_path=parquet_file,
            image_root=args.image_root,
            img_size=args.resolution,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            max_obj=args.max_obj,
            min_box_size=args.min_box_size,
        )
        train_dataloader = accelerator.prepare(train_dataloader)

        unet.train()
        text_proj_model.train()
        pos_net.train()
        file_losses = []

        file_start_step = global_step
        file_start_time = time.time()

        progress_bar = tqdm(
            train_dataloader,
            desc=f"File {file_idx + 1}/{total_files}",
            disable=not accelerator.is_main_process,
            dynamic_ncols=True,
            leave=True
        )

        for step, batch in enumerate(progress_bar):
            if file_idx == start_file_idx and step < skip_in_current_file:
                continue

            with accelerator.accumulate(unet):
                pixel_values = batch["images"].to(device, dtype=vae.dtype)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                          device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                text_input_ids = batch['text_input_ids'].to(device)
                text_input_ids_2 = batch['text_input_ids_2'].to(device)
                encoder_output = text_encoder(text_input_ids, output_hidden_states=True)
                text_embeds = encoder_output.hidden_states[-2]
                encoder_output_2 = text_encoder_2(text_input_ids_2, output_hidden_states=True)
                text_embeds_2 = encoder_output_2.hidden_states[-2]
                pooled_text_embeds = encoder_output_2[0]
                text_embeds = torch.cat([text_embeds, text_embeds_2], dim=-1)

                phrase_embeds, phrase_eot_embeds, all_obj_attention_mask, phrase_num_arr = prepare_phrase_embeddings(
                    batch, text_encoder, text_encoder_2, device, args.extract_embedding_layers
                )

                text_embeds, pooled_text_embeds, phrase_embeds, phrase_eot_embeds, all_obj_attention_mask = apply_condition_dropout(
                    text_embeds, pooled_text_embeds, phrase_embeds, phrase_eot_embeds, all_obj_attention_mask
                )

                all_boxes = torch.stack(list(itertools.chain(*batch["all_boxes"]))).to(device)

                model_pred = occlurank_forward(
                    unet=unet,
                    text_proj_model=text_proj_model,
                    pos_net=pos_net,
                    noisy_latents=noisy_latents,
                    timesteps=timesteps,
                    text_embeds=text_embeds,
                    pooled_text_embeds=pooled_text_embeds,
                    phrase_embeds=phrase_embeds,
                    all_obj_attention_mask=all_obj_attention_mask,
                    phrase_eot_embeds=phrase_eot_embeds,
                    phrase_num_arr=phrase_num_arr,
                    boxes=all_boxes,
                    device=device
                )

                latent_h, latent_w = noisy_latents.shape[2], noisy_latents.shape[3]
                num_channels = noisy_latents.shape[1]

                global_loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")

                union_mask = create_union_mask_from_batch_boxes(
                    batch["all_boxes"], latent_h, latent_w, device
                )
                union_mask = union_mask.unsqueeze(1)

                loss_unweighted = F.mse_loss(model_pred.float(), noise.float(), reduction="none")
                local_loss_numerator = (loss_unweighted * union_mask).sum()
                local_loss_denominator = union_mask.sum() * num_channels + 1e-8
                local_loss = local_loss_numerator / local_loss_denominator

                loss = (
                        args.global_loss_weight * global_loss +
                        args.local_loss_weight * local_loss
                )

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_to_optimize, 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    loss_value = accelerator.reduce(loss, reduction="mean").item()
                    global_loss_value = accelerator.reduce(global_loss, reduction="mean").item()
                    local_loss_value = accelerator.reduce(local_loss, reduction="mean").item()

                    file_losses.append(loss_value)

                    if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            save_path = os.path.join(args.output_dir, f"checkpoint_step_{global_step:08d}.ckpt")
                            torch.save({
                                "text_proj_model": accelerator.unwrap_model(text_proj_model).state_dict(),
                                "pos_net": accelerator.unwrap_model(pos_net).state_dict(),
                                "adapter_modules": accelerator.unwrap_model(adapter_modules).state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "global_step": global_step,
                                "file_idx": file_idx,
                                "next_batch_idx_in_file": step + 1,
                                "batches_in_file": len(train_dataloader),
                                "parquet_file": os.path.basename(parquet_file),
                                "checkpoint_format_version": 2,
                                "args": checkpoint_args(args),
                            }, save_path)
                            print(f"Saved checkpoint at step {global_step}: {save_path}")

                    if args.max_train_steps is not None and global_step >= args.max_train_steps:
                        if accelerator.is_main_process:
                            print(f"\nReached max_train_steps={args.max_train_steps}.")
                        break

                    if accelerator.is_main_process:
                        elapsed_since_file = time.time() - file_start_time
                        steps_in_file = global_step - file_start_step
                        if steps_in_file > 0 and elapsed_since_file > 0:
                            secs_per_step = elapsed_since_file / steps_in_file
                            if estimated_total_steps is not None:
                                remaining_steps = max(0, estimated_total_steps - global_step)
                                eta_secs = remaining_steps * secs_per_step
                                eta_str = f"{eta_secs / 3600:.1f}h" if eta_secs > 3600 else f"{eta_secs / 60:.1f}m"
                                progress_bar.set_postfix(
                                    Step=f"{global_step}/{estimated_total_steps}",
                                    ETA=eta_str,
                                    refresh=False
                                )
                            else:
                                progress_bar.set_postfix(
                                    Step=global_step,
                                    Speed=f"{secs_per_step:.2f}s/step",
                                    refresh=False
                                )

                    if accelerator.is_main_process:
                        current_lr = optimizer.param_groups[0]["lr"]
                        log_msg = (f"[File {file_idx + 1}/{total_files}] Step {global_step} | "
                                   f"Total: {loss_value:.6f} | "
                                   f"Global: {global_loss_value:.6f} | "
                                   f"Local: {local_loss_value:.6f} | "
                                   f"LR: {current_lr:.2e}\n")
                        sys.stdout.log_only(log_msg)

        if accelerator.is_main_process:
            avg_loss = sum(file_losses) / len(file_losses) if file_losses else float('nan')
            print(f"Finished file {file_idx + 1}. Average loss: {avg_loss:.6f}")

    if accelerator.is_main_process:
        print(f"Training completed. Final global_step={global_step}; processed {total_files} parquet files.")

if __name__ == "__main__":
    main()
