import os
import argparse
import itertools
import random
import torch
import numpy as np
from PIL import ImageDraw
from accelerate import Accelerator
from occlurank.inference_data import create_dataloader
from occlurank.conditioning import Resampler, PositionNet
from occlurank.model import OccluRankModel
from occlurank.pipeline import StableDiffusionXLPipeline

from occlurank.attention import (
    AttnProcessor2_0 as AttnProcessor,
)
from occlurank.attention import (
    OccluRankAttnProcessor2_0 as OccluRankAttnProcessor,
)

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

    return prompt_embeds, all_obj_attention_mask, eot_embeds


def prepare_phrase_embeddings(batch, text_encoder, text_encoder_2, device, extract_embedding_layers):
    all_obj_ids = batch["all_obj_ids"]
    all_obj_ids_2 = batch["all_obj_ids_2"]
    all_obj_attention_mask = batch["all_obj_attention_mask"]
    phrase_num_arr = [len(obj_ids) for obj_ids in all_obj_ids]

    all_obj_ids = list(itertools.chain(*all_obj_ids))
    all_obj_ids = torch.cat(all_obj_ids).to(device)

    all_obj_ids_2 = list(itertools.chain(*all_obj_ids_2))
    all_obj_ids_2 = torch.cat(all_obj_ids_2).to(device)

    all_obj_attention_mask = list(itertools.chain(*all_obj_attention_mask))
    all_obj_attention_mask = torch.cat(all_obj_attention_mask).to(device)

    prompt_embeds, all_obj_attention_mask, eot_embeds = encode_phrases(
        all_obj_ids, all_obj_ids_2, text_encoder, text_encoder_2, all_obj_attention_mask, device,
        extract_embedding_layers
    )

    return prompt_embeds, all_obj_attention_mask, eot_embeds, phrase_num_arr



def _reorder_instance_sequence(values, permutation):
    """
    Reorder one sample's instance-aligned sequence.

    Supported inner containers:
      - list
      - tuple
      - numpy.ndarray
      - torch.Tensor
    """
    if isinstance(values, torch.Tensor):
        index = torch.tensor(
            permutation,
            device=values.device,
            dtype=torch.long,
        )
        return values.index_select(0, index)

    if isinstance(values, np.ndarray):
        return values[np.asarray(permutation, dtype=np.int64)]

    if isinstance(values, tuple):
        return tuple(values[index] for index in permutation)

    if isinstance(values, list):
        return [values[index] for index in permutation]

    raise TypeError(
        "Unsupported instance sequence type: "
        f"{type(values).__name__}"
    )


def reverse_instance_order_in_batch(batch):
    """
    Reverse every sample's instance order.

    The following instance-aligned fields are reversed together:
      - all_obj_ids
      - all_obj_ids_2
      - all_obj_attention_mask
      - all_boxes

    The global prompt fields text_input_ids and text_input_ids_2 are not
    changed. Therefore, enabling this option changes only the foreground-to-
    background ordering of the instance conditions.

    For an original global order [0, 1, 2, ..., N-1], the new order becomes
    [N-1, ..., 2, 1, 0]. Under a globally consistent order, this flips the
    relative order of every pair of distinct instances.
    """
    aligned_keys = [
        "all_obj_ids",
        "all_obj_ids_2",
        "all_obj_attention_mask",
        "all_boxes",
    ]

    for key in aligned_keys:
        if key not in batch:
            raise KeyError(
                f"Batch is missing required instance field: {key}"
            )

    batch_size = len(batch["all_boxes"])
    permutations = []

    for sample_index in range(batch_size):
        num_objects = len(batch["all_boxes"][sample_index])
        permutation = list(reversed(range(num_objects)))
        permutations.append(permutation)

    for key in aligned_keys:
        outer = batch[key]
        reordered = [
            _reorder_instance_sequence(
                outer[sample_index],
                permutations[sample_index],
            )
            for sample_index in range(batch_size)
        ]

        if isinstance(outer, tuple):
            batch[key] = tuple(reordered)
        elif isinstance(outer, list):
            batch[key] = reordered
        else:
            raise TypeError(
                f"Unsupported outer batch container for {key}: "
                f"{type(outer).__name__}"
            )

    return permutations


def parse_args():
    parser = argparse.ArgumentParser(description="OccluRank inference on SDXL.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-xl-base-1.0",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to an OccluRank training checkpoint.",
    )
    parser.add_argument("--output_path", type=str, default="./outputs/occlurank_inference")
    parser.add_argument("--dataset_txt_path", type=str, required=True)

    # Defaults used for the paper experiments.
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--num_sample", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--control_ratio", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--max_obj", type=int, default=5)
    parser.add_argument("--min_box_size", type=float, default=0.01)
    parser.add_argument("--extract_embedding_layers", type=int, default=3)

    # Public aliases map to the internal names used by existing checkpoints.
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
        help="Must match the learned position-embedding shape in the checkpoint.",
    )

    parser.add_argument(
        "--reverse_instance_order",
        action="store_true",
        help="Diagnostic option: reverse instance conditions while keeping the global prompt unchanged.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device
    weight_dtype = torch.bfloat16
    vae_dtype = torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=weight_dtype,
        add_watermarker=False,
    ).to(device)

    # Keep the VAE in FP32 for stable SDXL image decoding.
    pipe.vae.to(device=device, dtype=vae_dtype)
    pipe.unet.to(device=device, dtype=weight_dtype)
    pipe.text_encoder.to(device=device, dtype=weight_dtype)
    pipe.text_encoder_2.to(device=device, dtype=weight_dtype)

    original_vae_decode = pipe.vae.decode

    def vae_decode_fp32(latents, *decode_args, **decode_kwargs):
        latents = latents.to(device=device, dtype=vae_dtype)
        return original_vae_decode(latents, *decode_args, **decode_kwargs)

    pipe.vae.decode = vae_decode_fp32

    resampler_dim = 2560
    resampler_depth = 4
    resampler_dim_head = 128
    resampler_num_heads = 20
    resampler_num_queries = 4
    resampler_ff_mult = 4
    fourier_freqs = 64

    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    text_encoder_2 = pipe.text_encoder_2
    tokenizer_2 = pipe.tokenizer_2
    unet = pipe.unet

    text_proj_model = Resampler(
        dim=resampler_dim,
        depth=resampler_depth,
        dim_head=resampler_dim_head,
        heads=resampler_num_heads,
        num_queries=resampler_num_queries,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=resampler_ff_mult,
        phrase_embeddings_dim=text_encoder.config.projection_dim + text_encoder_2.config.projection_dim,
    ).to(device, dtype=weight_dtype)
    pos_net = PositionNet(
        unet.config.cross_attention_dim,
        fourier_freqs=fourier_freqs,
    ).to(device=device, dtype=weight_dtype)

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
    occlurank_model = OccluRankModel(
        unet, text_proj_model, pos_net, adapter_modules=adapter_modules, device=device, num_tokens=4,
        ckpt_path=args.adapter_path
    )
    occlurank_model.to(device=device, dtype=weight_dtype)

    pipe.unet.eval()
    pipe.text_encoder.eval()
    pipe.text_encoder_2.eval()
    pipe.vae.eval()
    occlurank_model.eval()

    print("=" * 80)
    print("Inference dtypes")
    print(f"  UNet:           {next(pipe.unet.parameters()).dtype}")
    print(f"  Text encoder 1: {next(pipe.text_encoder.parameters()).dtype}")
    print(f"  Text encoder 2: {next(pipe.text_encoder_2.parameters()).dtype}")
    print(f"  Text projector: {next(text_proj_model.parameters()).dtype}")
    print(f"  PositionNet:    {next(pos_net.parameters()).dtype}")
    print(f"  VAE:            {next(pipe.vae.parameters()).dtype}")
    print("=" * 80)

    assert next(pipe.unet.parameters()).dtype == weight_dtype
    assert next(pipe.text_encoder.parameters()).dtype == weight_dtype
    assert next(pipe.text_encoder_2.parameters()).dtype == weight_dtype
    assert next(text_proj_model.parameters()).dtype == weight_dtype
    assert next(pos_net.parameters()).dtype == weight_dtype
    assert next(pipe.vae.parameters()).dtype == vae_dtype

    test_dataloader = create_dataloader(
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        img_size=args.resolution,
        txt_file=args.dataset_txt_path,
        batch_size=args.batch_size,
        num_workers=0,
        max_obj=args.max_obj,
        min_box_size=args.min_box_size,
    )

    with torch.no_grad():
        for step, batch in enumerate(test_dataloader):
            if args.reverse_instance_order:
                permutations = reverse_instance_order_in_batch(batch)

                if step < 3:
                    for sample_index, permutation in enumerate(permutations):
                        file_name = str(batch["file_names"][sample_index])
                        print(
                            "[instance-order reverse] "
                            f"file={file_name}, "
                            f"new_order_from_original_indices={permutation}"
                        )

            for idx in range(args.num_sample):
                phrase_embeds, all_obj_attention_mask, phrase_eot_embeds, phrase_num_arr = prepare_phrase_embeddings(
                    batch, text_encoder, text_encoder_2, device, args.extract_embedding_layers
                )

                phrase_embeds_uncond = torch.zeros_like(phrase_embeds)
                phrase_eot_embeds_uncond = torch.zeros_like(phrase_eot_embeds)

                encoder_output = text_encoder(batch['text_input_ids'].to(device), output_hidden_states=True)
                text_embeds = encoder_output.hidden_states[-2]
                encoder_output_2 = text_encoder_2(batch['text_input_ids_2'].to(device), output_hidden_states=True)
                pooled_text_embeds = encoder_output_2[0]
                text_embeds_2 = encoder_output_2.hidden_states[-2]
                text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1)

                all_boxes = torch.stack(
                    list(itertools.chain(*batch["all_boxes"]))
                ).to(device=device, dtype=weight_dtype)
                text_embeds_uncond = torch.zeros_like(text_embeds)
                pooled_text_embeds_uncond = torch.zeros_like(pooled_text_embeds)

                images = occlurank_model.generate(
                    pipe=pipe,
                    phrase_embeds=phrase_embeds.to(
                        device=device, dtype=weight_dtype
                    ),
                    negative_phrase_embeds=phrase_embeds_uncond.to(
                        device=device, dtype=weight_dtype
                    ),
                    phrase_eot_embeds=phrase_eot_embeds.to(
                        device=device, dtype=weight_dtype
                    ),
                    negative_phrase_eot_embeds=phrase_eot_embeds_uncond.to(
                        device=device, dtype=weight_dtype
                    ),
                    text_embeds=text_embeds.to(
                        device=device, dtype=weight_dtype
                    ),
                    negative_text_embeds=text_embeds_uncond.to(
                        device=device, dtype=weight_dtype
                    ),
                    pooled_text_embeds=pooled_text_embeds.to(
                        device=device, dtype=weight_dtype
                    ),
                    negative_pooled_text_embeds=pooled_text_embeds_uncond.to(
                        device=device, dtype=weight_dtype
                    ),
                    cond_ratio=args.control_ratio,
                    all_obj_attention_mask=all_obj_attention_mask,
                    phrase_num_arr=phrase_num_arr,
                    boxes=all_boxes,
                    scale=1,
                    seed=args.seed + idx,
                    guidance_scale=args.guidance_scale,
                    height=args.resolution,
                    width=args.resolution,
                    num_inference_steps=args.num_inference_steps
                )

                save_path = os.path.join(args.output_path)
                os.makedirs(save_path, exist_ok=True)
                color_list = ["red", "blue", "yellow", "purple", "green", "black", "brown", "orange", "white", "gray"]

                for i in range(len(images)):
                    file_name = batch["file_names"][i]
                    boxes = batch["all_boxes"][i]

                    images_dir = os.path.join(save_path, "images")
                    layout_dir = os.path.join(save_path, "layout")
                    os.makedirs(images_dir, exist_ok=True)
                    os.makedirs(layout_dir, exist_ok=True)

                    image = images[i]

                    try:
                        original_image = image.convert("RGB")
                        file_stem = os.path.splitext(file_name)[0]
                        original_output_path = os.path.join(images_dir, f"{file_stem}_{idx}.png")
                        original_image.save(original_output_path, "PNG")

                        editable_image = original_image.copy()
                        draw = ImageDraw.Draw(editable_image)
                        W, H = editable_image.size

                        for j, box in enumerate(boxes):
                            color = color_list[j % len(color_list)]
                            x1, y1, x2, y2 = box
                            adjusted_bbox = (
                                int(x1 * W),
                                int(y1 * H),
                                int(x2 * W),
                                int(y2 * H),
                            )
                            draw.rectangle(adjusted_bbox, outline=color, width=4)

                        layout_output_path = os.path.join(layout_dir, f"{file_stem}_{idx}.png")
                        editable_image.save(layout_output_path, "PNG")

                    except Exception as e:
                        print(f"Error processing file {file_name}: {e}")


if __name__ == "__main__":
    main()
