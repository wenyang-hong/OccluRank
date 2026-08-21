import os
import requests
from urllib.parse import urlparse
from tqdm import tqdm

import torch

from .attention import (
    OccluRankAttnProcessor2_0 as OccluRankAttnProcessor,
)



class OccluRankModel(torch.nn.Module):
    def __init__(self, unet, text_proj_model, pos_net, adapter_modules=None, device="cuda", num_tokens=4, ckpt_path=None):
        super().__init__()
        self.device = device
        self.adapter_modules = adapter_modules
        self.ckpt_path = ckpt_path
        self.num_tokens = num_tokens
        self.pos_net = pos_net

        self.unet = unet.to(self.device)
        self.text_proj_model = text_proj_model

        if ckpt_path is not None:
            self.load_checkpoint(ckpt_path)

    def load_checkpoint(self, ckpt_path: str):
        pretrained_models_dir = "./pretrained_models"
        os.makedirs(pretrained_models_dir, exist_ok=True)
        
        parsed_url = urlparse(ckpt_path)
        is_http_url = parsed_url.scheme in ['http', 'https']
        
        if is_http_url:
            filename = os.path.basename(parsed_url.path) or "occlurank_checkpoint.ckpt"
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
                    raise RuntimeError(f"Failed to download model: {e}") from e
        else:
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Model file not found: {ckpt_path}")
            if os.path.isdir(ckpt_path):
                raise IsADirectoryError("--adapter_path must point to a checkpoint file")

        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.text_proj_model.parameters()]))
        orig_pos_net_sum = torch.sum(torch.stack([torch.sum(p) for p in self.pos_net.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        print(f"Checkpoint entries: {list(state_dict.keys())}")
        self.text_proj_model.load_state_dict(state_dict["text_proj_model"], strict=True)
        adapter_load_info = self.adapter_modules.load_state_dict(state_dict["adapter_modules"], strict=False)
        self.pos_net.load_state_dict(state_dict["pos_net"], strict=True)
        if adapter_load_info.missing_keys or adapter_load_info.unexpected_keys:
            print(f"Adapter missing keys: {adapter_load_info.missing_keys}")
            print(f"Adapter unexpected keys: {adapter_load_info.unexpected_keys}")

        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.text_proj_model.parameters()]))
        new_pos_net_sum = torch.sum(torch.stack([torch.sum(p) for p in self.pos_net.parameters()]))

        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of text_proj_model did not change!"
        assert orig_pos_net_sum != new_pos_net_sum, "Weights of pos_net did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")

    def set_scale(self, scale): 
        for attn_processor in self.unet.attn_processors.values():
            if isinstance(attn_processor, OccluRankAttnProcessor):
                attn_processor.scale = scale
    
    def generate(
        self,
        pipe,
        phrase_embeds=None, 
        negative_phrase_embeds=None, 
        image=None,
        adapter_conditioning_scale=1,
        phrase_eot_embeds=None, 
        negative_phrase_eot_embeds=None, 
        text_embeds=None, 
        negative_text_embeds=None,
        pooled_text_embeds=None,
        negative_pooled_text_embeds=None,   
        all_obj_attention_mask=None, 
        phrase_num_arr=None, 
        boxes=None,
        scale=1.0,
        cond_ratio=1,
        guidance_scale=0,
        seed=None,
        height=1024,
        width=1024,
        num_inference_steps=30,
        **kwargs,
        ):
        self.set_scale(scale)


        if guidance_scale > 0:
            phrase_num_arr = phrase_num_arr * 2
            boxes = boxes.repeat(2,1)
            all_obj_attention_mask = torch.cat([torch.zeros_like(all_obj_attention_mask), all_obj_attention_mask], dim=0)
            phrase_embeds = torch.cat([negative_phrase_embeds, phrase_embeds], dim=0)
            phrase_eot_embeds = torch.cat([negative_phrase_eot_embeds, phrase_eot_embeds], dim=0)

        ap_tokens = self.text_proj_model(phrase_embeds, all_obj_attention_mask)
        B, n_embedding_layers, n_q, dim = ap_tokens.shape
        ap_tokens = ap_tokens.view(B, n_embedding_layers*n_q, dim)

        phrase_eot_embeds = phrase_eot_embeds.view(B,1,dim)
        ap_tokens = torch.concat([ap_tokens, phrase_eot_embeds], dim=1)
        boxes = boxes.to(ap_tokens)
        grounding_embeddings = self.pos_net(boxes).view(B, 1, -1)
        ap_tokens = ap_tokens + grounding_embeddings

        cross_attention_kwargs = {
            "phrase_num_arr": phrase_num_arr,
            "ap_tokens": ap_tokens,
            "boxes": boxes
        }
                
        generator = torch.Generator(self.device).manual_seed(seed) if seed is not None else None
        
        if image is not None and adapter_conditioning_scale is not None:

            images = pipe(
                prompt_embeds=text_embeds,
                negative_prompt_embeds=negative_text_embeds,
                image = image,
                adapter_conditioning_scale = adapter_conditioning_scale,
                pooled_prompt_embeds=pooled_text_embeds,
                negative_pooled_prompt_embeds=negative_pooled_text_embeds,
                cond_ratio=cond_ratio,
                num_inference_steps=num_inference_steps,
                generator=generator,
                guidance_scale=guidance_scale,
                cross_attention_kwargs=cross_attention_kwargs,
                height=height,
                width=width,
                **kwargs,
            ).images
        else:
            images = pipe(
                prompt_embeds=text_embeds,
                negative_prompt_embeds=negative_text_embeds,
                
                pooled_prompt_embeds=pooled_text_embeds,
                negative_pooled_prompt_embeds=negative_pooled_text_embeds,
                cond_ratio=cond_ratio,
                num_inference_steps=num_inference_steps,
                generator=generator,
                guidance_scale=guidance_scale,
                cross_attention_kwargs=cross_attention_kwargs,
                height=height,
                width=width,
                **kwargs,
            ).images

        return images

    def forward(self, 
        noisy_latents, timesteps, text_embeds, unet_added_cond_kwargs, phrase_embeds, all_obj_attention_mask, eot_embeds, phrase_num_arr, boxes):
        ap_tokens = self.text_proj_model(phrase_embeds, all_obj_attention_mask)
        B, n_embedding_layers, n_q, dim = ap_tokens.shape
        ap_tokens = ap_tokens.view(B, n_embedding_layers*n_q, dim)

        eot_embeds = eot_embeds.view(B,1,dim)
        ap_tokens = torch.concat([ap_tokens, eot_embeds], dim=1)

        grounding_embeddings = self.pos_net(boxes).view(B, 1, -1)
        ap_tokens = ap_tokens + grounding_embeddings

        cross_attention_kwargs = {
            "phrase_num_arr": phrase_num_arr,
            "ap_tokens": ap_tokens,
            "boxes": boxes
        }
        noise_pred = self.unet(noisy_latents, timesteps, text_embeds, added_cond_kwargs=unet_added_cond_kwargs, cross_attention_kwargs=cross_attention_kwargs).sample
        return noise_pred
