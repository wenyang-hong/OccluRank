#!/usr/bin/env bash
set -euo pipefail

# Edit these paths before running.
PRETRAINED_MODEL="stabilityai/stable-diffusion-xl-base-1.0"
CHECKPOINT="./checkpoints/occlurank_step_00001500.ckpt"
LAYOUT_LIST="./examples/infer.txt"
OUTPUT_DIR="./outputs/example"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python infer_occlurank.py \
  --pretrained_model_name_or_path "${PRETRAINED_MODEL}" \
  --adapter_path "${CHECKPOINT}" \
  --dataset_txt_path "${LAYOUT_LIST}" \
  --output_path "${OUTPUT_DIR}" \
  --resolution 1024 \
  --num_inference_steps 30 \
  --guidance_scale 7.5 \
  --num_sample 1 \
  --batch_size 1 \
  --seed 42 \
  --control_ratio 1.0 \
  --max_obj 5 \
  --min_box_size 0.01 \
  --model_variant lrt \
  --order_position_encoding learned \
  --interaction_depth 1 \
  --rank_embedding_capacity 10 \
  --extract_embedding_layers 3
