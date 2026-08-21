#!/usr/bin/env bash
set -euo pipefail

# Edit these paths before running.
PRETRAINED_MODEL="stabilityai/stable-diffusion-xl-base-1.0"
TRAIN_PARQUET="/path/to/layoutocclusion_train.parquet"
IMAGE_ROOT="/path/to/LayoutOcclusion"
OUTPUT_DIR="./outputs/occlurank_train"
INITIAL_ADAPTER="./checkpoints/initial_adapter.ckpt"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python train_occlurank.py \
  --pretrained_model_name_or_path "${PRETRAINED_MODEL}" \
  --parquet_path "${TRAIN_PARQUET}" \
  --image_root "${IMAGE_ROOT}" \
  --output_dir "${OUTPUT_DIR}" \
  --initial_adapter_checkpoint "${INITIAL_ADAPTER}" \
  --model_variant lrt \
  --order_position_encoding learned \
  --interaction_depth 1 \
  --rank_embedding_capacity 10 \
  --max_obj 5 \
  --min_box_size 0.01 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 40 \
  --resolution 1024 \
  --max_train_steps 1500 \
  --save_every_n_steps 500 \
  --learning_rate 1e-4 \
  --mixed_precision bf16 \
  --global_loss_weight 1.0 \
  --local_loss_weight 2.0 \
  --extract_embedding_layers 3 \
  --num_workers 4 \
  --seed 42
