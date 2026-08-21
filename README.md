# OccluRank

Official PyTorch implementation of **OccluRank: Controllable Occlusion-Aware Layout-to-Image Generation by Adding Just an Ordinal Rank**.

OccluRank extends bounding-box layouts with a foreground-to-background ordinal rank. Its Order-aware Instance Interaction (OII) module updates overlapping instance representations before they are aggregated and injected into SDXL.

## Release status

The training and inference code is available in this repository. Public download links for the OccluRank checkpoint and the initialization checkpoint used in the paper will be added when they are available.

## Installation

The reference environment uses Python 3.12, PyTorch 2.7.0, and CUDA 12.8.

```bash
git clone https://github.com/wenyang-hong/OccluRank.git
cd OccluRank

conda create -n occlurank python=3.12 -y
conda activate occlurank

pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Use the PyTorch command appropriate for your CUDA version if CUDA 12.8 is unavailable.

## Model weights

The inference entry point accepts either a local checkpoint path or an HTTP(S) checkpoint URL through `--adapter_path`.

| Checkpoint | Purpose | Download |
| --- | --- | --- |
| OccluRank checkpoint | Inference | To be released |
| Initialization checkpoint | Reproducing the paper training setup | To be released |

Downloaded checkpoints can be placed in `checkpoints/`; this directory is ignored by Git.

## Quick start

The repository includes one ordered-layout example at `examples/layouts/example.json`. After downloading the OccluRank checkpoint, edit `CHECKPOINT` in `configs/paper_infer.sh`, then run:

```bash
bash configs/paper_infer.sh
```

Generated images are written to `outputs/example/images/`. Layout visualizations with colored boxes are written to `outputs/example/layout/`.

## Inference data format

`--dataset_txt_path` points to a text file containing one layout JSON path per line. Paths are resolved from the directory in which the command is run.

Each JSON file contains a global caption and an ordered annotation list:

```json
{
  "caption": "A cream-colored sportbike is parked on a concrete floor, with a white metal shelving unit to its right and slightly behind it; a woven wicker bed with beige mattress and brown blanket is visible behind the shelving unit, against a plain white wall.",
  "annotations": [
    {
      "box": [177.0, 267.75, 242.25, 330.0],
      "caption": ["cream-colored sportbike viewed from rear, parked on concrete floor"]
    },
    {
      "box": [129.0, 93.75, 354.0, 390.75],
      "caption": ["white metal shelving unit with four horizontal shelves, empty"]
    },
    {
      "box": [231.0, 302.25, 348.75, 133.5],
      "caption": ["woven wicker frame with beige mattress and brown blanket, against wall"]
    }
  ]
}
```

Important conventions:

- `annotations` must be ordered from foreground to background.
- Each `box` uses COCO-style `xywh` coordinates.
- OccluLayout-Bench boxes are defined on a 768 x 768 annotation canvas and are normalized internally.
- The generation resolution is controlled independently by `--resolution`.
- Inputs with more than `--max_obj` valid instances are subsampled while preserving the relative order of the selected instances.

A direct inference command is:

```bash
python infer_occlurank.py \
  --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
  --adapter_path checkpoints/occlurank_step_00001500.ckpt \
  --dataset_txt_path examples/infer.txt \
  --output_path outputs/example \
  --resolution 1024 \
  --num_inference_steps 30 \
  --guidance_scale 7.5 \
  --model_variant lrt \
  --order_position_encoding learned \
  --interaction_depth 1 \
  --rank_embedding_capacity 10
```

## Training data format

`train_occlurank.py` reads one parquet file or every parquet file in a directory. Each row must contain:

- `image_path`: path relative to `--image_root`;
- `metadata.global_caption`: global image description;
- `metadata.bbox_info`: ordered instance annotations;
- optionally, `metadata.image_info.width` and `metadata.image_info.height`.

Each item in `metadata.bbox_info` contains:

- `bbox`: absolute `xyxy` coordinates in the source image;
- `detail_description`: instance description.

The order of `bbox_info` is interpreted as foreground to background. The data loader applies the same resize and center crop to the image and its boxes, then converts valid boxes to normalized `xyxy` coordinates.

## Training

Edit the paths at the top of `configs/paper_train.sh`, then run:

```bash
bash configs/paper_train.sh
```

The paper configuration uses:

- SDXL base model: `stabilityai/stable-diffusion-xl-base-1.0`;
- image resolution: 1024;
- batch size: 4;
- gradient accumulation: 40;
- optimizer steps: 1,500;
- learning rate: 1e-4;
- mixed precision: BF16;
- global/local loss weights: 1.0/2.0;
- maximum instances: 5;
- OII depth: 1;
- learned rank-embedding capacity: 10;
- seed: 42.

Checkpoints are saved as `checkpoint_step_XXXXXXXX.ckpt`. They contain the text projector, position network, attention processors, optimizer state, progress state, and reproducibility hyperparameters. Local dataset and output paths are not stored.

`--initial_adapter_checkpoint` is optional in the code. Supply the initialization checkpoint to reproduce the paper setup; omit the option to initialize the trainable modules from scratch.

## Repository structure

```text
.
|-- configs/
|   |-- paper_infer.sh
|   `-- paper_train.sh
|-- examples/
|   |-- infer.txt
|   `-- layouts/example.json
|-- occlurank/
|   |-- attention.py
|   |-- conditioning.py
|   |-- inference_data.py
|   |-- model.py
|   |-- pipeline.py
|   `-- training_data.py
|-- infer_occlurank.py
|-- train_occlurank.py
`-- requirements.txt
```

## Acknowledgements

This implementation builds on PyTorch, Hugging Face Diffusers, Transformers, Accelerate, OpenFlamingo, and imagen-pytorch. Attribution and upstream license headers are retained in the corresponding source files.

## Citation

The BibTeX entry will be added after the arXiv identifier is assigned.
