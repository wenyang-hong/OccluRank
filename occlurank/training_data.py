import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms


def center_crop_arr(pil_image, image_size):
    WW, HH = pil_image.size
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    performed_scale = image_size / min(WW, HH)
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    info = {
        "performed_scale": performed_scale,
        "crop_y": crop_y,
        "crop_x": crop_x,
    }
    cropped = arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]
    return cropped, info


def to_valid(x0, y0, x1, y1, image_size, min_box_size):
    if x0 > image_size or y0 > image_size or x1 < 0 or y1 < 0:
        return False, (None, None, None, None)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, image_size), min(y1, image_size)
    if (x1 - x0) * (y1 - y0) / (image_size * image_size) < min_box_size:
        return False, (None, None, None, None)
    return True, (x0, y0, x1, y1)


class OccluLayoutTrainDataset(torch.utils.data.Dataset):
    """Read OccluLayout training samples from parquet.

    `bbox_info` is consumed in its stored order, which represents the ordinal
    foreground-to-background sequence. Boxes removed by the
    `min_box_size` filter do not reorder the remaining instances.
    """

    def __init__(
        self,
        parquet_path,
        image_root,
        tokenizer,
        tokenizer_2,
        size=1024,
        max_obj=5,
        min_box_size=0.01,
    ):
        super().__init__()
        self.df = pd.read_parquet(parquet_path)
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.max_obj = max_obj
        self.min_box_size = min_box_size

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = os.path.join(self.image_root, row["image_path"])
        pil_image = Image.open(image_path).convert("RGB")

        arr, _ = center_crop_arr(pil_image, self.size)
        image_tensor = self.to_tensor(arr.copy())

        meta = row["metadata"]
        global_caption = meta["global_caption"]
        bbox_info = meta["bbox_info"]

        all_boxes = []
        all_text = []
        all_obj_ids = []
        all_obj_ids_2 = []
        all_obj_attention_mask = []

        image_info = meta.get("image_info", {})
        orig_w = image_info.get("width", pil_image.width)
        orig_h = image_info.get("height", pil_image.height)

        # Apply the image resize and center crop to every box.
        orig_min = min(orig_w, orig_h)
        scale_to_resize = self.size / orig_min
        resized_w = round(orig_w * scale_to_resize)
        resized_h = round(orig_h * scale_to_resize)
        crop_x = (resized_w - self.size) // 2
        crop_y = (resized_h - self.size) // 2

        for obj in bbox_info:
            x0_raw, y0_raw, x1_raw, y1_raw = obj["bbox"]

            x0_resized = x0_raw * scale_to_resize
            y0_resized = y0_raw * scale_to_resize
            x1_resized = x1_raw * scale_to_resize
            y1_resized = y1_raw * scale_to_resize

            x0_final = x0_resized - crop_x
            y0_final = y0_resized - crop_y
            x1_final = x1_resized - crop_x
            y1_final = y1_resized - crop_y

            valid, (x0_f, y0_f, x1_f, y1_f) = to_valid(
                x0_final,
                y0_final,
                x1_final,
                y1_final,
                image_size=self.size,
                min_box_size=self.min_box_size,
            )
            if not valid:
                continue

            all_boxes.append(torch.tensor([
                x0_f / self.size,
                y0_f / self.size,
                x1_f / self.size,
                y1_f / self.size,
            ], dtype=torch.float32))

            text = obj["detail_description"]
            all_text.append(text)

            out1 = self.tokenizer(
                text,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            out2 = self.tokenizer_2(
                text,
                max_length=self.tokenizer_2.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            all_obj_ids.append(out1["input_ids"])
            all_obj_ids_2.append(out2["input_ids"])
            all_obj_attention_mask.append(out1["attention_mask"])

        cap_out1 = self.tokenizer(
            global_caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        cap_out2 = self.tokenizer_2(
            global_caption,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = cap_out1["input_ids"]
        text_input_ids_2 = cap_out2["input_ids"]
        attention_mask = cap_out1["attention_mask"]

        # Ensure each sample supplies at least one local condition.
        if len(all_obj_ids) == 0:
            all_obj_ids.append(text_input_ids)
            all_obj_ids_2.append(text_input_ids_2)
            all_obj_attention_mask.append(attention_mask)
            all_boxes.append(torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32))
            all_text.append(global_caption)

        # Sample over-capacity layouts without changing the relative order of
        # the selected instances.
        if len(all_obj_ids) > self.max_obj:
            original_count = len(all_obj_ids)
            indices = torch.randperm(original_count)[:self.max_obj].tolist()
            indices.sort()
            print(
                f"[OccluRank warning] {os.path.basename(row['image_path'])}: "
                f"{original_count} valid instances exceed max_obj={self.max_obj}; "
                "sampling a subset while preserving relative order."
            )
            all_text = [all_text[i] for i in indices]
            all_boxes = [all_boxes[i] for i in indices]
            all_obj_ids = [all_obj_ids[i] for i in indices]
            all_obj_ids_2 = [all_obj_ids_2[i] for i in indices]
            all_obj_attention_mask = [all_obj_attention_mask[i] for i in indices]

        return {
            "images": image_tensor,
            "caption": global_caption,
            "all_text": all_text,
            "file_name": os.path.basename(row["image_path"]),
            "boxes": all_boxes,
            "text_input_ids": text_input_ids,
            "text_input_ids_2": text_input_ids_2,
            "attention_mask": attention_mask,
            "all_obj_ids": all_obj_ids,
            "all_obj_ids_2": all_obj_ids_2,
            "all_obj_attention_mask": all_obj_attention_mask,
        }


def collate_fn(batch):
    images = torch.stack([item["images"] for item in batch])
    file_names = [item["file_name"] for item in batch]
    caption = [item["caption"] for item in batch]
    text_input_ids = torch.cat([item["text_input_ids"] for item in batch], dim=0)
    text_input_ids_2 = torch.cat([item["text_input_ids_2"] for item in batch], dim=0)
    attention_mask = torch.cat([item["attention_mask"] for item in batch], dim=0)
    boxes = [item["boxes"] for item in batch]
    all_text = [item["all_text"] for item in batch]
    all_obj_ids = [item["all_obj_ids"] for item in batch]
    all_obj_ids_2 = [item["all_obj_ids_2"] for item in batch]
    all_obj_attention_mask = [item["all_obj_attention_mask"] for item in batch]
    return {
        "images": images,
        "file_names": file_names,
        "caption": caption,
        "all_text": all_text,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2,
        "attention_mask": attention_mask,
        "all_boxes": boxes,
        "all_obj_ids": all_obj_ids,
        "all_obj_ids_2": all_obj_ids_2,
        "all_obj_attention_mask": all_obj_attention_mask,
    }


def create_dataloader(
    tokenizer,
    tokenizer_2,
    parquet_path,
    image_root,
    img_size=1024,
    batch_size=4,
    num_workers=4,
    max_obj=5,
    min_box_size=0.01,
):
    dataset = OccluLayoutTrainDataset(
        parquet_path=parquet_path,
        image_root=image_root,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        size=img_size,
        max_obj=max_obj,
        min_box_size=min_box_size,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
