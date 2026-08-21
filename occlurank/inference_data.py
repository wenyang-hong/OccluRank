import json
import os
import random

import torch


ANNOTATION_CANVAS_SIZE = 768


class OccluLayoutBenchmarkDataset(torch.utils.data.Dataset):
    """Read ordered layouts from OccluLayout-Bench JSON files.

    The benchmark JSON files store COCO-style ``xywh`` boxes on a fixed
    768x768 annotation canvas. They are converted to normalized ``xyxy``
    coordinates before being passed to the model. The annotation list order is
    interpreted as the ordinal foreground-to-background sequence.
    """

    def __init__(
        self,
        text_file,
        tokenizer,
        tokenizer_2,
        size=1024,
        max_obj=5,
        min_box_size=0.01,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.min_box_size = min_box_size
        self.max_obj = max_obj
        self.size = size

        with open(text_file, "r", encoding="utf-8") as f:
            self.layout_files = [line.strip() for line in f if line.strip()]

    def __getitem__(self, idx):
        json_path = self.layout_files[idx]
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        file_name = os.path.basename(json_path)
        caption = data["caption"]

        all_boxes = []
        all_obj_ids = []
        all_obj_ids_2 = []
        all_obj_attention_mask = []
        all_text = []

        for anno in data["annotations"]:
            x, y, w, h = anno["box"]
            text = anno["caption"] if isinstance(anno["caption"], str) else anno["caption"][0]

            x0 = max(0.0, min(1.0, x / ANNOTATION_CANVAS_SIZE))
            y0 = max(0.0, min(1.0, y / ANNOTATION_CANVAS_SIZE))
            x1 = max(0.0, min(1.0, (x + w) / ANNOTATION_CANVAS_SIZE))
            y1 = max(0.0, min(1.0, (y + h) / ANNOTATION_CANVAS_SIZE))

            norm_area = (x1 - x0) * (y1 - y0)
            if norm_area < self.min_box_size:
                continue

            all_boxes.append(torch.tensor([x0, y0, x1, y1], dtype=torch.float32))
            all_text.append(text)

            output1 = self.tokenizer(
                text,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            output2 = self.tokenizer_2(
                text,
                max_length=self.tokenizer_2.model_max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )

            all_obj_ids.append(output1.input_ids)
            all_obj_ids_2.append(output2.input_ids)
            all_obj_attention_mask.append(output1.attention_mask)

        output1 = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids = output1.input_ids
        attention_mask = output1.attention_mask

        output2 = self.tokenizer_2(
            caption,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids_2 = output2.input_ids

        # Ensure each sample supplies at least one local condition.
        if len(all_obj_ids) == 0:
            all_obj_ids.append(text_input_ids)
            all_obj_ids_2.append(text_input_ids_2)
            all_obj_attention_mask.append(attention_mask)
            all_boxes.append(torch.tensor([0.0, 0.0, 1.0, 1.0], dtype=torch.float32))

        # Sample over-capacity layouts without changing the relative order of
        # the selected instances.
        if len(all_obj_ids) > self.max_obj:
            original_count = len(all_obj_ids)
            indices = random.sample(range(original_count), self.max_obj)
            indices.sort()
            print(
                f"[OccluRank warning] {file_name}: {original_count} valid instances "
                f"exceed max_obj={self.max_obj}; sampling a subset while preserving relative order."
            )
            all_text = [all_text[i] for i in indices]
            all_obj_ids = [all_obj_ids[i] for i in indices]
            all_boxes = [all_boxes[i] for i in indices]
            all_obj_ids_2 = [all_obj_ids_2[i] for i in indices]
            all_obj_attention_mask = [all_obj_attention_mask[i] for i in indices]

        return {
            "caption": caption,
            "all_text": all_text,
            "file_name": file_name,
            "boxes": all_boxes,
            "text_input_ids": text_input_ids,
            "all_obj_ids": all_obj_ids,
            "all_obj_ids_2": all_obj_ids_2,
            "all_obj_attention_mask": all_obj_attention_mask,
            "text_input_ids_2": text_input_ids_2,
            "attention_mask": attention_mask,
        }

    def __len__(self):
        return len(self.layout_files)


def collate_fn(data):
    file_names = [example["file_name"] for example in data]
    caption = [example["caption"] for example in data]
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    text_input_ids_2 = torch.cat([example["text_input_ids_2"] for example in data], dim=0)
    attention_mask = torch.cat([example["attention_mask"] for example in data], dim=0)
    boxes = [example["boxes"] for example in data]
    all_text = [example["all_text"] for example in data]
    all_obj_ids = [example["all_obj_ids"] for example in data]
    all_obj_ids_2 = [example["all_obj_ids_2"] for example in data]
    all_obj_attention_mask = [example["all_obj_attention_mask"] for example in data]

    return {
        "caption": caption,
        "file_names": file_names,
        "all_text": all_text,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2,
        "attention_mask": attention_mask,
        "all_obj_ids": all_obj_ids,
        "all_obj_ids_2": all_obj_ids_2,
        "all_obj_attention_mask": all_obj_attention_mask,
        "all_boxes": boxes,
    }


def create_dataloader(
    tokenizer,
    tokenizer_2,
    img_size=1024,
    txt_file="infer.txt",
    batch_size=1,
    num_workers=0,
    max_obj=5,
    min_box_size=0.01,
):
    dataset = OccluLayoutBenchmarkDataset(
        txt_file,
        tokenizer,
        tokenizer_2,
        size=img_size,
        max_obj=max_obj,
        min_box_size=min_box_size,
    )
    return torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
    )
