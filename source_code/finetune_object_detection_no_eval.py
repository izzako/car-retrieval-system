
from huggingface_hub import login
from datasets import load_dataset
import albumentations as A
from transformers import AutoImageProcessor
from transformers import TrainingArguments
from transformers import Trainer
from huggingface_hub import hf_hub_download
import numpy as np
import torch
from processor import helpers
from transformers import AutoModelForObjectDetection
import os
from dataclasses import dataclass
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from functools import partial
import json
import wandb

login(token=os.environ.get('HF_TOKEN'))  # Replace with your Hugging Face token

# Login to Hugging Face Hub

DATASET_REPO = "izzako/IDD_Detection_CPPE5"
MODEL_NAME = "microsoft/conditional-detr-resnet-50"  # or "facebook/detr-resnet-50"
HUB_MODEL_ID = "detr-resnet-50-finetuned-IDD_Detection"
IMAGE_SIZE = 480

MAX_SIZE = IMAGE_SIZE

id2label_file_path = hf_hub_download(
    repo_id="izzako/IDD_Detection_CPPE5",
    repo_type="dataset",
    filename="id2label.json"
)
with open(id2label_file_path, "r") as f:
    id2label = json.load(f)
id2label = {int(k): v for k, v in id2label.items()}  # Ensure keys are integers
label2id = {v: k for k, v in id2label.items()}


image_processor=AutoImageProcessor.from_pretrained(MODEL_NAME)
# Define the augmentations and transformations for training and validation datasets
train_augment_and_transform = A.Compose(
    [
        A.Resize(480, 480),
        A.HorizontalFlip(p=1.0),
        A.RandomBrightnessContrast(p=0.5),
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
)


# Make transform functions for batch and apply for dataset splits
train_transform_batch = partial(
    helpers.augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor
)

if __name__ == "__main__":
    
    wandb.login()
    print(f'Loading dataset from {DATASET_REPO}...')
    custom_dataset = load_dataset(DATASET_REPO,trust_remote_code=True)
   


    # Preproess the datasets with the defined transformations
    custom_dataset["train"] = custom_dataset["train"].with_transform(train_transform_batch)
    # custom_dataset["validation"] = custom_dataset["validation"].with_transform(validation_transform_batch)
    
    def collate_fn(batch):
        data = {}
        data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
        data["labels"] = [x["labels"] for x in batch]
        if "pixel_mask" in batch[0]:
            data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
        return data

    #train the model
    model = AutoModelForObjectDetection.from_pretrained(
        MODEL_NAME,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        f"{HUB_MODEL_ID}-outputs",
        num_train_epochs=2,
        fp16=True,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=2,
        dataloader_num_workers=1,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        weight_decay=1e-4,
        max_grad_norm=0.01,
        # metric_for_best_model="eval_map",
        # greater_is_better=True,
        # load_best_model_at_end=True,
        # eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        # eval_do_concat_batches=False,
        push_to_hub=True,
        report_to="wandb",
        hub_model_id=HUB_MODEL_ID,
        # eval_accumulation_steps=5,
        # eval_steps=50,
        logging_steps=100,
        # hub_strategy="end",
        resume_from_checkpoint=True,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=custom_dataset["train"],
        # eval_dataset=custom_dataset["validation"],
        processing_class=image_processor,
        data_collator=collate_fn,
        # compute_metrics=eval_compute_metrics_fn,
    )

    trainer.train()
    # Save the model to the hub
    kwargs = {
        "tags": ["vision", "object-detection"],
        "finetuned_from": MODEL_NAME,
        "dataset": "IDD 40K Object Detection Dataset",
    }
    trainer.push_to_hub(**kwargs)