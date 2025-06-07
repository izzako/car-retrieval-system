
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

# Login to Hugging Face Hub
login(token=os.environ['HF_TOKEN'])  # Replace with your Hugging Face token

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


image_processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME,
    do_resize=True,
    size={"max_height": MAX_SIZE, "max_width": MAX_SIZE},
    do_pad=True,
    pad_size={"height": MAX_SIZE, "width": MAX_SIZE},
)

# Define the augmentations and transformations for training and validation datasets
train_augment_and_transform = A.Compose(
    [
        A.Perspective(p=0.1),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.1),
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
)

validation_transform = A.Compose(
    [A.NoOp()],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
)

# Make transform functions for batch and apply for dataset splits
train_transform_batch = partial(
    helpers.augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor
)
validation_transform_batch = partial(
    helpers.augment_and_transform_batch, transform=validation_transform, image_processor=image_processor
)


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


@torch.no_grad()
def compute_metrics(evaluation_results, image_processor, threshold=0.0, id2label=None):
    """
    Compute mean average mAP, mAR and their variants for the object detection task.

    Args:
        evaluation_results (EvalPrediction): Predictions and targets from evaluation.
        threshold (float, optional): Threshold to filter predicted boxes by confidence. Defaults to 0.0.
        id2label (Optional[dict], optional): Mapping from class id to class name. Defaults to None.

    Returns:
        Mapping[str, float]: Metrics in a form of dictionary {<metric_name>: <metric_value>}
    """

    predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

    # For metric computation we need to provide:
    #  - targets in a form of list of dictionaries with keys "boxes", "labels"
    #  - predictions in a form of list of dictionaries with keys "boxes", "scores", "labels"

    image_sizes = []
    post_processed_targets = []
    post_processed_predictions = []

    # Collect targets in the required format for metric computation
    for batch in targets:
        # collect image sizes, we will need them for predictions post processing
        batch_image_sizes = torch.tensor(np.array([x["orig_size"] for x in batch]))
        image_sizes.append(batch_image_sizes)
        # collect targets in the required format for metric computation
        # boxes were converted to YOLO format needed for model training
        # here we will convert them to Pascal VOC format (x_min, y_min, x_max, y_max)
        for image_target in batch:
            boxes = torch.tensor(image_target["boxes"])
            boxes = helpers.convert_bbox_yolo_to_pascal(boxes, image_target["orig_size"])
            labels = torch.tensor(image_target["class_labels"])
            post_processed_targets.append({"boxes": boxes, "labels": labels})

    # Collect predictions in the required format for metric computation,
    # model produce boxes in YOLO format, then image_processor convert them to Pascal VOC format
    for batch, target_sizes in zip(predictions, image_sizes):
        batch_logits, batch_boxes = batch[1], batch[2]
        output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
        post_processed_output = image_processor.post_process_object_detection(
            output, threshold=threshold, target_sizes=target_sizes
        )
        post_processed_predictions.extend(post_processed_output)

    # Compute metrics
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
    metric.update(post_processed_predictions, post_processed_targets)
    metrics = metric.compute()

    # Replace list of per class metrics with separate metric for each class
    classes = metrics.pop("classes")
    map_per_class = metrics.pop("map_per_class")
    mar_100_per_class = metrics.pop("mar_100_per_class")
    for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
        class_name = id2label[class_id.item()] if id2label is not None else class_id.item()
        metrics[f"map_{class_name}"] = class_map
        metrics[f"mar_100_{class_name}"] = class_mar

    metrics = {k: round(v.item(), 4) for k, v in metrics.items()}
    return metrics

if __name__ == "__main__":
    wandb.login()
    print(f'Loading dataset from {DATASET_REPO}...')
    custom_dataset = load_dataset(DATASET_REPO,trust_remote_code=True)
    # custom_dataset = load_dataset("data/IDD_Detection_CPPE5",data_dir='data/IDD_Detection_CPPE5')
    if "validation" not in custom_dataset:
        split = custom_dataset["train"].train_test_split(0.15, seed=24)
        custom_dataset["train"] = split["train"]
        custom_dataset["validation"] = split["test"]


    eval_compute_metrics_fn = partial(
        compute_metrics, image_processor=image_processor, id2label=id2label, threshold=0.0
    )

    # Preproess the datasets with the defined transformations
    custom_dataset["train"] = custom_dataset["train"].with_transform(train_transform_batch)
    custom_dataset["validation"] = custom_dataset["validation"].with_transform(validation_transform_batch)


    #train the model
    model = AutoModelForObjectDetection.from_pretrained(
        MODEL_NAME,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )


    training_args = TrainingArguments(
        "{HUB_MODEL_ID}-outputs",
        num_train_epochs=30,
        fp16=False,
        per_device_train_batch_size=8,
        dataloader_num_workers=4,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        weight_decay=1e-4,
        max_grad_norm=0.01,
        metric_for_best_model="eval_map",
        greater_is_better=True,
        load_best_model_at_end=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        push_to_hub=True,
        report_to="wandb",
        hub_model_id=HUB_MODEL_ID,
        save_steps=100,
        eval_accumulation_steps=5,
        eval_steps=100,
        logging_steps=100,
        hub_strategy="end",
        resume_from_checkpoint=True,
        logging_dir="./logs",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=custom_dataset["train"],
        eval_dataset=custom_dataset["validation"],
        processing_class=image_processor,
        data_collator=helpers.collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )

    trainer.train()
    # Save the model to the hub
    kwargs = {
        "repo_id": "izzako/detr_finetuned_idd_cppe5",
        "repo_type": "model",
        "private": True,
        "use_auth_token": os.environ['HF_TOKEN']
    }
    trainer.push_to_hub(**kwargs)