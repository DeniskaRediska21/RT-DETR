import torch
# from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, pipeline
import mlflow
import numpy as np
from utils import get_model
from transformers import TrainingArguments
from data import LizaDataset
from transformers import Trainer
from dataclasses import dataclass
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers.image_transforms import center_to_corners_format

from config import (
    MLFLOW_URI,
    PROJECT_NAME,
    MODEL_NAME,
    DATASET_PATH,
    EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
)


def convert_bbox_yolo_to_pascal(boxes, image_size):
    """
    Convert bounding boxes from YOLO format (x_center, y_center, width, height) in range [0, 1]
    to Pascal VOC format (x_min, y_min, x_max, y_max) in absolute coordinates.

    Args:
        boxes (torch.Tensor): Bounding boxes in YOLO format
        image_size (Tuple[int, int]): Image size in format (height, width)

    Returns:
        torch.Tensor: Bounding boxes in Pascal VOC format (x_min, y_min, x_max, y_max)
    """
    # convert center to corners format
    boxes = center_to_corners_format(boxes)

    # convert to absolute coordinates
    height, width = image_size[0], image_size[0]
    boxes = boxes * torch.tensor([[width, height, width, height]])

    return boxes


def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    return data


@dataclass
class ModelOutput:
    logits: torch.Tensor
    pred_boxes: torch.Tensor


class MAPEvaluator:

    def __init__(self, image_processor, threshold=0.00, id2label=None):
        self.image_processor = image_processor
        self.threshold = threshold
        self.id2label = id2label

    def collect_image_sizes(self, targets):
        """Collect image sizes across the dataset as list of tensors with shape [batch_size, 2]."""
        image_sizes = []
        for batch in targets:
            tmp = torch.tensor(np.array([x["size"] for x in batch]))
            batch_image_sizes = []
            for batch_image_size in tmp:
                size = batch_image_size if len(batch_image_size) == 2 else [batch_image_size[0], batch_image_size[0]]
                batch_image_sizes.append(size)
            image_sizes.append(batch_image_sizes)
        return image_sizes

    def collect_targets(self, targets, image_sizes):
        post_processed_targets = []
        for target_batch, image_size_batch in zip(targets, image_sizes):
            for target, size in zip(target_batch, image_size_batch):
                boxes = torch.tensor(target["boxes"])
                boxes = convert_bbox_yolo_to_pascal(boxes, size)
                labels = torch.tensor(target["class_labels"])
                post_processed_targets.append({"boxes": boxes, "labels": labels})
        return post_processed_targets

    def collect_predictions(self, predictions, image_sizes):
        post_processed_predictions = []
        for batch, target_sizes in zip(predictions, image_sizes):
            batch_logits, batch_boxes = batch[1], batch[2]
            output = ModelOutput(logits=torch.tensor(batch_logits), pred_boxes=torch.tensor(batch_boxes))
            post_processed_output = self.image_processor.post_process_object_detection(output,
                                                                                       threshold=self.threshold,
                                                                                       target_sizes=target_sizes)
            post_processed_predictions.extend(post_processed_output)
        return post_processed_predictions

    @torch.no_grad()
    def __call__(self, evaluation_results):

        predictions, targets = evaluation_results.predictions, evaluation_results.label_ids

        # targets = [[tar for tar in target if len(tar) == 2] for target in targets]
        # targets = [target for target in targets if len(target) > 0]

        image_sizes = self.collect_image_sizes(targets)
        post_processed_targets = self.collect_targets(targets, image_sizes)
        post_processed_predictions = self.collect_predictions(predictions, image_sizes)

        evaluator = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
        evaluator.warn_on_many_detections = False
        evaluator.update(post_processed_predictions, post_processed_targets)

        metrics = evaluator.compute()

        # Replace list of per class metrics with separate metric for each class
        # classes = metrics.pop("classes")
        map_per_class = metrics.pop("map_per_class")
        metrics["map_person"] = map_per_class

        metrics = {k: round(v.item(), 4) for k, v in metrics.items()}

        return metrics


mlflow_uri = MLFLOW_URI
project_name = PROJECT_NAME
model_name = MODEL_NAME
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment(project_name)
lengths = [0.7, 0.3]

DEVICE = 'cuda'

pipeline_ = get_model(mlflow_uri, project_name, model_name)
model, image_processor = pipeline_.model, pipeline_.image_processor

dataset_path = DATASET_PATH
dataset = LizaDataset(dataset_path, image_processor=image_processor, transforms=None)
train_dataset, validation_dataset = torch.utils.data.random_split(dataset, lengths)

model = model.to(DEVICE)

categories = ['person']
id2label = {index: x for index, x in enumerate(categories, start=0)}
label2id = {v: k for k, v in id2label.items()}

eval_compute_metrics_fn = MAPEvaluator(image_processor=image_processor, threshold=0.01, id2label=id2label)

training_args = TrainingArguments(
    output_dir="rtdetr-r50-cppe5-finetune",
    num_train_epochs=EPOCHS,
    max_grad_norm=0.1,
    learning_rate=LEARNING_RATE,
    warmup_steps=300,
    per_device_train_batch_size=BATCH_SIZE,
    dataloader_num_workers=4,
    metric_for_best_model="eval_map",
    greater_is_better=True,
    load_best_model_at_end=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
    eval_do_concat_batches=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=image_processor,
    data_collator=collate_fn,
    compute_metrics=eval_compute_metrics_fn,
)

trainer.train()
