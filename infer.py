import torch

from torchvision.transforms.functional import to_pil_image
import os
import datetime
import matplotlib.pyplot as plt
# from transformers import RTDetrForObjectDetection, RTDetrImageProcessor, pipeline
import mlflow
import numpy as np
from utils import get_model
from transformers import TrainingArguments
from data import LizaDataset, get_transforms
from transformers import Trainer
from dataclasses import dataclass
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from utils.load_to_mlflow import log_model
from utils import convert_bbox_yolo_to_pascal, collate_fn

from config import (
    MLFLOW_URI,
    PROJECT_NAME,
    MODEL_NAME,
    DATASET_PATH,
)


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def plot_results(pil_img, scores, labels, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax), c in zip(scores, labels, boxes, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{model.config.id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    mlflow_uri = MLFLOW_URI
    project_name = PROJECT_NAME
    model_name = MODEL_NAME

    DEVICE = 'cuda'

    pipeline_ = get_model(mlflow_uri, project_name, model_name.split('@')[0] + '@trained')
    model, image_processor = pipeline_.model, pipeline_.image_processor

    dataset_path = DATASET_PATH
    dataset = LizaDataset(dataset_path, image_processor=image_processor, transforms=None)

    model = model.to(DEVICE)
    for n_image in range(len(dataset)):
        inputs = dataset.__getitem__(n_image)
        image = inputs['pixel_values']
        with torch.no_grad():
            outputs = model(image.to(DEVICE).unsqueeze(0))

        _, width, height = image.size()
        postprocessed_outputs = image_processor.post_process_object_detection(
            outputs,
            target_sizes=[(height, width)],
            threshold=0.3
        )
        results = postprocessed_outputs[0]
        plot_results(image.numpy().transpose((1,2,0)), results['scores'].tolist(), results['labels'].tolist(), results['boxes'].tolist())
