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
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import requests
from config import (
    MLFLOW_URI,
    PROJECT_NAME,
    MODEL_NAME,
    DATASET_PATH,
    VALIDATION_DATASET_PATH,
    INFERENCE_SIZE,
    OVERLAP,
)


# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]


def plot_results(pil_img, postprocessed_outputs, additions, width, height):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for output, addition in zip(postprocessed_outputs, additions):
        scores = output['scores'].to('cpu')
        labels = output['labels'].to('cpu')
        boxes = output['boxes'].to('cpu')
        boxes[:,0] += addition[1]
        boxes[:,1] += addition[0]
        boxes[:,2] += addition[1]
        boxes[:,3] += addition[0]

        colors = COLORS * 100
        for score, label, (xmin, ymin, xmax, ymax), c in zip(scores, labels, boxes, colors):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=c, linewidth=3))
            text = f'{model.config.id2label[int(label)]}: {score:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


if __name__ == '__main__':
    mlflow_uri = MLFLOW_URI
    project_name = PROJECT_NAME
    model_name = MODEL_NAME

    DEVICE = 'cuda'

    pipeline_ = get_model(mlflow_uri, project_name, model_name.split('@')[0] + '@trained')
    # pipeline_ = get_model(mlflow_uri, project_name, model_name.split('@')[0] + '@base_detr_visdrone')
    model, image_processor = pipeline_.model, pipeline_.image_processor

    image_processor.do_resize = False
    image_processor.do_normalize = False
    image_processor.do_pad = False

    dataset_path = VALIDATION_DATASET_PATH
    dataset = LizaDataset(dataset_path, image_processor=image_processor, transforms=None)

    model = model.to(DEVICE)
    for n_image in range(len(dataset)):
        inputs = dataset.__getitem__(n_image)
        image = inputs['pixel_values']
        _, height, width = image.size()
        step_size = int(INFERENCE_SIZE * (1 - OVERLAP))

        # Sliding window
        h_range = np.array(range(0, height, step_size))
        w_range = np.array(range(0, width, step_size))

        h_range[-1] = height - INFERENCE_SIZE if h_range[-1] + INFERENCE_SIZE > height else h_range[-1]
        w_range[-1] = width - INFERENCE_SIZE if w_range[-1] + INFERENCE_SIZE > width else w_range[-1]

        subs = [image[:, ymin: ymin + INFERENCE_SIZE, xmin: xmin + INFERENCE_SIZE] for ymin in h_range for xmin in w_range]
        additions = torch.tensor([(ymin, xmin) for ymin in h_range for xmin in w_range])

        subs = torch.stack(subs)
        batch_size = 1
        batches = torch.split(subs, batch_size)
        batched_additions = torch.split(additions, batch_size)

        outputs_all = AttrDict()
        outputs_all.logits = []
        outputs_all.pred_boxes = []
        outputs_all.last_hidden_state = []
        outputs_all.encoder_last_hidden_state = []

        for batch, addition in zip(batches, batched_additions):
            with torch.no_grad():
                outputs = model(batch.to(DEVICE))
                # for index_in_batch in range(len(batch)):
                #     outputs['pred_boxes'][index_in_batch,:,0] += addition[index_in_batch, 1] / width
                #     outputs['pred_boxes'][index_in_batch,:,1] += addition[index_in_batch, 0] / height
                [outputs_all[key].append(val) for key, value in outputs.items() for val in value]
                pass

        for key, value in outputs_all.items():
            outputs_all[key] = torch.stack(value)

        target_sizes = [(INFERENCE_SIZE, INFERENCE_SIZE)] * len(outputs_all['pred_boxes']),
        postprocessed_outputs = image_processor.post_process_object_detection(
            outputs_all,
            target_sizes=target_sizes[0],
            threshold=0.8
        )

        plot_results(image.numpy().transpose((1, 2, 0)), postprocessed_outputs, additions, width, height)
