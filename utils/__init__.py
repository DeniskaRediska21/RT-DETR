from .model_preprocessing import *

import torch
from transformers.image_transforms import center_to_corners_format


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
    pixel_values = []
    labels = []

    unannotated = False
    for x in batch:
        if len(x['labels']['class_labels']) > 0:
            pixel_values.append(x["pixel_values"])
            labels.append(x["labels"])
        else:
            if not unannotated:
                pixel_values.append(x["pixel_values"])
                labels.append(x["labels"])
                unannotated = True

    data = {'pixel_values': torch.stack(pixel_values), 'labels': labels}

    return data
