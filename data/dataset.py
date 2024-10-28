# import sys
import glob
import os
import re
import torch
from torch.utils.data import Dataset
import imagesize
import mlflow
from torchvision.io import decode_image
from pathlib import Path
import numpy as np

from utils import get_model


def split_sliding_window(image, formated_annotation, inference_size, overlap=0.2):
    _, h, w = image.shape
    step = int(inference_size * (1 - overlap))

    images = []
    formated_annotations = []

    for y in range(0, h, step):
        for x in range(0, w, step):
            window = torch.zeros((3, inference_size, inference_size), dtype=image.dtype)

            copy_h = min(inference_size, h - y)
            copy_w = min(inference_size, w - x)

            window[:, :copy_h, :copy_w] = image[:, y:y + copy_h, x:x + copy_w]
            images.append(window)

            window_annotations = []
            for ann in formated_annotation['annotations']:
                bbox = ann['bbox']
                new_bbox = [
                    max(0, bbox[0] - x),
                    max(0, bbox[1] - y),
                    min(copy_w, bbox[2] + bbox[0] - x) - max(0, bbox[0] - x),
                    min(copy_h, bbox[3] + bbox[1] - y) - max(0, bbox[1] - y)
                ]
                
                if new_bbox[2] > 0 and new_bbox[3] > 0:
                    new_ann = ann.copy()
                    new_ann['bbox'] = new_bbox
                    window_annotations.append(new_ann)

            formated_annotations.append({
                'image_id': formated_annotation['image_id'],
                'annotations': window_annotations
            })

    return images, formated_annotations


def format_to_coco(image_id, annotations, image_shape):
    formated = []
    for annotation in annotations:
        if len(annotation) == 0:
            continue
        _, h, w = image_shape
        category, xmin, ymin, bw, bh = annotation
        bbox = [xmin, ymin, bw, bh]
        formated.append({
            "image_id": image_id,
            "category_id": category,
            "bbox": [xmin * w, ymin * h, bw * w, bh * h],
            "iscrowd": 0,
            "area": bbox[2] * bbox[3],
        })

    return {"image_id": image_id, "annotations": formated}


class LizaDataset(Dataset):

    def __init__(self, dataset_path, image_processor, transforms=None, inference_size=640, overlap=0.2, do_slide=False):
        self.inference_size = inference_size
        self.overlap = overlap
        self.annotations = glob.glob(os.path.join(dataset_path, '*.txt'), recursive=True)
        self.annotation_ids = [int(Path(annotation).stem) for annotation in self.annotations]
        self.images = [file for file in glob.glob(os.path.join(dataset_path, '*'), recursive=True) if re.match(r'(.*\.jpg)|(.*\.JPG)', file)]
        self.image_ids = [int(Path(image).stem) for image in self.images]

        _, self.annotations = np.array(sorted(zip(self.annotation_ids, self.annotations))).T
        _, self.images = np.array(sorted(zip(self.image_ids, self.images))).T
        self.image_ids = sorted(self.image_ids)
        self.image_processor = image_processor
        self.transforms = transforms
        self.image_sizes = []
        print('Loading image sizes...')
        for index, image in enumerate(self.images):
            if index % 1000 == 0:
                print(f'{index} / {len(self.images)}', end='\r')
            self.image_sizes.append([*imagesize.get(image)])
        self.backlog = []

        repetitions = [
            int(h / (inference_size * (1 - overlap))) * int(w / (inference_size * (1 - overlap)))
            for w, h in self.image_sizes
        ]

        repeated_images = []
        [repeated_images.append(image) for image, reps in zip(self.images, repetitions) for _ in range(reps)]
        repeated_image_ids = []
        [
            repeated_image_ids.append(image_id)
            for image_id, reps in zip(self.image_ids, repetitions)
            for _ in range(reps)
        ]
        repeated_annotations = []
        [
            repeated_annotations.append(annotation)
            for annotation, reps in zip(self.annotations, repetitions)
            for _ in range(reps)
        ]

        self.images = repeated_images  # comment out if window not impl
        self.image_ids = repeated_image_ids  # comment out if window not impl
        self.annotations = repeated_annotations  # comment out if window not impl
        self.do_slide = do_slide

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        if not self.backlog:
            image = decode_image(self.images[idx])
            with open(self.annotations[idx], 'r') as file:
                annotations = file.read().splitlines()
                for index in range(len(annotations)):
                    annotations[index] = eval(f"[{annotations[index].replace(' ', ',')}]")

            formated_annotations = format_to_coco(self.image_ids[idx], annotations, image.shape)

            if self.do_slide:
                image, formated_annotations = split_sliding_window(
                     image,
                     formated_annotations,
                     inference_size=self.inference_size,
                     overlap=self.overlap
                )

            # Apply the image processor transformations: resizing, rescaling, normalization

            self.backlog = [
                [img, formated_annotetion] for img, formated_annotetion in zip(image, formated_annotations)
            ]

        image, formated_annotations = self.backlog.pop(0)

        # Apply the torchvision transforms if provided
        if self.transforms:
            image = self.transforms(image)

        result = self.image_processor(images=image, annotations=formated_annotations, return_tensors="pt")
        # Image processor expands batch dimension, lets squeeze it
        result = {k: v[0] for k, v in result.items()}

        return result


if __name__ == "__main__":
    mlflow_uri = 'http://localhost:5000'
    project_name = 'LIZA'
    model_name = 'LIZA-detector@base'
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(project_name)

    DEVICE = 'cuda'

    pipline = get_model(mlflow_uri, project_name, model_name)
    model, image_processor = pipline.model, pipline.image_processor

    dataset = LizaDataset(os.path.join('..', 'Dataset'), image_processor=image_processor, transforms=None)
    dataset.__getitem__(0)
    pass
