from torch.utils.data import Dataset
import mlflow
import glob
import os
from torchvision.io import decode_image
from pathlib import Path
import numpy as np
import sys 
sys.path.append('../')
sys.path.append('../RT-DETR')
from model_preprocessing import get_model



def format_to_coco(image_id, annotations, image_shape):
    formated = []
    for annotation in annotations:
        if len(annotation) == 0:
            continue
        _, h, w = image_shape
        category, xmin, ymin, bw, bh = annotation
        bbox = [xmin, ymin, bw, bh]
        formated.append(
            {
                "image_id": image_id,
                "category_id": category,
                "bbox": [xmin * w, ymin * h, bw * w, bh * h],
                "iscrowd": 0,
                "area": bbox[2] * bbox[3],
            }
        )

    return {"image_id": image_id, "annotations": formated}


class LizaDataset(Dataset):
    def __init__(self, dataset_path, image_processor, transforms):
        self.annotations = glob.glob(os.path.join(dataset_path, '*.txt'))
        self.annotation_ids = [int(Path(annotation).stem) for annotation in self.annotations]
        self.images = glob.glob(os.path.join(dataset_path, '*.jpg'))
        self.image_ids = [int(Path(image).stem) for image in self.images]

        _, self.annotations= np.array(sorted(zip(self.annotation_ids, self.annotations))).T
        _, self.images = np.array(sorted(zip(self.image_ids, self.images))).T
        self.image_ids = sorted(self.image_ids)
        self.image_processor = image_processor
        # self.image_ids = [int(''.join([ch for ch in name if 47<ord(ch)<58])) for name in self.images]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image = decode_image(self.images[idx])
        with open(self.annotations[idx], 'r') as file:
            annotations = file.read().splitlines()
            for index in range(len(annotations)):
                annotations[index]= eval(f"[{annotations[index].replace(' ', ',')}]")

        formated_annotations = format_to_coco(self.image_ids[idx], annotations, image.shape)
        # Apply the image processor transformations: resizing, rescaling, normalization
        result = self.image_processor(
            images=image, annotations=formated_annotations, return_tensors="pt"
        )

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
    
