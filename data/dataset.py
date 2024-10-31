# import sys
import glob
import os
import re
from natsort import natsorted
import torch
from torch.utils.data import Dataset
import imagesize
import mlflow
from torchvision.io import decode_image
from torchvision import tv_tensors
from pathlib import Path
import numpy as np
import sys
sys.path.append('..')
sys.path.append('../upgreat_detector')
sys.path.append('../RT-DETR')
from utils import get_model
from config import MLFLOW_URI, PROJECT_NAME, MODEL_NAME_VAL


def plot_image(pil_img, boxes):
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    H, W, _ = np.shape(pil_img)
    for (xcenter, ycenter, w, h), c in zip(boxes, colors):
        ax.add_patch(plt.Rectangle((W * xcenter, H * ycenter), W * w, H * h,
                                   fill=False, color=c, linewidth=3))
    plt.axis('off')
    plt.show()


def format_to_coco(image_id, annotations, image_shape, num_pedestrian):
    formated = []
    for annotation in annotations:
        if len(annotation) == 0:
            continue
        _, h, w = image_shape
        category, xcenter, ycenter, bw, bh = annotation
        xmin, ymin = xcenter - bw/2, ycenter - bh/2

        formated.append({
            "image_id": image_id,
            "category_id": num_pedestrian,
            "bbox": [(xmin) * w, (ymin) * h, (bw)* w,(bh) * h],
            "iscrowd": 0,
            "area": bh * bw * h * w,
        })

    return {"image_id": image_id, "annotations": formated}


class LizaDataset(Dataset):

    def __init__(self, dataset_path, image_processor, transforms=None, num_pedestrian=0):
        self.annotations = glob.glob(os.path.join(dataset_path, '*.txt'), recursive=True)
        self.images = [file for file in glob.glob(os.path.join(dataset_path, '*'), recursive=True) if re.match(r'(.*\.jpg)|(.*\.JPG)', file)]

        self.images = natsorted(self.images)
        self.annotations = natsorted(self.annotations)

        self.image_ids = list(range(len(self.images)))
        self.annotation_ids = list(range(len(self.annotations)))
        
        # self.image_ids = [int(Path(image).stem) for image in self.images]
        # self.annotation_ids = [int(Path(annotation).stem) for annotation in self.annotations]
        # _, self.annotations = np.array(sorted(zip(self.annotation_ids, self.annotations))).T
        # _, self.images = np.array(sorted(zip(self.image_ids, self.images))).T
        # self.image_ids = sorted(self.image_ids)

        self.image_processor = image_processor
        self.transforms = transforms
        self.num_pedestrian = num_pedestrian

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = decode_image(self.images[idx])
        with open(self.annotations[idx], 'r') as file:
            annotations = file.read().splitlines()
            for index in range(len(annotations)):
                annotations[index] = eval(f"[{annotations[index].replace(' ', ',')}]")

        formated_annotations = format_to_coco(self.image_ids[idx], annotations, image.shape,num_pedestrian=self.num_pedestrian)


        # Apply the torchvision transforms if provided
        if self.transforms is not None:
            # bboxes = tv_tensors.BoundingBoxes(
            #     [dict_['bbox'] for dict_ in formated_annotations['annotations']],
            #     format="CXCYWH",
            #     canvas_size=image.shape[-2:],
            # )
            # if bboxes.size()[1] != 0:
            #     image, out_boxes = self.transforms(image, bboxes)

            #     for index, box in enumerate(out_boxes):
            #         formated_annotations['annotations'][index]['bbox'] = box
            pass

        result = self.image_processor(images=image, annotations=formated_annotations, return_tensors="pt")
        # Image processor expands batch dimension, lets squeeze it
        result = {k: v[0] for k, v in result.items()}

        return result


if __name__ == "__main__":
    from config import DATASET_PATH
    import matplotlib.pyplot as plt
    from transforms import get_transforms
    mlflow_uri = 'http://localhost:5000'
    project_name = 'LIZA'
    model_name = 'LIZA-detector@base'
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(project_name)

    DEVICE = 'cuda'

    pipline = get_model(mlflow_uri, project_name, model_name)
    model, image_processor = pipline.model, pipline.image_processor
    dataset = LizaDataset(DATASET_PATH, image_processor=image_processor, transforms=None)
    for index in range(len(dataset)):
        a = dataset.__getitem__(index)
        plot_image(a['pixel_values'].numpy().transpose((1,2,0)), a['labels']['boxes'])
    pass
