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
from torchvision.io.image import decode_jpeg, read_file
from torchvision import tv_tensors
from pathlib import Path
import numpy as np
import sys
sys.path.append('..')
sys.path.append('../upgreat_detector')
sys.path.append('../RT-DETR')
from utils import get_transformers_model
from config import MLFLOW_URI, PROJECT_NAME, MODEL_NAME_VAL, VALIDATION_DATASET_PATH, CLASS_DATASET_PATH, DEVICE


def plot_image(pil_img, boxes):
    COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
              [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    H, W, _ = np.shape(pil_img)
    for (xcenter, ycenter, w, h), c in zip(boxes, colors):
        ax.add_patch(plt.Rectangle((W * (xcenter - w/2), H * (ycenter - h/2)), W * w, H * h,
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

    def __init__(self, dataset_path, image_processor, transforms=None, num_pedestrian=0, training=True):
        self.annotations = [file for file in glob.glob(os.path.join(dataset_path, '**'), recursive=True) if re.match(r'(.*\.txt)', file)]
        self.images = [file for file in glob.glob(os.path.join(dataset_path, '**'), recursive=True) if re.match(r'(.*\.jpg)|(.*\.JPG)', file)]

        self.images = natsorted(self.images)
        self.annotations = natsorted(self.annotations)

        self.image_ids = list(range(len(self.images)))
        self.annotation_ids = list(range(len(self.annotations)))
        
        self.image_processor = image_processor
        self.transforms = transforms
        self.num_pedestrian = num_pedestrian
        self.training = training
        # if not self.training:
        #     sizes = [imagesize.get(image) for image in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = decode_image(self.images[idx])
        # im = read_file(self.images[idx])
        # image = decode_jpeg(im, device=DEVICE)

        with open(self.annotations[idx], 'r') as file:
            annotations = file.read().splitlines()
            for index in range(len(annotations)):
                annotations[index] = eval(f"[{annotations[index].replace(' ', ',')}]")

        formated_annotations = format_to_coco(self.image_ids[idx], annotations, image.shape,num_pedestrian=self.num_pedestrian)


        # Apply the torchvision transforms if provided
        # if self.training:
        if self.transforms is not None:
            bboxes = tv_tensors.BoundingBoxes(
                [dict_['bbox'] for dict_ in formated_annotations['annotations']],
                format="XYWH",
                canvas_size=image.shape[-2:],
            )
            if bboxes.size()[1] != 0:
                image, out_boxes = self.transforms(image, bboxes)

                for index, box in enumerate(out_boxes):
                    formated_annotations['annotations'][index]['bbox'] = box
            # pass

        result = self.image_processor(images=image, annotations=formated_annotations, return_tensors="pt")
        # else: 
        #     result = {'pixel_values': (image/255).unsqueeze(0).to(torch.bfloat16)}
          # result = self.image_processor(images=image, return_tensors="pt")
            
        result = {k: v[0] for k, v in result.items()}

        return result


class LizaClassDataset(Dataset):

    def __init__(self, dataset_path, image_processor, transforms=None):
        self.images_humans = [file for file in glob.glob(os.path.join(dataset_path, 'human', '*'), recursive=False) if re.match(r'(.*\.jpg)|(.*\.JPG)', file)]

        self.images_na = [file for file in glob.glob(os.path.join(dataset_path, 'not-human', '*'), recursive=False) if re.match(r'(.*\.jpg)|(.*\.JPG)', file)]

        self.annotations_na = list(torch.zeros(len(self.images_na), dtype=int))
        self.annotations_humans = list(torch.ones(len(self.images_humans), dtype=int))

        self.images = self.images_humans + self.images_na
        self.annotations = self.annotations_humans + self.annotations_na

        self.annotation = torch.tensor(self.annotations)

        self.transforms = transforms
        self.image_processor = image_processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = decode_image(self.images[idx])
        image = self.image_processor(image, return_tensors='pt')['pixel_values'][0]
        target = self.annotations[idx]

        return image, target


if __name__ == "__main__":
    dataset_class = LizaClassDataset(CLASS_DATASET_PATH)

    from config import DATASET_PATH
    import matplotlib.pyplot as plt
    from transforms import get_transforms
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(PROJECT_NAME)

    DEVICE = 'cuda'

    pipline = get_model(MLFLOW_URI, PROJECT_NAME, MODEL_NAME_VAL)
    model, image_processor = pipline.model, pipline.image_processor
    dataset = LizaDataset(DATASET_PATH, image_processor=image_processor, transforms=None)
    for index in range(len(dataset)):
        a = dataset.__getitem__(index)
        plot_image(a['pixel_values'].numpy().transpose((1,2,0)), a['labels']['boxes'])
    pass
