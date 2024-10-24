from torch.utils.data import Dataset
import glob
import os
from torchvision.io import decode_image
from pathlib import Path
import numpy as np


def format_to_coco(image_id, annotations, image_shape):
    formated = []
    for category, *bbox in annotations:
        _, h, w = image_shape
        xmin, ymin, bw, bh = bbox
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
    def __init__(self, dataset_path, transforms):
        self.annotations = glob.glob(os.path.join(dataset_path, '*.txt'))
        self.annotation_ids = [int(Path(annotation).stem) for annotation in self.annotations]
        self.images = glob.glob(os.path.join(dataset_path, '*.jpg'))
        self.image_ids = [int(Path(image).stem) for image in self.images]

        _, self.annotations= np.array(sorted(zip(self.annotation_ids, self.annotations))).T
        _, self.images = np.array(sorted(zip(self.image_ids, self.images))).T
        self.image_ids = sorted(self.image_ids)
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
        return image, formated_annotations


if __name__ == '__main__':
    dataset = LizaDataset(os.path.join('..', 'Dataset'), None)
    image, annotation = dataset.__getitem__(0)
    pass
