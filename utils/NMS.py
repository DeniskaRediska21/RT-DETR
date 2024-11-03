from torchvision.ops import box_iou
import numpy as np
import torch
from torchvision.ops import batched_nms
import torchvision.transforms.v2 as transforms

import sys
sys.path.append('..')
sys.path.append('../upgreat_detector')
sys.path.append('../RT-DETR')
from config import CLASS_INFERENCE_SIZE, CLASS_BATCH_SIZE, CLASS_DEVICE
from train_classifier import get_valid_transform

def delete(arr: torch.Tensor, indexes: int) -> torch.Tensor:
    return torch.stack([x for index, x in enumerate(arr) if index not in indexes])


def remove_overlaping(result, iou_treshold, exclude_diag=True, ratio_tresh=10):
    neibor = result
    if result and result['boxes'].__len__() > 0:
        if ratio_tresh is not None:
            boxes = result['boxes']
            ratio = (boxes[:,0] - boxes[:,2]) / (boxes[:,1] - boxes[:,3])
            keep = torch.where(
                    ratio < 1 + ratio_tresh
            )
            for key in result:
                result[key] = result[key][keep]

        keep = batched_nms(result['boxes'], result['scores'], torch.ones_like(result['scores']), iou_treshold)

        for key in result:
            result[key] = result[key][keep]

    return result

def reclassify(classifier, T, result):
    if result and result['boxes'].__len__() > 0:
        boxes = result['boxes']
        
        subs = []
        transform = get_valid_transform(CLASS_INFERENCE_SIZE, True)
        for box in boxes:
            xmin, ymin, xmax, ymax = box.clone().to(torch.int32).detach()
            xmin, ymin, xmax, ymax = xmin.item(), ymin.item(), xmax.item(), ymax.item()
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(xmax, T.shape[2]), min(ymax, T.shape[1])
            
            # if xmin and ymin:
            patch = T[:, ymin:ymax, xmin:xmax]
            # elif xmin:
            #     patch = T[:, :ymax, xmin:xmax]
            # elif ymin:
            #     patch = T[:, ymin:ymax, :xmax]
            # else:
            #     patch = T[:, :ymax, :xmax]
                
            sub = transform(patch)
            # sub = transforms.Resize((CLASS_INFERENCE_SIZE, CLASS_INFERENCE_SIZE))(patch)
            subs.append(sub)
        
        subs = torch.stack(subs)
        batch_size = CLASS_BATCH_SIZE
        batches = torch.split(subs, batch_size)
        
        classes = []
        for batch in batches:
            with torch.no_grad():
                batch = batch.to(CLASS_DEVICE)
                outputs = classifier(batch)
            
            for output in outputs:
                classes.append(torch.argmax(output))
                
        for i in range(len(classes)):
            result['labels'][i] = classes[i]
        
        return result
