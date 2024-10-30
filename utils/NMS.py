from torchvision.ops import box_iou
import numpy as np
import torch


def delete(arr: torch.Tensor, indexes: int) -> torch.Tensor:
    return torch.stack([x for index, x in enumerate(arr) if index not in indexes])


def remove_overlaping(result, iou_treshold, exclude_diag=True):
    neibor = result
    if result['boxes'].__len__() > 0:
        iou = box_iou(torch.Tensor(result['boxes']), torch.Tensor(neibor['boxes']))
        in_current, in_next = torch.where(iou > iou_treshold)
        if exclude_diag:
            in_current, in_next = torch.tensor([[curr, nex] for curr, nex in zip(in_current, in_next) if curr != nex]).T
        I = torch.tensor([result['scores'][current_idx] > neibor['scores'][next_idx] for current_idx, next_idx in zip(in_current, in_next)])
        in_current = in_current[torch.logical_not(I)]
        in_next = in_next[I]

        for key in result:
            result[key] = delete(result[key], in_current)
    return result
