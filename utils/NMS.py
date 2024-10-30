from torchvision.ops import box_iou
import numpy as np
import torch


def delete(arr: torch.Tensor, indexes: int) -> torch.Tensor:
    return torch.stack([x for index, x in enumerate(arr) if index not in indexes])


def remove_overlaping(result, iou_treshold, exclude_diag=True, ratio_tresh=10):
    neibor = result
    if result['boxes'].__len__() > 0:
        if ratio_tresh is not None:
            boxes = result['boxes']
            ratio = (boxes[:,0] - boxes[:,2]) / (boxes[:,1] - boxes[:,3])
            ratio_idx = torch.where(
                torch.logical_not(
                    ratio < 1 + ratio_tresh
                )
            )
            for key in result:
                result[key] = delete(result[key], ratio_idx[0])

        iou = box_iou(torch.Tensor(result['boxes']).to('cpu'), torch.Tensor(neibor['boxes']).to('cpu'))
        in_current, in_next = torch.where(iou > iou_treshold)
        if exclude_diag:
            in_current, in_next = torch.tensor([[curr, nex] for curr, nex in zip(in_current, in_next) if curr != nex]).T
        I = torch.tensor([result['scores'][current_idx] > neibor['scores'][next_idx] for current_idx, next_idx in zip(in_current, in_next)])
        in_current = in_current[torch.logical_not(I)]
        in_next = in_next[I]

        for key in result:
            result[key] = delete(result[key], in_current)
    return result
