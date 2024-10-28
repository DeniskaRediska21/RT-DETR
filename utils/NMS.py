from torchvision.ops import box_iou
import numpy as np
import torch


def remove_overlaping(result, neibor, iou_treshold, exclude_diag = False):
    iou = box_iou(torch.Tensor(result['boxes']), torch.Tensor(neibor['boxes']))
    in_current, in_next = torch.where(iou > iou_treshold)
    if exclude_diag:
        in_current, in_next = np.array([[curr, nex] for curr, nex in zip(in_current, in_next) if curr != nex]).T
    I = np.array([result['scores'][current_idx] > neibor['scores'][next_idx] for current_idx, next_idx in zip(in_current, in_next)])
    in_current = in_current[np.logical_not(I)]
    in_next = in_next[I]

    for key in result:
        if key != 'int_coords' and key != 'targets' and 'strict' not in key:
            result[key] = np.delete(result[key], in_current, axis = 0)
            neibor[key] = np.delete(neibor[key], in_next, axis = 0)
        elif 'strict' in key:
            pass
            # warnings.warn('NMS is not supported for strict')
