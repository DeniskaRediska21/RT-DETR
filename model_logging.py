import os
import numpy as np


if __name__ == '__main__':
    savedir = os.path.join(
        os.sep,
        'home',
        'user',
        'LIZA',
        'RT-DETR',
        'rtdetr-r50-cppe5-finetune',
        'October_28_2024_10_11_54',
    )

    checkpoint_dirs = os.listdir(savedir)

    last_checkpoint = np.argmax([int(''.join([ch for ch in filename if 47<ord(ch)<58])) for filename in checkpoint_dirs])

    last_checkpoint_dir = checkpoint_dirs[last_checkpoint]

    PATH = os.path.join(savedir, last_checkpoint_dir)

    pass
