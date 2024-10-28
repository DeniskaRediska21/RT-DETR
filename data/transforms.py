import torchvision.transforms.v2 as T


def get_transforms():
    transforms = T.Compose([
        T.RandomVerticalFlip(p=0.5),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomResize(480, 840),
#        T.ColorJitter(brightness=0.2),
#        T.ColorJitter(contrast=0.2)
    ])

    return transforms
