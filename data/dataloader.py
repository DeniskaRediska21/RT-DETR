import os
from torch.utils.data import DataLoader, random_split
from data.dataset import LizaClassDataset


def collate_fn(batch: list) -> tuple:
    images, classes = zip(*batch)
    return images, classes


def get_train_val_dataloader(dataset_dir: str, batch_size: int, num_workers: int, shuffle=True, train_size:float=0.7, transforms=None, image_processor=None) -> DataLoader:

    dataset = LizaClassDataset(dataset_dir,transforms=transforms, image_processor=image_processor)

    train_dataset, val_dataset = random_split(dataset=dataset, lengths = (train_size, 1-train_size))

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn,shuffle=shuffle)

    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=shuffle)
    return train_dataloader, val_dataloader
