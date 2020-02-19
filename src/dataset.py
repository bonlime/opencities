import os
import cv2
import numpy as np
import pandas as pd
from functools import reduce
import albumentations as albu
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from src.utils import ToCudaLoader
from src.augmentations import get_aug


def get_dataloaders(FLAGS):
    """Returns:
    train_dataloader, val_dataloader
    """

    name_to_dataset = {
        "tier1" : OpenCitiesDataset,
        "inria" : InriaTilesDataset
    }

    ## Get augmentations 
    train_aug = get_aug(FLAGS.augmentation, size=FLAGS.size)
    val_aug = get_aug("val", size=FLAGS.size)

    ## Get datasets
    val_datasets = []
    train_datasets = []
    for name in FLAGS.datasets:
        val_datasets.append(name_to_dataset[name](split="val", transform=val_aug, tile_size=FLAGS.tile_size))
        train_datasets.append(name_to_dataset[name](split="train", transform=train_aug, tile_size=FLAGS.tile_size))

    # concat all datasets into one
    val_dtst = reduce(lambda x, y: x + y, val_datasets)
    val_dtld = DataLoader(val_dtst, batch_size=FLAGS.bs, shuffle=False, num_workers=8)
    val_dtld = ToCudaLoader(val_dtld)

    train_dtst = reduce(lambda x, y: x + y, train_datasets)
    train_dtld = DataLoader(train_dtst, batch_size=FLAGS.bs, shuffle=True, num_workers=8)
    train_dtld = ToCudaLoader(train_dtld)

    print(f"\nUsing datasets: {FLAGS.datasets}. Train size: {len(train_dtst)}. Val size {len(val_dtst)}.")
    return train_dtld, val_dtld

class OpenCitiesDataset(Dataset):
    def __init__(
        self, 
        split='all', 
        transform=None, 
        imgs_path="data/tier1/images", 
        masks_path="data/tier1/masks", 
        tile_size=512,
        buildings_only=True
        ):
      
        """
        Args:
            split (str): one of `val`, `train`, `all`
            transform (albu.Compose): albumentation transformation for images
            buildings_only (bool): Flag to return only masks for building without borders and contact
        """
        
        ids = os.listdir(imgs_path + f"-{tile_size}")
        if split == "train":
            self.ids = [i for i in ids if "train" in i]
        elif split == "val":
            self.ids = [i for i in ids if "val" in i]

        self.imgs_path = imgs_path + f"-{tile_size}"
        self.masks_path = masks_path + f"-{tile_size}"
        self.transform = albu.Compose([]) if transform is None else transform
        self.buildings_only = buildings_only

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.imgs_path, self.ids[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.masks_path, self.ids[idx]))
        augmented = self.transform(image=img, mask=mask)
        aug_img, aug_mask = augmented['image'], augmented['mask'] / 255.0

        if self.buildings_only:
            ch_last = aug_mask.size(2) == 3
            aug_mask = aug_mask[:, :, 2:] if ch_last else aug_mask[2:, :, :]
            
        return aug_img, aug_mask

class OpenCitiesTestDataset(Dataset):
    def __init__(
        self, 
        transform=None, 
        test_path="/home/zakirov/datasets/opencities/test"
        ):
        
        ids = os.listdir(test_path)
        ids.remove("catalog.json")
        self.tiff_names = ids
        self.ids = [os.path.join(test_path, i, i + ".tif") for i in ids]
        self.transform = albu.Compose([]) if transform is None else transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = cv2.imread(self.ids[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug_img = self.transform(image=img)["image"]
        return img, aug_img, self.tiff_names[idx]


class InriaTilesDataset(Dataset):
    def __init__(self, split="all", transform=None):
        super().__init__()
        df = pd.read_csv("/home/zakirov/datasets/AerialImageDataset/inria_tiles.csv", index_col=0)
        if split == "val":
            df = df[df.train == 0]
        elif split == "train":
            df = df[df.train == 1]
        self.img_ids = df["image"].values
        self.mask_ids = df["mask"].values
        self.transform = albu.Compose([]) if transform is None else transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_ids[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_ids[idx], cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, 2)
        augmented = self.transform(image=img, mask=mask)
        aug_img, aug_mask = augmented["image"], augmented["mask"] / 255.0
        return aug_img, aug_mask
