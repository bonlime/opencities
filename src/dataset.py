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


def get_dataloaders(datasets, augmentation="medium", batch_size=16, size=384, buildings_only=False):
    """Returns:
    train_dataloader, val_dataloader
    """

    ## get augmentations
    train_aug = get_aug(augmentation, size=size)
    val_aug = get_aug("val", size=size)

    # get datasets
    val_datasets = []
    train_datasets = []
    if "tier1" in datasets:
        val_datasets.append(
            OpenCitiesDataset(
                split="val",
                transform=val_aug,
                imgs_path="data/tier_1-images-512",
                masks_path="data/tier_1-masks-512",
                buildings_only=buildings_only,
            )
        )
        train_datasets.append(
            OpenCitiesDataset(
                split="train",
                transform=train_aug,
                imgs_path="data/tier_1-images-512",
                masks_path="data/tier_1-masks-512",
                buildings_only=buildings_only,
            )
        )
    if "tier2" in datasets:
        val_datasets.append(
            OpenCitiesDataset(
                split="val",
                transform=val_aug,
                imgs_path="data/tier_2-images-512",
                masks_path="data/tier_2-masks-512",
                buildings_only=buildings_only,
            )
        )
        train_datasets.append(
            OpenCitiesDataset(
                split="train",
                transform=train_aug,
                imgs_path="data/tier_2-images-512",
                masks_path="data/tier_2-masks-512",
                buildings_only=buildings_only,
            )
        )
    if "inria" in datasets:
        val_datasets.append(
            InriaTilesDataset(
                split="val",
                transform=val_aug,
                buildings_only=buildings_only
            )
        )
        train_datasets.append(
            InriaTilesDataset(
                split="train",
                transform=train_aug,
                buildings_only=buildings_only,
            )
        )

    # concat all datasets into one
    val_dtst = reduce(lambda x, y: x + y, val_datasets)
    val_dtld = DataLoader(val_dtst, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    val_dtld = ToCudaLoader(val_dtld)

    train_dtst = reduce(lambda x, y: x + y, train_datasets)
    # without `drop_last` last batch consists of 1 element and BN fails
    train_dtld = DataLoader(train_dtst, batch_size=batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)
    train_dtld = ToCudaLoader(train_dtld)

    print(f"\nUsing datasets: {datasets}. With {augmentation} augmentation. Train size: {len(train_dtst)}. Val size {len(val_dtst)}.")
    return train_dtld, val_dtld


class OpenCitiesDataset(Dataset):
    def __init__(
        self,
        split="all",
        transform=None,
        imgs_path="data/tier_1-images-512",
        masks_path="data/tier_1-masks-512",
        buildings_only=False,
    ):
        """Args:
            split (str): one of `val`, `train`, `all`
            transform (albu.Compose): albumentation transformation for images
            buildings_only (bool): Flag to return only images with buildings
        """
        ids = os.listdir(imgs_path)
        if split == "train":
            ids = [i for i in ids if "train" in i]
        elif split == "val":
            ids = [i for i in ids if "val" in i]
        img_ids = [os.path.join(imgs_path, i) for i in ids]
        mask_ids = [os.path.join(masks_path, i) for i in ids]
        self.img_ids, self.mask_ids = np.array(img_ids), np.array(mask_ids)
        self.transform = albu.Compose([]) if transform is None else transform
        if buildings_only:
            # empty mask has size 842 bytes for images of size 512. Use this ibfo for filtering
            has_buildings = [os.path.getsize(i) > 1000 for i in self.mask_ids]
            self.img_ids, self.mask_ids = self.img_ids[has_buildings], self.mask_ids[has_buildings]

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_ids[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_ids[idx])
        augmented = self.transform(image=img, mask=mask)
        aug_img, aug_mask = augmented["image"], augmented["mask"] / 255.0
        # change distance map from [0, 1] to [-1, 1]
        aug_mask[0, :, :] = aug_mask[0, :, :] * 2 - 1
        return aug_img, aug_mask


class OpenCitiesTestDataset(Dataset):
    def __init__(self, transform=None, test_path="/home/zakirov/datasets/opencities/test"):
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
    def __init__(self, split="all", transform=None, buildings_only=False):
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
        mask = cv2.imread(self.mask_ids[idx]) #  cv2.IMREAD_GRAYSCALE # remove graysa
        # mask = np.expand_dims(mask, 2)
        augmented = self.transform(image=img, mask=mask)
        aug_img, aug_mask = augmented["image"], augmented["mask"] / 255.0
        return aug_img, aug_mask
