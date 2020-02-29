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
    name_to_dataset = {"opencities": OpenCitiesDataset, "inria": InriaTilesDataset}

    ## get augmentations
    train_aug = get_aug(FLAGS.augmentation, size=FLAGS.size)
    val_aug = get_aug("val", size=FLAGS.size)

    # get datasets
    val_datasets = []
    train_datasets = []
    for name in FLAGS.datasets:
        val_datasets.append(name_to_dataset[name](split="val", transform=val_aug))
        train_datasets.append(name_to_dataset[name](split="train", transform=train_aug))

    # concat all datasets into one
    val_dtst = reduce(lambda x, y: x + y, val_datasets)
    val_dtld = DataLoader(val_dtst, batch_size=FLAGS.bs, shuffle=False, num_workers=8)
    val_dtld = ToCudaLoader(val_dtld)

    train_dtst = reduce(lambda x, y: x + y, train_datasets)
    # without `drop_last` last batch consists of 1 element and BN fails
    train_dtld = DataLoader(train_dtst, batch_size=FLAGS.bs, shuffle=True, num_workers=8, drop_last=True)
    train_dtld = ToCudaLoader(train_dtld)

    print(f"\nUsing datasets: {FLAGS.datasets}. Train size: {len(train_dtst)}. Val size {len(val_dtst)}.")
    return train_dtld, val_dtld


class OpenCitiesDataset(Dataset):
    def __init__(
        self,
        split="all",
        transform=None,
        imgs_path="data/images-512",
        masks_path="data/masks-512",
        buildings_only=False,
        return_distance=False,
    ):
        """Args:
            split (str): one of `val`, `train`, `all`
            transform (albu.Compose): albumentation transformation for images
            buildings_only (bool): Flag to return only masks for building without borders and contact
            return_distance (bool): overwrirtes buildings_only and returns only normalized distance maps
        """
        ids = os.listdir(imgs_path)
        if split == "train":
            ids = [i for i in ids if "train" in i]
        elif split == "val":
            ids = [i for i in ids if "val" in i]
        self.img_ids = [os.path.join(imgs_path, i) for i in ids]
        self.mask_ids = [os.path.join(masks_path, i) for i in ids]
        self.transform = albu.Compose([]) if transform is None else transform
        self.buildings_only = buildings_only
        self.return_distance = return_distance

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
        # ch_last = aug_mask.size(2) == 3
        # if self.return_distance:
        #     aug_mask = aug_mask[:, :, 0:1] if ch_last else aug_mask[0:1, :, :]
        #     aug_mask = aug_mask * 2 - 1 # from [0, 1] to [-1, 1]
        # elif self.buildings_only:
        #     aug_mask = aug_mask[:, :, 2:] if ch_last else aug_mask[2:, :, :]
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
        mask = cv2.imread(self.mask_ids[idx]) #  cv2.IMREAD_GRAYSCALE # remove graysa
        # mask = np.expand_dims(mask, 2)
        augmented = self.transform(image=img, mask=mask)
        aug_img, aug_mask = augmented["image"], augmented["mask"] / 255.0
        return aug_img, aug_mask
