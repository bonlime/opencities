import os
import cv2
import numpy as np
import albumentations as albu
from torch.utils.data import Dataset


# TODO: put default augmentation here

class OpenCitiesDataset(Dataset):
    def __init__(self, split='all', transform=None, imgs_path="data/images-512", masks_path="data/masks-512", buildings_only=True):
        """Args:
            split (str): one of `val`, `train`, `all` 
            transform (albu.Compose): albumentation transformation for images
            buildings_only (bool): Flag to return only masks for building without borders and contact
        """
        ids = os.listdir(imgs_path)
        if split == "train":
            self.ids = [i for i in ids if "train" in i]
        elif split == "val":
            self.ids = [i for i in ids if "val" in i]
        self.imgs_path = imgs_path
        self.masks_path = masks_path
        self.transform = albu.Compose([]) if transform is None else transform
        self.buildings_only = buildings_only

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.imgs_path, self.ids[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.masks_path, self.ids[idx]))
        augmented = self.transform(image=img, mask=mask)
        aug_img, aug_mask = augmented['image'], augmented['mask'] / 255.
        if self.buildings_only:
            ch_last = aug_mask.size(2) == 3
            aug_mask = aug_mask[:, :, 2:] if ch_last else aug_mask[2:, :, :]
        return aug_img, aug_mask


class OpenCitiesTestDataset(Dataset):
    def __init__(self, transform=None, test_path="/home/zakirov/datasets/opencities/test"):
        ids = os.listdir(test_path)
        ids.remove("catalog.json")
        self.tiff_names = ids
        self.ids = [os.path.join(test_path, i, i+'.tif') for i in ids]
        self.transform = albu.Compose([]) if transform is None else transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = cv2.imread(self.ids[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug_img = self.transform(image=img)["image"]
        return img, aug_img, self.tiff_names[idx]
