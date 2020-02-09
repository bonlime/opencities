import cv2
import torch
import albumentations as albu
import albumentations.pytorch as albu_pt

class ToTensor(albu_pt.ToTensorV2):
    def apply_to_mask(self, mask, **params):
        return torch.from_numpy(mask.transpose(2, 0, 1))
    

def get_aug(aug_type="val", size=256):
    """aug_type (str): one of `val`, `test`, `light`, `medium`, `hard`
       size (int): final size of the crop"""

    VAL_AUG = albu.Compose([
        albu.CenterCrop(size, size),
        albu.Normalize(),
        ToTensor(),
    ])

    TEST_AUG = albu.Compose([
        albu.Normalize(),
        ToTensor(),
    ])

    LIGHT_AUG = albu.Compose([
        albu.Flip(),
        albu.ShiftScaleRotate(scale_limit=0.2),
        albu.RandomCrop(size, size),
        albu.RandomRotate90(),
        albu.Normalize(),
        ToTensor(),
    ])

    MEDIUM_AUG = albu.Compose(
        [
            albu.Flip(),
            albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=15, border_mode=cv2.BORDER_CONSTANT),
            albu.RandomCrop(size, size),
            # Add occasion blur/sharpening
            albu.OneOf([albu.GaussianBlur(), albu.IAASharpen(), albu.NoOp()]),
            # Spatial-preserving augmentations:
            # albu.OneOf([albu.CoarseDropout(), albu.MaskDropout(max_objects=5), albu.NoOp()]),
            albu.GaussNoise(),
            albu.OneOf([albu.RandomBrightnessContrast(), albu.CLAHE(), albu.HueSaturationValue(), albu.RGBShift(), albu.RandomGamma()]),
            # Weather effects
            albu.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.3, p=0.1),
            albu.Normalize(),
            ToTensor(), 
        ]
    )

    if aug_type == "val":
        return VAL_AUG
    elif aug_type == "test":
        return TEST_AUG
    elif aug_type == "light":
        return LIGHT_AUG
    elif aug_type == "medium":
        return MEDIUM_AUG
    elif aug_type == "hard":
        raise NotImplementedError
