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

    NORM_TO_TENSOR = albu.Compose([albu.Normalize(), ToTensor()])
    CROP_AUG = albu.RandomResizedCrop(size, size, scale=(0.08, 0.3))
    VAL_AUG = albu.Compose([
        albu.CenterCrop(size, size),
        NORM_TO_TENSOR,
    ])

    TEST_AUG = NORM_TO_TENSOR

    LIGHT_AUG = albu.Compose([
        CROP_AUG,
        albu.Flip(),
        albu.RandomRotate90(),
        NORM_TO_TENSOR,
    ])

    MEDIUM_AUG = albu.Compose(
        [   
            CROP_AUG,
            albu.Flip(),
            albu.ShiftScaleRotate(), # border_mode=cv2.BORDER_CONSTANT
            # Add occasion blur/sharpening
            albu.OneOf([albu.GaussianBlur(), albu.IAASharpen(), albu.NoOp()]),
            # Spatial-preserving augmentations:
            # albu.OneOf([albu.CoarseDropout(), albu.MaskDropout(max_objects=5), albu.NoOp()]),
            albu.GaussNoise(),
            albu.OneOf([albu.RandomBrightnessContrast(), albu.CLAHE(), albu.HueSaturationValue(), albu.RGBShift(), albu.RandomGamma()]),
            # Weather effects
            albu.RandomFog(fog_coef_lower=0.01, fog_coef_upper=0.3, p=0.1),
            NORM_TO_TENSOR,
        ]
    )

    types = {
        "val" : VAL_AUG,
        "test" : TEST_AUG,
        "light" : LIGHT_AUG,
        "medium" : MEDIUM_AUG
    }

    return types[aug_type]
