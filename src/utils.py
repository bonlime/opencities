import torch
from functools import reduce
import albumentations.pytorch as albu_pt
import segmentation_models_pytorch as sm
import pytorch_tools as pt

MODEL_FROM_NAME = {
    "unet": pt.segmentation_models.Unet,
    "linknet": pt.segmentation_models.Linknet,
    "deeplab": pt.segmentation_models.DeepLabV3,
    "unet_sm": sm.Unet,
    "linknet_sm": sm.Linknet,
    "fpn_sm": sm.FPN,
    "segm_fpn": pt.segmentation_models.SegmentationFPN,
    "segm_bifpn": pt.segmentation_models.SegmentationBiFPN,
}

LOSS_FROM_NAME = {
    "bce": pt.losses.CrossEntropyLoss(mode="binary"),
    "dice": pt.losses.DiceLoss(mode="binary"),
    "jaccard": pt.losses.JaccardLoss(mode="binary"),
    "hinge": pt.losses.BinaryHinge(),
    "focal": pt.losses.BinaryFocalLoss(),
}

def criterion_from_list(crit_list):
    """expects something like `bce 0.5 dice 0.5` to construct loss"""
    losses = [ LOSS_FROM_NAME[l] * float(w) for l, w in zip(crit_list[::2], crit_list[1::2])]
    return reduce(lambda x, y: x + y, losses)

# want also to transform mask
class ToTensor(albu_pt.ToTensorV2):
    def apply_to_mask(self, mask, **params):
        return torch.from_numpy(mask.transpose(2, 0, 1))


# loads batches to cuda. TODO: check custom collate_fn instead. it should be faster
# or just move to DALI and forget about all this
class ToCudaLoader:
    def __init__(self, loader):
        self.loader = loader
        
    def __iter__(self):
        return ((i.cuda() for i in batch) for batch in self.loader)
    
    def __len__(self):
        return len(self.loader)