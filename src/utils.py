import torch
import torchvision.models.segmentation as tv_segm
from functools import reduce
import albumentations.pytorch as albu_pt
import segmentation_models_pytorch as sm
import pytorch_tools as pt

def tv_deeplab(arch, **kwargs):
    if arch == "resnet50":
        return tv_segm.deeplabv3_resnet50(pretrained=False, progress=False, num_classes=1)
    elif arch == "resnet101":
        return tv_segm.deeplabv3_resnet101(pretrained=True, progress=False, num_classes=1)

class ClipMSELoss(pt.losses.MSELoss):
    """same as MSE but predictions are clipped in [-1, 1]"""
    def forward(self, input, target):
        input = torch.clamp(input, -1, 1)
        return torch.nn.functional.mse_loss(input, target, reduction=self.reduction)

class ClipL1Loss(pt.losses.L1Loss):
    """same as MSE but predictions are clipped in [-1, 1]"""
    def forward(self, input, target):
        input = torch.clamp(input, -1, 1)
        if target.min() == 0:
            target = target * 2 - 1
        return torch.nn.functional.mse_loss(input, target, reduction=self.reduction)

MODEL_FROM_NAME = {
    "unet": pt.segmentation_models.Unet,
    "linknet": pt.segmentation_models.Linknet,
    "deeplab": pt.segmentation_models.DeepLabV3,
    "tv_deeplab": tv_deeplab,
    "unet_sm": sm.Unet,
    "linknet_sm": sm.Linknet,
    "fpn_sm": sm.FPN,
    "deeplab_sm": sm.DeepLabV3,
    "segm_fpn": pt.segmentation_models.SegmentationFPN,
    "segm_bifpn": pt.segmentation_models.SegmentationBiFPN,
}

LOSS_FROM_NAME = {
    "bce": pt.losses.CrossEntropyLoss(mode="binary"),
    "wbce": pt.losses.CrossEntropyLoss(mode="binary", weight=[5]),
    "dice": pt.losses.DiceLoss(mode="binary"),
    "jaccard": pt.losses.JaccardLoss(mode="binary"),
    "log_jaccard": pt.losses.JaccardLoss(mode="binary", log_loss=True),
    "hinge": pt.losses.BinaryHinge(),
    "whinge": pt.losses.BinaryHinge(pos_weight=3),
    "focal": pt.losses.BinaryFocalLoss(),
    "reduced_focal": pt.losses.BinaryFocalLoss(reduced=True),
    "mse": pt.losses.MSELoss(),
    "clip_mse": ClipMSELoss(),
    "mae": pt.losses.L1Loss(),
    "clip_mae": ClipL1Loss(),
}

CHANNEL_FROM_TARGET_TYPE = {
    "distance_map": 0,
    "mask": 2,
}

TARGET_TYPE_FROM_NAME = {
    "bce": "mask",
    "wbce": "mask",
    "dice": "mask",
    "jaccard": "mask",
    "hinge": "mask",
    "whinge": "mask",
    "focal": "mask",
    "reduced_focal": "mask",
    "mse": "distance_map",
    "clip_mse": "distance_map",
    "mae": "distance_map",
    "clip_mae": "mask", # "distance_map", # change mae to operate on mask
}
class TargetWrapper(pt.losses.Loss):
    """wrapper which gets particular channel from target for computing loss
    Args:
        loss: any losss
        target_type: one of `mask`, `distance_map`"""
    def __init__(self, loss, target_type):
        super().__init__()
        self.loss = loss
        if hasattr(loss, "name"):
            self.name = loss.name
        self.chn = CHANNEL_FROM_TARGET_TYPE[target_type]

    def forward(self, pred, target):
        return self.loss(pred, target[:, self.chn: self.chn+1, ...])

def criterion_from_list(crit_list):
    """expects something like `bce 0.5 dice 0.5` to construct loss"""
    losses = [TargetWrapper(LOSS_FROM_NAME[l], TARGET_TYPE_FROM_NAME[l]) for l in crit_list[::2]]
    losses = [l * float(w) for l, w in zip(losses, crit_list[1::2])]
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
        return ([img.cuda(non_blocking=True), target.cuda(non_blocking=True)] for img, target in self.loader)
    
    def __len__(self):
        return len(self.loader)
