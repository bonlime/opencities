import torch
from torch import nn
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

class DoubleLoss(pt.losses.Loss):
    def __init__(self, loss, cls_weights=[10, 1]):
        super().__init__()
        self.loss = loss 
        self.register_buffer("cls_weights", torch.Tensor(cls_weights))

    def forward(self, y_pred, y_true):
        l1 = self.loss(y_pred[:, 0:1], y_true[:, 0:1]) * self.cls_weights[0]
        l2 = self.loss(y_pred[:, 1:2], y_true[:, 1:2]) * self.cls_weights[1]
        return l1 + l2

class DoubleLossOCR(DoubleLoss):
    def forward(self, y_pred, y_true):
        l1 = super().forward(y_pred[0], y_true)
        l2 = super().forward(y_pred[1], y_true)
        return l1 * 0.4 + l2

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
    "hrnet": pt.segmentation_models.hrnet.hrnet_w48,
    "hrnet_ocr": pt.segmentation_models.hrnet.hrnet_w48_ocr
}

LOSS_FROM_NAME = {
    "bce": pt.losses.CrossEntropyLoss(mode="binary"),
    "wbce": pt.losses.CrossEntropyLoss(mode="binary", weight=[5]),
    "dice": pt.losses.DiceLoss(mode="binary"),
    "jaccard": pt.losses.JaccardLoss(mode="binary"),
    "log_jaccard": pt.losses.JaccardLoss(mode="binary", log_loss=True),
    "hinge": pt.losses.BinaryHinge(),
    "whinge": pt.losses.BinaryHinge(pos_weight=3),
    "focal": pt.losses.FocalLoss(mode="binary", alpha=None),
    "double_focal": DoubleLoss(pt.losses.FocalLoss(mode="binary", alpha=None)),
    "reduced_focal": pt.losses.FocalLoss(mode="binary", combine_thr=0.5, alpha=None),
    "reduced_double_focal": DoubleLoss(pt.losses.FocalLoss(mode="binary", combine_thr=0.5, alpha=None)),
    "reduced_double_focal_ocr": DoubleLossOCR(pt.losses.FocalLoss(mode="binary", combine_thr=0.5, alpha=None)),
    "double_bce": DoubleLoss(pt.losses.CrossEntropyLoss(mode="binary")),
    "mse": pt.losses.MSELoss(),
    "clip_mse": ClipMSELoss(),
    "mae": pt.losses.L1Loss(),
    "clip_mae": ClipL1Loss(),
}

CHANNEL_FROM_TARGET_TYPE = {
    "distance_map": 0,
    "mask": 1,
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
        if isinstance(pred, (list, tuple)) and len(pred) == 2:
            pred = pred[1] # take only second output of OCR
        return self.loss(pred[:, self.chn: self.chn+1, ...], target[:, self.chn: self.chn+1, ...])

def criterion_from_list(crit_list):
    """expects something like `bce 0.5 dice 0.5` to construct loss"""
    # losses = [TargetWrapper(LOSS_FROM_NAME[l], TARGET_TYPE_FROM_NAME[l]) for l in crit_list[::2]]
    losses = [LOSS_FROM_NAME[l] for l in crit_list[::2]]
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
