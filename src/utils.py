import torch
import albumentations.pytorch as albu_pt
import pytorch_tools as pt
import segmentation_models_pytorch as sm

MODEL_FROM_NAME = {
    "unet": pt.segmentation_models.Unet,
    "linknet": pt.segmentation_models.Linknet,
    "deeplab": pt.segmentation_models.DeepLabV3,
    "unet_sm": sm.Unet,
    "linknet_sm": sm.Linknet,
    "fpn_sm": sm.FPN,
}

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