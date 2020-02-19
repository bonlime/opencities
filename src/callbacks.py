import torch
import numpy as np
import pytorch_tools as pt
from torchvision.utils import make_grid


class ThrJaccardScore(pt.metrics.JaccardScore):
    """Calculate Jaccard on Thresholded by `thr` prediction. This function applyis sigmoid to prediction first"""

    def __init__(self, *args, thr=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.from_logits = False
        self.name = "ThrJaccard@" + str(thr)
        self.thr = thr

    def forward(self, y_pred, y_true):
        y_pred = (y_pred.sigmoid() > self.thr).float()
        return super().forward(y_pred, y_true)


class PredictViewer(pt.fit_wrapper.callbacks.TensorBoard):
    """Saves first batch and visualizes model predictions on it for every epoch"""

    def __init__(self, log_dir, log_every=20, num_images=4):
        """num_images (int): number of images to visualize"""
        super().__init__(log_dir, log_every=20)
        self.has_saved = False  # Flag to save first batch
        self.img_batch = None
        self.num_images = num_images

    def on_batch_end(self):
        super().on_batch_end()
        # save first val batch
        if not self.has_saved and not self.state.is_train:
            self.img_batch = self.state.input[0].detach()[:self.num_images]  # only take first 9 images
            self.has_saved = True
            target_batch = self.state.input[1].detach()[:self.num_images]
            self.target_grid = make_grid((target_batch * 255).type(torch.uint8), nrow=self.num_images)

    def on_epoch_end(self):
        super().on_epoch_end()
        self.state.model.eval()  # not sure if needed but just in case
        pred = self.state.model(self.img_batch)
        pred = (pred.sigmoid() * 255).type(torch.uint8)
        grid = make_grid(pred, nrow=self.num_images)
        grid = torch.cat([grid, self.target_grid], axis=1)
        self.writer.add_image("val/prediction", grid, self.current_step)
