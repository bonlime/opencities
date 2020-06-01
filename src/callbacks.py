import torch
import pytorch_tools as pt
from torchvision.utils import make_grid

class BalancedAccuracy2: 
     """ BalancedAccuracy == mean of recalls for each class 
         >>> y_true = [0, 1, 0, 0, 1, 0] 
         >>> y_pred = [0, 1, 0, 0, 0, 1] 
         >>> BalancedAccuracy()(y_true, y_pred) 
         0.625 
     """ 
  
     def __init__(self, balanced=True): 
        self.name = "BalancedAcc" if balanced else "Acc"
        self.balanced = balanced

     def __call__(self, output, target): 
        """Args: 
        output (Tensor): raw logits of shape (N, C) or already argmaxed of shape (N,) 
        target (Long Tensor): true classes of shape (N,) or one-hot encoded of shape (N, C )""" 
        if len(target.shape) > 1 and (target.size(1) > 1): 
            target = target.argmax(1)
        if len(output.shape) > 1:
            if output.size(1) == 1:
                # binary predictions case
                output = output.gt(0).long()  
            else:
                output = output.argmax(1)
        correct = output.eq(target)
        if not self.balanced:
            return correct.float().mean()
        result = 0 
        for cls in target.unique(): 
            tp = (correct * target.eq(cls)).sum().float() 
            tp_fn = target.eq(cls).sum() 
            result += tp / tp_fn 
        return result.mul(100.0 / target.unique().size(0)) 

class ScheduledDropout(pt.fit_wrapper.callbacks.Callback):
    def __init__(self, drop_rate=0.1, epochs=30, attr_name="dropout.p", verbose=True):
        """
        Slowly changes dropout value for `attr_name` each epoch. 
        Args:
            drop_rate (float): max dropout rate
            epochs (int): num epochs to max dropout to fully take effect
            attr_name (str): name of dropout block in model
        """
        super().__init__()
        self.drop_rate = drop_rate
        self.epochs = epochs
        self.attr_name = attr_name

    def on_epoch_end(self):
        current_rate = self.drop_rate * min(1, self.state.epoch / self.epochs)
        setattr(self.state.model, self.attr_name, current_rate)



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
        if isinstance(pred, (list, tuple)):
            pred = pred[1]
        pred = (pred.sigmoid() * 255).type(torch.uint8)
        grid = make_grid(pred, nrow=self.num_images)
        grid = torch.cat([grid, self.target_grid], axis=1)
        self.writer.add_image("val/prediction", grid, self.current_step)
