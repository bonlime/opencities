import torch
import numpy as np
import pytorch_tools as pt

class ThrJaccardScore(pt.metrics.JaccardScore):
    def __init__(self, *args, thr=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.from_logits = False
        self.name = "ThrJaccard@" + str(thr)
        self.thr = thr

    def forward(self, y_pred, y_true):
        y_pred = (y_pred.sigmoid() > self.thr).float()
        return super().forward(y_pred, y_true)