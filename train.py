import os
import yaml
import time
# import wandb
import numpy as np
import albumentations as albu
import albumentations.pytorch as albu_pt

import apex
import torch
from torch.utils.data import DataLoader

import pytorch_tools as pt
from pytorch_tools.optim import optimizer_from_name
from pytorch_tools.fit_wrapper.callbacks import Callback as NoClb
from pytorch_tools.fit_wrapper.callbacks import SegmCutmix

from src.arg_parser import parse_args
from src.dataset import get_dataloaders
from src.utils import ToTensor
from src.utils import MODEL_FROM_NAME
from src.utils import criterion_from_list
from src.callbacks import ThrJaccardScore
from src.callbacks import PredictViewer


def main():
    FLAGS = parse_args()

    pt.utils.misc.set_random_seed(123) # fix all seeds

    ## Save config file
    os.makedirs(FLAGS.outdir, exist_ok=True)
    yaml.dump(vars(FLAGS), open(FLAGS.outdir + '/config.yaml', 'w'))

    ## Get dataloaders
    train_dtld, val_dtld = get_dataloaders(FLAGS)

    ## Get model and optimizer
    model = MODEL_FROM_NAME[FLAGS.segm_arch](FLAGS.arch, **FLAGS.model_params).cuda()
    optimizer = optimizer_from_name(FLAGS.optim)(
        model.parameters(),
        lr=FLAGS.lr,
        weight_decay=FLAGS.weight_decay,  # **FLAGS.optim_params TODO: add additional optim params if needed
    )
    if FLAGS.resume:
        checkpoint = torch.load(FLAGS.resume, map_location=lambda storage, loc: storage.cuda())
        model.load_state_dict(checkpoint["state_dict"])
        # FLAGS.start_epoch = checkpoint["epoch"]
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except:  # may raise an error if another optimzer was used
            print("Failed to load state dict into optimizer")
    num_params = pt.utils.misc.count_parameters(model)[0]
    print(f"Number of parameters: {num_params / 1e6:.02f}M")

    ## TODO: add lookahead
    ## train on fp16 by default
    model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1", verbosity=0, loss_scale=2048)

    ## get loss. fixed for now.
    bce_loss = pt.losses.CrossEntropyLoss(mode="binary").cuda()
    bce_loss.name = "BCE"
    loss = criterion_from_list(FLAGS.criterion).cuda()
    print("Loss for this run is: ", loss)
    ## get runner
    tb_logger = PredictViewer(FLAGS.outdir)
    runner = pt.fit_wrapper.Runner(
        model,
        optimizer,
        criterion=loss,
        callbacks=[
            pt.fit_wrapper.callbacks.Timer(),
            pt.fit_wrapper.callbacks.ConsoleLogger(),
            # pt.fit_wrapper.callbacks.ReduceLROnPlateau(patience=10),
            SegmCutmix(1, 1) if FLAGS.cutmix else NoClb(),
            # pt.fit_wrapper.callbacks.FileLogger(FLAGS.outdir),
            tb_logger,
            pt.fit_wrapper.callbacks.CheckpointSaver(FLAGS.outdir, save_name="model.chpn"),
        ],
        metrics=[bce_loss, pt.metrics.JaccardScore(mode="binary").cuda(), ThrJaccardScore(thr=0.5),],
    )

    if FLAGS.decoder_warmup_epochs > 0:
        ## freeze encoder
        for p in model.encoder.parameters():
            p.requires_grad = False
        runner.fit(
            train_dtld,
            val_loader=val_dtld,
            epochs=FLAGS.decoder_warmup_epochs,
            steps_per_epoch=50 if FLAGS.short_epoch else None,
            val_steps=50 if FLAGS.short_epoch else None,
        )

        ## unfreeze all
        for p in model.parameters():
            p.requires_grad = True

    runner.fit(
        train_dtld,
        val_loader=val_dtld,
        start_epoch=FLAGS.decoder_warmup_epochs,
        epochs=FLAGS.epochs,
        steps_per_epoch=50 if FLAGS.short_epoch else None,
        val_steps=50 if FLAGS.short_epoch else None,
    )

    # log training hyperparameters to TensorBoard
    # This produces extra folder in logs. don't want this.
    # hparam_dict=vars(FLAGS)
    # hparam_dict.pop("config_file")
    # metric_dict={"ValJaccardScore": round(runner.state.val_metrics[0].avg, 4)}
    # tb_logger.writer.add_hparams(hparam_dict, metric_dict)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Finished Training. Took: {(time.time() - start_time) / 60:.02f}m")
