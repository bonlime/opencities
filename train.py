import os
import yaml
import time
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
from src.utils import ToTensor, MODEL_FROM_NAME
from src.callbacks import ThrJaccardScore



def main():
    FLAGS = parse_args()
    print(FLAGS)
    pt.utils.misc.set_random_seed(42) # fix all seeds
    ## dump config
    os.makedirs(FLAGS.outdir, exist_ok=True)
    yaml.dump(vars(FLAGS), open(FLAGS.outdir + '/config.yaml', 'w'))

    ## get dataloaders
    train_dtld, val_dtld = get_dataloaders(FLAGS)

    ## get model and optimizer
    model = MODEL_FROM_NAME[FLAGS.segm_arch](FLAGS.arch, **FLAGS.model_params).cuda()
    optimizer = optimizer_from_name(FLAGS.optim)(
        model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay, # **FLAGS.optim_params TODO: add additional optim params if needed
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
    
    ## get loss. fixed for now. TODO: add loss combinations
    jacc_loss = pt.losses.JaccardLoss(mode="binary").cuda()
    bce_loss = pt.losses.CrossEntropyLoss(mode="binary").cuda()
    bce_loss.name = "BCE"
    # loss = 0.5 * bce_loss + 0.5 * jacc_loss
    loss = pt.losses.BinaryHinge()
    # loss = jacc_loss

    ## get runner
    tb_logger = pt.fit_wrapper.callbacks.TensorBoard(FLAGS.outdir)
    runner = pt.fit_wrapper.Runner(
        model,
        optimizer,
        criterion=loss,
        callbacks=[
            pt.fit_wrapper.callbacks.Timer(),
            pt.fit_wrapper.callbacks.ConsoleLogger(),
            # pt.fit_wrapper.callbacks.ReduceLROnPlateau(patience=10),
            SegmCutmix(1, 1) if FLAGS.cutmix else NoClb(),
            pt.fit_wrapper.callbacks.FileLogger(FLAGS.outdir),
            tb_logger,
            pt.fit_wrapper.callbacks.CheckpointSaver(FLAGS.outdir, save_name="model.chpn")
        ],
        metrics=[
            bce_loss,
            pt.metrics.JaccardScore(mode="binary").cuda(),
            ThrJaccardScore(thr=0.5),
        ]
    )

    if FLAGS.decoder_warmup_epochs > 0:
        ## freeze encoder
        for p in model.encoder.parameters():
            p.requires_grad = False
        runner.fit(train_dtld, val_loader=val_dtld, epochs=FLAGS.decoder_warmup_epochs)
    
        ## unfreeze all
        for p in model.parameters():
            p.requires_grad = True
            
    runner.fit(train_dtld, val_loader=val_dtld, start_epoch=FLAGS.decoder_warmup_epochs, epochs=FLAGS.epochs)


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