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

from src.arg_parser import parse_args
from src.dataset import OpenCitiesDataset
from src.utils import ToCudaLoader, ToTensor, MODEL_FROM_NAME



def main():
    FLAGS = parse_args()
    print(FLAGS)
    pt.utils.misc.set_random_seed(42) # fix all seeds
    ## dump config
    os.makedirs(FLAGS.outdir, exist_ok=True)
    yaml.dump(vars(FLAGS), open(FLAGS.outdir + '/config.yaml', 'w'))

    ## get augmentations 
    train_aug = albu.Compose([
        albu.Flip(),
        albu.ShiftScaleRotate(scale_limit=0.2),
        albu.RandomResizedCrop(FLAGS.size, FLAGS.size, scale=(0.15, 0.8)),
        albu.RandomBrightnessContrast(),
        albu.HueSaturationValue(),
        albu.RandomRotate90(),
        albu.Normalize(),
        ToTensor(),
    ])

    val_aug = albu.Compose([
        albu.CenterCrop(FLAGS.size, FLAGS.size),
        albu.Normalize(),
        ToTensor(),
    ])

    ## get dataloaders
    val_dtst = OpenCitiesDataset(split="val", transform=val_aug, buildings_only=True)
    val_dtld = DataLoader(val_dtst, batch_size=FLAGS.bs, shuffle=False, num_workers=8)
    val_dtld = ToCudaLoader(val_dtld)

    train_dtst = OpenCitiesDataset(split="train", transform=train_aug, buildings_only=True)
    train_dtld = DataLoader(train_dtst, batch_size=FLAGS.bs, shuffle=True, num_workers=8)
    train_dtld = ToCudaLoader(train_dtld)

    ## get model and optimizer
    model = MODEL_FROM_NAME[FLAGS.segm_arch](FLAGS.arch).cuda()
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
    loss = pt.losses.JaccardLoss(mode="binary").cuda()
    loss = 0.5 * pt.losses.CrossEntropyLoss(mode="binary").cuda() + 0.5 * loss

    ## get runner
    tb_logger = pt.fit_wrapper.callbacks.TensorBoard(FLAGS.outdir)
    runner = pt.fit_wrapper.Runner(
        model,
        optimizer,
        criterion=loss,
        callbacks=[
            pt.fit_wrapper.callbacks.Timer(),
            pt.fit_wrapper.callbacks.ConsoleLogger(),
            pt.fit_wrapper.callbacks.ReduceLROnPlateau(patience=10),
            pt.fit_wrapper.callbacks.FileLogger(FLAGS.outdir),
            tb_logger,
            pt.fit_wrapper.callbacks.CheckpointSaver(FLAGS.outdir, save_name="model.chpn")
        ],
        metrics=pt.metrics.JaccardScore(mode="binary").cuda(),
    )

    ## freeze encoder
    for p in model.encoder.parameters():
        p.requires_grad = False
    runner.fit(train_dtld, val_loader=val_dtld, epochs=FLAGS.decoder_warmup_epochs)
    
    ## unfreeze all
    for p in model.parameters():
        p.requires_grad = True
    runner.fit(train_dtld, val_loader=val_dtld, start_epoch=FLAGS.decoder_warmup_epochs, epochs=FLAGS.epochs)


    # log training hyperparameters to TensorBoard
    hparam_dict=vars(FLAGS)
    hparam_dict.pop("config_file")
    metric_dict={"ValJaccardScore": round(runner.state.val_metrics[0].avg, 4)}
    tb_logger.writer.add_hparams(hparam_dict, metric_dict)

if __name__ == "__main__":
    ## TODO: add info about how long did training take
    start_time = time.time()
    main()
    print(f"Finished Training. Took: {(time.time() - start_time) / 60:.02f}m")