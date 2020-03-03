import os
import yaml
import time

import apex
import torch

import pytorch_tools as pt
from pytorch_tools.optim import optimizer_from_name
from pytorch_tools.fit_wrapper.callbacks import Callback as NoClb

from src.arg_parser import parse_args
from src.dataset import get_dataloaders
from src.utils import MODEL_FROM_NAME
from src.utils import criterion_from_list
from src.utils import TargetWrapper
from src.callbacks import ThrJaccardScore
from src.callbacks import PredictViewer


def main():
    FLAGS = parse_args()
    print(FLAGS)
    pt.utils.misc.set_random_seed(42)  # fix all seeds
    ## dump config
    os.makedirs(FLAGS.outdir, exist_ok=True)
    yaml.dump(vars(FLAGS), open(FLAGS.outdir + '/config.yaml', 'w'))

    ## get dataloaders
    train_dtld, val_dtld = get_dataloaders(FLAGS.datasets, FLAGS.augmentation, FLAGS.bs, FLAGS.size)

    ## get model and optimizer
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
        # don't want to load optimizer to get to maybe better point
        # try:
        #     optimizer.load_state_dict(checkpoint["optimizer"])
        # except:  # may raise an error if another optimzer was used
        #     print("Failed to load state dict into optimizer")
    num_params = pt.utils.misc.count_parameters(model)[0]
    print(f"Number of parameters: {num_params / 1e6:.02f}M")

    ## TODO: add lookahead
    ## train on fp16 by default
    model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1", verbosity=0, loss_scale=1024)

    ## get loss. fixed for now.
    bce_loss = TargetWrapper(pt.losses.CrossEntropyLoss(mode="binary").cuda(), "mask")
    bce_loss.name = "BCE"
    loss = criterion_from_list(FLAGS.criterion).cuda()
    # loss = 0.5 * pt.losses.CrossEntropyLoss(mode="binary", weight=[5]).cuda()
    print("Loss for this run is: ", loss)
    ## get runner
    sheduler = pt.fit_wrapper.callbacks.PhasesScheduler(FLAGS.phases)
    runner = pt.fit_wrapper.Runner(
        model,
        optimizer,
        criterion=loss,
        callbacks=[
            pt.fit_wrapper.callbacks.Timer(),
            pt.fit_wrapper.callbacks.ConsoleLogger(),
            sheduler,
            pt.fit_wrapper.callbacks.SegmCutmix(1, 1) if FLAGS.cutmix else NoClb(),
            pt.fit_wrapper.callbacks.FileLogger(FLAGS.outdir),
            PredictViewer(FLAGS.outdir, num_images=8),
            pt.fit_wrapper.callbacks.CheckpointSaver(FLAGS.outdir, save_name="model.chpn"),
        ],
        metrics=[
            bce_loss,
            TargetWrapper(pt.metrics.JaccardScore(mode="binary").cuda(), "mask"),
            TargetWrapper(ThrJaccardScore(thr=0.5), "mask"),
        ],
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

        # need to init again to avoid nan's in loss
        optimizer = optimizer_from_name(FLAGS.optim)(
            model.parameters(),
            lr=FLAGS.lr,
            weight_decay=FLAGS.weight_decay,  # **FLAGS.optim_params TODO: add additional optim params if needed
        )
        model, optimizer = apex.amp.initialize(model, optimizer, opt_level="O1", verbosity=0, loss_scale=2048)
        runner.state.model = model
        runner.state.optimizer = optimizer

    runner.fit(
        train_dtld,
        val_loader=val_dtld,
        start_epoch=FLAGS.decoder_warmup_epochs,
        epochs=sheduler.tot_epochs,
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
