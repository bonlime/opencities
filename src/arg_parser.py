import os
import configargparse as argparse
from pytorch_tools.utils.misc import get_timestamp

def get_parser():
    parser = argparse.ArgumentParser(
        description="Opencities",
        default_config_files=["configs/base.yaml"],
        args_for_setting_config_path=["-c", "--config_file"],
        config_file_parser_class=argparse.YAMLConfigFileParser,
    )
    add_arg = parser.add_argument

    # slicer args
    add_arg("--zoom_level", default=19, type=int, help="Zoom level for tile slicing")
    add_arg("--tile_size", default=512, type=int, help="Size of tile chip in pixels")
    add_arg("--val_percent", default=0.15, type=float, help="How many tiles to leave for validation")

    # training args
    add_arg("--arch", "-a", default="se_resnet50", help="Backbone architecture")
    add_arg("--model_params", type=eval, default={}, help="Additional model params as kwargs")
    add_arg(
        "--segm_arch",
        default="unet",
        type=str,
        # choices=["unet", "linknet", "deeplab"],
        help="Segmentation architecture to use",
    )
    add_arg(
        "--optim",
        type=str,
        default="adamw",  # choices=['sgd', 'sgdw', 'adam', 'adamw', 'rmsprop', 'radam'],
        help="Optimizer to use (default: adamw)",
    )
    add_arg(
        "-j",
        "--workers",
        default=8,
        type=int,
        help="number of data loading workers (default: 4)",
    )
    add_arg("--resume", default="", type=str, help="path to checkpoint to start from (default: none)")
    add_arg("-n", "--name", default="", type=str, help="Name of this run")
    add_arg("--decoder_warmup_epochs", default=0, type=int, help="Number of epochs for training only decoder")
    add_arg("--epochs", default=100, type=int, help="Total number of epochs")
    add_arg("--weight_decay", "--wd", default=1e-4, type=float, help="weight decay (default: 1e-4)")
    add_arg("--size", default=256, type=int, help="Size of crops to train at")
    add_arg("--bs", default=16, type=int, help="Batch size")
    add_arg("--lr", default=1e-3, type=float, help="starting learning rate")
    add_arg("--outdir", default="", type=str, help="Do not pass it manually")
    add_arg(
        "--augmentation",
        default="medium",
        type=str,
        choices=["light", "medium", "hard"],
        help="How hard augs are"
    )
    # inference args
    return parser
    
     
def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    # add timestamp to name and create this run folder
    assert args.outdir == "", "Do not pass `outdir` param manually"
    timestamp = get_timestamp()
    args.name = args.name + "_" + timestamp if args.name else timestamp
    args.outdir = os.path.join("logs/", args.name)
    return args