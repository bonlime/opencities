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

    # TIFF slicer parameters
    
    add_arg("--border_thickness", default=20, type=int, help="Border for distance transform")
    add_arg("--data_path", type=str, help="Path to raw tier 1/2 data")
    add_arg("--sub_folder", type=str, default='tier1', help='Subfolder to store data')
    add_arg("--zoom_level", default=19, type=int, help="Zoom level for tile slicing")
    add_arg("--tile_size", default=512, type=int, help="Size of tile chip in pixels")
    add_arg("--val_percent", default=0.15, type=float, help="How many tiles to leave for validation")
    add_arg("--random_val", action='store_true', help="Method to generate val set. If False use 15% by Y-coord")

    # training args
    add_arg("--arch", "-a", default="se_resnet50", help="Backbone architecture")
    add_arg("--model_params", type=eval, default={}, help="Additional model params as kwargs")
    add_arg("--segm_arch", default="unet", type=str, help="Segmentation architecture to use")
    add_arg("--optim", type=str, default="adamw", help="Optimizer to use (default: adamw)")
    add_arg("--lookahead", action="store_true", help="Flag to wrap optimizer with lookahead")
    add_arg(
        "-j", "--workers", default=8, type=int, help="number of data loading workers (default: 4)",
    )
    add_arg("--resume", default="", type=str, help="path to checkpoint to start from (default: none)")
    add_arg("-n", "--name", default="", type=str, help="Name of this run")
    add_arg("--decoder_warmup_epochs", default=0, type=int, help="Number of epochs for training only decoder")
    add_arg("--epochs", default=100, type=int, help="Total number of epochs")
    add_arg("--weight_decay", "--wd", default=1e-4, type=float, help="weight decay (default: 1e-4)")
    add_arg("--size", default=256, type=int, help="Size of crops to train at")
    add_arg("--val_size", default=256, type=int, help="Size of crops for validation")
    add_arg("--bs", "--batch_size", default=16, type=int, help="Batch size")
    add_arg("--lr", default=1e-3, type=float, help="starting learning rate")
    add_arg("--outdir", default="", type=str, help="Do not pass it manually")
    add_arg(
        "--augmentation",
        default="medium",
        type=str,
        choices=["light", "medium", "hard"],
        help="How hard augs are",
    )
    add_arg("--cutmix", action="store_true", help="Turns on cutmix aug on input")
    add_arg(
        "--datasets",
        default=["opencities"],
        type=str,
        nargs="+",
        help="Datasets to use for training. Default is only opencities",
    )
    # inference args
    add_arg("--short_epoch", action="store_true", help="Flag to enable debug mod and make very short epochs")
    add_arg(
        "--criterion",
        type=str,
        required=True,
        nargs="+",
        help="List of criterions to use. Should be like `bce 0.5 dice 0.5`",
    )
    add_arg("--opt_level", default="O0", type=str, help="Optimization level for apex")
    add_arg(
        "--phases",
        type=eval,
        action='append',
        help="Specify epoch order of data resize and learning rate schedule:"
        '[{"ep":0,"sz":128,"bs":64},{"ep":5,"lr":1e-2}]',
    )
    add_arg("--dropout", type=float, default=0, help="Max spatial dropout value")
    add_arg("--dropout_epochs", type=int, default=20, help="Number of epochs for dropout to fully take effect")
    add_arg("--train_tta", action="store_true", help="Enables TTA during training")
    add_arg("--buildings_only", action="store_true", help="Flag to filter images without buildings away")
    return parser


def parse_args():
    parser = get_parser()
    args = parser.parse_args()
    # add timestamp to name and create this run folder
    # assert args.outdir == "", "Do not pass `outdir` param manually"
    timestamp = get_timestamp()
    name = args.name + "_" + timestamp if args.name else timestamp
    args.outdir = os.path.join("logs/", name)
    return args
