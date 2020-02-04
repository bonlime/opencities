import configargparse as argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Opencities",
        args_for_setting_config_path=["-c", "--config_file"],
        config_file_parser_class=argparse.YAMLConfigFileParser,
    )
    add_arg = parser.add_argument
    add_arg("--arch", "-a", default="densenet121", help="Backbone architecture")
    add_arg(
        "--segm-arch",
        default="unet",
        type=str,
        choices=["unet", "linknet", "deeplab"],
        help="Segmentation architecture to use",
    )
    add_arg(
        "-j",
        "--workers",
        default=4,
        type=int,
        help="number of data loading workers (default: 4)",
    )
    add_arg("--wd", default=1e-4, type=float, help="weight decay (default: 1e-4)")
    add_arg("--size", default=256, type=int, help="Size of crops to train at")
    return parser.parse_args()
     