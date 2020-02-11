"""Make predict on validation and measure metric or just make predict for test"""

import os
import cv2
import torch
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path

import apex
import pytorch_tools as pt
import albumentations as albu
from torch.utils.data import DataLoader

# local imports
from arg_parser import get_parser
from dataset import OpenCitiesDataset, OpenCitiesTestDataset
from utils import ToCudaLoader, ToTensor, MODEL_FROM_NAME
from callbacks import ThrJaccardScore

def main():
    parser = get_parser()
    parser.add_argument("--no_val", action="store_true", help="Disable validation")
    parser.add_argument("--no_test", action="store_true", help="Disable prediction on test")
    parser.add_argument("--short_predict", action="store_true", help="Predict only first 10 images")
    parser.add_argument("--thr", default=0.5, type=float, help="Threshold for cutting")
    FLAGS = parser.parse_args()
    assert os.path.exists(FLAGS.outdir), "You have to pass config after training to inference script"
    # get model
    print("Loading model")
    model = MODEL_FROM_NAME[FLAGS.segm_arch](FLAGS.arch, **FLAGS.model_params)#.cuda()
    sd = torch.load(os.path.join(FLAGS.outdir, "model.chpn"))["state_dict"]
    model.load_state_dict(sd)
    model = model.cuda()
    model = apex.amp.initialize(model, verbosity=0)
    print("Loaded model")
    # get validation dataloaders
    val_aug = albu.Compose([
        albu.CenterCrop(FLAGS.size, FLAGS.size),
        albu.Normalize(),
        ToTensor(),
    ])
    val_dtst = OpenCitiesDataset(split="val", transform=val_aug, buildings_only=True)
    val_dtld = DataLoader(val_dtst, batch_size=FLAGS.bs, shuffle=False, num_workers=8)
    val_dtld = ToCudaLoader(val_dtld)

    if not FLAGS.no_val:
        runner = pt.fit_wrapper.Runner(
            model, 
            None, 
            pt.losses.JaccardLoss(), 
            [pt.metrics.JaccardScore(), ThrJaccardScore(thr=FLAGS.thr)])
        _, (jacc_score, thr_jacc_score) = runner.evaluate(val_dtld)
        print(f"Validation Jacc Score: {thr_jacc_score:.4f}")

    if FLAGS.no_test:
        return

    # Predict on test
    # for now simply resize it to proper size
    test_aug = albu.Compose([
        albu.Resize(FLAGS.size, FLAGS.size), # TODO: check how does intrepolation affect results
        albu.Normalize(),
        ToTensor(),
    ])
    test_dtst = OpenCitiesTestDataset(transform=test_aug)
    # test_dtld = DataLoader(val_dtst, batch_size=FLAGS.bs, shuffle=False, num_workers=8)

    preds_path = Path("data/preds")
    preds_preview_path = Path(FLAGS.outdir) / "preds_preview"
    shutil.rmtree(preds_preview_path, ignore_errors=True)
    preds_path.mkdir(exist_ok=True)
    preds_preview_path.mkdir(exist_ok=True)

    for imgs_count, (img, aug_img, idx) in enumerate(tqdm(test_dtst)):
        aug_img = aug_img.view(1, *aug_img.shape) # add batch dimension
        pred = pt.utils.misc.to_numpy(model(aug_img.cuda())).squeeze()
        pred = cv2.resize(pred, (1024, 1024))
        pred = (pred > FLAGS.thr).astype(np.uint8)
        # make copy of the image with houses in red and save them both together to check that it's valid
        img2 = img.copy()
        img2[pred.astype(bool)] = [255, 0, 0]
        combined = cv2.cvtColor(np.hstack([img, img2]), cv2.COLOR_RGB2BGR)
        # only save preview with --short_predict. only save predicts for full test run. 
        if FLAGS.short_predict:
            cv2.imwrite(str(preds_preview_path / (idx + ".jpg")), combined)
            if imgs_count > 10:
                break
        else:
            cv2.imwrite(str(preds_path / (idx + ".tif")), pred)
        


if __name__ == "__main__":
    main()