"""Make predict on validation and measure metric or just make predict for test"""

import os
import cv2
import apex
import torch
import shutil
import numpy as np
from tqdm import tqdm
from pathlib import Path
from multiprocessing import pool

import apex
import pytorch_tools as pt
from pytorch_tools.utils.misc import to_numpy
import albumentations as albu
from torch.utils.data import DataLoader

# local imports
from src.arg_parser import get_parser
from src.dataset import OpenCitiesDataset, OpenCitiesTestDataset
from src.augmentations import get_aug
from src.utils import ToCudaLoader, ToTensor, MODEL_FROM_NAME, TargetWrapper
from src.callbacks import ThrJaccardScore

PREDS_PATH = None
THR = None
def save_pred(single_pred_idx):
    single_pred, idx = single_pred_idx
    global PREDS_PATH
    global THR
    single_pred = cv2.resize(single_pred, (1024, 1024))
    single_pred = (single_pred > THR).astype(np.uint8)
    cv2.imwrite(str(PREDS_PATH / (idx + ".tif")), single_pred)

@torch.no_grad()
def main():
    parser = get_parser()
    parser.add_argument("--no_val", action="store_true", help="Disable validation")
    parser.add_argument("--no_test", action="store_true", help="Disable prediction on test")
    parser.add_argument("--short_predict", default=0, type=int, help="Number of first images to show predict for")
    parser.add_argument("--thr", default=0.5, type=float, help="Threshold for cutting")
    parser.add_argument("--tta", action="store_true", help="Enables TTA")
    FLAGS = parser.parse_args()
    assert os.path.exists(FLAGS.outdir), "You have to pass config after training to inference script"
    # get model
    print("Loading model")
    model = MODEL_FROM_NAME[FLAGS.segm_arch](FLAGS.arch, **FLAGS.model_params)  # .cuda()
    sd = torch.load(os.path.join(FLAGS.outdir, "model.chpn"))["state_dict"]
    model.load_state_dict(sd)
    model = model.cuda().eval()
    if FLAGS.tta:
        model = pt.tta_wrapper.TTA(
            model, segm=True, h_flip=True, rotation=[90], merge="gmean", activation="sigmoid"
        )
    model = apex.amp.initialize(model, verbosity=0)
    print("Loaded model")
    # get validation dataloaders
    val_aug = albu.Compose([albu.CenterCrop(FLAGS.size, FLAGS.size), albu.Normalize(), ToTensor(),])
    val_dtst = OpenCitiesDataset(split="val", transform=val_aug, buildings_only=True)
    val_dtld = DataLoader(val_dtst, batch_size=FLAGS.bs, shuffle=False, num_workers=8)
    val_dtld = ToCudaLoader(val_dtld)

    if not FLAGS.no_val:
        runner = pt.fit_wrapper.Runner(
            model, 
            None, 
            TargetWrapper(pt.losses.JaccardLoss(), "mask"), 
            [
                TargetWrapper(pt.metrics.JaccardScore(), "mask"), 
                TargetWrapper(ThrJaccardScore(thr=FLAGS.thr), "mask"),
            ],
        )
        _, (jacc_score, thr_jacc_score) = runner.evaluate(val_dtld)
        print(f"Validation Jacc Score: {thr_jacc_score:.4f}")

    if FLAGS.no_test:
        return

    # Predict on test
    # for now simply resize it to proper size
    test_aug = get_aug("test", size=FLAGS.size) 

    test_dataset = OpenCitiesTestDataset(transform=test_aug)
    test_loader = DataLoader(test_dataset, batch_size=FLAGS.bs, shuffle=False, num_workers=8, )

    global PREDS_PATH
    global THR
    THR = FLAGS.thr
    PREDS_PATH = Path("data/preds")
    preds_preview_path = Path(FLAGS.outdir) / "preds_preview"
    shutil.rmtree(preds_preview_path, ignore_errors=True)
    PREDS_PATH.mkdir(exist_ok=True)
    preds_preview_path.mkdir(exist_ok=True)
    workers_pool = pool.Pool()
    cnt = 0
    for imgs, aug_imgs, idxs in tqdm(test_loader):
        # aug_img = aug_img.view(1, *aug_img.shape)  # add batch dimension
        preds = model(aug_imgs.cuda())
        if not FLAGS.tta:
            preds = preds.sigmoid()
        preds = to_numpy(preds).squeeze()
        workers_pool.map(save_pred, zip(preds, idxs))

        if FLAGS.short_predict:
            for img, idx, pred in zip(imgs, idxs, preds):
                img2 = to_numpy(img).copy()
                pred = cv2.resize(pred, (1024, 1024))
                img2[(pred > THR).astype(bool)] = [255, 0, 0]
                combined = cv2.cvtColor(np.hstack([img, img2]), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(preds_preview_path / (idx + ".jpg")), combined)
            if cnt < FLAGS.short_predict:
                cnt += FLAGS.bs
            else:
                break
        # pred = cv2.resize(pred, (1024, 1024))
        # pred = (pred > FLAGS.thr).astype(np.uint8)
        # make copy of the image with houses in red and save them both together to check that it's valid
        # img2 = img.copy()
        # img2[pred.astype(bool)] = [255, 0, 0]
        # combined = cv2.cvtColor(np.hstack([img, img2]), cv2.COLOR_RGB2BGR)
        # only save preview with --short_predict. only save predicts for full test run.
        # if FLAGS.short_predict:
        #     cv2.imwrite(str(preds_preview_path / (idx + ".jpg")), combined)
        #     if imgs_count > 30:
        #         break
        # else:
            # cv2.imwrite(str(preds_path / (idx + ".tif")), pred)
    workers_pool.close()

if __name__ == "__main__":
    main()
