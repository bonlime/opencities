"""Make predict on validation and measure metric or just make predict for test"""

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

import apex
import pytorch_tools as pt
import albumentations as albu
from torch.utils.data import DataLoader

# local imports
from arg_parser import get_parser
from dataset import OpenCitiesDataset, OpenCitiesTestDataset
from utils import ToCudaLoader, ToTensor, MODEL_FROM_NAME


MODEL_FROM_NAME = {
    "unet": pt.segmentation_models.Unet,
    "linknet": pt.segmentation_models.Linknet,
    "deeplab": pt.segmentation_models.DeepLabV3,
}

def main():
    parser = get_parser()
    parser.add_argument("--no_val", action="store_true", help="Disable validation")
    parser.add_argument("--no_test", action="store_true", help="Disable prediction on test")
    parser.add_argument("--short_predict", action="store_true", help="Predict only first 10 images")
    FLAGS = parser.parse_args()
    assert os.path.exists(FLAGS.outdir), "You have to pass config after training to inference script"
    # get model
    print("Loading model")
    model = MODEL_FROM_NAME[FLAGS.segm_arch](FLAGS.arch, num_classes=1)#.cuda()
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
        runner = pt.fit_wrapper.Runner(model, None, pt.losses.JaccardLoss(), pt.metrics.JaccardScore())
        _, (jacc_score,) = runner.evaluate(val_dtld)
        print(f"Validation Jacc Score: {jacc_score:.4f}")

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

    preds_path = "data/preds"
    preds_preview_path = "data/preds_preview"
    os.makedirs(preds_path, exist_ok=True)
    os.makedirs(preds_preview_path, exist_ok=True)

    for imgs_count, (img, aug_img, idx) in enumerate(tqdm(test_dtst)):
        aug_img = aug_img.view(1, *aug_img.shape) # add batch dimension
        pred = pt.utils.misc.to_numpy(model(aug_img.cuda())).squeeze()
        pred = cv2.resize(pred, (1024, 1024))
        pred = (pred > 0.5).astype(np.uint8)
        # make copy of the image with houses in red and save them both together to check that it's valid
        img2 = img.copy()
        img2[pred.astype(bool)] = [255, 0, 0]
        combined = cv2.cvtColor(np.hstack([img, img2]), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(preds_preview_path, idx+".png"), combined)
        cv2.imwrite(os.path.join(preds_path, idx+".tif"), pred)
        
        if FLAGS.short_predict and imgs_count > 10:
            break


if __name__ == "__main__":
    main()