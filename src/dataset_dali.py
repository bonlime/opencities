import math
import numpy as np
import pandas as pd
from nvidia import dali
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIClassificationIterator

class ExternalInputIterator:
    """Loads raw bytes of images and masks"""
    def __init__(self, train=False, batch_size=8):
        df = pd.read_csv("/home/zakirov/datasets/AerialImageDataset/inria_tiles.csv", index_col=0)
        df = df[df.train == int(train)]
        self.train = train

        self.img_ids = df["image"].values
        self.mask_ids = df["mask"].values
        self.batch_size = batch_size
        self.n = len(self.img_ids)
        self._reset()
        
    def __iter__(self):
        return self
    
    def _reset(self):
        self.i = 0
        # only shuffle train images
        if self.train:
            indexes = np.arange(len(self.img_ids))
            np.random.shuffle(indexes)
            self.img_ids = self.img_ids[indexes]
            self.mask_ids = self.mask_ids[indexes]

    def __next__(self):
        if self.i >= self.n:
            self._reset()
        imgs = []
        masks = []
        for _ in range(self.batch_size):
            img_f = open(self.img_ids[self.i], 'rb')
            imgs.append(np.frombuffer(img_f.read(), dtype=np.uint8))
            mask_f = open(self.mask_ids[self.i], 'rb')
            masks.append(np.frombuffer(mask_f.read(), dtype=np.uint8))
            self.i = (self.i + 1) % self.n
        return (imgs, masks)
    next = __next__

    def __len__(self):
        return self.n


class ExternalSourcePipeline(dali.pipeline.Pipeline):
    def __init__(
        self,
        train=False,
        batch_size=16,
        size=384,
        num_threads=4, 
        device_id=0
    ):
        super(ExternalSourcePipeline, self).__init__(
            batch_size, num_threads, device_id, seed=42
        )
        self.eii = iter(
            ExternalInputIterator(train, batch_size)
        )
        self.images_input = ops.ExternalSource()
        self.masks_input = ops.ExternalSource()
        if train:
            fixed_area = (size / 784)**2
            self.decode = ops.ImageDecoderRandomCrop(
                device="mixed",
                random_area=[fixed_area*0.7, fixed_area*1.3], 
                random_aspect_ratio=[0.7, 1.3],
            )
        else:
            self.decode = ops.ImageDecoderCrop(
                device="mixed", 
                crop=(size, size)
            )
        self.resize = ops.Resize(
            device="gpu", 
            interp_type=types.INTERP_TRIANGULAR,
            resize_x=size, 
            resize_y=size
        )
        self.mask_resize = ops.Resize(
            device="gpu", 
            interp_type=types.INTERP_NN,
            resize_x=size, 
            resize_y=size
        )
        self.normalize = ops.CropMirrorNormalize(
            device="gpu",
            mean=[0.5 * 255],  # 0.456 * 255, 0.406 * 255],
            std=[0.5 * 255],  # , 0.224 * 255, 0.225 * 255],
            output_layout=types.NCHW
        )
        self.mask_normalize = ops.CropMirrorNormalize(
            device="gpu",
            mean=[0],
            std=[255],
            output_layout=types.NCHW,
            image_type=types.GRAY,
        )
        # extra augmentations
        self.to_gray = ops.ColorSpaceConversion(
            device="gpu", image_type=types.RGB, output_type=types.GRAY
        )
        self.contrast = ops.BrightnessContrast(device="gpu")
        self.hsv = ops.Hsv(device="gpu")
        self.jitter = ops.Jitter(device ="gpu")
        # self.rng1 = ops.Uniform(range=[0, 1])
        self.rng2 = ops.Uniform(range=[0.8,1.2])
        self.rng3 = ops.Uniform(range=[-30, 30]) # for hue
        self.coin03 = ops.CoinFlip(probability=0.3)
        self.train = train

    def define_graph(self):
        # get raw bytes
        self.images = self.images_input()
        self.masks = self.masks_input()
        # decode. for train it's random crop, for val - center crop
        out_imgs, out_masks = self.decode([self.images, self.masks])
        # dont need it, because we have TargetWrapper
        # out_masks = self.to_gray(out_masks)

        out_imgs = self.resize(out_imgs)
        out_masks = self.mask_resize(out_masks)

        # return not augmented images
        if not self.train:
            return self.normalize(out_imgs), self.mask_normalize(out_masks)

        # color augmentations
        out_imgs = self.contrast(
            out_imgs, contrast=self.rng2(), brightness=self.rng2()
        )
        out_imgs = self.hsv(
            out_imgs, hue=self.rng3(), saturation=self.rng2(), value=self.rng2(),
        )

        # jitter with 0.5 prob
        out_imgs = self.jitter(out_imgs, mask=self.coin03()) 

        # normalize and flip
        flip = self.coin03()
        out_imgs = self.normalize(out_imgs, mirror=flip)
        out_masks = self.mask_normalize(out_masks, mirror=flip)
        return out_imgs, out_masks

    def iter_setup(self):
        (imgs, masks) = self.eii.next()
        self.feed_input(self.images, imgs)
        self.feed_input(self.masks, masks)


class DaliLoader:
    """Wrap dali to look like torch dataloader"""
    
    def __init__(
        self, 
        train=False, 
        batch_size=32, 
        # workers=4, 
        size=384, 
    ):
        """Returns train or val iterator over Imagenet data"""
        pipe = ExternalSourcePipeline(
            train=train, 
            size=size, 
            batch_size=batch_size,
        )
        pipe.build()
        self.loader = DALIClassificationIterator(
            pipe,
            size=len(ExternalInputIterator(train, batch_size)),
            auto_reset=True,
            fill_last_batch=train,  # want real accuracy on validiation
            last_batch_padded=True,  # want epochs to have the same length
        )

    def __len__(self):
        return math.ceil(self.loader._size / self.loader.batch_size)
    
    def __iter__(self):
        return ((batch[0]['data'], batch[0]['label']) for batch in self.loader)

