from functools import partial
from typing import Union, Callable, List

import torch
from pytorch_toolbelt.modules import ABN, ACT_RELU
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules.decoders import DecoderModule
from pytorch_toolbelt.modules.encoders import EncoderModule
from torch import nn
from torch.nn import functional as F

OUTPUT_MASK_KEY = "mask"


__all__ = ["seresnext50_runet64", "hrnet18_runet64", "hrnet34_runet64", "hrnet48_runet64", "densenet121_runet64"]


class ResidualDoubleConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.residual_path = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        )

        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        skip = self.residual_path(x)
        x = self.main_path(x)
        return F.relu(skip + x, inplace=True)


class RUnetBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, block=ResidualDoubleConvRelu, num_blocks=1):
        super().__init__()

        blocks = []
        for i in range(num_blocks):
            blocks.append(block(in_channels, out_channels))
            in_channels = out_channels

        self.blocks = nn.Sequential(*blocks)

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        x = self.blocks(x)
        return x


class RUnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layer(x)


class RUNetDecoderV2(DecoderModule):
    def __init__(
        self,
        feature_maps: List[int],
        decoder_features: List[int],
        runet_blocks: List[int],
        mask_channels: int,
        last_upsample_filters=None,
        dropout=0.0,
        abn_block=ABN,
    ):
        super().__init__()

        if not isinstance(decoder_features, list):
            decoder_features = [decoder_features * (2 ** i) for i in range(len(feature_maps))]

        if last_upsample_filters is None:
            last_upsample_filters = decoder_features[0]

        self.encoder_features = feature_maps
        self.decoder_features = decoder_features
        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for idx in range(0, len(self.decoder_features))])

        self.bottlenecks = nn.ModuleList(
            [
                RUnetBottleneckBlock(self.encoder_features[-i - 2] + f, f, num_blocks=runet_blocks[i])
                for i, f in enumerate(reversed(self.decoder_features[:]))
            ]
        )

        self.output_filters = decoder_features

        self.last_upsample = RUnetDecoderBlock(decoder_features[0], last_upsample_filters, last_upsample_filters)

        self.final = nn.Conv2d(last_upsample_filters, mask_channels, kernel_size=1)

    def get_decoder(self, layer):
        in_channels = (
            self.encoder_features[layer + 1]
            if layer + 1 == len(self.decoder_features)
            else self.decoder_features[layer + 1]
        )
        return RUnetDecoderBlock(in_channels, self.decoder_features[layer], self.decoder_features[max(layer, 0)])

    def forward(self, feature_maps):

        last_dec_out = feature_maps[-1]

        x = last_dec_out
        for idx, bottleneck in enumerate(self.bottlenecks):
            rev_idx = -(idx + 1)
            decoder = self.decoder_stages[rev_idx]
            x = decoder(x)
            x = bottleneck(x, feature_maps[rev_idx - 1])

        x = self.last_upsample(x)

        f = self.final(x)

        return f


class RUnetV2SegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        num_classes: int,
        unet_channels: Union[int, List[int]],
        runet_blocks: List[int] = (1, 1, 1, 1),
        last_upsample_filters=None,
        dropout=0.25,
        abn_block: Union[ABN, Callable[[int], nn.Module]] = ABN,
        full_size_mask=True,
    ):
        super().__init__()
        self.encoder = encoder

        self.decoder = RUNetDecoderV2(
            feature_maps=encoder.output_filters,
            decoder_features=unet_channels,
            runet_blocks=runet_blocks,
            last_upsample_filters=last_upsample_filters,
            mask_channels=num_classes,
            dropout=dropout,
            abn_block=abn_block,
        )

        self.full_size_mask = full_size_mask

    def forward(self, x):
        features = self.encoder(x)

        # Decode mask
        mask = self.decoder(features)

        if self.full_size_mask:
            mask = F.interpolate(mask, size=x.size()[2:], mode="bilinear", align_corners=False)

        output = {OUTPUT_MASK_KEY: mask}
        # return output
        return mask


def seresnext50_runet64(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.SEResNeXt50Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return RUnetV2SegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[64, 128, 256, 256],
        runet_blocks=[3, 3, 3, 3],
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )


def hrnet18_runet64(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.HRNetV2Encoder18(pretrained=pretrained, layers=[1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return RUnetV2SegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[64, 128, 256],
        runet_blocks=[3, 3, 3, 3],
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )


def hrnet34_runet64(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.HRNetV2Encoder34(pretrained=pretrained, layers=[1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return RUnetV2SegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[128, 128, 256],
        runet_blocks=[3, 3, 3, 3],
        last_upsample_filters=64,
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )


def hrnet48_runet64(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.HRNetV2Encoder48(pretrained=pretrained, layers=[1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return RUnetV2SegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[128, 128, 256],
        runet_blocks=[3, 3, 3, 3],
        last_upsample_filters=64,
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )


def densenet121_runet64(input_channels=3, num_classes=1, dropout=0.0, pretrained=True):
    encoder = E.DenseNet121Encoder(pretrained=pretrained, layers=[1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return RUnetV2SegmentationModel(
        encoder,
        num_classes=num_classes,
        unet_channels=[128, 128, 256],
        runet_blocks=[3, 3, 3, 3],
        last_upsample_filters=64,
        dropout=dropout,
        abn_block=partial(ABN, activation=ACT_RELU),
    )