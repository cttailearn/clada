# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from torch.nn.utils import spectral_norm

from .registry import MODELS


def _get_efficientnet(arch: str):
    """Load a pre-trained EfficientNet and return its features module."""
    import torchvision.models as models

    weight_map = {
        'efficientnet-b0': models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        'efficientnet-b1': models.EfficientNet_B1_Weights.IMAGENET1K_V1,
        'efficientnet-b2': models.EfficientNet_B2_Weights.IMAGENET1K_V1,
        'efficientnet-b3': models.EfficientNet_B3_Weights.IMAGENET1K_V1,
        'efficientnet-b4': models.EfficientNet_B4_Weights.IMAGENET1K_V1,
    }
    model_fn = getattr(models, arch.replace('-', '_'))
    net = model_fn(weights=weight_map[arch])
    return net.features


# EfficientNet-B0 feature map channels and effective strides
# after each entry in the features Sequential.
# Stride 2  -> H/2
# Stride 4  -> H/4
# Stride 8  -> H/8
# Stride 16 -> H/16
# Stride 32 -> H/32
_EFFICIENTNET_LEVELS = {
    'efficientnet-b0': {
        'indices': [2, 4, 6, 7],
        'channels': [24, 80, 192, 320],
    },
    'efficientnet-b1': {
        'indices': [3, 5, 7, 8],
        'channels': [24, 96, 240, 448],
    },
    'efficientnet-b2': {
        'indices': [3, 5, 7, 8],
        'channels': [24, 104, 264, 528],
    },
    'efficientnet-b3': {
        'indices': [3, 5, 7, 8],
        'channels': [32, 112, 296, 624],
    },
    'efficientnet-b4': {
        'indices': [4, 6, 8, 9],
        'channels': [32, 128, 336, 752],
    },
}

_EFFICIENTNET_NORM_MEAN = torch.tensor([0.485, 0.456, 0.406])
_EFFICIENTNET_NORM_STD = torch.tensor([0.229, 0.224, 0.225])


@MODELS.register_module()
class ProjectedDiscriminator(BaseModule):
    """Projected GAN Discriminator using frozen pre-trained feature networks.

    Uses a frozen EfficientNet backbone to extract multi-scale features,
    projects them to a common channel dimension, and applies a small
    trainable CNN head for real/fake discrimination.

    Ref:
        Projected GANs Converge Faster (Sauer et al., NeurIPS 2021)
        https://arxiv.org/abs/2111.01007

    Args:
        in_channels (int): Number of input channels. Default: 3.
        mid_channels (int): Channel dimension after projection.
            Default: 64.
        feature_network (str): EfficientNet variant to use as frozen
            feature extractor. Supported: efficientnet-b0 through b4.
            Default: 'efficientnet-b0'.
        use_spectral_norm (bool): Whether to apply spectral normalization
            to the discriminator head convolutions. Default: False.
    """

    def __init__(self,
                 in_channels=3,
                 mid_channels=64,
                 feature_network='efficientnet-b0',
                 use_spectral_norm=False):

        super().__init__()

        if feature_network not in _EFFICIENTNET_LEVELS:
            raise ValueError(
                f'Unsupported feature network: {feature_network}. '
                f'Choose from: {list(_EFFICIENTNET_LEVELS.keys())}')

        level_cfg = _EFFICIENTNET_LEVELS[feature_network]
        self.feature_indices = level_cfg['indices']
        self.feature_channels = level_cfg['channels']

        # frozen feature extractor
        self.feature_extractor = _get_efficientnet(feature_network)
        self.feature_extractor.requires_grad_(False)
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        # per-level 1x1 projection layers
        self.projectors = nn.ModuleList([
            nn.Conv2d(ch, mid_channels, kernel_size=1)
            for ch in self.feature_channels
        ])

        total_channels = mid_channels * len(self.feature_indices)

        conv = spectral_norm if use_spectral_norm else lambda m: m

        self.head = nn.Sequential(
            conv(nn.Conv2d(total_channels, mid_channels * 4, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            conv(nn.Conv2d(mid_channels * 4, mid_channels * 2, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            conv(nn.Conv2d(mid_channels * 2, mid_channels, 3, 1, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channels, 1, 3, 1, 1),
        )

    def forward(self, img):
        """Forward pass.

        Args:
            img (Tensor): Input image tensor of shape (N, C, H, W).
                Values expected in [0, 1] range.

        Returns:
            Tensor: Real/fake logits map of shape (N, 1, H', W').
        """
        # normalize from [0,1] to ImageNet stats for the frozen backbone
        mean = _EFFICIENTNET_NORM_MEAN.to(img.device).view(1, -1, 1, 1)
        std = _EFFICIENTNET_NORM_STD.to(img.device).view(1, -1, 1, 1)
        x = (img - mean) / std

        feats = []
        for i, layer in enumerate(self.feature_extractor):
            x = layer(x)
            if i in self.feature_indices:
                idx = self.feature_indices.index(i)
                projected = self.projectors[idx](x)
                feats.append(projected)

        # upsample all feature maps to the largest spatial resolution
        h, w = feats[0].shape[2:]
        for i in range(1, len(feats)):
            if feats[i].shape[2:] != feats[0].shape[2:]:
                feats[i] = F.interpolate(
                    feats[i], size=(h, w),
                    mode='bilinear', align_corners=False)

        out = torch.cat(feats, dim=1)
        return self.head(out)
