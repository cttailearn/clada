# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from lada.models.basicvsrpp.mmagic.registry import MODELS
from .guided_deform_attention import GuidedDeformAttention


class RVRTBlock(nn.Module):
    """Single RVRT transformer block.

    Structure: LN → GDA (+residual) → LN → FFN (+residual).

    Args:
        dim (int): Feature channel dimension.
        num_heads (int): Attention heads for GDA. Default: 8.
        num_points (int): Sampling points per head. Default: 9.
        mlp_ratio (float): FFN expansion ratio. Default: 2.66.
    """

    def __init__(self, dim, num_heads=8, num_points=9, mlp_ratio=2.66):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.gda = GuidedDeformAttention(dim, num_heads, num_points)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1),
        )

    def forward(self, x):
        # x: (B, C, H, W)
        # GDA
        residual = x
        x_norm = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.gda(x_norm) + residual
        # FFN
        residual = x
        x_norm = self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.ffn(x_norm) + residual
        return x


@MODELS.register_module()
class RVRTNet(BaseModule):
    """RVRT (Recurrent Video Restoration Transformer) generator.

    Processes video frames in overlapping groups with a shared
    transformer backbone and recurrent hidden state propagation.
    Uses Guided Deformable Attention (GDA) instead of optical flow
    for content-adaptive feature alignment.

    Paper: "Recurrent Video Restoration Transformer with Guided
           Deformable Attention", Liang et al., NeurIPS 2022.

    Args:
        mid_channels (int): Feature channel dimension. Default: 64.
        num_blocks (int): RVRTBlocks per group. Default: 8.
        num_heads (int): GDA attention heads. Default: 8.
        num_points (int): GDA sampling points per head. Default: 9.
        group_size (int): Frames per group. Default: 3.
        group_overlap (int): Overlap between consecutive groups.
            Default: 1.
        mlp_ratio (float): FFN expansion ratio. Default: 2.66.
    """

    def __init__(self,
                 mid_channels=64,
                 num_blocks=8,
                 num_heads=8,
                 num_points=9,
                 group_size=3,
                 group_overlap=1,
                 mlp_ratio=2.66):
        super().__init__()
        self.mid_channels = mid_channels
        self.num_blocks = num_blocks
        self.group_size = group_size
        self.group_overlap = group_overlap

        # Shallow feature extraction (full resolution, no downsampling)
        self.feat_extract = nn.Conv2d(3, mid_channels, 3, 1, 1)

        # Shared RVRT blocks
        self.blocks = nn.ModuleList([
            RVRTBlock(mid_channels, num_heads, num_points, mlp_ratio)
            for _ in range(num_blocks)
        ])

        # Recurrent hidden state projection
        self.hidden_proj = nn.Conv2d(mid_channels, mid_channels, 1)

        # Reconstruction (same-size output)
        self.reconstruction = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, 3, 3, 1, 1),
        )

    def _build_groups(self, T):
        """Build overlapping frame group indices."""
        stride = self.group_size - self.group_overlap
        groups = []
        start = 0
        while start + self.group_size <= T:
            groups.append(list(range(start, start + self.group_size)))
            start += stride
        if start < T and start > 0:
            groups.append(list(range(T - self.group_size, T)))
        return groups

    def forward(self, lqs):
        """Forward pass.

        Args:
            lqs (Tensor): Low-quality video frames of shape
                (N, T, 3, H, W), values in [0, 1].

        Returns:
            Tensor: Restored frames of shape (N, T, 3, H, W).
        """
        N, T, C, H, W = lqs.shape

        # 1. Shallow feature extraction (per-frame)
        feats = self.feat_extract(lqs.reshape(N * T, C, H, W))
        feats = feats.reshape(N, T, self.mid_channels, H, W)

        # 2. Build overlapping groups
        groups = self._build_groups(T)

        # 3. Process groups recurrently
        hidden = None
        accumulated = [None] * T
        counts = [0] * T

        for indices in groups:
            # Extract group features: (N, G, mid_ch, H, W) → (N*G, mid_ch, H, W)
            group_feat = feats[:, indices, :, :, :]
            group_feat = group_feat.reshape(
                N * len(indices), self.mid_channels, H, W)

            # Add recurrent hidden from previous group
            if hidden is not None:
                group_feat = group_feat + self.hidden_proj(hidden)

            # RVRT blocks
            x = group_feat
            for block in self.blocks:
                x = block(x)

            # Detach for truncated BPTT — prevents memory from scaling
            # with number of groups
            hidden = x.detach()

            # Distribute back to per-frame storage with averaging
            x = x.reshape(N, len(indices), self.mid_channels, H, W)
            for local_i, global_i in enumerate(indices):
                if accumulated[global_i] is None:
                    accumulated[global_i] = x[:, local_i]
                else:
                    accumulated[global_i] = accumulated[global_i] + x[:, local_i]
                counts[global_i] += 1

        # Average overlapping frames
        out_feats_list = []
        for i in range(T):
            out_feats_list.append(accumulated[i] / counts[i])
        out_feats = torch.stack(out_feats_list, dim=1)  # (N, T, mid_ch, H, W)

        # 4. Reconstruction
        out_feats = out_feats.reshape(N * T, self.mid_channels, H, W)
        outputs = self.reconstruction(out_feats)
        outputs = outputs.reshape(N, T, C, H, W)

        # 5. Global residual
        return outputs + lqs


@MODELS.register_module()
class RVRTGanNet(RVRTNet):
    """GAN-compatible RVRT generator wrapper.

    Extends RVRTNet to support ``return_lqs=True``, returning a
    (outputs, lqs) tuple as expected by RealBasicVSR.forward_train.
    """

    def forward(self, lqs, return_lqs=False):
        outputs = super().forward(lqs)
        if return_lqs:
            return outputs, lqs
        return outputs
