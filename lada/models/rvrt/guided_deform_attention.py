# SPDX-FileCopyrightText: Lada Authors
# SPDX-License-Identifier: AGPL-3.0

import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidedDeformAttention(nn.Module):
    """Guided Deformable Attention (GDA) from RVRT.

    For each spatial query position, predicts K sampling offsets and
    attention weights via conv layers, samples features at those
    positions by bilinear interpolation, and aggregates with softmax.

    Replaces optical-flow-based alignment with content-adaptive
    deformable sampling. No pre-computed flow required.

    Args:
        dim (int): Input channel dimension.
        num_heads (int): Number of attention heads. Default: 8.
        num_points (int): Sampling points per query per head. Default: 9.
    """

    def __init__(self, dim, num_heads=8, num_points=9):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_points = num_points
        self.head_dim = dim // num_heads

        self.norm = nn.LayerNorm(dim)

        self.offset_conv = nn.Conv2d(
            dim, num_heads * num_points * 2,
            kernel_size=3, stride=1, padding=1)

        self.attn_conv = nn.Conv2d(
            dim, num_heads * num_points,
            kernel_size=3, stride=1, padding=1)

        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

        self.offset_scale = nn.Parameter(torch.tensor(0.0))

        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        nn.init.constant_(self.attn_conv.weight, 0)
        nn.init.constant_(self.attn_conv.bias, 0)

    def forward(self, x):
        """Forward pass.

        Args:
            x (Tensor): Feature map of shape (B, C, H, W).

        Returns:
            Tensor: Attended features of shape (B, C, H, W).
        """
        B, C, H, W = x.shape

        # LayerNorm on channel dim: (B, H, W, C) → (B, H, W, C)
        x_norm = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        # Predict offsets: (B, n_heads*n_points*2, H, W)
        offset = self.offset_conv(x_norm)
        offset = offset * self.offset_scale.tanh()
        offset = offset.reshape(B, self.num_heads, self.num_points * 2, H, W)

        # Predict attention weights: (B, n_heads*n_points, H, W)
        attn = self.attn_conv(x_norm)
        attn = attn.reshape(B, self.num_heads, self.num_points, H, W)

        # Split offset: (B, heads, points, 2, H, W)
        offset = offset.reshape(
            B, self.num_heads, self.num_points, 2, H, W)

        # Reference grid at each spatial position: (H, W, 2)
        ref_y, ref_x = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=x.dtype),
            torch.arange(W, device=x.device, dtype=x.dtype),
            indexing='ij')
        ref = torch.stack((ref_x, ref_y), dim=-1)  # (H, W, 2)
        ref = ref.view(1, 1, H, W, 1, 2)
        # → (1, 1, H, W, 1, 2)

        # Permute offset to (B, heads, H, W, points, 2)
        offset = offset.permute(0, 1, 4, 5, 2, 3)

        # Sampling locations: ref + offset = (B, heads, H, W, points, 2)
        sample_locs = ref + offset

        # Normalize to [-1, 1] for grid_sample
        sample_locs[..., 0] = 2.0 * sample_locs[..., 0] / max(W - 1, 1) - 1.0
        sample_locs[..., 1] = 2.0 * sample_locs[..., 1] / max(H - 1, 1) - 1.0

        # Reshape for batched grid_sample:
        # Stack "points" into the spatial width dimension
        # grid_sample input: (B*heads, head_dim, H, W)
        # grid_sample grid:  (B*heads, H, W*points, 2)
        sample_locs = sample_locs.reshape(
            B * self.num_heads, H, W * self.num_points, 2)

        x_heads = x.reshape(B, self.num_heads, self.head_dim, H, W)
        x_heads = x_heads.reshape(B * self.num_heads, self.head_dim, H, W)

        # Sample features: (B*heads, head_dim, H, W*points)
        sampled = F.grid_sample(
            x_heads, sample_locs,
            mode='bilinear', padding_mode='zeros', align_corners=True)
        sampled = sampled.reshape(
            B, self.num_heads, self.head_dim, H, W, self.num_points)
        # → (B, heads, head_dim, H, W, points)

        # Softmax attention over points
        attn = F.softmax(attn, dim=2)  # (B, heads, points, H, W)
        attn = attn.permute(0, 1, 3, 4, 2).unsqueeze(2)
        # → (B, heads, 1, H, W, points)

        # Weighted sum: (B, heads, head_dim, H, W)
        output = (sampled * attn).sum(dim=-1)
        output = output.reshape(B, C, H, W)

        return self.proj(output)
