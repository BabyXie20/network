from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Layer normalization that supports channels-first 3D tensors.

    Parameters
    ----------
    normalized_shape : int
        Number of feature channels. For channels-last inputs, this is passed to
        :func:`torch.nn.functional.layer_norm`. For channels-first inputs, the
        normalization is computed manually across channels.
    eps : float, optional
        Small constant to avoid division by zero.
    data_format : str, optional
        Either ``"channels_last"`` (``B, D, H, W, C``) or ``"channels_first"``
        (``B, C, D, H, W``). Only these two formats are supported.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6, data_format: str = "channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


class MLKA(nn.Module):
    """2-branch Multi-Scale Large Kernel Attention for 3D feature maps.

    使用 3x3x3 + 5x5x5 两个尺度的大核 depthwise 3D 卷积。
    通道被二等分，因此 n_feats 只需能被 2 整除。
    """

    def __init__(self, n_feats: int):
        super().__init__()
        if n_feats % 2 != 0:
            raise ValueError("n_feats must be divisible by 2 for MLKA2Branch.")
        i_feats = 2 * n_feats

        self.norm = LayerNorm(n_feats, data_format="channels_first")

        self.proj_first = nn.Conv3d(n_feats, i_feats, kernel_size=1, stride=1, padding=0)

        # 每个分支的通道数
        c = n_feats // 2

        self.LKA3 = nn.Sequential(
            nn.Conv3d(c, c, kernel_size=3, stride=1, padding=1, groups=c),
            nn.Conv3d(
                c,
                c,
                kernel_size=5,
                stride=1,
                padding=(5 // 2) * 2,  # 4
                dilation=2,
                groups=c,
            ),
            nn.Conv3d(c, c, kernel_size=1, stride=1, padding=0),
        )
        self.X3 = nn.Conv3d(c, c, kernel_size=3, stride=1, padding=1, groups=c)

        self.LKA5 = nn.Sequential(
            nn.Conv3d(c, c, kernel_size=5, stride=1, padding=5 // 2, groups=c),
            nn.Conv3d(
                c,
                c,
                kernel_size=7,
                stride=1,
                padding=(7 // 2) * 3,  # 9
                dilation=3,
                groups=c,
            ),
            nn.Conv3d(c, c, kernel_size=1, stride=1, padding=0),
        )
        self.X5 = nn.Conv3d(c, c, kernel_size=5, stride=1, padding=5 // 2, groups=c)

        self.proj_last = nn.Conv3d(n_feats, n_feats, kernel_size=1, stride=1, padding=0)

        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x  

        x = self.norm(x)             # [B, C, D, H, W]
        x = self.proj_first(x)       # [B, 2C, D, H, W]

        a, x = torch.chunk(x, 2, dim=1)   # a: [B, C, D, H, W], x: [B, C, D, H, W]
        a_1, a_2 = torch.chunk(a, 2, dim=1)  # 每个 [B, C/2, D, H, W]

        a = torch.cat(
            [
                self.LKA3(a_1) * self.X3(a_1),
                self.LKA5(a_2) * self.X5(a_2),
            ],
            dim=1,
        )  # a: [B, C, D, H, W]

        x = self.proj_last(x * a) * self.scale + shortcut
        return x
