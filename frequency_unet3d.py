"""Wavelet-based 3D U-Net segmentation model using project-provided blocks."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.blocks import ResidualConvBlock, FreqFuse3D
from network.utils import DWT3D, IDWT3D
from network.MLKA import MLKA


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class HighProjector(nn.Module):
    """Project 7 wavelet high-frequency subbands to a target channel size."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_channels * 7, out_channels * 7, kernel_size=1)

    def forward(self, highs: torch.Tensor) -> torch.Tensor:
        b, c, s, d, h, w = highs.shape
        highs = highs.view(b, c * s, d, h, w)
        highs = self.proj(highs)
        highs = highs.view(b, -1, 7, d, h, w)
        return highs


class LowProjector(nn.Module):
    """Project low-frequency subband to target channels."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, low: torch.Tensor) -> torch.Tensor:
        return self.proj(low)


class WaveletUNet3D(nn.Module):
    """Four-stage encoder-decoder with Haar DWT/IDWT skips and MLKA fusion."""

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 4,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        # shared transforms
        self.dwt1, self.dwt2, self.dwt3, self.dwt4 = DWT3D(), DWT3D(), DWT3D(), DWT3D()
        self.idwt1, self.idwt2, self.idwt3, self.idwt4 = IDWT3D(), IDWT3D(), IDWT3D(), IDWT3D()

        # projections for wavelet outputs
        self.low_proj1 = LowProjector(in_channels, base_channels)
        self.high_proj1 = HighProjector(in_channels, base_channels)

        self.low_proj2 = LowProjector(base_channels, base_channels * 2)
        self.high_proj2 = HighProjector(base_channels, base_channels * 2)

        self.low_proj3 = LowProjector(base_channels * 2, base_channels * 4)
        self.high_proj3 = HighProjector(base_channels * 2, base_channels * 4)

        self.low_proj4 = LowProjector(base_channels * 4, base_channels * 8)
        self.high_proj4 = HighProjector(base_channels * 4, base_channels * 8)

        # MLKA on low frequencies
        self.mlka1 = MLKA(base_channels)
        self.mlka2 = MLKA(base_channels * 2)
        self.mlka3 = MLKA(base_channels * 4)
        self.mlka4 = MLKA(base_channels * 8)

        # frequency fusion (returns fused low/high features)
        self.fuse1 = FreqFuse3D(base_channels)
        self.fuse2 = FreqFuse3D(base_channels * 2)
        self.fuse3 = FreqFuse3D(base_channels * 4)
        self.fuse4 = FreqFuse3D(base_channels * 8)

        # spatial path residual blocks and downsampling
        self.res1 = ResidualConvBlock(in_channels, base_channels, n_stages=1, normalization="instancenorm")
        self.down1 = nn.Conv3d(base_channels, base_channels, kernel_size=3, stride=2, padding=1)

        self.res2 = ResidualConvBlock(base_channels, base_channels * 2, n_stages=2, normalization="instancenorm")
        self.down2 = nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1)

        self.res3 = ResidualConvBlock(base_channels * 2, base_channels * 4, n_stages=3, normalization="instancenorm")
        self.down3 = nn.Conv3d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1)

        self.res4 = ResidualConvBlock(base_channels * 4, base_channels * 8, n_stages=3, normalization="instancenorm")

        # decoder residual blocks
        self.dec4 = ResidualConvBlock(base_channels * 8, base_channels * 8, n_stages=3, normalization="instancenorm")
        self.dec3 = ResidualConvBlock(base_channels * 8 + base_channels * 4, base_channels * 4, n_stages=3, normalization="instancenorm")
        self.dec2 = ResidualConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2, n_stages=2, normalization="instancenorm")
        self.dec1 = ResidualConvBlock(base_channels * 2 + base_channels, base_channels, n_stages=1, normalization="instancenorm")

        self.head = nn.Conv3d(base_channels, n_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _wavelet_path(
        self, dwt: DWT3D, low_proj: LowProjector, high_proj: HighProjector, mlka: MLKA, fuse: FreqFuse3D, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        low, highs = dwt(x)
        low = low_proj(low)
        highs = high_proj(highs)
        low_mlka = mlka(low)
        fused = fuse(low_mlka, highs)
        return fused, highs, low

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder 1
        fused1, highs1, low1 = self._wavelet_path(self.dwt1, self.low_proj1, self.high_proj1, self.mlka1, self.fuse1, x)
        spatial1 = self.res1(x)
        spatial1_down = self.down1(spatial1)
        total1 = spatial1_down + fused1

        # Encoder 2
        fused2, highs2, low2 = self._wavelet_path(self.dwt2, self.low_proj2, self.high_proj2, self.mlka2, self.fuse2, total1)
        spatial2 = self.res2(total1)
        spatial2_down = self.down2(spatial2)
        total2 = spatial2_down + fused2

        # Encoder 3
        fused3, highs3, low3 = self._wavelet_path(self.dwt3, self.low_proj3, self.high_proj3, self.mlka3, self.fuse3, total2)
        spatial3 = self.res3(total2)
        spatial3_down = self.down3(spatial3)
        total3 = spatial3_down + fused3

        # Encoder 4 (bottleneck)
        fused4, highs4, low4 = self._wavelet_path(self.dwt4, self.low_proj4, self.high_proj4, self.mlka4, self.fuse4, total3)
        spatial4 = self.res4(total3)
        total4 = spatial4 + fused4

        # Decoder 4 (upsample via IDWT)
        dec4_res = self.dec4(total4)
        x_dec = self.idwt4(dec4_res, highs4)

        # Decoder 3
        if x_dec.shape[2:] != total3.shape[2:]:
            x_dec = F.interpolate(x_dec, size=total3.shape[2:], mode="trilinear", align_corners=False)
        dec3_in = torch.cat([x_dec, total3], dim=1)
        dec3_res = self.dec3(dec3_in)
        x_dec = self.idwt3(dec3_res, highs3)

        # Decoder 2
        if x_dec.shape[2:] != total2.shape[2:]:
            x_dec = F.interpolate(x_dec, size=total2.shape[2:], mode="trilinear", align_corners=False)
        dec2_in = torch.cat([x_dec, total2], dim=1)
        dec2_res = self.dec2(dec2_in)
        x_dec = self.idwt2(dec2_res, highs2)

        # Decoder 1
        if x_dec.shape[2:] != total1.shape[2:]:
            x_dec = F.interpolate(x_dec, size=total1.shape[2:], mode="trilinear", align_corners=False)
        dec1_in = torch.cat([x_dec, total1], dim=1)
        dec1_res = self.dec1(dec1_in)
        x_dec = self.idwt1(dec1_res, highs1)

        # Final projection
        if x_dec.shape[2:] != x.shape[2:]:
            x_dec = F.interpolate(x_dec, size=x.shape[2:], mode="trilinear", align_corners=False)
        logits = self.head(x_dec)
        return logits


if __name__ == "__main__":
    torch.manual_seed(0)
    model = WaveletUNet3D(in_channels=1, n_classes=5, base_channels=32)
    x = torch.randn(2, 1, 64, 128, 128)
    y = model(x)
    print(f"Parameter count: {count_parameters(model):,}")
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    assert y.shape == (2, 5, 64, 128, 128)

