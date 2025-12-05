"""Wavelet-based 3D U-Net style segmentation model.

This file implements a self contained 3D CT multi-organ segmentation network
with simplified wavelet frequency decomposition and fusion. The architecture
follows the specification provided in the task description.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_norm_layer(norm: str, num_features: int) -> nn.Module:
    """Utility to create a normalization layer.

    Args:
        norm: Normalization type ("instance" or "batch").
        num_features: Number of channels for the norm layer.
    """
    if norm == "batch":
        return nn.BatchNorm3d(num_features)
    return nn.InstanceNorm3d(num_features, affine=True)


class ResidualUnit3D(nn.Module):
    """Basic residual unit with two Conv3d layers."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = "instance",
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = get_norm_layer(norm, out_channels)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = get_norm_layer(norm, out_channels)
        self.act2 = nn.LeakyReLU(inplace=True)

        self.proj: nn.Module | None = None
        if in_channels != out_channels:
            self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Forward pass with residual connection."""
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.proj is not None:
            identity = self.proj(identity)

        out = out + identity
        out = self.act2(out)
        return out


class ResidualConvBlock(nn.Module):
    """Stack of residual units."""

    def __init__(self, in_channels: int, out_channels: int, n_stages: int, norm: str = "instance") -> None:
        super().__init__()
        stages: List[nn.Module] = []
        for i in range(n_stages):
            stages.append(
                ResidualUnit3D(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    norm=norm,
                )
            )
        self.blocks = nn.Sequential(*stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Apply residual units sequentially."""
        return self.blocks(x)


class DWT3D(nn.Module):
    """Simplified 3D discrete wavelet transform placeholder.

    This module preserves spatial resolution and splits input into low/high
    frequency components using convolutions.
    """

    def __init__(self, in_channels: int, out_channels: int, norm: str = "instance") -> None:
        super().__init__()
        self.low_conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.low_norm = get_norm_layer(norm, out_channels)
        self.high_conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.high_norm = get_norm_layer(norm, out_channels)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: D401
        """Return low and high frequency components."""
        low = self.low_conv(x)
        low = self.low_norm(low)
        low = self.act(low)

        high = self.high_conv(x)
        high = self.high_norm(high)
        high = self.act(high)
        return low, high


class IDWT3D(nn.Module):
    """Inverse transform placeholder that fuses low and aggregated high."""

    def __init__(self, in_channels: int, out_channels: int, norm: str = "instance") -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            get_norm_layer(norm, out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, low: torch.Tensor, high_agg: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Fuse low and high aggregated features."""
        x = torch.cat([low, high_agg], dim=1)
        return self.fuse(x)


class MLKA(nn.Module):
    """Simplified Multi-scale Large Kernel Attention for 3D volumes."""

    def __init__(self, channels: int, norm: str = "instance") -> None:
        super().__init__()
        # Two depthwise convolutions with different kernel sizes to emulate large receptive field.
        self.dw_conv5 = nn.Conv3d(channels, channels, kernel_size=5, padding=2, groups=channels, bias=False)
        self.dw_conv7 = nn.Conv3d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=False)
        self.point = nn.Conv3d(channels * 2, channels, kernel_size=1, bias=False)
        self.norm = get_norm_layer(norm, channels)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Apply large-kernel attention and return modulated features."""
        context5 = self.dw_conv5(x)
        context7 = self.dw_conv7(x)
        merged = torch.cat([context5, context7], dim=1)
        attn = self.point(merged)
        attn = self.norm(attn)
        attn = self.act(attn)
        return x * attn


class FreFusion(nn.Module):
    """Fuse low-frequency (after MLKA) and high-frequency features."""

    def __init__(self, in_channels_low: int, in_channels_high: int, out_channels: int, norm: str = "instance") -> None:
        super().__init__()
        self.low_proj = nn.Conv3d(in_channels_low, out_channels, kernel_size=1, bias=False)
        self.high_proj = nn.Conv3d(in_channels_high, out_channels, kernel_size=1, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            get_norm_layer(norm, out_channels),
            nn.LeakyReLU(inplace=True),
        )
        self.high_agg = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            get_norm_layer(norm, out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, low_feat: torch.Tensor, high_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: D401
        """Return fused features and aggregated high frequency."""
        low_p = self.low_proj(low_feat)
        high_p = self.high_proj(high_feat)
        merged = torch.cat([low_p, high_p], dim=1)
        fused = self.fuse(merged)
        high_agg = self.high_agg(merged)
        return fused, high_agg


class WaveletUNet3D(nn.Module):
    """Wavelet-based encoder-decoder network for 3D segmentation."""

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 4,
        base_channels: int = 32,
        norm: str = "instance",
    ) -> None:
        super().__init__()
        # Encoder stage 1
        self.dwt1 = DWT3D(in_channels, base_channels, norm=norm)
        self.mlka1 = MLKA(base_channels, norm=norm)
        self.frefuse1 = FreFusion(base_channels, base_channels, base_channels, norm=norm)
        self.res1 = ResidualConvBlock(in_channels, base_channels, n_stages=1, norm=norm)
        self.down1 = nn.Conv3d(base_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.fusedown1 = nn.Conv3d(base_channels, base_channels, kernel_size=3, stride=2, padding=1, bias=False)

        # Encoder stage 2
        self.dwt2 = DWT3D(base_channels, base_channels * 2, norm=norm)
        self.mlka2 = MLKA(base_channels * 2, norm=norm)
        self.frefuse2 = FreFusion(base_channels * 2, base_channels * 2, base_channels * 2, norm=norm)
        self.res2 = ResidualConvBlock(base_channels, base_channels * 2, n_stages=2, norm=norm)
        self.down2 = nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.fusedown2 = nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1, bias=False)

        # Encoder stage 3
        self.dwt3 = DWT3D(base_channels * 2, base_channels * 4, norm=norm)
        self.mlka3 = MLKA(base_channels * 4, norm=norm)
        self.frefuse3 = FreFusion(base_channels * 4, base_channels * 4, base_channels * 4, norm=norm)
        self.res3 = ResidualConvBlock(base_channels * 2, base_channels * 4, n_stages=3, norm=norm)
        self.down3 = nn.Conv3d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.fusedown3 = nn.Conv3d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1, bias=False)

        # Encoder stage 4 (bottleneck, no further downsampling)
        self.dwt4 = DWT3D(base_channels * 4, base_channels * 8, norm=norm)
        self.mlka4 = MLKA(base_channels * 8, norm=norm)
        self.frefuse4 = FreFusion(base_channels * 8, base_channels * 8, base_channels * 8, norm=norm)
        self.res4 = ResidualConvBlock(base_channels * 4, base_channels * 8, n_stages=3, norm=norm)
        self.fuseproj4 = nn.Conv3d(base_channels * 8, base_channels * 8, kernel_size=3, padding=1, bias=False)

        # Decoder blocks
        self.dec4 = ResidualConvBlock(base_channels * 8, base_channels * 8, n_stages=3, norm=norm)
        self.idwt4 = IDWT3D(base_channels * 16, base_channels * 8, norm=norm)

        self.dec3 = ResidualConvBlock(base_channels * 8 + base_channels * 4, base_channels * 4, n_stages=3, norm=norm)
        self.idwt3 = IDWT3D(base_channels * 8, base_channels * 4, norm=norm)

        self.dec2 = ResidualConvBlock(base_channels * 4 + base_channels * 2, base_channels * 2, n_stages=2, norm=norm)
        self.idwt2 = IDWT3D(base_channels * 4, base_channels * 2, norm=norm)

        self.dec1 = ResidualConvBlock(base_channels * 2 + base_channels, base_channels, n_stages=1, norm=norm)
        self.idwt1 = IDWT3D(base_channels * 2, base_channels, norm=norm)

        self.head = nn.Conv3d(base_channels, n_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Forward pass of the WaveletUNet3D."""
        # Encoder 1
        low1, high1 = self.dwt1(x)
        low1_mlka = self.mlka1(low1)
        fused1, high_agg1 = self.frefuse1(low1_mlka, high1)

        spatial1 = self.res1(x)
        spatial1_down = self.down1(spatial1)
        fused1_proj = self.fusedown1(fused1)
        total1 = spatial1_down + fused1_proj

        # Encoder 2
        low2, high2 = self.dwt2(total1)
        low2_mlka = self.mlka2(low2)
        fused2, high_agg2 = self.frefuse2(low2_mlka, high2)

        spatial2 = self.res2(total1)
        spatial2_down = self.down2(spatial2)
        fused2_proj = self.fusedown2(fused2)
        total2 = spatial2_down + fused2_proj

        # Encoder 3
        low3, high3 = self.dwt3(total2)
        low3_mlka = self.mlka3(low3)
        fused3, high_agg3 = self.frefuse3(low3_mlka, high3)

        spatial3 = self.res3(total2)
        spatial3_down = self.down3(spatial3)
        fused3_proj = self.fusedown3(fused3)
        total3 = spatial3_down + fused3_proj

        # Encoder 4 (bottleneck, no further downsample)
        low4, high4 = self.dwt4(total3)
        low4_mlka = self.mlka4(low4)
        fused4, high_agg4 = self.frefuse4(low4_mlka, high4)

        spatial4 = self.res4(total3)
        fused4_proj = self.fuseproj4(fused4)
        total4 = spatial4 + fused4_proj

        # Decoder 4
        dec4_res = self.dec4(total4)
        x_dec = self.idwt4(dec4_res, high_agg4)

        # Decoder 3
        x_dec_up = F.interpolate(x_dec, size=total3.shape[2:], mode="trilinear", align_corners=False)
        dec3_in = torch.cat([x_dec_up, total3], dim=1)
        dec3_res = self.dec3(dec3_in)
        x_dec = self.idwt3(dec3_res, high_agg3)

        # Decoder 2
        x_dec_up = F.interpolate(x_dec, size=total2.shape[2:], mode="trilinear", align_corners=False)
        dec2_in = torch.cat([x_dec_up, total2], dim=1)
        dec2_res = self.dec2(dec2_in)
        x_dec = self.idwt2(dec2_res, high_agg2)

        # Decoder 1
        x_dec_up = F.interpolate(x_dec, size=total1.shape[2:], mode="trilinear", align_corners=False)
        dec1_in = torch.cat([x_dec_up, total1], dim=1)
        dec1_res = self.dec1(dec1_in)
        x_dec = self.idwt1(dec1_res, high_agg1)

        # Final upsample to input resolution if needed
        if x_dec.shape[2:] != x.shape[2:]:
            x_dec = F.interpolate(x_dec, size=x.shape[2:], mode="trilinear", align_corners=False)

        logits = self.head(x_dec)
        return logits


def count_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Basic smoke test
    torch.manual_seed(0)
    model = WaveletUNet3D(in_channels=1, n_classes=5, base_channels=32)
    x = torch.randn(2, 1, 64, 128, 128)
    y = model(x)
    print(f"Parameter count: {count_parameters(model):,}")
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    assert y.shape == (2, 5, 64, 128, 128), "Output shape mismatch"
