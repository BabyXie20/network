"""Custom 3D wavelet UNet with MLKA fusion and IDWT skips."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.blocks import ResidualConvBlock, FreqFuse3D
from network.utils import DWT3D, IDWT3D
from network.MLKA import MLKA


class HighProjector(nn.Module):
    """Project wavelet high-frequency subbands to the target channel size."""

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
    """Project the low-frequency wavelet subband to the target channel size."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, low: torch.Tensor) -> torch.Tensor:
        return self.proj(low)


class FreqFusionWithHighSkip(FreqFuse3D):
    """FreqFuse3D that also returns the weighted high-frequency subbands."""

    def forward(self, low: torch.Tensor, highs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, s, d, h, w = highs.shape
        assert s == 7, f"Expected 7 high-frequency subbands, got {s}"

        band_desc = highs.mean(dim=(3, 4, 5)).mean(dim=1)
        band_weights = self.band_mlp(band_desc).view(b, 1, s, 1, 1, 1)
        highs_weighted = highs * band_weights

        highs_reshaped = highs_weighted.view(b, c * s, d, h, w)
        high_agg = self.high_agg_conv(highs_reshaped)

        base = low + high_agg
        ch_att = self.channel_att(base)
        fused = base * ch_att

        return fused, highs_weighted


class WaveletFusionUNet3D(nn.Module):
    """
    Encoder-decoder network that fuses spatial and frequency paths via DWT/IDWT.

    Encoder:
        Each stage applies DWT3D, projects subbands, processes low-frequency with MLKA,
        fuses with high-frequency bands, and adds the result to a spatial residual path
        followed by strided convolution downsampling (except the final stage).

    Decoder:
        Uses encoder fused outputs as skip connections for ResidualConvBlocks and the
        weighted high-frequency subbands as the IDWT3D skip inputs.
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 4,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        channel_list = [base_channels * (2 ** i) for i in range(4)]
        self.dwt_layers = nn.ModuleList([DWT3D() for _ in range(4)])
        self.idwt_layers = nn.ModuleList([IDWT3D() for _ in range(4)])

        self.low_projectors = nn.ModuleList(
            [LowProjector(in_channels if i == 0 else channel_list[i - 1], ch) for i, ch in enumerate(channel_list)]
        )
        self.high_projectors = nn.ModuleList(
            [HighProjector(in_channels if i == 0 else channel_list[i - 1], ch) for i, ch in enumerate(channel_list)]
        )

        self.mlka_blocks = nn.ModuleList([MLKA(ch) for ch in channel_list])
        self.fusion_blocks = nn.ModuleList([FreqFusionWithHighSkip(ch) for ch in channel_list])

        self.encoder_blocks = nn.ModuleList(
            [
                ResidualConvBlock(in_channels, channel_list[0], n_stages=1, normalization="instancenorm"),
                ResidualConvBlock(channel_list[0], channel_list[1], n_stages=2, normalization="instancenorm"),
                ResidualConvBlock(channel_list[1], channel_list[2], n_stages=3, normalization="instancenorm"),
                ResidualConvBlock(channel_list[2], channel_list[3], n_stages=3, normalization="instancenorm"),
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                nn.Conv3d(channel_list[0], channel_list[0], kernel_size=3, stride=2, padding=1),
                nn.Conv3d(channel_list[1], channel_list[1], kernel_size=3, stride=2, padding=1),
                nn.Conv3d(channel_list[2], channel_list[2], kernel_size=3, stride=2, padding=1),
            ]
        )

        self.decoder_blocks = nn.ModuleList(
            [
                ResidualConvBlock(channel_list[3], channel_list[3], n_stages=3, normalization="instancenorm"),
                ResidualConvBlock(channel_list[3] + channel_list[2], channel_list[2], n_stages=3, normalization="instancenorm"),
                ResidualConvBlock(channel_list[2] + channel_list[1], channel_list[1], n_stages=2, normalization="instancenorm"),
                ResidualConvBlock(channel_list[1] + channel_list[0], channel_list[0], n_stages=1, normalization="instancenorm"),
            ]
        )

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
        self,
        stage_idx: int,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        low, highs = self.dwt_layers[stage_idx](x)
        low = self.low_projectors[stage_idx](low)
        highs = self.high_projectors[stage_idx](highs)
        low = self.mlka_blocks[stage_idx](low)
        fused, highs_weighted = self.fusion_blocks[stage_idx](low, highs)
        return fused, highs_weighted

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_totals: List[torch.Tensor] = []
        high_skips: List[torch.Tensor] = []

        # Encoder stages
        for idx in range(4):
            fused, highs = self._wavelet_path(idx, x)
            spatial = self.encoder_blocks[idx](x)
            if idx < len(self.downsamples):
                spatial = self.downsamples[idx](spatial)
            total = spatial + fused
            enc_totals.append(total)
            high_skips.append(highs)
            x = total

        # Decoder stage 4
        x = self.decoder_blocks[0](enc_totals[3])
        x = self.idwt_layers[3](x, high_skips[3])

        # Decoder stage 3
        if x.shape[2:] != enc_totals[2].shape[2:]:
            x = F.interpolate(x, size=enc_totals[2].shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, enc_totals[2]], dim=1)
        x = self.decoder_blocks[1](x)
        x = self.idwt_layers[2](x, high_skips[2])

        # Decoder stage 2
        if x.shape[2:] != enc_totals[1].shape[2:]:
            x = F.interpolate(x, size=enc_totals[1].shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, enc_totals[1]], dim=1)
        x = self.decoder_blocks[2](x)
        x = self.idwt_layers[1](x, high_skips[1])

        # Decoder stage 1
        if x.shape[2:] != enc_totals[0].shape[2:]:
            x = F.interpolate(x, size=enc_totals[0].shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, enc_totals[0]], dim=1)
        x = self.decoder_blocks[3](x)
        x = self.idwt_layers[0](x, high_skips[0])

        logits = self.head(x)
        return logits


if __name__ == "__main__":
    torch.manual_seed(0)
    model = WaveletFusionUNet3D(in_channels=1, n_classes=5, base_channels=32)
    x = torch.randn(1, 1, 32, 64, 64)
    with torch.no_grad():
        y = model(x)
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    assert y.shape == (1, 5, 32, 64, 64)
