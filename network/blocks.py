import torch
from torch import nn
import torch.nn.functional as F
from .utils import norm3d


class ResidualConvBlock(nn.Module):
    """
    结构:
        输入 -> [Conv3d(3x3x3) -> Norm -> ReLU] * (n_stages - 1)
             ->  Conv3d(3x3x3) -> Norm
             ->  [可选] 1x1x1 投影 (当 in_ch != out_ch 时)
             ->  残差相加 -> ReLU
    """
    def __init__(self, in_ch, out_ch, n_stages=2, normalization='none'):
        super().__init__()
        assert n_stages >= 1, "n_stages 必须 >= 1"
        self.n_stages = n_stages

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.acts  = nn.ModuleList()

        for i in range(n_stages):
            # 第一层: in_ch -> out_ch，其余层: out_ch -> out_ch
            conv_in_ch = in_ch if i == 0 else out_ch
            conv = nn.Conv3d(conv_in_ch, out_ch, kernel_size=3, padding=1, bias=False)
            self.convs.append(conv)

            norm = norm3d(normalization, out_ch)
            self.norms.append(norm)

            if i < n_stages - 1:
                self.acts.append(nn.ReLU(inplace=True))
            else:
                self.acts.append(None)

        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)

        self.act_out = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = x

        # 主分支: 多层 Conv3d + Norm (+ ReLU)
        for i in range(self.n_stages):
            out = self.convs[i](out)
            out = self.norms[i](out)
            if self.acts[i] is not None:
                out = self.acts[i](out)

        # 残差分支: 可选 1x1 投影
        if self.proj is not None:
            identity = self.proj(identity)

        out = out + identity
        out = self.act_out(out)
        return out


class FreqFuse3D(nn.Module):
    """
    输入:
        low: [B, C, D, H, W]      低频子带（LLL）
        highs: [B, C, 7, D, H, W] 高频子带（7个）
    
    输出:
        [B, C, D, H, W] 融合后的特征
    """
    def __init__(self, channels, normalization='instancenorm', reduction=4):
        super().__init__()
        self.channels = channels

        # 7 个高频子带聚合的 1x1x1 卷积
        self.high_agg_conv = nn.Conv3d(channels * 7, channels, kernel_size=1, bias=False)

        # 频段注意力：[B,7] -> [B,7] 的权重
        hidden = max(7 * 2, 8)
        self.band_mlp = nn.Sequential(
            nn.Linear(7, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 7),
            nn.Sigmoid()
        )

        # 通道注意力：SE-style
        mid_ch = max(channels // reduction, 8)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),                  # [B, C, 1, 1, 1]
            nn.Conv3d(channels, mid_ch, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_ch, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, low, highs):
        """
        low: [B, C, D, H, W]
        highs: [B, C, 7, D, H, W]
        返回: [B, C, D, H, W]
        """
        B, C, S, D, H, W = highs.shape
        assert S == 7, f"Expect 7 high-frequency subbands, got {S}"

        # 1) 频段描述：全局平均池化
        band_desc = highs.mean(dim=(3, 4, 5))   # [B, C, 7]
        band_desc = band_desc.mean(dim=1)       # [B, 7]

        # 2) MLP 生成 band-wise 权重
        band_weights = self.band_mlp(band_desc)         # [B, 7]
        band_weights = band_weights.view(B, 1, S, 1, 1, 1)  # [B, 1, 7, 1, 1, 1]

        # 3) 对高频子带加权
        highs_weighted = highs * band_weights           # [B, C, 7, D, H, W]

        # 4) 聚合 7 个子带：reshape -> 1x1x1 Conv
        highs_reshaped = highs_weighted.view(B, C * S, D, H, W)  # [B, 7C, D, H, W]
        high_agg = self.high_agg_conv(highs_reshaped)            # [B, C, D, H, W]

        # 5) 与低频融合 + 通道注意力 + 残差细化
        base = low + high_agg                       # [B, C, D, H, W]
        ch_att = self.channel_att(base)             # [B, C, 1, 1, 1]
        fused = base * ch_att                       # [B, C, D, H, W]
    
        return fused




__all__ = [
    'ResidualConvBlock',

]

