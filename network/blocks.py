import torch
from torch import nn
import torch.nn.functional as F
from .utils import norm3d


class ResidualConvBlock(nn.Module):
    """
    结构:
        ├─ Conv3d(3x3x3) -> Norm -> ReLU
        ├─ Conv3d(3x3x3) -> Norm
        ├─ [可选] 1x1x1 投影 (当 in_ch != out_ch 时)
        └─ 残差相加 -> ReLU
    """
    def __init__(self, in_ch, out_ch, normalization='none'):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.norm1 = norm3d(normalization, out_ch)
        self.act1  = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = norm3d(normalization, out_ch)

        self.proj = None
        if in_ch != out_ch:
            self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)

        self.act_out = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        
        if self.proj is not None:
            identity = self.proj(identity)

        out = out + identity
        out = self.act_out(out)
        return out





__all__ = [
    'ResidualConvBlock',
]