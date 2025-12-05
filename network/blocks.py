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






__all__ = [
    'ResidualConvBlock',

]
