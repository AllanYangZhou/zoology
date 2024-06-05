import torch
import torch.nn as nn
from torch_dct import dct, idct
from einops import rearrange

from .hyena import Filter


class DiscreteTransform(nn.Module):
    def __init__(self, mode="dct", norm="ortho", dim=1):
        super().__init__()
        self.dim = dim
        self.mode = mode
        self.norm = norm

    def forward(self, x):
        if self.mode == "dct":
            return dct(x, dim=self.dim, n=x.shape[self.dim])
        elif self.mode == "fft":
            return torch.fft.rfft(x, dim=self.dim)

    def inverse(self, x):
        if self.mode == "dct":
            return idct(x, dim=self.dim, n=x.shape[self.dim])
        elif self.mode == "fft":
            return torch.fft.irfft(x, dim=self.dim)


class Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x)


class OrchidOperator(nn.Module):
    def __init__(
            self,
            d_model,
            seq_len,
            d_filter=64,
            l_conv1d=3,
            dxt_mode="fft",
            to_out_proj=True,
            **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len

        # setup projections
        self.in_linear = nn.Linear(d_model, 3 * d_model)
        self.to_out_proj = to_out_proj
        if to_out_proj:
            self.out_linear = nn.Linear(d_model, d_model)

        # setup short conv filter
        width = d_model * 3  # NOTE: was d_model * 2 in paper
        self.short_filter = nn.Conv1d(
            in_channels=width,
            out_channels=width,
            kernel_size=l_conv1d,
            groups=width,
            padding=(l_conv1d-1) // 2  # NOTE: was l_conv1d - 1 in paper
        )

        # setup static conv
        self.static_conv = Filter(
            d_model,
            order=d_filter,
            seq_len=self.seq_len,
            dropout=0,
            bidirectional=False,  # TODO: check
            l_max=self.seq_len,
            num_inner_mlps=1,
        )

        # NOTE: padding was l_conv1d - 1 in paper.
        self.conditioning_nn = nn.Sequential(
            nn.Conv1d(
                in_channels=d_model, out_channels=d_model, kernel_size=l_conv1d,
                groups=d_model, padding=(l_conv1d-1) // 2),
            DiscreteTransform(mode=dxt_mode, dim=2),
            Abs(),
            nn.Conv1d(
                in_channels=d_model, out_channels=d_model, kernel_size=l_conv1d,
                groups=d_model, padding=(l_conv1d-1) // 2),
        )

        self.transform = DiscreteTransform(mode=dxt_mode, dim=-1)

    def forward(self, x_BxSxD, **kwargs):
        x_BxSx3D = self.in_linear(x_BxSxD)
        _, _, v_BxSxD = torch.split(x_BxSx3D, self.d_model, dim=-1)
        # E = D // 2 + 1, due to RFFT
        h_adapt_f_BxDxE = self.conditioning_nn(rearrange(v_BxSxD, "b s d -> b d s"))
        h0_1xSxD = self.static_conv.filter(self.seq_len)
        h0_1xDxS = rearrange(h0_1xSxD, "c s d -> c d s")

        # short conv1d() filter
        x_Bx3DxS = rearrange(x_BxSx3D, "b s d -> b d s")
        x_Bx3DxS = self.short_filter(x_Bx3DxS)  # [..., :self.seq_len]

        s1_BxDxS, s2_BxDxS, v_BxDxS = x_Bx3DxS.split(self.d_model, dim=1)

        y_BxDxS = v_BxDxS * s1_BxDxS
        y_BxDxS = self.adaptive_conv(y_BxDxS, h0_1xDxS, h_adapt_f_BxDxE)
        y_BxDxS = y_BxDxS * s2_BxDxS
        y_BxDxS = rearrange(y_BxDxS, "b d l -> b l d")
        if self.to_out_proj:
            y_BxDxS = self.out_linear(y_BxDxS)
        return y_BxDxS

    def adaptive_conv(self, x_BxDxS, h0_1xDxS, h_adapt_f_BxDxE):
        h0_f_1xDxS = self.transform(h0_1xDxS)
        x_f_BxDxE = self.transform(x_BxDxS)
        y = torch.fft.irfft(x_f_BxDxE * (h0_f_1xDxS + h_adapt_f_BxDxE))
        return y
