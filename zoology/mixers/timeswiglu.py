import torch
from torch import nn 
import torch.nn.functional as F
from einops import rearrange


class TimeSwiGLU(nn.Module):
    def __init__(self, attention_dropout=0.0, causal=True):
        super().__init__()
        self.dropout_p = attention_dropout
        self.silu = nn.SiLU()
        self.causal = causal

    def forward(self, qkv_BxSx7xHxD):
        S = qkv_BxSx7xHxD.shape[1]
        (q1_BxSxHxD, k1_BxSxHxD,
         q2_BxSxHxD, k2_BxSxHxD,
         q3_BxSxHxD, k3_BxSxHxD, v_BxSxHxD) = qkv_BxSx7xHxD.unbind(dim=2)
        a1_BxHxSxS = torch.einsum("bthd,bshd->bhts", q1_BxSxHxD, k1_BxSxHxD)
        a2_BxHxSxS = torch.einsum("bthd,bshd->bhts", q2_BxSxHxD, k2_BxSxHxD)
        a3_BxHxSxS = torch.einsum("bthd,bshd->bhts", q3_BxSxHxD, k3_BxSxHxD)
        if self.causal:
            mask_SxS = torch.tril(torch.ones((S, S), device=a1_BxHxSxS.device))
            a1_BxHxSxS = mask_SxS * a1_BxHxSxS
            a2_BxHxSxS = mask_SxS * a2_BxHxSxS
            a3_BxHxSxS = mask_SxS * a3_BxHxSxS
        a1_BxHxSxS = F.dropout(a1_BxHxSxS, self.dropout_p if self.training else 0.0)
        a2_BxHxSxS = F.dropout(a2_BxHxSxS, self.dropout_p if self.training else 0.0)
        a3_BxHxSxS = F.dropout(a3_BxHxSxS, self.dropout_p if self.training else 0.0)
        t1_BxSxHxD = self.silu(torch.einsum("bhts,bshd->bthd", a1_BxHxSxS, v_BxSxHxD))
        t2_BxSxHxD = torch.einsum("bhts,bshd->bthd", a2_BxHxSxS, v_BxSxHxD)
        return torch.einsum("bhts,bshd->bthd", a3_BxHxSxS, t1_BxSxHxD * t2_BxSxHxD)


class MHTimeSwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int=1,
        bias: bool=True,
        dropout: float=0.0,
        causal: bool=True,
        **kwargs
    ):
        super().__init__()
        self.Wproj = nn.Linear(d_model, 7 * d_model, bias=bias)
        self.d_model = d_model
        self.out_proj = nn.Linear(d_model, d_model)
        self.head_dim = self.d_model // num_heads
        self.inner_attn = TimeSwiGLU(attention_dropout=dropout, causal=causal)
        self.norm = nn.GroupNorm(num_heads, d_model)
        self.num_heads = num_heads

    def forward(self, x_BxSxHD):
        qkv_BxSx7HD = self.Wproj(x_BxSxHD)
        qkv_BxSx7xHxD = rearrange(
            qkv_BxSx7HD, "... (seven h d) -> ... seven h d", seven=7, d=self.head_dim
        )
        ctx_BxSxHD = rearrange(self.inner_attn(qkv_BxSx7xHxD), "... h d -> ... (h d)")
        # independently normalize each head
        ctx_BxSxHD = torch.transpose(self.norm(torch.transpose(ctx_BxSxHD, -1, -2)), -1, -2)
        return self.out_proj(ctx_BxSxHD)

    def state_size(self, **kwargs):
        return 3 * self.head_dim * self.head_dim * self.num_heads
