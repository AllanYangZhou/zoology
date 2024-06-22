import torch
from torch import nn 
import torch.nn.functional as F
from einops import rearrange


class LinearSelfAttention(nn.Module):
    def __init__(self, attention_dropout=0.0, causal=True):
        super().__init__()
        self.dropout_p = attention_dropout
        self.causal = causal

    def forward(self, qkv_BxSx3xHxD):
        """Implements the multihead linear attention."""
        S = qkv_BxSx3xHxD.shape[1]
        q_BxSxHxD, k_BxSxHxD, v_BxSxHxD = qkv_BxSx3xHxD.unbind(dim=2)
        a_BxHxSxS = torch.einsum("bthd,bshd->bhts", q_BxSxHxD, k_BxSxHxD)
        if self.causal:
            mask_SxS = torch.tril(torch.ones((S, S), device=a_BxHxSxS.device))
            a_BxHxSxS = mask_SxS * a_BxHxSxS
        a_BxHxSxS = F.dropout(a_BxHxSxS, self.dropout_p if self.training else 0.0)
        out_BxSxHxD = torch.einsum("bhts,bshd->bthd", a_BxHxSxS, v_BxSxHxD)
        return out_BxSxHxD


class MHLA(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int=1,
        bias: bool=True,
        dropout: float=0.0,
        causal: bool=True,
        **kwargs,
    ):
        super().__init__()
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.d_model = d_model
        self.out_proj = nn.Linear(d_model, d_model)
        self.head_dim = self.d_model // num_heads
        self.inner_attn = LinearSelfAttention(attention_dropout=dropout, causal=causal)
        self.norm = nn.GroupNorm(num_heads, d_model)
        self.num_heads = num_heads

    def forward(self, x_BxSxHD):
        qkv_BxSx3HD = self.Wqkv(x_BxSxHD)
        qkv_BxSx3xHxD = rearrange(
            qkv_BxSx3HD, "... (three h d) -> ... three h d", three=3, d=self.head_dim
        )
        ctx_BxSxHD = rearrange(self.inner_attn(qkv_BxSx3xHxD), "... h d -> ... (h d)")
        # independently normalize each head
        ctx_BxSxHD = torch.transpose(self.norm(torch.transpose(ctx_BxSxHD, -1, -2)), -1, -2)
        return self.out_proj(ctx_BxSxHD)

    def state_size(self, **kwargs):
        return self.head_dim * self.head_dim * self.num_heads


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


class TTT(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super().__init__()
        self.dropout_p = attention_dropout

    def forward(self, qkv_BxSx3xHxD):
        """Implements 2 steps of TTT with linear inner loop."""
        q_BxSxHxD, k_BxSxHxD, v_BxSxHxD = qkv_BxSx3xHxD.unbind(dim=2)
        # mask(QK')
        a1_BxHxSxS = torch.tril(torch.einsum("bthd,bshd->bhts", q_BxSxHxD, k_BxSxHxD))
        # mask(QK')K
        o2_BxHxSxD = torch.einsum("bhts,bshd->bhtd", a1_BxHxSxS, k_BxSxHxD)
        # mask(QK')KK'
        a2_BxHxSxS = torch.tril(torch.einsum("bhtd,bshd->bhts", o2_BxHxSxD, k_BxSxHxD))
        a1_BxHxSxS = F.dropout(
            2 * a1_BxHxSxS - a2_BxHxSxS,
            self.dropout_p if self.training else 0.0)
        return torch.einsum("bhts,bshd->bthd", a1_BxHxSxS, v_BxSxHxD)


class MHTTT(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int=1,
        bias: bool=True,
        dropout: float=0.0,
        **kwargs
    ):
        super().__init__()
        self.Wproj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.d_model = d_model
        self.out_proj = nn.Linear(d_model, d_model)
        self.head_dim = self.d_model // num_heads
        self.inner_attn = TTT(attention_dropout=dropout)
        self.norm = nn.GroupNorm(num_heads, d_model)
        self.num_heads = num_heads

    def forward(self, x_BxSxHD):
        qkv_BxSx3HD = self.Wproj(x_BxSxHD)
        qkv_BxSx3xHxD = rearrange(
            qkv_BxSx3HD, "... (three h d) -> ... three h d", three=3, d=self.head_dim
        )
        ctx_BxSxHD = rearrange(self.inner_attn(qkv_BxSx3xHxD), "... h d -> ... (h d)")
        # independently normalize each head
        ctx_BxSxHD = torch.transpose(self.norm(torch.transpose(ctx_BxSxHD, -1, -2)), -1, -2)
        return self.out_proj(ctx_BxSxHD)

    def state_size(self, **kwargs):
        return 2 * self.head_dim * self.head_dim * self.num_heads


def get_ttt_inner(ln):
    def ttt_inner_2step(W0_DxD, Q_TxD, K_TxD, V_TxD):
        Z_TxD = K_TxD @ W0_DxD
        dy = ln(Z_TxD) - V_TxD  # y = ln(z)
        dz1, = torch.func.vjp(ln, Z_TxD)[1](dy)
        # K @ W1, where W1 = W0 - K.T @ dz1, with causal masking.
        Z_TxD = Z_TxD - torch.tril(K_TxD @ K_TxD.T) @ dz1
        dy = ln(Z_TxD) - V_TxD
        dz2, = torch.func.vjp(ln, Z_TxD)[1](dy)

        A_TxT = torch.tril(Q_TxD @ K_TxD.T)
        # Q @ (W0 - K.T @ dz1 - K.T @ dz2), with causal masking.
        return Q_TxD @ W0_DxD + A_TxT @ (-dz1) + A_TxT @ (-dz2)
    return ttt_inner_2step


class MHTTTWithLN(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int=1,
        bias: bool=True,
        dropout: float=0.0,
        **kwargs
    ):
        super().__init__()
        self.Wproj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.d_model = d_model
        self.out_proj = nn.Linear(d_model, d_model)
        self.head_dim = self.d_model // num_heads
        self.inner_attn = TTT(attention_dropout=dropout)
        self.norm = nn.GroupNorm(num_heads, d_model)
        self.num_heads = num_heads
        self.ln = nn.LayerNorm(self.head_dim)
        self.ttt_inner = torch.vmap(
            torch.vmap(get_ttt_inner(self.ln), in_dims=(None, 0, 0, 0)),
            in_dims=(None, 0, 0, 0))
        self.W0_DxD = nn.Parameter(torch.randn(
            self.head_dim, self.head_dim) / self.head_dim**0.5)

    def forward(self, x_BxSxHD):
        qkv_BxSx3HD = self.Wproj(x_BxSxHD)
        qkv_BxHx3xSxD = rearrange(
            qkv_BxSx3HD, "b s (three h d) -> b h three s d", three=3, d=self.head_dim
        )
        q_BxHxSxD, k_BxHxSxD, v_BxHxSxD = qkv_BxHx3xSxD.unbind(dim=2)
        ctx_BxHxSxD = self.ttt_inner(self.W0_DxD, q_BxHxSxD, k_BxHxSxD, v_BxHxSxD)
        ctx_BxSxHD = rearrange(ctx_BxHxSxD, "b h s d -> b s (h d)")
        # independently normalize each head
        ctx_BxSxHD = torch.transpose(self.norm(torch.transpose(ctx_BxSxHD, -1, -2)), -1, -2)
        return self.out_proj(ctx_BxSxHD)

    def state_size(self, **kwargs):
        return 2 * self.head_dim * self.head_dim * self.num_heads
