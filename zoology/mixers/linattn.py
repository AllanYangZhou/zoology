import torch
from torch import nn 
import torch.nn.functional as F
from einops import rearrange


class LinearSelfAttention(nn.Module):
    def __init__(self, attention_dropout=0.0):
        super().__init__()
        self.dropout_p = attention_dropout

    def forward(self, qkv_BxSx3xHxD):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D)
        """
        q_BxSxHxD, k_BxSxHxD, v_BxSxHxD = qkv_BxSx3xHxD.unbind(dim=2)
        a_BxHxSxS = torch.einsum("bthd,bshd->bhts", q_BxSxHxD, k_BxSxHxD)
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
        **kwargs
    ):
        super().__init__()
        self.Wqkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.d_model = d_model
        self.out_proj = nn.Linear(d_model, d_model)
        self.head_dim = self.d_model // num_heads
        self.inner_attn = LinearSelfAttention(attention_dropout=dropout)
        self.norm = nn.GroupNorm(num_heads, d_model)

    def forward(self, x_BxSxHD):
        qkv_BxSx3HD = self.Wqkv(x_BxSxHD)
        qkv_BxSx3xHxD = rearrange(
            qkv_BxSx3HD, "... (three h d) -> ... three h d", three=3, d=self.head_dim
        )
        ctx_BxSxHD = rearrange(self.inner_attn(qkv_BxSx3xHxD), "... h d -> ... (h d)")
        # independently normalize each head
        ctx_BxSxHD = torch.transpose(self.norm(torch.transpose(ctx_BxSxHD, -1, -2)), -1, -2)
        return self.out_proj(ctx_BxSxHD)