import typing as tp
import math
import torch
from torch import nn
from einops import rearrange


def deeper_attn_naive(Q_TxD, K_TxD, V_TxD):
    D = Q_TxD.shape[1]
    A1_TxT = torch.tril(torch.exp(Q_TxD @ K_TxD.T / math.sqrt(D)))
    A2_TxT = torch.exp(K_TxD @ K_TxD.T / math.sqrt(D))
    A_TxT = 2 * A1_TxT - torch.tril(A1_TxT @ A2_TxT)
    # normalize
    A_TxT = A_TxT / A_TxT.sum(dim=-1, keepdim=True)
    return A_TxT @ V_TxD


def deeper_attn(Q_TxD, K_TxD, V_TxD):
    T, D = Q_TxD.shape[0], Q_TxD.shape[1]
    mask = torch.triu(torch.ones(T, T, device=Q_TxD.device), diagonal=1).bool()
    QK_TxT = Q_TxD @ K_TxD.T / math.sqrt(D)
    QK_TxT = torch.where(mask, float('-inf'), QK_TxT)
    KK_TxT = K_TxD @ K_TxD.T / math.sqrt(D)
    QK_plus_KK_TxTxT = QK_TxT[:, :, None] + KK_TxT[None, :, :]
    C_T = torch.maximum(
        torch.max(QK_TxT, dim=-1).values,
        torch.max(torch.max(QK_plus_KK_TxTxT, dim=-1).values, dim=-1).values,
    )
    A1_TxT = torch.exp(QK_TxT - C_T[:, None])
    A2_TxT = torch.tril(torch.sum(torch.exp(QK_plus_KK_TxTxT - C_T[:, None, None]), dim=1))
    A_TxT = (2 * A1_TxT - A2_TxT)
    A_TxT = A_TxT / A_TxT.sum(dim=-1, keepdim=True)
    return A_TxT @ V_TxD


class DeeperMHA(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int=1,
        bias: bool=True,
        dropout: float=0.0,
        groupnorm: bool=True,
        **kwargs
    ):
        super().__init__()
        assert not (dropout > 0.0), "dropout not supported"
        self.Wproj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.d_model = d_model
        self.out_proj = nn.Linear(d_model, d_model)
        self.head_dim = self.d_model // num_heads
        self.num_heads = num_heads
        self.norm = None
        if groupnorm:
            self.norm = nn.GroupNorm(num_heads, d_model)
        self.deeper_attn = torch.vmap(
            torch.vmap(deeper_attn, in_dims=(0, 0, 0)),
            in_dims=(0, 0, 0))

    def forward(self, x_BxSxHD):
        qkv_BxSx3HD = self.Wproj(x_BxSxHD)
        qkv_BxHx3xSxD = rearrange(
            qkv_BxSx3HD, "b s (three h d) -> b h three s d", three=3, d=self.head_dim
        )
        q_BxHxSxD, k_BxHxSxD, v_BxHxSxD = qkv_BxHx3xSxD.unbind(dim=2)
        ctx_BxHxSxD = self.deeper_attn(q_BxHxSxD, k_BxHxSxD, v_BxHxSxD)
        ctx_BxSxHD = rearrange(ctx_BxHxSxD, "b h s d -> b s (h d)")
        if self.norm is not None:
            # independently normalize each head
            ctx_BxSxHD = torch.transpose(self.norm(torch.transpose(ctx_BxSxHD, -1, -2)), -1, -2)
        return self.out_proj(ctx_BxSxHD)

    def state_size(self, sequence_length: tp.Optional[int]=None, **kwargs):
        assert sequence_length is not None, "sequence_length must be provided"
        return 2 * self.d_model * sequence_length


if __name__ == '__main__':
    torch.manual_seed(42)
    T, D = 5, 3
    Q_TxD = torch.randn(T, D)
    K_TxD = torch.randn(T, D)
    V_TxD = torch.randn(T, D)

    out1 = deeper_attn_naive(Q_TxD, K_TxD, V_TxD)
    out2 = deeper_attn(Q_TxD, K_TxD, V_TxD)
    print(out1)
    print(out2)