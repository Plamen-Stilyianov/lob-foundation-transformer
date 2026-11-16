from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LOBTransformerConfig:
    feature_dim: int = 23
    patch_size: int = 4
    d_model: int = 192
    n_heads: int = 6
    n_layers: int = 4
    max_patches: int = 128
    dropout: float = 0.1
    horizons: tuple[int, ...] = (1, 5, 15)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: LOBTransformerConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Flash / SDPA with strict causal mask (no future leakage)
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: LOBTransformerConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model)
        self.ffn = SwiGLU(cfg.d_model, int(cfg.d_model * 8 / 3), cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class LOBFoundationTransformer(nn.Module):
    """Causal Time-Series Foundation Model pre-trained from scratch on L2 Order Book patches."""

    def __init__(self, cfg: LOBTransformerConfig):
        super().__init__()
        self.cfg = cfg
        patch_dim = cfg.feature_dim * cfg.patch_size
        self.patch_embed = nn.Linear(patch_dim, cfg.d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.max_patches, cfg.d_model) * 0.02)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm_out = RMSNorm(cfg.d_model)

        # Self-supervised pre-training head: predict next patch + multi-horizon forward returns
        self.next_patch_head = nn.Linear(cfg.d_model, patch_dim)
        self.horizon_head = nn.Linear(cfg.d_model, len(cfg.horizons))

    def forward(self, patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, _ = patches.shape
        x = self.patch_embed(patches) + self.pos_embed[:, :T, :]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_out(x)
        return self.next_patch_head(x), self.horizon_head(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
