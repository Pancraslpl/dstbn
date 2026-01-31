from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock1D(nn.Module):
    """A simple 1D residual block."""

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.0) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class BaseBranch(nn.Module):
    """Base CNN branch (ResNet-style) for global feature extraction."""

    def __init__(self, in_ch: int = 1, channels: int = 64, blocks: int = 4, out_dim: int = 128) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.blocks = nn.Sequential(*[ResBlock1D(channels, kernel_size=3, dropout=0.1) for _ in range(blocks)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,1,L)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x  # (B,out_dim)


class SinusoidalPositionalEncoding(nn.Module):
    """Classic sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,d)
        t = x.size(1)
        return x + self.pe[:t].unsqueeze(0)


class TransformerEncoderLayerWithAttn(nn.Module):
    """Transformer encoder layer that can return attention weights."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor, *, need_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention
        attn_out, attn_weights = self.self_attn(
            x, x, x, need_weights=need_attn, average_attn_weights=False
        )  # attn_weights: (B, heads, T, T)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        # FFN
        y = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(y)
        x = self.norm2(x)

        return x, attn_weights


class SimilaritySensitiveTransformerBranch(nn.Module):
    """Transformer branch intended to capture fine-grained differences among similar classes.

    It uses a 1D Conv patch embedding to convert (B,1,L) into a token sequence (B,T,d_model).
    """

    def __init__(
        self,
        in_ch: int = 1,
        seq_len: int = 1024,
        patch_size: int = 16,
        patch_stride: int = 10,
        d_model: int = 16,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 64,
        out_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch = nn.Conv1d(in_ch, d_model, kernel_size=patch_size, stride=patch_stride, bias=False)
        # infer token length
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, seq_len)
            t = self.patch(dummy).shape[-1]
        self.token_len = int(t)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=max(512, self.token_len + 1))

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayerWithAttn(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.proj = nn.Linear(d_model, out_dim)

    def forward(self, x: torch.Tensor, *, return_attn: bool = False) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        # x: (B,1,L)
        x = self.patch(x)  # (B,d_model,T)
        x = x.transpose(1, 2)  # (B,T,d_model)
        x = self.pos(x)

        attn_all: Optional[List[torch.Tensor]] = [] if return_attn else None
        for layer in self.layers:
            x, attn = layer(x, need_attn=return_attn)
            if return_attn and attn_all is not None:
                attn_all.append(attn)

        x = self.proj(x)         # (B,T,out_dim)
        feat = x.mean(dim=1)     # (B,out_dim)
        return feat, attn_all


class GlobalGRUBranch(nn.Module):
    """GRU branch to model temporal dependency at a global scale."""

    def __init__(
        self,
        in_ch: int = 1,
        seq_len: int = 1024,
        patch_size: int = 16,
        patch_stride: int = 10,
        d_model: int = 16,
        hidden_size: int = 64,
        num_layers: int = 1,
        bidirectional: bool = True,
        out_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch = nn.Conv1d(in_ch, d_model, kernel_size=patch_size, stride=patch_stride, bias=False)
        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        gru_out = hidden_size * (2 if bidirectional else 1)
        self.proj = nn.Linear(gru_out, out_dim)

        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, seq_len)
            t = self.patch(dummy).shape[-1]
        self.token_len = int(t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x).transpose(1, 2)  # (B,T,d_model)
        y, _ = self.gru(x)                 # (B,T,gru_out)
        y = self.proj(y)                   # (B,T,out_dim)
        feat = y.mean(dim=1)               # (B,out_dim)
        return feat
