from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

try:
    from typing import Literal  # Py>=3.8
except ImportError:
    from typing_extensions import Literal  # Py<=3.7

import torch
import torch.nn as nn
import torch.nn.functional as F

from .branches import BaseBranch, GlobalGRUBranch, SimilaritySensitiveTransformerBranch

BranchMode = Literal["full", "base", "gru", "trans"]


@dataclass
class DsTBNOutputs:
    logits: torch.Tensor            # (B, K)
    embedding: torch.Tensor         # (B, D)
    transformer_attn: Optional[List[torch.Tensor]] = None

    # optional debug/analysis
    f_base: Optional[torch.Tensor] = None
    f_trans: Optional[torch.Tensor] = None
    f_gru: Optional[torch.Tensor] = None


class DsTBN(nn.Module):
    """Dual-stage Tri-Branch Network."""

    def __init__(
        self,
        num_classes: int,
        *,
        seq_len: int = 1024,
        emb_dim: int = 128,
        base_dim: int = 128,
        trans_dim: int = 64,
        gru_dim: int = 64,
        proj_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)

        self.base = BaseBranch(in_ch=1, out_dim=base_dim)
        self.trans = SimilaritySensitiveTransformerBranch(in_ch=1, seq_len=seq_len, out_dim=trans_dim)
        self.gru = GlobalGRUBranch(in_ch=1, seq_len=seq_len, out_dim=gru_dim)

        self.base_dim = int(base_dim)
        self.trans_dim = int(trans_dim)
        self.gru_dim = int(gru_dim)

        fusion_dim = base_dim + trans_dim + gru_dim
        self.proj = nn.Sequential(
            nn.Linear(fusion_dim, proj_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(proj_hidden, emb_dim),
        )
        self.cls = nn.Linear(emb_dim, self.num_classes)

    @staticmethod
    def _mask_branches(
        f_base: torch.Tensor,
        f_trans: torch.Tensor,
        f_gru: torch.Tensor,
        branch: BranchMode,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if branch == "full":
            return f_base, f_trans, f_gru
        if branch == "base":
            return f_base, torch.zeros_like(f_trans), torch.zeros_like(f_gru)
        if branch == "trans":
            return torch.zeros_like(f_base), f_trans, torch.zeros_like(f_gru)
        if branch == "gru":
            return torch.zeros_like(f_base), torch.zeros_like(f_trans), f_gru
        raise ValueError(f"Unknown branch mode: {branch}")

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_attn: bool = False,
        branch: BranchMode = "full",
        return_feats: bool = False,
    ) -> DsTBNOutputs:
        # x: (B,1,L)
        f_base = self.base(x)
        f_trans, attn = self.trans(x, return_attn=return_attn)
        f_gru = self.gru(x)

        f_base_m, f_trans_m, f_gru_m = self._mask_branches(f_base, f_trans, f_gru, branch)

        f = torch.cat([f_base_m, f_trans_m, f_gru_m], dim=1)
        emb = self.proj(f)
        emb = F.normalize(emb, p=2, dim=1)
        logits = self.cls(emb)

        return DsTBNOutputs(
            logits=logits,
            embedding=emb,
            transformer_attn=attn,
            f_base=f_base if return_feats else None,
            f_trans=f_trans if return_feats else None,
            f_gru=f_gru if return_feats else None,
        )
