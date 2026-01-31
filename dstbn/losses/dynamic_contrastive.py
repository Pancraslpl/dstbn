from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

try:
    from typing import Literal  # Py>=3.8
except ImportError:
    from typing_extensions import Literal  # Py<=3.7

import torch
import torch.nn as nn


@dataclass
class DCLStats:
    l_intra: float
    l_inter: float
    alpha: float
    beta: float
    num_classes_in_batch: int
    mode: str


class DynamicContrastiveLoss(nn.Module):
    """
    Dynamic Contrastive Learning (DCL) loss.

    For a mini-batch of embeddings z (B,D) with labels y (B,),
    compute:
      - L_intra: intra-class compactness (mean squared distance to class centroid)
      - L_inter: inter-class overlap proxy via mean RBF similarity between class centroids

    Dynamic weights (alpha, beta) are computed from batch statistics (stop-grad):
      alpha_b = L_intra / (L_intra + L_inter + eps)
      beta_b  = L_inter / (L_intra + L_inter + eps)

    To avoid trivial dominance (esp. early training), we clamp alpha/beta into [clamp_min, clamp_max]
    and re-normalize them to sum-to-1. Optionally, EMA smoothing is applied.

    Additionally supports a "fixed" mode (for ablation baseline):
      alpha = fixed_alpha, beta = fixed_beta (renormalized), without batch adaptivity.
    """

    def __init__(
        self,
        *,
        sigma: float = 1.0,
        ema_momentum: float = 0.9,
        eps: float = 1e-12,
        ab_mode: Literal["dynamic", "fixed"] = "dynamic",
        fixed_alpha: float = 0.5,
        fixed_beta: float = 0.5,
        clamp_min: float = 0.05,
        clamp_max: float = 0.95,
    ) -> None:
        super().__init__()
        if sigma <= 0:
            raise ValueError("sigma must be > 0")
        if not (0.0 <= ema_momentum < 1.0):
            raise ValueError("ema_momentum must be in [0,1)")
        if ab_mode not in ("dynamic", "fixed"):
            raise ValueError("ab_mode must be 'dynamic' or 'fixed'")
        if clamp_min <= 0 or clamp_max >= 1 or clamp_min >= clamp_max:
            raise ValueError("clamp_min/clamp_max must satisfy 0 < clamp_min < clamp_max < 1")

        self.sigma = float(sigma)
        self.ema_momentum = float(ema_momentum)
        self.eps = float(eps)

        self.ab_mode = str(ab_mode)
        self.fixed_alpha = float(fixed_alpha)
        self.fixed_beta = float(fixed_beta)

        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)

        # EMA states (buffers)
        self.register_buffer("_alpha_ema", torch.tensor(0.5), persistent=True)
        self.register_buffer("_beta_ema", torch.tensor(0.5), persistent=True)

    @staticmethod
    def _class_centroids(z: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (unique_labels, centroids)."""
        labels = torch.unique(y)
        centroids = []
        for c in labels:
            centroids.append(z[y == c].mean(dim=0))
        return labels, torch.stack(centroids, dim=0)

    def _renorm_ab(self, a: torch.Tensor, b: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        s = a + b + self.eps
        return a / s, b / s

    def forward(self, z: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, DCLStats]:
        if z.ndim != 2:
            raise ValueError(f"Expected z to be (B,D), got {z.shape}")
        if y.ndim != 1:
            raise ValueError(f"Expected y to be (B,), got {y.shape}")
        if z.size(0) != y.size(0):
            raise ValueError(f"Batch mismatch: z={z.shape}, y={y.shape}")

        labels, centroids = self._class_centroids(z, y)
        k = int(labels.numel())

        # If only one class in batch, inter-class is undefined -> fallback to intra only.
        if k <= 1:
            l_intra = torch.mean(torch.sum((z - z.mean(dim=0, keepdim=True)) ** 2, dim=1))
            alpha = torch.tensor(1.0, device=z.device)
            beta = torch.tensor(0.0, device=z.device)
            loss = l_intra
            stats = DCLStats(
                l_intra=float(l_intra.detach().cpu().item()),
                l_inter=0.0,
                alpha=1.0,
                beta=0.0,
                num_classes_in_batch=k,
                mode=self.ab_mode,
            )
            return loss, stats

        # -------- L_intra: mean squared distance to its class centroid --------
        centroid_map = {int(c.item()): centroids[i] for i, c in enumerate(labels)}
        diffs = []
        for i in range(z.size(0)):
            diffs.append(z[i] - centroid_map[int(y[i].item())])
        diffs = torch.stack(diffs, dim=0)
        l_intra = torch.mean(torch.sum(diffs ** 2, dim=1))

        # -------- L_inter: mean RBF similarity among class centroids --------
        # sim_ij = exp(-||mu_i - mu_j||^2 / (2*sigma^2)), minimizing encourages larger centroid distances.
        mu = centroids
        sim_sum = 0.0
        count = 0
        for i in range(k):
            for j in range(i + 1, k):
                dist2 = torch.sum((mu[i] - mu[j]) ** 2)
                sim = torch.exp(-dist2 / (2.0 * (self.sigma ** 2)))
                sim_sum = sim_sum + sim
                count += 1
        l_inter = sim_sum / max(count, 1)

        # -------- alpha/beta selection --------
        if self.ab_mode == "fixed":
            a = torch.tensor(self.fixed_alpha, device=z.device)
            b = torch.tensor(self.fixed_beta, device=z.device)
            a, b = self._renorm_ab(a, b)
            alpha = a.detach()
            beta = b.detach()

        else:
            # dynamic from batch statistics (stop-grad)
            denom = (l_intra + l_inter).detach() + self.eps
            alpha_batch = (l_intra.detach() / denom)
            beta_batch = (l_inter.detach() / denom)

            # clamp + renorm
            alpha_batch = alpha_batch.clamp(self.clamp_min, self.clamp_max)
            beta_batch = beta_batch.clamp(self.clamp_min, self.clamp_max)
            alpha_batch, beta_batch = self._renorm_ab(alpha_batch, beta_batch)

            if self.ema_momentum > 0.0:
                m = self.ema_momentum
                self._alpha_ema = m * self._alpha_ema + (1.0 - m) * alpha_batch
                self._beta_ema = m * self._beta_ema + (1.0 - m) * beta_batch
                alpha_ema, beta_ema = self._renorm_ab(self._alpha_ema, self._beta_ema)
                alpha = alpha_ema.detach()
                beta = beta_ema.detach()
            else:
                alpha = alpha_batch.detach()
                beta = beta_batch.detach()

        loss = alpha * l_intra + beta * l_inter

        stats = DCLStats(
            l_intra=float(l_intra.detach().cpu().item()),
            l_inter=float(l_inter.detach().cpu().item()),
            alpha=float(alpha.detach().cpu().item()),
            beta=float(beta.detach().cpu().item()),
            num_classes_in_batch=k,
            mode=self.ab_mode,
        )
        return loss, stats
