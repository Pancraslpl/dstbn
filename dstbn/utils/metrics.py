from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class OpenSetMetrics:
    os_star: float
    uk: float
    h_score: float


def compute_open_set_metrics(y_true: np.ndarray, y_pred: np.ndarray, unknown_id: int) -> OpenSetMetrics:
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    assert y_true.shape == y_pred.shape

    is_unknown_true = (y_true == unknown_id)
    known_mask = ~is_unknown_true
    is_unknown_pred = (y_pred == unknown_id)

    os_star = float((y_pred[known_mask] == y_true[known_mask]).mean()) if known_mask.any() else 0.0
    uk = float((is_unknown_pred[is_unknown_true]).mean()) if is_unknown_true.any() else 0.0
    h = 0.0 if (os_star + uk) == 0 else float(2.0 * os_star * uk / (os_star + uk))
    return OpenSetMetrics(os_star=os_star, uk=uk, h_score=h)
