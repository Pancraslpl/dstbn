# dstbn/openset/weibull.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import numpy as np
import torch
import libmr

# -------------------------
# Module-level caches
# -------------------------
_MR_FLIP_CACHE: Dict[int, bool] = {}


# -------------------------
# Data structures
# -------------------------
@dataclass
class WeibullClassModel:
    centroid: np.ndarray  # (D,)
    mr: object            # libmr.MR
    tau: float            # distance threshold for stage-1
    tail_size: int

# -------------------------
# Helpers
# -------------------------
def _euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _mr_raw_score(mr: object, dist: float) -> float:
    """
    Return raw libmr score in [0,1] (best effort).
    Different libmr builds interpret w_score/cdf differently.
    """
    if hasattr(mr, "w_score"):
        return float(mr.w_score(dist))
    if hasattr(mr, "cdf"):
        return float(mr.cdf(dist))
    raise AttributeError("libmr.MR object lacks w_score/cdf method")


def _mr_outlier_prob(mr: object, dist: float) -> float:
    """
    Convert libmr score into an OUTLIER probability that is (approximately)
    monotonic increasing w.r.t distance.

    We auto-detect whether raw score increases or decreases with distance.
    If it decreases (likely inlier-likeness), we flip: outlier = 1 - score.

    NOTE: we cache the flip flag in a module-level dict because libmr.MR
    objects may not allow setattr.
    """
    key = id(mr)
    if key not in _MR_FLIP_CACHE:
        probes = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
        vals = []
        for d in probes:
            try:
                v = _mr_raw_score(mr, float(d))
            except Exception:
                v = float("nan")
            vals.append(v)

        finite = [v for v in vals if np.isfinite(v)]
        if len(finite) >= 2:
            v0 = finite[0]
            v1 = finite[-1]
            # If score DECREASES as distance increases -> likely inlier score -> flip.
            flip = bool(v1 < v0 - 1e-6)
        else:
            flip = False

        _MR_FLIP_CACHE[key] = flip

    raw = _mr_raw_score(mr, float(dist))
    raw = float(np.clip(raw, 0.0, 1.0))

    flip = _MR_FLIP_CACHE[key]
    out = (1.0 - raw) if flip else raw
    return float(np.clip(out, 0.0, 1.0))


def _find_tau_by_bisect(mr: object, p_threshold: float, *, hi0: float, iters: int = 60) -> float:
    """
    Find distance tau such that outlier_prob(tau) ~= p_threshold via bisection.
    Uses _mr_outlier_prob() (robust to libmr direction).
    """
    lo = 0.0
    hi = float(hi0)

    # expand hi until outlier_prob(hi) >= p_threshold
    for _ in range(50):
        if _mr_outlier_prob(mr, hi) >= p_threshold:
            break
        hi *= 2.0
        if hi > 1e6:
            break

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        s = _mr_outlier_prob(mr, mid)
        if s < p_threshold:
            lo = mid
        else:
            hi = mid

    return float(hi)


def _to_2d_embedding_np(z: torch.Tensor, batch_size: int) -> np.ndarray:
    """
    Convert embedding tensor to numpy with shape (B, D).

    Handles:
      - (B, D)
      - (D, B)  -> transpose
      - (B, C, T, ...) -> flatten to (B, D)
    """
    if z.ndim > 2:
        z = z.flatten(1)

    if z.ndim == 2 and z.shape[0] != batch_size and z.shape[1] == batch_size:
        z = z.transpose(0, 1)

    if z.ndim != 2 or z.shape[0] != batch_size:
        raise RuntimeError(f"[EmbeddingShapeError] expected (B,D) with B={batch_size}, got {tuple(z.shape)}")

    return z.detach().cpu().numpy().astype(np.float32, copy=False)


# -------------------------
# Fitting
# -------------------------
def fit_weibull_per_class(
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    *,
    tail_frac: float,
    p_threshold: float,
) -> Dict[int, WeibullClassModel]:
    if libmr is None:
        raise ImportError("libmr is not installed. Please install libmr to use Weibull calibration.")

    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be (N,D), got {embeddings.shape}")
    if labels.ndim != 1:
        raise ValueError(f"labels must be (N,), got {labels.shape}")
    if not (0 < tail_frac < 1):
        raise ValueError("tail_frac must be in (0,1)")
    if not (0 < p_threshold < 1):
        raise ValueError("p_threshold must be in (0,1)")

    models: Dict[int, WeibullClassModel] = {}
    for c in range(num_classes):
        zc = embeddings[labels == c]
        if zc.shape[0] < 5:
            raise ValueError(f"Not enough samples for class {c}: {zc.shape[0]}")

        centroid = zc.mean(axis=0).astype(np.float32)
        dists = np.linalg.norm(zc - centroid[None, :], axis=1).astype(np.float64)
        d_sorted = np.sort(dists)

        tail_size = max(1, int(round(float(tail_frac) * len(d_sorted))))
        tail_data = d_sorted[-tail_size:]

        mr = libmr.MR()
        mr.fit_high(tail_data.tolist(), tail_size)

        hi0 = float(d_sorted.max() * 2.0 + 1e-6)
        tau = _find_tau_by_bisect(mr, float(p_threshold), hi0=hi0)

        models[c] = WeibullClassModel(
            centroid=centroid,
            mr=mr,
            tau=float(tau),
            tail_size=int(tail_size),
        )

    return models


def extract_embeddings(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    zs: List[np.ndarray] = []
    ys: List[np.ndarray] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)

            z = _to_2d_embedding_np(out.embedding, batch_size=x.shape[0])
            z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)

            zs.append(z)
            ys.append(y.cpu().numpy())

    return np.concatenate(zs, axis=0), np.concatenate(ys, axis=0)


# -------------------------
# Dual-stage open-set model
# -------------------------
@dataclass
class DualStageOpenSetModel:
    class_models: Dict[int, WeibullClassModel]
    tau_prime: float
    unknown_id: int

    def stage1_predict(self, emb: np.ndarray) -> int:
        dists = {c: _euclidean(emb, m.centroid) for c, m in self.class_models.items()}
        c_hat = min(dists, key=dists.get)
        if dists[c_hat] <= self.class_models[c_hat].tau:
            return int(c_hat)
        return int(self.unknown_id)

    def stage2_scores(self, emb: np.ndarray, base_probs: np.ndarray) -> Tuple[np.ndarray, float]:
        p_adj = []
        for c, m in self.class_models.items():
            d = _euclidean(emb, m.centroid)
            outlier_p = _mr_outlier_prob(m.mr, d)
            p_adj.append(float(base_probs[c]) * (1.0 - outlier_p))

        p_adj = np.array(p_adj, dtype=np.float64)
        p_known_mass = float(np.clip(p_adj.sum(), 0.0, 1.0))
        p_unknown = float(np.clip(1.0 - p_known_mass, 0.0, 1.0))
        return p_adj, p_unknown

    def stage2_predict(self, emb: np.ndarray, base_probs: np.ndarray) -> Tuple[int, float]:
        p_adj, p_unknown = self.stage2_scores(emb, base_probs)
        if p_unknown >= self.tau_prime:
            return int(self.unknown_id), p_unknown
        return int(np.argmax(p_adj)), p_unknown

    def predict_batch(self, embeddings: np.ndarray, base_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        preds = np.zeros((embeddings.shape[0],), dtype=np.int64)
        p_unknowns = np.zeros((embeddings.shape[0],), dtype=np.float64)

        for i in range(embeddings.shape[0]):
            s1 = self.stage1_predict(embeddings[i])
            if s1 == self.unknown_id:
                preds[i] = self.unknown_id
                p_unknowns[i] = 1.0
            else:
                s2, pu = self.stage2_predict(embeddings[i], base_probs[i])
                preds[i] = s2
                p_unknowns[i] = pu

        return preds, p_unknowns


# -------------------------
# Hyperparam selection
# -------------------------
def choose_weibull_hyperparams(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_known_loader: torch.utils.data.DataLoader,
    val_unknown_loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    num_classes: int,
    tail_fracs: Sequence[float],
    p_thresholds: Sequence[float],
    omega: float = 0.5,
) -> Tuple[float, float]:
    if libmr is None:
        raise ImportError("libmr is not installed.")

    z_train, y_train = extract_embeddings(model, train_loader, device)
    z_vk, _ = extract_embeddings(model, val_known_loader, device)
    z_vu, _ = extract_embeddings(model, val_unknown_loader, device)

    best_obj = float("inf")
    best = (float(tail_fracs[0]), float(p_thresholds[0]))

    for tail_frac in tail_fracs:
        temp_models: Dict[int, Tuple[np.ndarray, object, int, float]] = {}

        for c in range(num_classes):
            zc = z_train[y_train == c]
            if zc.shape[0] < 5:
                raise ValueError(f"Not enough samples for class {c}: {zc.shape[0]}")

            centroid = zc.mean(axis=0).astype(np.float32)
            dists = np.linalg.norm(zc - centroid[None, :], axis=1).astype(np.float64)
            d_sorted = np.sort(dists)

            tail_size = max(1, int(round(float(tail_frac) * len(d_sorted))))
            tail_data = d_sorted[-tail_size:]

            mr = libmr.MR()
            mr.fit_high(tail_data.tolist(), tail_size)

            temp_models[c] = (centroid, mr, int(tail_size), float(d_sorted.max()))

        for pth in p_thresholds:
            class_models: Dict[int, WeibullClassModel] = {}
            for c, (centroid, mr, tail_size, dmax) in temp_models.items():
                hi0 = float(dmax * 2.0 + 1e-6)
                tau = _find_tau_by_bisect(mr, float(pth), hi0=hi0)
                class_models[c] = WeibullClassModel(
                    centroid=centroid,
                    mr=mr,
                    tau=float(tau),
                    tail_size=int(tail_size),
                )

            s1 = DualStageOpenSetModel(class_models=class_models, tau_prime=1.0, unknown_id=num_classes)

            fp = sum(1 for i in range(z_vk.shape[0]) if s1.stage1_predict(z_vk[i]) == num_classes)
            fpr = fp / max(1, z_vk.shape[0])

            fn = sum(1 for i in range(z_vu.shape[0]) if s1.stage1_predict(z_vu[i]) != num_classes)
            fnr = fn / max(1, z_vu.shape[0])

            obj = float(omega) * float(fpr) + (1.0 - float(omega)) * float(fnr)
            if obj < best_obj:
                best_obj = obj
                best = (float(tail_frac), float(pth))

    return best


def choose_tau_prime(
    model: torch.nn.Module,
    class_models: Dict[int, WeibullClassModel],
    val_mix_loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    num_classes: int,
    tau_candidates: Sequence[float],
    unknown_id: int,
) -> float:
    model.eval()

    z_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    p_list: List[np.ndarray] = []

    with torch.no_grad():
        for x, y in val_mix_loader:
            x = x.to(device, non_blocking=True)
            out = model(x)

            prob = torch.softmax(out.logits, dim=1).detach().cpu().numpy()

            z = _to_2d_embedding_np(out.embedding, batch_size=x.shape[0])
            z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)

            z_list.append(z)
            y_list.append(y.cpu().numpy())
            p_list.append(prob)

    z_all = np.concatenate(z_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    p_all = np.concatenate(p_list, axis=0)

    if not (z_all.shape[0] == y_all.shape[0] == p_all.shape[0]):
        raise RuntimeError(f"[ValMixAlignError] z={z_all.shape}, y={y_all.shape}, p={p_all.shape}")
    if p_all.shape[1] != num_classes:
        raise RuntimeError(f"[ValMixAlignError] p has shape {p_all.shape}, expected (*, {num_classes})")

    best_tau = float(tau_candidates[0])
    best_h = -1.0

    for tau in tau_candidates:
        os_model = DualStageOpenSetModel(class_models=class_models, tau_prime=float(tau), unknown_id=unknown_id)
        preds, _ = os_model.predict_batch(z_all, p_all)

        is_unknown_true = (y_all == unknown_id)
        is_unknown_pred = (preds == unknown_id)
        known_mask = ~is_unknown_true

        os_star = float((preds[known_mask] == y_all[known_mask]).mean()) if known_mask.any() else 0.0
        uk = float((is_unknown_pred[is_unknown_true]).mean()) if is_unknown_true.any() else 0.0
        h = 0.0 if (os_star + uk) == 0 else float(2.0 * os_star * uk / (os_star + uk))

        if h > best_h:
            best_h = h
            best_tau = float(tau)

    return best_tau
