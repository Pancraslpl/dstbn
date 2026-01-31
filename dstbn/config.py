from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

try:
    from typing import Literal  # Py>=3.8
except ImportError:
    from typing_extensions import Literal  # Py<=3.7

@dataclass(frozen=True)
class DsTBNConfig:
    # ---------------- Basic ----------------
    gpu_id: str
    seed: int
    num_workers: int

    # ---------------- Data -----------------
    data_name: str
    condition: str
    split_index: int
    data_root: Path

    # ---------------- Train ---------------
    batch_size: int
    epochs: int
    lr: float
    weight_decay: float
    grad_clip: float

    # ---------------- Loss ----------------
    lambda1: float
    dcl_sigma: float
    dcl_ema: float

    # alpha/beta mechanism
    ab_mode: Literal["dynamic", "fixed"]
    fixed_alpha: float
    fixed_beta: float
    ab_clamp_min: float
    ab_clamp_max: float

    # ---------------- Open-set ------------
    tail_frac_candidates: List[float]
    p_threshold_candidates: List[float]
    bo_weight: float
    tau_prime_candidates: List[float]

    # ---------------- Logging/Output ------
    out_dir: Path
    experiment_name: str


def get_args() -> DsTBNConfig:
    parser = argparse.ArgumentParser(description="Ds-TBN for Open Set Fault Diagnosis")

    # ========================= Basic =========================
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU id to use (e.g., '0')")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")

    # ========================= Dataset =========================
    parser.add_argument("--data_name", type=str, default="dps", choices=["cwru", "gearbox", "dps"], help="Dataset name")
    parser.add_argument("--condition", type=str, default="1920", help="Working condition")
    parser.add_argument("--split_index", type=int, default=0, help="Task index within the condition (e.g., A0/B0/C0)")

    parser.add_argument(
        "--data_root",
        type=str,
        default="./data/processed",
        help="Root folder that contains per-task processed .npy files",
    )

    # ========================= Training =========================
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    # alpha/beta mechanism
    parser.add_argument("--ab_mode", type=str, default="dynamic", choices=["dynamic", "fixed"])

    # ========================= Output =========================
    parser.add_argument("--out_dir", type=str, default="./runs", help="Output directory")
    parser.add_argument("--experiment_name", type=str, default="dstbn", help="Experiment name")

    args = parser.parse_args()

    def parse_floats(csv_str: str) -> List[float]:
        vals = []
        for s in csv_str.split(","):
            s = s.strip()
            if s:
                vals.append(float(s))
        return vals

    return DsTBNConfig(
        gpu_id=args.gpu_id,
        seed=args.seed,
        num_workers=args.num_workers,
        data_name=args.data_name,
        condition=args.condition,
        split_index=args.split_index,
        data_root=Path(args.data_root),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        lambda1=args.lambda1,
        dcl_sigma=args.dcl_sigma,
        dcl_ema=args.dcl_ema,
        ab_mode=args.ab_mode,
        fixed_alpha=args.fixed_alpha,
        fixed_beta=args.fixed_beta,
        ab_clamp_min=args.ab_clamp_min,
        ab_clamp_max=args.ab_clamp_max,
        tail_frac_candidates=parse_floats(args.tail_fracs),
        p_threshold_candidates=parse_floats(args.p_thresholds),
        bo_weight=args.bo_weight,
        tau_prime_candidates=parse_floats(args.tau_primes),
        out_dir=Path(args.out_dir),
        experiment_name=args.experiment_name,
    )
