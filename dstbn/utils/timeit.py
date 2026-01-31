from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

import torch


@dataclass
class Timing:
    seconds: float


@contextmanager
def timer() -> Iterator[Timing]:
    start = time.perf_counter()
    t = Timing(seconds=0.0)
    yield t
    t.seconds = time.perf_counter() - start


def measure_inference_ms(model: torch.nn.Module, x: torch.Tensor, device: torch.device, repeats: int = 30) -> float:
    """Rudimentary inference latency (ms/sample) measurement."""
    model.eval()
    x = x.to(device)
    # warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

    if device.type == "cuda":
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        with torch.no_grad():
            for _ in range(repeats):
                _ = model(x)
        ender.record()
        torch.cuda.synchronize()
        total_ms = starter.elapsed_time(ender)
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(repeats):
                _ = model(x)
        total_ms = (time.perf_counter() - t0) * 1000.0

    # average per batch, then per sample
    b = x.shape[0]
    return float(total_ms / repeats / max(1, b))
