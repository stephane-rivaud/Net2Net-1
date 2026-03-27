"""Device helpers for CPU, CUDA, and Apple MPS execution."""

from __future__ import annotations

import numpy as np
import torch


def select_device(force_cpu: bool = False) -> torch.device:
    """Select the best available compute device.

    Args:
        force_cpu: If True, always return CPU.

    Returns:
        The preferred torch.device, choosing MPS first, then CUDA, then CPU.
    """

    if force_cpu:
        return torch.device("cpu")

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def seed_everything(seed: int, device: torch.device) -> None:
    """Seed PyTorch, NumPy, and any accelerator-specific RNGs."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
