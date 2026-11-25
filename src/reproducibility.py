from __future__ import annotations

import random
import numpy as np
import torch


def set_reproducibility(seed: int) -> None:
    """Seed the relevant random number generators.

    Controls Python, NumPy and PyTorch (CPU and, when present, every CUDA
    device), and asks cuDNN for deterministic kernels instead of
    auto-tuned ones.
    """
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Without these, cuDNN may pick different (non-deterministic) convolution
    # algorithms from run to run. They have no effect on CPU-only runs.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed a DataLoader worker from PyTorch's worker seed.

    PyTorch already gives each worker its own torch seed (base_seed + worker_id),
    derived from the DataLoader generator. NumPy and Python random are not
    covered, so derive them from that seed. NumPy only accepts 32-bit seeds.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    """Return a seeded torch.Generator used for deterministic shuffling."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
