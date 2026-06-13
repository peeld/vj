"""generate.py — random 3D point cloud generation."""

import numpy as np


def generate_points(n: int = 5000, seed: int = 42, scale: float = 100.0) -> np.ndarray:
    """Return (n, 3) float32 array of random points in [0, scale]^3."""
    rng = np.random.default_rng(seed)
    return (rng.random((n, 3), dtype=np.float32) * scale).astype(np.float32)
