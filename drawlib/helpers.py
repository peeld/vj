"""
helpers.py
Reusable geometry builders.

All functions return (vertices, colors, indices) tuples ready to pass
directly to LinesDrawable.setup() or PointsDrawable.setup().

Functions
---------
  wireframe_cube(half, color)   — 12 edges of an axis-aligned cube
  grid(size, steps, color)      — flat XZ grid
  axes(length)                  — RGB XYZ axis lines
"""

from __future__ import annotations

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  Cube wireframe
# ══════════════════════════════════════════════════════════════════════════════

def wireframe_cube(
    half:  float                  = 1.0,
    color: tuple[float, ...]      = (0.3, 0.3, 0.5, 0.6),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    12-edge wireframe of an axis-aligned cube centred at the origin.

    Returns
    -------
    vertices : float32 (8, 3)
    colors   : float32 (8, 4)   — all corners share the same RGBA color
    indices  : uint32  (24,)    — index pairs for LINES draw mode

    Usage
    -----
        wire = LinesDrawable(ctx)
        wire.setup(*wireframe_cube(half=1.0))
    """
    h = half
    verts = np.array([
        [-h, -h, -h], [ h, -h, -h], [ h,  h, -h], [-h,  h, -h],  # back face
        [-h, -h,  h], [ h, -h,  h], [ h,  h,  h], [-h,  h,  h],  # front face
    ], dtype=np.float32)

    colors = np.tile(np.array(color, dtype=np.float32), (8, 1))

    indices = np.array([
        0, 1,  1, 2,  2, 3,  3, 0,   # back
        4, 5,  5, 6,  6, 7,  7, 4,   # front
        0, 4,  1, 5,  2, 6,  3, 7,   # connecting edges
    ], dtype=np.uint32)

    return verts, colors, indices


# ══════════════════════════════════════════════════════════════════════════════
#  XZ grid
# ══════════════════════════════════════════════════════════════════════════════

def grid(
    size:  float                  = 2.0,
    steps: int                    = 10,
    color: tuple[float, ...]      = (0.2, 0.2, 0.2, 0.5),
    y:     float                  = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Flat grid of lines in the XZ plane at height ``y``.

    Returns
    -------
    vertices : float32 (N, 3)
    colors   : float32 (N, 4)
    indices  : uint32  (M,)   — index pairs for LINES draw mode

    Usage
    -----
        g = LinesDrawable(ctx)
        g.setup(*grid(size=2.0, steps=10))
    """
    verts   = []
    indices = []
    idx     = 0

    coords = np.linspace(-size, size, steps + 1)

    for x in coords:
        verts.append([x, y, -size])
        verts.append([x, y,  size])
        indices += [idx, idx + 1]
        idx += 2

    for z in coords:
        verts.append([-size, y, z])
        verts.append([ size, y, z])
        indices += [idx, idx + 1]
        idx += 2

    verts_np   = np.array(verts,   dtype=np.float32)
    colors_np  = np.tile(np.array(color, dtype=np.float32), (len(verts_np), 1))
    indices_np = np.array(indices, dtype=np.uint32)

    return verts_np, colors_np, indices_np


# ══════════════════════════════════════════════════════════════════════════════
#  XYZ axes
# ══════════════════════════════════════════════════════════════════════════════

def axes(
    length: float = 1.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Three RGB axis lines (X=red, Y=green, Z=blue), origin → tip.

    Returns
    -------
    vertices : float32 (6, 3)
    colors   : float32 (6, 4)
    indices  : uint32  (6,)   — index pairs for LINES draw mode

    Usage
    -----
        ax = LinesDrawable(ctx)
        ax.setup(*axes(length=1.5))
    """
    origin = [0.0, 0.0, 0.0]
    verts  = np.array([
        origin, [length, 0, 0],   # X — red
        origin, [0, length, 0],   # Y — green
        origin, [0, 0, length],   # Z — blue
    ], dtype=np.float32)

    colors = np.array([
        [1, 0, 0, 1], [1, 0, 0, 1],
        [0, 1, 0, 1], [0, 1, 0, 1],
        [0, 0, 1, 1], [0, 0, 1, 1],
    ], dtype=np.float32)

    indices = np.array([0, 1,  2, 3,  4, 5], dtype=np.uint32)

    return verts, colors, indices


def build_wireframe(half: float):
    """Return (vertices, colors, indices) numpy arrays for a unit wireframe cube."""
    h = half
    vertices = np.array([
        [-h,-h,-h], [ h,-h,-h], [ h, h,-h], [-h, h,-h],
        [-h,-h, h], [ h,-h, h], [ h, h, h], [-h, h, h],
    ], dtype=np.float32)
    indices = np.array([
        0,1, 1,2, 2,3, 3,0,
        4,5, 5,6, 6,7, 7,4,
        0,4, 1,5, 2,6, 3,7,
    ], dtype=np.uint32)
    colors = np.full((8, 4), [1.0, 1.0, 1.0, 0.45], dtype=np.float32)
    return vertices, colors, indices
