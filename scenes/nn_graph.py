"""
scenes/nn_graph.py
500 random points connected to their 5 nearest neighbours via animated lines.

All heavy work runs on the GPU each frame:
  - init_nn_points      : randomise positions + colours via warp RNG
  - animate_nn_points   : per-point sinusoidal drift around base positions
  - find_knn_5          : O(N^2) brute-force K=5 nearest-neighbour search
  - build_nn_edges      : write flat edge VBO (pos + col) from nn indices

Run with:
    python run_nn_graph.py
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import warp as wp
import moderngl

from drawlib.scene import Scene
from drawlib.drawable import PointsDrawable, DynamicLinesDrawable
from drawlib.post_effect import FeedbackPostEffect
from drawlib.warp_feedback import FeedbackParams


# ── Constants ──────────────────────────────────────────────────────────────────

N = 500
K = 5
N_EDGES      = N * K        # directed edges (some duplicates — fine)
N_EDGE_VERTS = N_EDGES * 2  # two vertices per segment


# ── GPU Kernels ────────────────────────────────────────────────────────────────

@wp.kernel
def init_nn_points(
    positions: wp.array(dtype=wp.vec3),
    base_pos:  wp.array(dtype=wp.vec3),
    colors:    wp.array(dtype=wp.vec4),
    seed:      int,
):
    """Randomise point positions and record as animation base. Colour by XYZ."""
    i = wp.tid()
    r0 = wp.rand_init(seed, i * 3 + 0)
    r1 = wp.rand_init(seed, i * 3 + 1)
    r2 = wp.rand_init(seed, i * 3 + 2)
    x = wp.randf(r0) * 2.0 - 1.0
    y = wp.randf(r1) * 2.0 - 1.0
    z = wp.randf(r2) * 2.0 - 1.0
    p = wp.vec3(x, y, z)
    positions[i] = p
    base_pos[i]  = p
    r = wp.clamp(x * 0.5 + 0.7, 0.2, 1.0)
    g = wp.clamp(y * 0.5 + 0.6, 0.2, 1.0)
    b = wp.clamp(z * 0.5 + 0.9, 0.3, 1.0)
    colors[i] = wp.vec4(r, g, b, 1.0)


@wp.kernel
def animate_nn_points(
    positions: wp.array(dtype=wp.vec3),
    base_pos:  wp.array(dtype=wp.vec3),
    t:         float,
    amplitude: float,
):
    """Oscillate each point independently around its base position."""
    i = wp.tid()
    bp = base_pos[i]
    fi = float(i)
    px = fi * 0.137
    py = fi * 0.251
    pz = fi * 0.389
    dx = wp.sin(t * 0.6 + px) * amplitude
    dy = wp.cos(t * 0.45 + py) * amplitude
    dz = wp.sin(t * 0.8 + pz) * amplitude
    positions[i] = wp.vec3(bp[0] + dx, bp[1] + dy, bp[2] + dz)


@wp.kernel
def find_knn_5(
    positions:  wp.array(dtype=wp.vec3),
    nn_indices: wp.array(dtype=int),
    nn_dists:   wp.array(dtype=float),
    n:          int,
):
    """
    Brute-force K=5 nearest-neighbour search.
    Each thread handles one point: scan all N points, maintain sorted top-5
    using local variables. At N=500 this is 499 checks per thread — trivial.
    """
    i = wp.tid()
    pi = positions[i]

    bi0 = int(-1); bd0 = float(1.0e18)
    bi1 = int(-1); bd1 = float(1.0e18)
    bi2 = int(-1); bd2 = float(1.0e18)
    bi3 = int(-1); bd3 = float(1.0e18)
    bi4 = int(-1); bd4 = float(1.0e18)

    for j in range(n):
        if j == i:
            continue
        pj = positions[j]
        dx = pi[0] - pj[0]
        dy = pi[1] - pj[1]
        dz = pi[2] - pj[2]
        d2 = dx*dx + dy*dy + dz*dz

        if d2 < bd0:
            bi4 = bi3; bd4 = bd3
            bi3 = bi2; bd3 = bd2
            bi2 = bi1; bd2 = bd1
            bi1 = bi0; bd1 = bd0
            bi0 = j;   bd0 = d2
        elif d2 < bd1:
            bi4 = bi3; bd4 = bd3
            bi3 = bi2; bd3 = bd2
            bi2 = bi1; bd2 = bd1
            bi1 = j;   bd1 = d2
        elif d2 < bd2:
            bi4 = bi3; bd4 = bd3
            bi3 = bi2; bd3 = bd2
            bi2 = j;   bd2 = d2
        elif d2 < bd3:
            bi4 = bi3; bd4 = bd3
            bi3 = j;   bd3 = d2
        elif d2 < bd4:
            bi4 = j;   bd4 = d2

    base = i * 5
    nn_indices[base + 0] = bi0;  nn_dists[base + 0] = wp.sqrt(bd0)
    nn_indices[base + 1] = bi1;  nn_dists[base + 1] = wp.sqrt(bd1)
    nn_indices[base + 2] = bi2;  nn_dists[base + 2] = wp.sqrt(bd2)
    nn_indices[base + 3] = bi3;  nn_dists[base + 3] = wp.sqrt(bd3)
    nn_indices[base + 4] = bi4;  nn_dists[base + 4] = wp.sqrt(bd4)


@wp.kernel
def build_nn_edges(
    positions:  wp.array(dtype=wp.vec3),
    colors:     wp.array(dtype=wp.vec4),
    nn_indices: wp.array(dtype=int),
    nn_dists:   wp.array(dtype=float),
    edge_pos:   wp.array(dtype=wp.vec3),
    edge_col:   wp.array(dtype=wp.vec4),
    fade_dist:  float,
):
    """
    Build flat edge-vertex buffer from precomputed KNN.
    Thread i writes 2 vertices for each of its K=5 edges.
    Alpha fades quadratically with segment length.
    """
    i = wp.tid()
    pi = positions[i]
    ci = colors[i]

    nn_base   = i * 5
    edge_base = i * 5 * 2

    for k in range(5):
        j    = nn_indices[nn_base + k]
        dist = nn_dists[nn_base + k]
        v    = edge_base + k * 2

        if j < 0:
            edge_pos[v]     = pi
            edge_pos[v + 1] = pi
            edge_col[v]     = wp.vec4(0.0, 0.0, 0.0, 0.0)
            edge_col[v + 1] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        else:
            pj    = positions[j]
            cj    = colors[j]
            t     = wp.clamp(1.0 - dist / fade_dist, 0.0, 1.0)
            alpha = t * t * 0.85

            edge_pos[v]     = pi
            edge_pos[v + 1] = pj
            edge_col[v]     = wp.vec4(ci[0], ci[1], ci[2], alpha)
            edge_col[v + 1] = wp.vec4(cj[0], cj[1], cj[2], alpha)


# ── Scene ──────────────────────────────────────────────────────────────────────

class NNGraphScene(Scene):
    """
    500 GPU points, each connected to its 5 nearest neighbours.
    KNN search and edge buffer rebuilt entirely on GPU every frame.
    Trippy feedback post-effect: zoom spiral, rotation echo, hue cycling,
    chromatic aberration halos, ripple distortion, long oversaturated trails.
    """

    title       = "Warp — Nearest-Neighbour Graph"
    auto_rotate = True
    cam_dist    = 3.2
    cam_pitch   = -20.0

    post_effect = FeedbackPostEffect(
        params=FeedbackParams(
            base_zoom        = 1.007,   # strong inward zoom -> spiral vortex
            zoom_sensitivity = 0.0,
            base_rot         = 0.007,   # continuous echo rotation
            rot_sensitivity  = 0.0,
            decay            = 0.978,   # slow decay -> long glowing trails
            ripple_strength  = 14.0,    # radial wave distortion
            ripple_freq      = 7.0,
            hue_shift        = 0.030,   # fast hue cycling -> rainbow ghosts
            chroma_offset    = 0.022,   # heavy R/B fringe -> colour halos
            sat_boost        = 1.55,    # vivid oversaturated trails
        ),
        scene_alpha=0.12,               # fresh frame blends gently -> long echo
    )

    # ── Scene interface ────────────────────────────────────────────────────────

    def setup(self, ctx: moderngl.Context) -> None:
        self._wp_pos      = wp.zeros(N, dtype=wp.vec3)
        self._wp_base_pos = wp.zeros(N, dtype=wp.vec3)
        self._wp_col      = wp.zeros(N, dtype=wp.vec4)
        self._wp_nn_idx   = wp.zeros(N * K, dtype=int)
        self._wp_nn_dist  = wp.zeros(N * K, dtype=float)
        self._wp_edge_pos = wp.zeros(N_EDGE_VERTS, dtype=wp.vec3)
        self._wp_edge_col = wp.zeros(N_EDGE_VERTS, dtype=wp.vec4)

        wp.launch(
            init_nn_points,
            dim=N,
            inputs=[self._wp_pos, self._wp_base_pos, self._wp_col, 42],
        )

        pos_np = self._wp_pos.numpy()
        col_np = self._wp_col.numpy()

        self._pts_draw  = PointsDrawable(ctx)
        self._pts_draw.setup(pos_np, col_np)

        self._edge_draw = DynamicLinesDrawable(ctx)
        self._edge_draw.setup(N_EDGES)

        # fade_dist: 500 pts in [-1,1]^3 -> mean spacing ~0.25; use 0.5
        self._fade_dist = 0.5

    def step(self, t: float, dt: float) -> None:
        wp.launch(
            animate_nn_points,
            dim=N,
            inputs=[self._wp_pos, self._wp_base_pos, t, 0.08],
        )
        wp.launch(
            find_knn_5,
            dim=N,
            inputs=[self._wp_pos, self._wp_nn_idx, self._wp_nn_dist, N],
        )
        wp.launch(
            build_nn_edges,
            dim=N,
            inputs=[
                self._wp_pos, self._wp_col,
                self._wp_nn_idx, self._wp_nn_dist,
                self._wp_edge_pos, self._wp_edge_col,
                self._fade_dist,
            ],
        )
        self._pts_draw.write_warp(self._wp_pos, self._wp_col)
        self._edge_draw.write_warp(self._wp_edge_pos, self._wp_edge_col)

    def draw(self, mvp: np.ndarray) -> None:
        self._edge_draw.draw(mvp)
        self._pts_draw.draw(mvp)

    def on_key(self, key, action, keys) -> None:
        if action != keys.ACTION_PRESS:
            return
        if key == keys.R:
            import random
            seed = random.randint(0, 2**31 - 1)
            wp.launch(
                init_nn_points,
                dim=N,
                inputs=[self._wp_pos, self._wp_base_pos, self._wp_col, seed],
            )
            print(f"[nn_graph] re-randomised (seed={seed})")
