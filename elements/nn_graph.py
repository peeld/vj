"""
drawlib/nn_graph.py
NNGraph -- animated K-nearest-neighbour graph rendered in Warp.

Responsibilities:
  - Own all Warp arrays (positions, base positions, colours, KNN indices/dists,
    edge position/colour buffers)
  - Run per-frame kernel pipeline: animate → find KNN → build edges
  - Upload results to the two ModernGL drawables (points + lines)
  - Draw both drawables with a single call

Usage::

    from drawlib.nn_graph import NNGraph

    # create once
    nn = NNGraph(ctx, device="cuda")

    # each frame
    nn.step(time)          # run Warp kernels
    nn.upload()            # push GPU data → GL buffers
    nn.draw(mvp)           # issue draw calls

    # on R-key
    nn.randomize(seed)
"""

import warp as wp
import numpy as np
import moderngl

from drawlib.drawable import PointsDrawable, DynamicLinesDrawable


# ── Constants ──────────────────────────────────────────────────────────────────

N          = 500
K          = 5
N_EDGES    = N * K
N_VERTS    = N_EDGES * 2   # two vertices per line segment


# ── Warp kernels ───────────────────────────────────────────────────────────────

@wp.kernel
def _init_points(
    positions: wp.array(dtype=wp.vec3),
    base_pos:  wp.array(dtype=wp.vec3),
    colors:    wp.array(dtype=wp.vec4),
    seed:      int,
):
    """Randomise positions, record as animation base, colour by XYZ."""
    i  = wp.tid()
    r0 = wp.rand_init(seed, i * 3 + 0)
    r1 = wp.rand_init(seed, i * 3 + 1)
    r2 = wp.rand_init(seed, i * 3 + 2)
    x  = wp.randf(r0) * 2.0 - 1.0
    y  = wp.randf(r1) * 2.0 - 1.0
    z  = wp.randf(r2) * 2.0 - 1.0
    p  = wp.vec3(x, y, z)
    positions[i] = p
    base_pos[i]  = p
    r = wp.clamp(x * 0.5 + 0.7, 0.2, 1.0)
    g = wp.clamp(y * 0.5 + 0.6, 0.2, 1.0)
    b = wp.clamp(z * 0.5 + 0.9, 0.3, 1.0)
    colors[i] = wp.vec4(r, g, b, 1.0)


@wp.kernel
def _animate_points(
    positions: wp.array(dtype=wp.vec3),
    base_pos:  wp.array(dtype=wp.vec3),
    t:         float,
    amplitude: float,
):
    """Sinusoidal drift of each point around its base position."""
    i  = wp.tid()
    bp = base_pos[i]
    fi = float(i)
    dx = wp.sin(t * 0.6  + fi * 0.137) * amplitude
    dy = wp.cos(t * 0.45 + fi * 0.251) * amplitude
    dz = wp.sin(t * 0.8  + fi * 0.389) * amplitude
    positions[i] = wp.vec3(bp[0] + dx, bp[1] + dy, bp[2] + dz)


@wp.kernel
def _find_knn5(
    positions:  wp.array(dtype=wp.vec3),
    nn_indices: wp.array(dtype=int),
    nn_dists:   wp.array(dtype=float),
    n:          int,
):
    """Brute-force K=5 nearest-neighbour search (O(N²), trivial at N=500)."""
    i  = wp.tid()
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
        d2 = dx * dx + dy * dy + dz * dz

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
def _build_edges(
    positions:  wp.array(dtype=wp.vec3),
    colors:     wp.array(dtype=wp.vec4),
    nn_indices: wp.array(dtype=int),
    nn_dists:   wp.array(dtype=float),
    edge_pos:   wp.array(dtype=wp.vec3),
    edge_col:   wp.array(dtype=wp.vec4),
    fade_dist:  float,
):
    """Write flat edge VBO from precomputed KNN; alpha fades with segment length."""
    i         = wp.tid()
    pi        = positions[i]
    ci        = colors[i]
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
            alpha = t * t * 0.95
            edge_pos[v]     = pi
            edge_pos[v + 1] = pj
            edge_col[v]     = wp.vec4(ci[0], ci[1], ci[2], alpha)
            edge_col[v + 1] = wp.vec4(cj[0], cj[1], cj[2], alpha)


# ── NNGraph class ──────────────────────────────────────────────────────────────

class NNGraph:
    """Animated K-nearest-neighbour graph: Warp simulation + ModernGL rendering.

    Parameters
    ----------
    ctx:
        Active ModernGL context.
    device:
        Warp device string, e.g. ``"cuda"`` or ``"cpu"``.
    seed:
        RNG seed for the initial point layout.
    amplitude:
        Maximum sinusoidal drift of each point from its base position.
    fade_dist:
        Edge alpha drops to 0 at this world-space distance.
    """

    def __init__(
        self,
        ctx:       moderngl.Context,
        device:    str,
        seed:      int   = 42,
        amplitude: float = 0.08,
        fade_dist: float = 0.5,
    ):
        self._device   = device
        self.amplitude = amplitude   # PropertyManager binds to this
        self.fade_dist = fade_dist   # PropertyManager binds to this

        # Warp arrays
        self._wp_pos      = wp.zeros(N,        dtype=wp.vec3)
        self._wp_base_pos = wp.zeros(N,        dtype=wp.vec3)
        self._wp_col      = wp.zeros(N,        dtype=wp.vec4)
        self._wp_nn_idx   = wp.zeros(N * K,    dtype=int)
        self._wp_nn_dist  = wp.zeros(N * K,    dtype=float)
        self._wp_edge_pos = wp.zeros(N_VERTS,  dtype=wp.vec3)
        self._wp_edge_col = wp.zeros(N_VERTS,  dtype=wp.vec4)

        # Seed initial layout
        wp.launch(
            _init_points, dim=N,
            inputs=[self._wp_pos, self._wp_base_pos, self._wp_col, seed],
            device=self._device,
        )

        # ModernGL drawables
        self._pts_draw  = PointsDrawable(ctx)
        self._pts_draw.setup(self._wp_pos.numpy(), self._wp_col.numpy())

        self._edge_draw = DynamicLinesDrawable(ctx)
        self._edge_draw.setup(N_EDGES)

    # ── Public API ────────────────────────────────────────────────────────────

    def step(self, time: float) -> None:
        """Run the full Warp kernel pipeline for one frame."""
        wp.launch(_animate_points, dim=N,
                  inputs=[self._wp_pos, self._wp_base_pos, time, self.amplitude],
                  device=self._device)
        wp.launch(_find_knn5, dim=N,
                  inputs=[self._wp_pos, self._wp_nn_idx, self._wp_nn_dist, N],
                  device=self._device)
        wp.launch(_build_edges, dim=N,
                  inputs=[self._wp_pos, self._wp_col,
                          self._wp_nn_idx, self._wp_nn_dist,
                          self._wp_edge_pos, self._wp_edge_col, self.fade_dist],
                  device=self._device)

    def upload(self) -> None:
        """Push Warp arrays → ModernGL buffers (CUDA-GL interop)."""
        self._pts_draw.write_warp(self._wp_pos, self._wp_col)
        self._edge_draw.write_warp(self._wp_edge_pos, self._wp_edge_col)

    def draw(self, mvp: np.ndarray) -> None:
        """Issue draw calls for edges then points."""
        self._edge_draw.draw(mvp)
        self._pts_draw.draw(mvp)

    def randomize(self, seed: int) -> None:
        """Re-seed point positions and colours."""
        wp.launch(
            _init_points, dim=N,
            inputs=[self._wp_pos, self._wp_base_pos, self._wp_col, seed],
            device=self._device,
        )
