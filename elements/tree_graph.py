"""
elements/tree_graph.py
TreeGraph — organic BFS spanning-tree grown from nearest-neighbour connections.

GPU responsibilities
--------------------
- _init_tree_points   randomise N positions + colours (one-shot per build)
- _animate_points     sinusoidal per-point drift each frame
- _build_tree_vbo     assemble edge VBO from static topology + live positions;
                      applies pulsing wavefront alpha  (runs every frame)

CPU responsibilities
--------------------
- BFS spanning-tree   find root, expand frontier, record edge list + depths
                      (numpy, O(N²) — negligible at N ≤ MAX_POINTS = 1000)
- Wavefront state     grow → hold → shrink → gap  phase machine

Tunable parameters (all live-writable, feed into rebuild() or step()):
    n_points      int    seeded/animated points            default 200
    branch_min    int    min children per BFS node         default 1
    branch_max    int    max children per BFS node         default 4
    amplitude     float  sinusoidal drift amplitude        default 0.06
    grow_speed    float  wavefront advance depth-units/s   default 3.0
    shrink_speed  float  wavefront retreat depth-units/s   default 5.0
    fade_width    float  alpha ramp width in depth units   default 0.8
    hold_time     float  seconds fully-grown before shrink default 0.5
    gap_time      float  seconds fully-shrunk before grow  default 0.3

Triggering a rebuild (e.g. key press or property change):
    tree.rebuild()              # new random seed, same params
    tree.rebuild(seed=42)       # deterministic
    tree.rebuild(n_points=350)  # change count and rebuild
"""

from __future__ import annotations

import random
import numpy as np
import warp as wp
import moderngl

from drawlib.drawable import DynamicLinesDrawable


# ── Pre-allocation limits ──────────────────────────────────────────────────────

MAX_POINTS = 10000          # upper bound for n_points
MAX_EDGES  = MAX_POINTS    # a tree has N-1 edges; +1 margin is fine


# ── GPU kernels ────────────────────────────────────────────────────────────────

@wp.kernel
def _init_tree_points(
    positions: wp.array(dtype=wp.vec3),
    base_pos:  wp.array(dtype=wp.vec3),
    colors:    wp.array(dtype=wp.vec4),
    seed:      int,
):
    """Randomise positions in [-1,1]³; colour by XYZ."""
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
    """Sinusoidal per-point drift around base positions."""
    i  = wp.tid()
    bp = base_pos[i]
    fi = float(i)
    dx = wp.sin(t * 0.6  + fi * 0.137) * amplitude
    dy = wp.cos(t * 0.45 + fi * 0.251) * amplitude
    dz = wp.sin(t * 0.8  + fi * 0.389) * amplitude
    positions[i] = wp.vec3(bp[0] + dx, bp[1] + dy, bp[2] + dz)


@wp.kernel
def _build_tree_vbo(
    positions:  wp.array(dtype=wp.vec3),
    colors:     wp.array(dtype=wp.vec4),
    edge_from:  wp.array(dtype=int),
    edge_to:    wp.array(dtype=int),
    edge_depth: wp.array(dtype=int),
    edge_pos:   wp.array(dtype=wp.vec3),
    edge_col:   wp.array(dtype=wp.vec4),
    wavefront:  float,
    fade_width: float,
    n_edges:    int,
):
    """
    Build flat edge VBO each frame.

    Launched with dim=MAX_EDGES. Slots e < n_edges carry live tree data with
    wavefront alpha; slots e >= n_edges are zeroed so unused VBO space is
    invisible.
    """
    e = wp.tid()

    if e >= n_edges:
        edge_pos[e * 2]     = wp.vec3(0.0, 0.0, 0.0)
        edge_pos[e * 2 + 1] = wp.vec3(0.0, 0.0, 0.0)
        edge_col[e * 2]     = wp.vec4(0.0, 0.0, 0.0, 0.0)
        edge_col[e * 2 + 1] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        return

    fi = edge_from[e]
    ti = edge_to[e]
    d  = float(edge_depth[e])

    fw = wp.max(fade_width, 0.01)
    alpha = wp.clamp((wavefront - d) / fw, 0.0, 1.0)

    cf = colors[fi]
    ct = colors[ti]

    edge_pos[e * 2]     = positions[fi]
    edge_pos[e * 2 + 1] = positions[ti]
    edge_col[e * 2]     = wp.vec4(cf[0], cf[1], cf[2], alpha)
    edge_col[e * 2 + 1] = wp.vec4(ct[0], ct[1], ct[2], alpha)


# ── TreeGraph ──────────────────────────────────────────────────────────────────

class TreeGraph:
    """
    Animated organic spanning-tree.  See module docstring for full details.

    Parameters
    ----------
    ctx : moderngl.Context
    n_points : int
    branch_min : int
    branch_max : int
    seed : int
    amplitude : float
    grow_speed : float
    shrink_speed : float
    fade_width : float
    hold_time : float
    gap_time : float
    """

    def __init__(
        self,
        ctx:          moderngl.Context,
        n_points:     int   = 200,
        branch_min:   int   = 1,
        branch_max:   int   = 2,
        seed:         int   = 42,
        amplitude:    float = 0.06,
        grow_speed:   float = 3.0,
        shrink_speed: float = 5.0,
        fade_width:   float = 0.8,
        hold_time:    float = 0.5,
        gap_time:     float = 0.3,
    ):
        # Live-writable params
        self.n_points    = n_points
        self.branch_min  = branch_min
        self.branch_max  = branch_max
        self.amplitude   = amplitude
        self.grow_speed  = grow_speed
        self.shrink_speed = shrink_speed
        self.fade_width  = fade_width
        self.hold_time   = hold_time
        self.gap_time    = gap_time

        # Pre-allocate GPU arrays at max size so n_points can be changed
        # at runtime without reallocating the registered GL buffers.
        self._wp_pos      = wp.zeros(MAX_POINTS,       dtype=wp.vec3)
        self._wp_base_pos = wp.zeros(MAX_POINTS,       dtype=wp.vec3)
        self._wp_col      = wp.zeros(MAX_POINTS,       dtype=wp.vec4)
        self._wp_ef       = wp.zeros(MAX_EDGES,        dtype=int)   # edge_from
        self._wp_et       = wp.zeros(MAX_EDGES,        dtype=int)   # edge_to
        self._wp_ed       = wp.zeros(MAX_EDGES,        dtype=int)   # edge_depth
        self._wp_edge_pos = wp.zeros(MAX_EDGES * 2,    dtype=wp.vec3)
        self._wp_edge_col = wp.zeros(MAX_EDGES * 2,    dtype=wp.vec4)

        # ModernGL drawable — fixed at MAX_EDGES segments
        self._draw = DynamicLinesDrawable(ctx)
        self._draw.setup(MAX_EDGES)

        # Tree state (set by rebuild)
        self._n_edges  = 0
        self._max_depth = 1
        self._seed = seed

        # Wavefront state machine
        self._wavefront = -fade_width
        self._phase = 'grow'
        self._timer = 0.0

        self.rebuild(seed=seed)

    # ── Public API ─────────────────────────────────────────────────────────────

    def rebuild(
        self,
        seed:       int | None = None,
        n_points:   int | None = None,
        branch_min: int | None = None,
        branch_max: int | None = None,
    ) -> None:
        """Re-seed points and recompute tree topology.

        Any keyword args override the corresponding instance attribute for this
        build and all subsequent ones.
        """
        if n_points   is not None: self.n_points   = max(2, min(n_points,   MAX_POINTS))
        if branch_min is not None: self.branch_min = max(1, branch_min)
        if branch_max is not None: self.branch_max = max(self.branch_min, branch_max)
        if seed is None:           seed = random.randint(0, 2**31 - 1)
        self._seed = seed

        # 1. Randomise positions on GPU
        wp.launch(
            _init_tree_points, dim=self.n_points,
            inputs=[self._wp_pos, self._wp_base_pos, self._wp_col, seed],
        )

        # 2. Pull positions to CPU for BFS (one small copy, ~12 KB at N=1000)
        pos = self._wp_pos.numpy()[:self.n_points]

        # 3. Build spanning tree on CPU
        ef, et, ed, max_depth = self._bfs_tree(pos, seed)
        self._n_edges  = len(ef)
        self._max_depth = max_depth

        # 4. Upload topology to GPU
        if self._n_edges > 0:
            n = self._n_edges
            np_ef = np.array(ef, dtype=np.int32)
            np_et = np.array(et, dtype=np.int32)
            np_ed = np.array(ed, dtype=np.int32)
            # Write into the pre-allocated arrays (first n slots)
            wp.copy(self._wp_ef, wp.array(np_ef, dtype=int), count=n)
            wp.copy(self._wp_et, wp.array(np_et, dtype=int), count=n)
            wp.copy(self._wp_ed, wp.array(np_ed, dtype=int), count=n)

        # 5. Reset wavefront to start of grow phase
        self._wavefront = -self.fade_width
        self._phase = 'grow'
        self._timer = 0.0

        print(f"[tree_graph] rebuilt  seed={seed}  n_points={self.n_points}"
              f"  edges={self._n_edges}  max_depth={self._max_depth}"
              f"  branch={self.branch_min}–{self.branch_max}")

    def step(self, time: float, dt: float) -> None:
        """Run GPU kernels for one frame."""
        # Animate point positions
        wp.launch(
            _animate_points, dim=self.n_points,
            inputs=[self._wp_pos, self._wp_base_pos, time, self.amplitude],
        )

        # Advance wavefront state machine
        self._tick(dt)

        # Build edge VBO (always MAX_EDGES threads; zeros unused slots)
        wp.launch(
            _build_tree_vbo, dim=MAX_EDGES,
            inputs=[
                self._wp_pos, self._wp_col,
                self._wp_ef, self._wp_et, self._wp_ed,
                self._wp_edge_pos, self._wp_edge_col,
                float(self._wavefront), float(self.fade_width),
                self._n_edges,
            ],
        )

    def upload(self) -> None:
        """Push Warp arrays → ModernGL buffers via CUDA-GL interop."""
        self._draw.write_warp(self._wp_edge_pos, self._wp_edge_col)

    def draw(self, mvp: np.ndarray) -> None:
        """Issue draw call."""
        self._draw.draw(mvp)

    # ── Wavefront state machine ────────────────────────────────────────────────

    def _tick(self, dt: float) -> None:
        peak   = float(self._max_depth) + self.fade_width
        trough = -self.fade_width

        if self._phase == 'grow':
            self._wavefront += dt * self.grow_speed
            if self._wavefront >= peak:
                self._wavefront = peak
                self._phase = 'hold'
                self._timer = self.hold_time

        elif self._phase == 'hold':
            self._timer -= dt
            if self._timer <= 0.0:
                self._phase = 'shrink'

        elif self._phase == 'shrink':
            self._wavefront -= dt * self.shrink_speed
            if self._wavefront <= trough:
                self._wavefront = trough
                self._phase = 'gap'
                self._timer = self.gap_time

        elif self._phase == 'gap':
            self._timer -= dt
            if self._timer <= 0.0:
                self._phase = 'grow'
                self._wavefront = trough

    # ── BFS tree construction (CPU / numpy) ────────────────────────────────────

    def _bfs_tree(
        self,
        pos: np.ndarray,
        seed: int,
    ) -> tuple[list, list, list, int]:
        """
        BFS spanning tree from the point closest to the origin.

        Each node in the frontier claims branch_min..branch_max nearest
        unvisited points as children.

        Returns (edge_from, edge_to, edge_depth_values, max_depth).
        """
        n   = len(pos)
        rng = np.random.default_rng(seed)

        visited = np.zeros(n, dtype=bool)

        # Root = point nearest to origin
        root = int(np.argmin(np.linalg.norm(pos, axis=1)))
        visited[root] = True

        edge_from:  list[int] = []
        edge_to:    list[int] = []
        edge_depth: list[int] = []
        max_depth = 0

        frontier  = [root]
        depth     = 0

        while frontier and np.any(~visited):
            next_frontier: list[int] = []

            for node in frontier:
                unvisited = np.where(~visited)[0]
                if len(unvisited) == 0:
                    break

                # Distances from this node to all unvisited points
                diffs = pos[unvisited] - pos[node]
                dists = np.linalg.norm(diffs, axis=1)

                # Random branching factor clamped to available neighbours
                k = int(rng.integers(self.branch_min, self.branch_max + 1))
                k = min(k, len(unvisited))

                # Indices of k nearest (partial sort — fast at large N)
                nearest_local = np.argpartition(dists, k - 1)[:k]
                children = unvisited[nearest_local]

                for child in children:
                    child = int(child)
                    if visited[child]:
                        continue        # another frontier node beat us to it
                    visited[child] = True
                    edge_from.append(node)
                    edge_to.append(child)
                    edge_depth.append(depth + 1)
                    next_frontier.append(child)
                    if depth + 1 > max_depth:
                        max_depth = depth + 1

            frontier = next_frontier
            depth   += 1

        return edge_from, edge_to, edge_depth, max(max_depth, 1)
