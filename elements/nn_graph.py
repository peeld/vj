"""
elements/nn_graph.py
NNGraph -- animated K-nearest-neighbour graph with DFS traversal reveal.

Always running (step/upload/draw called every frame), but nothing is visible
unless activated.

Root-cause fix for edge pops
-----------------------------
The original code ran _find_knn5 every frame and passed the live _wp_nn_idx
to _build_edges.  The visibility array is indexed by (node, k) -- the k-th
slot in the KNN.  As points animate, borderline neighbours swap KNN slots
every few frames.  When the live KNN reorders slot k to a different neighbour,
visibility[node*K+k] = 1.0 means a *different* edge is drawn than the one
that was revealed.  The original neighbour disappears; the new one appears.
That is the pop.

Fix: snapshot the KNN once at activate() time and store it in _wp_knn_snap
(GPU) and _knn_snap (CPU).  _build_edges uses _wp_knn_snap.  _advance_build
uses _knn_snap.  The slot assignments are frozen for the entire traversal
cycle; topology cannot churn.

_find_knn5 is called ONLY inside activate(), not in step().

Activation behaviour
--------------------
activate()   -- run KNN once from current positions, snapshot, pick a random
                start node, begin a depth-first walk.  Any in-progress build
                or unwind state is hard-reset so old slot indices (from the
                previous snapshot) cannot bleed into the new one.

deactivate() -- stop building; push the current path to the unwind queue.
                Each step() then hides edges FIFO at the same
                ``edges_per_frame`` rate.
"""

import random
import warp as wp
import numpy as np
import moderngl

from drawlib.drawable import DynamicLinesDrawable


# -- Constants -----------------------------------------------------------------

N       = 1400
K       = 5
N_EDGES = N * K
N_VERTS = N_EDGES * 2


# -- Warp kernels --------------------------------------------------------------

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
    """Brute-force K=5 nearest-neighbour search."""
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
    knn_snap:   wp.array(dtype=int),     # frozen snapshot -- NOT the live KNN
    visibility: wp.array(dtype=float),
    edge_pos:   wp.array(dtype=wp.vec3),
    edge_col:   wp.array(dtype=wp.vec4),
    fade_dist:  float,
):
    """
    Write flat edge VBO from the KNN snapshot.

    Uses knn_snap (frozen at activate() time) so topology cannot churn.
    Hidden edges (vis < 0.5) become zero-alpha degenerate segments.
    Revealed edges draw at constant alpha; position updates as points drift.
    """
    i         = wp.tid()
    pi        = positions[i]
    ci        = colors[i]
    nn_base   = i * 5
    edge_base = i * 5 * 2

    for k in range(5):
        j   = knn_snap[nn_base + k]     # stable slot assignment
        v   = edge_base + k * 2
        vis = visibility[nn_base + k]

        if j < 0 or vis < 0.5:
            # hidden: collapse to zero-alpha degenerate segment
            edge_pos[v]     = pi
            edge_pos[v + 1] = pi
            edge_col[v]     = wp.vec4(0.0, 0.0, 0.0, 0.0)
            edge_col[v + 1] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        else:
            pj    = positions[j]
            cj    = colors[j]
            alpha = float(0.85)
            edge_pos[v]     = pi
            edge_pos[v + 1] = pj
            edge_col[v]     = wp.vec4(ci[0], ci[1], ci[2], alpha)
            edge_col[v + 1] = wp.vec4(cj[0], cj[1], cj[2], alpha)


# -- NNGraph class -------------------------------------------------------------

class NNGraph:
    """
    Animated KNN graph with DFS traversal reveal.

    Always simulates (points drift, edge buffer written) every frame, but
    nothing is drawn until activate() is called.

    The KNN is computed ONCE per activation cycle (inside activate()), not
    every frame.  The snapshot is held in _wp_knn_snap / _knn_snap and used
    for both the DFS traversal and the VBO kernel.

    Parameters
    ----------
    ctx:
        Active ModernGL context.
    device:
        Warp device string, e.g. "cuda" or "cpu".
    seed:
        RNG seed for the initial point layout.
    amplitude:
        Maximum sinusoidal drift of each point from its base position.
    fade_dist:
        Kept for API compatibility; no longer used in the revealed-edge path.
    edges_per_frame:
        How many edges to reveal (and separately, to hide) each frame.
    """

    def __init__(
        self,
        ctx:             moderngl.Context,
        device:          str,
        seed:            int   = 42,
        amplitude:       float = 0.08,
        fade_dist:       float = 0.5,
        edges_per_frame: int   = 3,
    ):
        self._device         = device
        self.amplitude       = amplitude
        self.fade_dist       = fade_dist
        self.edges_per_frame = edges_per_frame

        # Live simulation arrays
        self._wp_pos      = wp.zeros(N,       dtype=wp.vec3,  device=device)
        self._wp_base_pos = wp.zeros(N,       dtype=wp.vec3,  device=device)
        self._wp_col      = wp.zeros(N,       dtype=wp.vec4,  device=device)
        self._wp_nn_idx   = wp.zeros(N * K,   dtype=int,      device=device)
        self._wp_nn_dist  = wp.zeros(N * K,   dtype=float,    device=device)

        # Frozen KNN snapshot -- set at activate(), never touched in step()
        self._wp_knn_snap: wp.array       = wp.zeros(N * K, dtype=int, device=device)
        self._knn_snap:    np.ndarray | None = None

        # Edge VBO arrays
        self._wp_edge_pos = wp.zeros(N_VERTS, dtype=wp.vec3,  device=device)
        self._wp_edge_col = wp.zeros(N_VERTS, dtype=wp.vec4,  device=device)

        # Per directed-edge visibility: 0.0=hidden, 1.0=visible.
        # Indexed by (node * K + k) matching the snapshot slot.
        self._visibility = np.zeros(N * K, dtype=np.float32)

        wp.launch(
            _init_points, dim=N,
            inputs=[self._wp_pos, self._wp_base_pos, self._wp_col, seed],
            device=device,
        )

        # ModernGL drawable (edges only -- points are never drawn)
        self._edge_draw = DynamicLinesDrawable(ctx)
        self._edge_draw.setup(N_EDGES)

        # Traversal state
        self._build_path:     list              = []
        self._build_path_set: set               = set()
        self._unwind_queue:   list              = []
        self._dfs_stack:      list              = []
        self._dfs_visited:    set               = set()
        self._building:       bool              = False

    # -- Activation API -------------------------------------------------------

    def is_active(self) -> bool:
        return self._building

    def activate(self) -> None:
        """
        Hard-reset any in-progress state, compute a fresh KNN snapshot from
        the current point positions, then start a new DFS traversal.

        The hard-reset ensures that visibility entries indexed against the
        *previous* snapshot cannot corrupt the new one.
        """
        print('ACTIVATE')

        # Zero out all in-progress visibility so old slot indices don't bleed
        # into the new snapshot's slot assignments.
        for (node, k) in self._build_path + self._unwind_queue:
            self._visibility[node * K + k] = 0.0
        self._build_path     = []
        self._build_path_set = set()
        self._unwind_queue   = []
        self._dfs_stack      = []
        self._dfs_visited    = set()
        self._building       = False

        # Compute KNN once from the current (live) point positions.
        wp.launch(
            _find_knn5, dim=N,
            inputs=[self._wp_pos, self._wp_nn_idx, self._wp_nn_dist, N],
            device=self._device,
        )
        wp.synchronize()  # must complete before we read + copy

        # Snapshot to GPU (for _build_edges) and CPU (for _advance_build DFS)
        wp.copy(self._wp_knn_snap, self._wp_nn_idx)
        self._knn_snap = self._wp_nn_idx.numpy().reshape(N, K)

        # Start DFS from a random node
        start    = random.randint(0, N - 1)
        shuffled = list(range(K))
        random.shuffle(shuffled)
        self._dfs_stack   = [(start, shuffled, 0)]
        self._dfs_visited = set()
        self._building    = True

    def deactivate(self) -> None:
        """Stop building; queue the current path for FIFO unwinding."""
        print('deactivate')
        self._building = False
        if self._build_path:
            self._unwind_queue   = self._build_path + self._unwind_queue
            self._build_path     = []
            self._build_path_set = set()
        self._dfs_stack   = []
        self._dfs_visited = set()

    # -- Frame pipeline -------------------------------------------------------

    def step(self, time: float) -> None:
        """
        Animate points, advance the traversal, and rebuild the edge VBO.

        _find_knn5 is NOT called here.  The KNN snapshot is stable for the
        entire build+unwind cycle; only activate() refreshes it.
        """
        wp.launch(
            _animate_points, dim=N,
            inputs=[self._wp_pos, self._wp_base_pos, time, self.amplitude],
            device=self._device,
        )

        for _ in range(self.edges_per_frame):
            if self._building:
                self._advance_build()
            if self._unwind_queue:
                self._advance_unwind()

        wp_vis = wp.array(self._visibility, dtype=wp.float32, device=self._device)
        wp.launch(
            _build_edges, dim=N,
            inputs=[
                self._wp_pos, self._wp_col,
                self._wp_knn_snap,   # ← frozen snapshot, not live nn_indices
                wp_vis,
                self._wp_edge_pos, self._wp_edge_col,
                self.fade_dist,
            ],
            device=self._device,
        )

    def upload(self) -> None:
        """Push Warp arrays -> ModernGL buffers (CUDA-GL interop)."""
        self._edge_draw.write_warp(self._wp_edge_pos, self._wp_edge_col)

    def draw(self, mvp: np.ndarray) -> None:
        """Issue draw call for edges (points are never drawn)."""
        self._edge_draw.draw(mvp)

    def randomize(self, seed: int) -> None:
        """Re-seed point positions and colours; reset all state."""
        wp.launch(
            _init_points, dim=N,
            inputs=[self._wp_pos, self._wp_base_pos, self._wp_col, seed],
            device=self._device,
        )
        self._visibility[:]  = 0.0
        self._build_path     = []
        self._build_path_set = set()
        self._unwind_queue   = []
        self._dfs_stack      = []
        self._dfs_visited    = set()
        self._building       = False
        self._knn_snap       = None

    # -- Internal traversal ---------------------------------------------------

    def _advance_build(self) -> bool:
        """
        Walk the DFS one step forward, revealing one edge.
        Uses _knn_snap (CPU) -- the frozen topology from activate().
        """
        while self._dfs_stack:
            node, shuffled_ks, k_idx = self._dfs_stack[-1]

            while k_idx < K:
                k        = shuffled_ks[k_idx]
                neighbor = int(self._knn_snap[node, k])
                k_idx   += 1
                edge_key = (node, k)

                if neighbor >= 0 and edge_key not in self._dfs_visited:
                    self._dfs_visited.add(edge_key)
                    self._build_path.append(edge_key)
                    self._build_path_set.add(edge_key)
                    self._visibility[node * K + k] = 1.0

                    self._dfs_stack[-1] = (node, shuffled_ks, k_idx)

                    new_ks = list(range(K))
                    random.shuffle(new_ks)
                    self._dfs_stack.append((neighbor, new_ks, 0))
                    return True

            # All K edges from this node tried -> backtrack
            self._dfs_stack.pop()

        self._building = False
        return False

    def _advance_unwind(self) -> bool:
        """
        Hide the oldest visible edge in the unwind queue.

        Edges re-claimed by the current active build are skipped.
        """
        found=0
        while self._unwind_queue:
            node, k = self._unwind_queue.pop(0)
            if (node, k) not in self._build_path_set:
                self._visibility[node * K + k] = 0.0
                found += 1
                if found > 3:
                    return True
        return False
