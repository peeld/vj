"""
scenes/nn_graph.py

Lifecycle
─────────
  DARK      Points animate freely.  KNN is computed once from the current
            snapshot; undirected edge list built on CPU (set, no duplicates).
            → BUILDING

  BUILDING  Everything frozen (no animation, no KNN).  Edges fade in one by
            one in index order; alpha computed purely from frame number — no
            accumulated state that can drift.
            → ACTIVE

  ACTIVE    Completely static.  VBO is not touched.  Pure render.
            → HIDING  (R key)

  HIDING    Edges fade out from the frozen snapshot.  VBO updated each frame.
            Once fully dark → DARK.

Key design decisions
────────────────────
  • animate_nn_points  runs ONLY in DARK.
  • find_knn_5 / edge-list build  run ONLY in DARK (once per cycle).
  • build_edge_vbo  runs ONLY in BUILDING and HIDING.
  • During ACTIVE nothing GPU-side changes — no pops possible.
  • Alpha is a pure function (phase, frame_number, edge_index) — no per-edge
    state array that could be out-of-sync with the edge list.

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

N_MAX_EDGES = N * K          # safe upper bound; actual count is ~N*K/2 after dedup
N_MAX_VERTS = N_MAX_EDGES * 2

FADE_SPEED      = 0.04       # alpha units per frame  (25 frames = full fade)
EDGES_PER_FRAME = 6          # edges activated per BUILDING frame
HIDE_FRAMES     = int(np.ceil(1.0 / FADE_SPEED)) + 2   # guarantees alpha hits 0

PHASE_DARK     = 0
PHASE_BUILDING = 1
PHASE_ACTIVE   = 2
PHASE_HIDING   = 3

# Kernel-side phase ints (passed as plain int arguments)
_DARK     = 0
_BUILDING = 1
_ACTIVE   = 2
_HIDING   = 3


# ── Kernels ────────────────────────────────────────────────────────────────────

@wp.kernel
def init_nn_points(
    positions: wp.array(dtype=wp.vec3),
    base_pos:  wp.array(dtype=wp.vec3),
    colors:    wp.array(dtype=wp.vec4),
    seed:      int,
):
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
    i  = wp.tid()
    bp = base_pos[i]
    fi = float(i)
    dx = wp.sin(t * 0.6  + fi * 0.137) * amplitude
    dy = wp.cos(t * 0.45 + fi * 0.251) * amplitude
    dz = wp.sin(t * 0.8  + fi * 0.389) * amplitude
    positions[i] = wp.vec3(bp[0] + dx, bp[1] + dy, bp[2] + dz)


@wp.kernel
def find_knn_5(
    positions:  wp.array(dtype=wp.vec3),
    nn_indices: wp.array(dtype=int),
    nn_dists:   wp.array(dtype=float),
    n:          int,
):
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
        d2 = dx*dx + dy*dy + dz*dz

        if d2 < bd0:
            bi4 = bi3; bd4 = bd3; bi3 = bi2; bd3 = bd2
            bi2 = bi1; bd2 = bd1; bi1 = bi0; bd1 = bd0
            bi0 = j;   bd0 = d2
        elif d2 < bd1:
            bi4 = bi3; bd4 = bd3; bi3 = bi2; bd3 = bd2
            bi2 = bi1; bd2 = bd1; bi1 = j;   bd1 = d2
        elif d2 < bd2:
            bi4 = bi3; bd4 = bd3; bi3 = bi2; bd3 = bd2
            bi2 = j;   bd2 = d2
        elif d2 < bd3:
            bi4 = bi3; bd4 = bd3; bi3 = j; bd3 = d2
        elif d2 < bd4:
            bi4 = j;   bd4 = d2

    base = i * 5
    nn_indices[base + 0] = bi0;  nn_dists[base + 0] = wp.sqrt(bd0)
    nn_indices[base + 1] = bi1;  nn_dists[base + 1] = wp.sqrt(bd1)
    nn_indices[base + 2] = bi2;  nn_dists[base + 2] = wp.sqrt(bd2)
    nn_indices[base + 3] = bi3;  nn_dists[base + 3] = wp.sqrt(bd3)
    nn_indices[base + 4] = bi4;  nn_dists[base + 4] = wp.sqrt(bd4)


@wp.kernel
def build_edge_vbo(
    positions:       wp.array(dtype=wp.vec3),
    colors:          wp.array(dtype=wp.vec4),
    edge_src:        wp.array(dtype=int),
    edge_dst:        wp.array(dtype=int),
    edge_pos:        wp.array(dtype=wp.vec3),
    edge_col:        wp.array(dtype=wp.vec4),
    n_edges:         int,
    phase:           int,
    frame_param:     int,    # build_frame when BUILDING; hide_frame when HIDING
    edges_per_frame: int,
    fade_speed:      float,
):
    """
    One thread per max-edge slot.  Alpha computed purely from (phase, frame, e):

        BUILDING : alpha = clamp( (build_frame - e // edges_per_frame) * fade_speed, 0, 1 )
        HIDING   : alpha = clamp( 1 - hide_frame * fade_speed,                       0, 1 )
        ACTIVE   : alpha = 1
        DARK     : alpha = 0

    No per-edge state.  Cannot drift.  Cannot pop.
    """
    e = wp.tid()
    v = e * 2

    alpha = float(0.0)
    if e < n_edges:
        if phase == 2:      # ACTIVE
            alpha = 1.0
        elif phase == 1:    # BUILDING
            birth = e // edges_per_frame
            alpha = wp.clamp(float(frame_param - birth) * fade_speed, 0.0, 1.0)
        elif phase == 3:    # HIDING
            alpha = wp.clamp(1.0 - float(frame_param) * fade_speed, 0.0, 1.0)
        # phase 0 (DARK): alpha stays 0.0

    z = wp.vec3(0.0, 0.0, 0.0)
    if alpha <= 0.0:
        edge_pos[v]     = z
        edge_pos[v + 1] = z
        edge_col[v]     = wp.vec4(0.0, 0.0, 0.0, 0.0)
        edge_col[v + 1] = wp.vec4(0.0, 0.0, 0.0, 0.0)
    else:
        src = edge_src[e];  dst = edge_dst[e]
        ps  = positions[src];  cs = colors[src]
        pd  = positions[dst];  cd = colors[dst]
        edge_pos[v]     = ps
        edge_pos[v + 1] = pd
        edge_col[v]     = wp.vec4(cs[0], cs[1], cs[2], alpha)
        edge_col[v + 1] = wp.vec4(cd[0], cd[1], cd[2], alpha)


# ── Scene ──────────────────────────────────────────────────────────────────────

class NNGraphScene(Scene):
    """
    500 points.  KNN graph built once when dark; edges revealed sequentially;
    everything frozen during reveal and hold; R hides and triggers a rebuild.
    """

    title       = "Warp — Nearest-Neighbour Graph"
    auto_rotate = True
    cam_dist    = 3.2
    cam_pitch   = -20.0

    post_effect = FeedbackPostEffect(
        params=FeedbackParams(
            base_zoom        = 1.007,
            zoom_sensitivity = 0.0,
            base_rot         = 0.007,
            rot_sensitivity  = 0.0,
            decay            = 0.978,
            ripple_strength  = 14.0,
            ripple_freq      = 7.0,
            hue_shift        = 0.030,
            chroma_offset    = 0.022,
            sat_boost        = 1.55,
        ),
        scene_alpha=0.12,
    )

    # ── Setup ──────────────────────────────────────────────────────────────────

    def setup(self, ctx: moderngl.Context) -> None:
        self._wp_pos      = wp.zeros(N, dtype=wp.vec3)
        self._wp_base_pos = wp.zeros(N, dtype=wp.vec3)
        self._wp_col      = wp.zeros(N, dtype=wp.vec4)

        self._wp_nn_idx  = wp.zeros(N * K, dtype=int)
        self._wp_nn_dist = wp.zeros(N * K, dtype=float)

        self._wp_edge_src = wp.zeros(N_MAX_EDGES, dtype=int)
        self._wp_edge_dst = wp.zeros(N_MAX_EDGES, dtype=int)
        self._n_edges     = 0

        self._wp_edge_pos = wp.zeros(N_MAX_VERTS, dtype=wp.vec3)
        self._wp_edge_col = wp.zeros(N_MAX_VERTS, dtype=wp.vec4)

        self._phase            = PHASE_DARK
        self._build_frame      = 0
        self._build_done_frame = 0
        self._hide_frame       = 0
        self._pending_seed     = None

        wp.launch(init_nn_points, dim=N,
                  inputs=[self._wp_pos, self._wp_base_pos, self._wp_col, 42])

        pos_np = self._wp_pos.numpy()
        col_np = self._wp_col.numpy()

        self._pts_draw = PointsDrawable(ctx)
        self._pts_draw.setup(pos_np, col_np)

        self._edge_draw = DynamicLinesDrawable(ctx)
        self._edge_draw.setup(N_MAX_EDGES)

    # ── Rebuild (called only when dark) ────────────────────────────────────────

    def _rebuild(self, t: float) -> None:
        """
        Animate points once to get the current snapshot, compute KNN on GPU,
        build the deduplicated sorted undirected edge list on CPU, upload to GPU.
        Called exclusively from the DARK phase — nothing is visible at this point.
        """
        # Re-seed if requested
        if self._pending_seed is not None:
            wp.launch(init_nn_points, dim=N,
                      inputs=[self._wp_pos, self._wp_base_pos, self._wp_col,
                               self._pending_seed])
            self._pending_seed = None

        # One animation step to get the current live snapshot
        wp.launch(animate_nn_points, dim=N,
                  inputs=[self._wp_pos, self._wp_base_pos, t, 0.08])

        # KNN on GPU — synchronise so we can read nn_indices on CPU
        wp.launch(find_knn_5, dim=N,
                  inputs=[self._wp_pos, self._wp_nn_idx, self._wp_nn_dist, N])
        wp.synchronize()

        # Deduplicate on CPU — a set makes duplicate edges impossible
        nn_cpu = self._wp_nn_idx.numpy()   # (N*K,)
        seen, srcs, dsts = set(), [], []
        for e in range(N * K):
            src = e // K
            dst = int(nn_cpu[e])
            if dst >= 0 and src < dst:
                key = (src, dst)
                if key not in seen:
                    seen.add(key)
                    srcs.append(src)
                    dsts.append(dst)

        # Sort → spatially coherent traversal (short local edges first)
        pairs = sorted(zip(srcs, dsts))
        if pairs:
            srcs, dsts = zip(*pairs)
        else:
            srcs, dsts = [], []

        self._n_edges = len(srcs)
        if self._n_edges > 0:
            wp.copy(self._wp_edge_src,
                    wp.array(np.array(srcs, dtype=np.int32), dtype=int))
            wp.copy(self._wp_edge_dst,
                    wp.array(np.array(dsts, dtype=np.int32), dtype=int))

        # Frame at which the last edge reaches alpha = 1.0
        if self._n_edges > 0:
            last_birth = (self._n_edges - 1) // EDGES_PER_FRAME
            self._build_done_frame = last_birth + int(np.ceil(1.0 / FADE_SPEED)) + 1
        else:
            self._build_done_frame = 1

        self._build_frame = 0

        # Freeze the current point snapshot into the drawables
        pos_np = self._wp_pos.numpy()
        col_np = self._wp_col.numpy()
        self._pts_draw.update(pos_np, col_np)

        print(f"[nn_graph] {self._n_edges} edges  (done_frame={self._build_done_frame})")

    # ── Step ───────────────────────────────────────────────────────────────────

    def step(self, t: float, dt: float) -> None:
        # ── Phase transitions & per-phase work ────────────────────────────────

        if self._phase == PHASE_DARK:
            # Build structure from live positions, then freeze everything
            self._rebuild(t)
            self._phase = PHASE_BUILDING
            # VBO is still all-zeros from end of HIDING → correct for frame 0
            return   # nothing more to do this frame

        elif self._phase == PHASE_BUILDING:
            self._build_frame += 1
            if self._build_frame >= self._build_done_frame:
                # All edges fully opaque — bake ACTIVE state into VBO once
                self._bake_active_vbo()
                self._phase = PHASE_ACTIVE
                return

            # Update VBO: edges fading in
            self._launch_vbo(_BUILDING, self._build_frame)

        elif self._phase == PHASE_ACTIVE:
            return   # static — VBO already correct, nothing to do

        elif self._phase == PHASE_HIDING:
            self._hide_frame += 1
            if self._hide_frame >= HIDE_FRAMES:
                self._phase    = PHASE_DARK
                self._hide_frame = 0
                return   # VBO is now all-zeros; DARK will rebuild next frame

            # Update VBO: edges fading out
            self._launch_vbo(_HIDING, self._hide_frame)

    # ── VBO helpers ────────────────────────────────────────────────────────────

    def _launch_vbo(self, phase_int: int, frame_param: int) -> None:
        if self._n_edges == 0:
            return
        wp.launch(
            build_edge_vbo, dim=N_MAX_EDGES,
            inputs=[
                self._wp_pos, self._wp_col,
                self._wp_edge_src, self._wp_edge_dst,
                self._wp_edge_pos, self._wp_edge_col,
                self._n_edges, phase_int, frame_param,
                EDGES_PER_FRAME, FADE_SPEED,
            ],
        )
        self._edge_draw.write_warp(self._wp_edge_pos, self._wp_edge_col)

    def _bake_active_vbo(self) -> None:
        """Write the fully-opaque edge VBO once when transitioning to ACTIVE."""
        self._launch_vbo(_ACTIVE, 0)

    # ── Draw ───────────────────────────────────────────────────────────────────

    def draw(self, mvp: np.ndarray) -> None:
        self._edge_draw.draw(mvp)
        self._pts_draw.draw(mvp)

    # ── Input ──────────────────────────────────────────────────────────────────

    def on_key(self, key, action, keys) -> None:
        if action != keys.ACTION_PRESS:
            return
        if key == keys.R:
            import random
            self._pending_seed = random.randint(0, 2**31 - 1)
            if self._phase in (PHASE_BUILDING, PHASE_ACTIVE):
                self._phase      = PHASE_HIDING
                self._hide_frame = 0
            print(f"[nn_graph] rebuild queued (seed={self._pending_seed})")
