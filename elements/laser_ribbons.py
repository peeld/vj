"""
elements/laser_ribbons.py
LaserRibbons -- a pool of coloured billboard-quad ribbons that fly from the
camera position in whatever direction the camera was facing at spawn time.
They are *not* parented to the camera; each ribbon keeps the velocity vector
frozen at birth.  Looks like laser fire from a spaceship.

Design
------
* Fixed-size ring-buffer pool of MAX_RIBBONS slots (int32 alive flags).
* Two Warp kernels run every frame:
    _advance_ribbons  -- moves each live ribbon, kills it past MAX_DIST.
    _build_tris       -- packs 6 billboard-quad vertices per ribbon into
                         output arrays (2 triangles, camera-facing width).
* Output arrays are uploaded to a DynamicTrianglesDrawable via CUDA-GL interop.
* Spawning is CPU-side (tiny array round-trip ~256 x 12 B); happens at most
  once per SPAWN_INTERVAL seconds.
* draw() temporarily switches to additive blending for the neon glow look.

Usage::

    laser = LaserRibbons(ctx)

    # each frame -- pass the four camera vectors from OrbitCamera.position_and_axes()
    laser.step(frame_time, cam_eye, cam_forward, cam_right, cam_up)
    laser.draw(mvp)
"""

from __future__ import annotations

import numpy as np
import warp as wp
import moderngl

from drawlib.drawable import DynamicTrianglesDrawable

# ── Tuneable constants ────────────────────────────────────────────────────────

MAX_RIBBONS    = 256     # pool size (power-of-two)
RIBBON_LENGTH  = 0.30    # world-space tail length behind the head
HALF_WIDTH     = 0.018   # half-width of the ribbon quad in world space
RIBBON_SPEED   = 7.0     # units / second
MAX_DIST       = 20.0    # travel distance at which a ribbon is killed
SPAWN_INTERVAL = 0.045   # seconds between spawns  (~22 / second)
SPAWN_SPREAD   = 0.5     # lateral jitter radius at spawn point
BEHIND_OFFSET  = 0.20    # how far behind the camera eye the ribbon is born

# Neon laser palette (RGBA, full brightness)
_PALETTE = np.array([
    [1.00, 0.15, 0.15, 1.0],   # red
    [0.15, 1.00, 0.15, 1.0],   # green
    [0.20, 0.55, 1.00, 1.0],   # blue
    [0.00, 1.00, 1.00, 1.0],   # cyan
    [1.00, 0.00, 1.00, 1.0],   # magenta
    [1.00, 1.00, 0.00, 1.0],   # yellow
    [1.00, 0.50, 0.00, 1.0],   # orange
    [0.55, 0.00, 1.00, 1.0],   # violet
    [1.00, 1.00, 1.00, 1.0],   # white
], dtype=np.float32)


# ── Warp kernels ──────────────────────────────────────────────────────────────

@wp.kernel
def _advance_ribbons(
    pos:      wp.array(dtype=wp.vec3),
    vel:      wp.array(dtype=wp.vec3),
    dist:     wp.array(dtype=wp.float32),
    alive:    wp.array(dtype=wp.int32),
    dt:       float,
    max_dist: float,
):
    """Move each live ribbon forward and kill it once it has travelled max_dist."""
    i = wp.tid()
    if alive[i] == 0:
        return
    step    = vel[i] * dt
    pos[i]  = pos[i] + step
    dist[i] = dist[i] + wp.length(step)
    if dist[i] >= max_dist:
        alive[i] = 0


@wp.kernel
def _build_tris(
    pos:           wp.array(dtype=wp.vec3),
    vel:           wp.array(dtype=wp.vec3),
    col:           wp.array(dtype=wp.vec4),
    alive:         wp.array(dtype=wp.int32),
    ribbon_length: float,
    half_width:    float,
    cam_eye:       wp.vec3,
    out_pos:       wp.array(dtype=wp.vec3),
    out_col:       wp.array(dtype=wp.vec4),
):
    """Write 6 billboard-quad vertices per ribbon (2 triangles, CCW).

    The width vector is perpendicular to both the ribbon direction and the
    camera-to-ribbon vector, so the quad always faces the camera.
    Head vertices carry the full ribbon colour; tail vertices fade to alpha=0.

    Dead ribbons collapse to degenerate zero-area triangles far off-screen.

    Vertex slots per ribbon (base = i*6):
        0  head_left   1  head_right  2  tail_right   <- triangle A
        3  head_left   4  tail_right  5  tail_left    <- triangle B
    """
    i    = wp.tid()
    base = i * 6

    if alive[i] == 1:
        head = pos[i]

        # Ribbon forward direction
        spd = wp.length(vel[i])
        if spd > 1.0e-6:
            fwd = vel[i] * (1.0 / spd)
        else:
            fwd = wp.vec3(0.0, 0.0, 1.0)
        tail = head - fwd * ribbon_length

        # Billboard width: perp to ribbon dir and camera-to-head vector
        to_cam = cam_eye - head
        tc_len = wp.length(to_cam)
        if tc_len > 1.0e-6:
            to_cam = to_cam * (1.0 / tc_len)
        else:
            to_cam = wp.vec3(0.0, 1.0, 0.0)

        w = wp.cross(fwd, to_cam)
        w_len = wp.length(w)
        if w_len > 1.0e-6:
            w = w * (half_width / w_len)
        else:
            w = wp.vec3(half_width, 0.0, 0.0)

        hl = head + w    # head left
        hr = head - w    # head right
        tl = tail + w    # tail left
        tr = tail - w    # tail right

        c = col[i]
        hc = c                                       # head colour: full
        tc = wp.vec4(c[0], c[1], c[2], 0.0)         # tail colour: transparent

        # Triangle A
        out_pos[base + 0] = hl;  out_col[base + 0] = hc
        out_pos[base + 1] = hr;  out_col[base + 1] = hc
        out_pos[base + 2] = tr;  out_col[base + 2] = tc
        # Triangle B
        out_pos[base + 3] = hl;  out_col[base + 3] = hc
        out_pos[base + 4] = tr;  out_col[base + 4] = tc
        out_pos[base + 5] = tl;  out_col[base + 5] = tc

    else:
        far  = wp.vec3(1.0e6, 1.0e6, 1.0e6)
        zero = wp.vec4(0.0, 0.0, 0.0, 0.0)
        out_pos[base + 0] = far;  out_col[base + 0] = zero
        out_pos[base + 1] = far;  out_col[base + 1] = zero
        out_pos[base + 2] = far;  out_col[base + 2] = zero
        out_pos[base + 3] = far;  out_col[base + 3] = zero
        out_pos[base + 4] = far;  out_col[base + 4] = zero
        out_pos[base + 5] = far;  out_col[base + 5] = zero


# ── Element ───────────────────────────────────────────────────────────────────

class LaserRibbons:
    """Pool of coloured laser ribbons fired from behind the camera.

    Parameters
    ----------
    ctx:
        Active ModernGL context.
    """

    def __init__(self, ctx: moderngl.Context):
        self._ctx = ctx
        self._rng = np.random.default_rng()

        # Tuneable instance attributes (PropertyManager binds to these;
        # they shadow the module-level constants and can be changed at runtime)
        self.ribbon_speed   = RIBBON_SPEED
        self.ribbon_length  = RIBBON_LENGTH
        self.half_width     = HALF_WIDTH
        self.spawn_interval = SPAWN_INTERVAL
        self.spawn_spread   = SPAWN_SPREAD
        self.max_dist       = MAX_DIST

        # Active spawn palette — updated by set_palette(); defaults to the neon set
        self._palette = _PALETTE.copy()

        # CPU shadow arrays (source of truth for spawning)
        n = MAX_RIBBONS
        self._np_pos   = np.zeros((n, 3), dtype=np.float32)
        self._np_vel   = np.zeros((n, 3), dtype=np.float32)
        self._np_col   = np.zeros((n, 4), dtype=np.float32)
        self._np_alive = np.zeros(n, dtype=np.int32)
        self._np_dist  = np.zeros(n, dtype=np.float32)

        self._sync_to_warp()

        # Intermediate arrays that _build_tris writes into before GL upload
        # 6 vertices per ribbon
        self._wp_tris_pos = wp.zeros(n * 6, dtype=wp.vec3)
        self._wp_tris_col = wp.zeros(n * 6, dtype=wp.vec4)

        # GL drawable -- n ribbons x 6 verts each
        self._drawable = DynamicTrianglesDrawable(ctx)
        self._drawable.setup(n)

        # Spawn state
        self._next_slot   = 0
        self._spawn_timer = 0.0

        # Cache cam_eye for the kernel (updated each step)
        self._cam_eye = wp.vec3(0.0, 0.0, 5.0)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _sync_to_warp(self) -> None:
        """Upload all CPU shadow arrays to fresh Warp arrays."""
        self._wp_pos   = wp.array(self._np_pos,   dtype=wp.vec3)
        self._wp_vel   = wp.array(self._np_vel,   dtype=wp.vec3)
        self._wp_col   = wp.array(self._np_col,   dtype=wp.vec4)
        self._wp_alive = wp.array(self._np_alive, dtype=wp.int32)
        self._wp_dist  = wp.array(self._np_dist,  dtype=wp.float32)

    def _sync_from_warp(self) -> None:
        """Download Warp arrays back to CPU shadows (needed before spawning)."""
        self._np_pos   = self._wp_pos.numpy()
        self._np_vel   = self._wp_vel.numpy()
        self._np_col   = self._wp_col.numpy()
        self._np_alive = self._wp_alive.numpy()
        self._np_dist  = self._wp_dist.numpy()

    def _spawn(
        self,
        cam_eye:     np.ndarray,
        cam_forward: np.ndarray,
        cam_right:   np.ndarray,
        cam_up:      np.ndarray,
    ) -> None:
        """Write one new ribbon into the ring-buffer slot and re-upload."""
        self._sync_from_warp()

        slot = self._next_slot
        self._next_slot = (self._next_slot + 1) % MAX_RIBBONS

        # Random lateral jitter so ribbons fan out slightly
        jr = self._rng.uniform(-self.spawn_spread, self.spawn_spread)
        ju = self._rng.uniform(-self.spawn_spread, self.spawn_spread)

        spawn_pos = (cam_eye
                     - cam_forward * BEHIND_OFFSET
                     + cam_right   * jr
                     + cam_up      * ju)

        # Velocity: strictly in the forward direction frozen at birth
        spawn_vel = cam_forward * self.ribbon_speed

        color_idx = int(self._rng.integers(0, len(self._palette)))

        self._np_pos[slot]   = spawn_pos
        self._np_vel[slot]   = spawn_vel
        self._np_col[slot]   = self._palette[color_idx]
        self._np_alive[slot] = 1
        self._np_dist[slot]  = 0.0

        self._sync_to_warp()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_palette(self, palette: list) -> None:
        """Replace the spawn colour palette with the given RGB list."""
        self._palette = np.array(
            [[r, g, b, 1.0] for r, g, b in palette], dtype=np.float32
        )

    def step(
        self,
        dt:          float,
        cam_eye:     np.ndarray,
        cam_forward: np.ndarray,
        cam_right:   np.ndarray,
        cam_up:      np.ndarray,
        enabled:     bool,
    ) -> None:
        """Spawn, advance, and build triangle geometry for this frame."""

        # Cache cam_eye as a Warp vec3 for the kernel
        self._cam_eye = wp.vec3(
            float(cam_eye[0]), float(cam_eye[1]), float(cam_eye[2])
        )

        # Spawn (may fire multiple times if dt > spawn_interval)
        if enabled:
            self._spawn_timer += dt
            while self._spawn_timer >= self.spawn_interval:
                self._spawn_timer -= self.spawn_interval
                self._spawn(cam_eye, cam_forward, cam_right, cam_up)
        else:
            self._spawn_timer = self.spawn_interval

        # Advance all live ribbons on GPU
        wp.launch(
            _advance_ribbons,
            dim=MAX_RIBBONS,
            inputs=[
                self._wp_pos, self._wp_vel, self._wp_dist, self._wp_alive,
                float(dt), float(self.max_dist),
            ],
        )

        # Build billboard-quad vertex data on GPU
        wp.launch(
            _build_tris,
            dim=MAX_RIBBONS,
            inputs=[
                self._wp_pos, self._wp_vel, self._wp_col, self._wp_alive,
                float(self.ribbon_length), float(self.half_width),
                self._cam_eye,
                self._wp_tris_pos, self._wp_tris_col,
            ],
        )

        # GPU -> GL upload via CUDA interop
        self._drawable.write_warp(self._wp_tris_pos, self._wp_tris_col)

    def draw(self, mvp: np.ndarray) -> None:
        """Draw all ribbons with additive blending for the neon glow effect."""
        self._ctx.blend_func = moderngl.ONE, moderngl.ONE
        self._drawable.draw(mvp)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
