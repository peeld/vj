"""
gui3.py – CircleAxisGUI
Random-sized circles centred on the X axis (in YZ-planes perpendicular to it),
with radial spokes through each circle and diagonal traversal lines woven
between them.

Controls:
  R          -- regenerate with a new seed
  P          -- toggle post-effect on / off
  O          -- toggle orbit camera on / off
  Mouse drag -- orbit camera
  Scroll     -- zoom
  ESC        -- quit

Post-effect tweaks (post-effect must be ON):
  Z/X  -- scene blend ↓/↑     D/F  -- decay ↓/↑
  Q/W  -- rotation ↓/↑        A/S  -- zoom ↓/↑
  H/J  -- hue shift ↓/↑       C/V  -- chromatic aberration ↓/↑
  B/N  -- saturation ↓/↑      K/L  -- smear ↓/↑
  M    -- cycle smear pattern  G    -- cycle blend mode
"""

import colorsys
from dataclasses import dataclass

import numpy as np
import moderngl
import moderngl_window as mglw
import warp as wp

from drawlib.drawable import RibbonDrawable
from drawlib.post_effect import FeedbackPostEffect
from drawlib.warp_feedback import FeedbackParams
from drawlib.camera import OrbitCamera
from color_harmony import ColorScheme, generate_palette

wp.init()

BOUND          = 1.0
N_CIRCLES      = 24
N_SEG          = 90    # segments per circle arc
N_SPOKES       = 6     # radial spokes per circle
N_TRAV_LINES   = 35    # diagonal traversal lines
CIRCLE_HALF_W  = 0.0045
SPOKE_HALF_W   = 0.0014
LINE_HALF_W    = 0.0022


# ── Feedback preset ─────────────────────────────────────────────────────────

@dataclass
class CircleParams(FeedbackParams):
    base_zoom        : float = 1.002
    zoom_sensitivity : float = 0.0
    base_rot         : float = 0.0008
    rot_sensitivity  : float = 0.0
    decay            : float = 0.993
    ripple_strength  : float = 0.0
    ripple_freq      : float = 10.0
    hue_shift        : float = 0.005
    chroma_offset    : float = 0.005
    sat_boost        : float = 1.12
    smear_strength   : float = 0.0


# ── Helpers ─────────────────────────────────────────────────────────────────

def _hsv(h: float, s: float, v: float, a: float = 1.0) -> np.ndarray:
    h = h % 1.0
    i = int(h * 6)
    f = h * 6 - i
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r, g, b = [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)][i % 6]
    return np.array([r, g, b, a], dtype=np.float32)


_REF_VECS = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
], dtype=np.float32)


def _width_dirs(tangents: np.ndarray) -> np.ndarray:
    N   = len(tangents)
    out = np.zeros((N, 3), dtype=np.float32)
    assigned = np.zeros(N, dtype=bool)
    for ref in _REF_VECS:
        w   = np.cross(tangents, ref)
        mag = np.linalg.norm(w, axis=1)
        ok  = (~assigned) & (mag > 0.1)
        w[ok] /= mag[ok, None]
        out[ok] = w[ok]
        assigned |= ok
        if assigned.all():
            break
    return out


def _ribbon_mesh(
    pts:    np.ndarray,   # (N, 3)
    colors: np.ndarray,   # (N, 4)
    half_w: float,
    offset: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N    = len(pts)
    seg  = pts[1:] - pts[:-1]
    tans = np.empty((N, 3), dtype=np.float32)
    tans[0]    = seg[0]
    tans[1:-1] = seg[:-1] + seg[1:]
    tans[-1]   = seg[-1]
    mag = np.linalg.norm(tans, axis=1, keepdims=True)
    tans /= np.where(mag < 1e-8, 1.0, mag)

    w_dirs = _width_dirs(tans)

    left  = pts - w_dirs * half_w
    right = pts + w_dirs * half_w

    verts       = np.empty((N * 2, 3), dtype=np.float32)
    verts[0::2] = left
    verts[1::2] = right

    cols       = np.empty((N * 2, 4), dtype=np.float32)
    cols[0::2] = colors * np.array([0.60, 0.60, 0.60, 1.0])
    cols[1::2] = colors

    n_quads = N - 1
    i  = np.arange(n_quads, dtype=np.uint32)
    v0 = offset + 2 * i
    v1 = offset + 2 * i + 1
    v2 = offset + 2 * i + 2
    v3 = offset + 2 * i + 3
    idx = np.empty(n_quads * 6, dtype=np.uint32)
    idx[0::6] = v0;  idx[1::6] = v1;  idx[2::6] = v3
    idx[3::6] = v0;  idx[4::6] = v3;  idx[5::6] = v2

    return verts, cols, idx


# ── Geometry builders ────────────────────────────────────────────────────────

def _circle_pts(cx: float, radius: float, n_seg: int, cy: float = 0.0) -> np.ndarray:
    """Closed circle in the YZ plane at x=cx, centred at (cx, cy, 0)."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_seg + 1, dtype=np.float32)
    pts   = np.zeros((n_seg + 1, 3), dtype=np.float32)
    pts[:, 0] = cx + cy  * 10
    pts[:, 1] = radius * np.cos(theta)
    pts[:, 2] = radius * np.sin(theta)
    return pts


def _ribbon_positions(pts: np.ndarray, half_w: float) -> np.ndarray:
    """Like _ribbon_mesh but returns only the (N*2, 3) vertex positions."""
    N    = len(pts)
    seg  = pts[1:] - pts[:-1]
    tans = np.empty((N, 3), dtype=np.float32)
    tans[0]    = seg[0]
    tans[1:-1] = seg[:-1] + seg[1:]
    tans[-1]   = seg[-1]
    mag = np.linalg.norm(tans, axis=1, keepdims=True)
    tans /= np.where(mag < 1e-8, 1.0, mag)
    w_dirs = _width_dirs(tans)
    verts       = np.empty((N * 2, 3), dtype=np.float32)
    verts[0::2] = pts - w_dirs * half_w
    verts[1::2] = pts + w_dirs * half_w
    return verts


@dataclass
class CircleAnimMeta:
    cx:        float
    radius:    float
    phase:     float   # random phase offset (radians)
    speed:     float   # angular speed (rad/s) — slow, lava-lamp
    amplitude: float   # max displacement along X
    n_verts:   int     # ribbon vertex count for this circle


@dataclass
class TravAnimMeta:
    x1:        float   # base x of endpoint 1
    x2:        float   # base x of endpoint 2
    y:         float   # shared Y (rv1 * cos(a1))
    z:         float   # shared Z (rv1 * sin(a1))
    phase:     float
    speed:     float
    amplitude: float   # max X shift


def build_geometry(
    n_circles: int = N_CIRCLES,
    bound:     float = BOUND,
    seed:      int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
    """
    Returns (verts, colors, indices, circle_metas, trav_metas).
    Both meta lists drive per-frame position rebuilds.
    """
    rng = np.random.default_rng(seed)
    pseed = int(seed) if seed is not None else 0

    # Cool palette for circles/spokes; complementary warm palette for traverse lines.
    cool = generate_palette(ColorScheme.TRIADIC,       seed=pseed)
    warm = generate_palette(ColorScheme.COMPLEMENTARY, seed=pseed + 13)
    cool_hs = [colorsys.rgb_to_hsv(*rgb)[:2] for rgb in cool]
    warm_hs = [colorsys.rgb_to_hsv(*rgb)[:2] for rgb in warm]

    all_verts:  list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_idx:    list[np.ndarray] = []
    vert_offset = 0

    def add(pts: np.ndarray, colors: np.ndarray, half_w: float) -> None:
        nonlocal vert_offset
        if len(pts) < 2:
            return
        v, c, ix = _ribbon_mesh(pts, colors, half_w, vert_offset)
        all_verts.append(v)
        all_colors.append(c)
        all_idx.append(ix)
        vert_offset += v.shape[0]

    # ── Circles ──────────────────────────────────────────────────────────────
    circle_metas: list[CircleAnimMeta] = []

    for ci in range(n_circles):
        cx     = float(rng.uniform(-bound * 0.88, bound * 0.88))
        radius = float(rng.uniform(0.04, bound * 0.88))

        # Lava-lamp animation params — slow, each circle on its own clock
        phase     = float(rng.uniform(0.0, 2.0 * np.pi))
        speed     = float(rng.uniform(0.08, 0.28))   # rad/s
        amplitude = float(rng.uniform(0.04, 0.22))

        pts = _circle_pts(cx, radius, N_SEG)  # cy=0 for initial build
        t   = np.linspace(0.0, 1.0, len(pts), dtype=np.float32)
        bh, bs = cool_hs[ci % len(cool_hs)]
        bh = (bh + ci * 0.041) % 1.0
        cols = np.stack([
            _hsv((bh + ti * 0.10) % 1.0, max(0.55, bs), 0.40 + 0.60 * ti)
            for ti in t
        ])
        add(pts, cols, CIRCLE_HALF_W)

        circle_metas.append(CircleAnimMeta(
            cx=cx, radius=radius,
            phase=phase, speed=speed, amplitude=amplitude,
            n_verts=(N_SEG + 1) * 2,  # ribbon doubles each source point
        ))

    # ── Traversal lines ──────────────────────────────────────────────────────
    trav_metas: list[TravAnimMeta] = []

    for li in range(N_TRAV_LINES):
        lh, ls = warm_hs[li % len(warm_hs)]
        lh = (lh + li * 0.057) % 1.0

        x1  = float(rng.uniform(-bound * 0.95, bound * 0.95))
        x2  = float(rng.uniform(-bound * 0.95, bound * 0.95))
        rv1 = float(rng.uniform(0.0, bound * 0.60))
        a1  = float(rng.uniform(0.0, 2.0 * np.pi))
        y   = float(rv1 * np.cos(a1))
        z   = float(rv1 * np.sin(a1))

        phase     = float(rng.uniform(0.0, 2.0 * np.pi))
        speed     = float(rng.uniform(0.06, 0.22))
        amplitude = float(rng.uniform(0.08, 0.35))

        p1  = np.array([x1, y, z], dtype=np.float32)
        p2  = np.array([x2, y, z], dtype=np.float32)
        pts = np.stack([p1, p2])

        lv = float(li / N_TRAV_LINES)
        cols = np.array([
            _hsv(lh, max(0.45, ls), 0.30 + 0.25 * lv),
            _hsv((lh + 0.09) % 1.0, max(0.45, ls), 0.90),
        ], dtype=np.float32)
        add(pts, cols, LINE_HALF_W)

        trav_metas.append(TravAnimMeta(
            x1=x1, x2=x2, y=y, z=z,
            phase=phase, speed=speed, amplitude=amplitude,
        ))

    return (
        np.vstack(all_verts).astype(np.float32),
        np.vstack(all_colors).astype(np.float32),
        np.concatenate(all_idx).astype(np.uint32),
        circle_metas,
        trav_metas,
    )


# ── GUI ──────────────────────────────────────────────────────────────────────

class CircleAxisGUI(mglw.WindowConfig):
    title        = "Warp -- Circle Axis"
    window_size  = (1280, 720)
    gl_version   = (3, 3)
    resizable    = True
    aspect_ratio = None

    SCENE_ALPHA: float = 0.18

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._seed = 0
        self._geo: RibbonDrawable | None = None
        self._circle_metas: list[CircleAnimMeta] = []
        self._trav_metas:   list[TravAnimMeta]   = []
        self._regen()

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self.camera = OrbitCamera()
        self._drag  = False

        self.post_effect_on = True
        w, h = self.window_size
        self._post = FeedbackPostEffect(
            params        = CircleParams(),
            scene_alpha   = self.SCENE_ALPHA,
            smear_pattern = "outward",
        )
        self._post.setup(self.ctx, w, h)

        print("[gui3] ready  —  R: regenerate  P: post-effect  O: orbit  ESC: quit")

    def _regen(self):
        if self._geo is None:
            self._geo = RibbonDrawable(self.ctx)
        verts, colors, idx, circle_metas, trav_metas = build_geometry(seed=self._seed)
        self._geo.setup(verts, colors, idx)
        self._circle_metas = circle_metas
        self._trav_metas   = trav_metas
        print(f"[gui3] geometry generated  (seed={self._seed})")
        self._seed += 1

    def _animated_positions(self, t: float) -> np.ndarray:
        """Rebuild all ribbon positions at time t."""
        parts: list[np.ndarray] = []
        for m in self._circle_metas:
            cx  = m.cx + m.amplitude * np.sin(t * m.speed + m.phase) * 10
            pts = _circle_pts(cx, m.radius, N_SEG)
            parts.append(_ribbon_positions(pts, CIRCLE_HALF_W))
        for m in self._trav_metas:
            dx  = m.amplitude * np.sin(t * m.speed + m.phase) * 15
            p1  = np.array([m.x1 + dx, m.y, m.z], dtype=np.float32)
            p2  = np.array([m.x2 + dx, m.y, m.z], dtype=np.float32)
            pts = np.stack([p1, p2])
            parts.append(_ribbon_positions(pts, LINE_HALF_W))
        return np.vstack(parts).astype(np.float32)

    # -- Per-frame ------------------------------------------------------------

    def on_render(self, current_time: float, frame_time: float):
        self.camera.tick(frame_time)
        mvp = self.camera.mvp(self.window_size)

        # Animate circle positions
        new_verts = self._animated_positions(current_time)
        self._geo.update(vertices=new_verts)

        if self.post_effect_on:
            self._post.bind_scene_fbo()
            self._geo.draw(mvp)
            self._post.process(self._post.fbo, current_time, dt=0.0)
            self._post.blit_to_screen(self.ctx.screen)
        else:
            self.ctx.screen.use()
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.clear(0.04, 0.04, 0.06, 1.0)
            self._geo.draw(mvp)

    # -- Input ----------------------------------------------------------------

    def on_mouse_press_event(self, x, y, button):
        if button == 1: self._drag = True

    def on_mouse_release_event(self, x, y, button):
        if button == 1: self._drag = False

    def on_mouse_drag_event(self, x, y, dx, dy):
        if self._drag: self.camera.on_drag(dx, dy)

    def on_mouse_scroll_event(self, x_offset, y_offset):
        self.camera.on_scroll(y_offset)

    def on_key_event(self, key, action, modifiers):
        if action != self.wnd.keys.ACTION_PRESS:
            return
        keys = self.wnd.keys

        if key == keys.ESCAPE:
            self.wnd.close()
        elif key == keys.R:
            self._regen()
        elif key == keys.P:
            self.post_effect_on = not self.post_effect_on
            print(f"[gui3] post-effect: {'ON' if self.post_effect_on else 'OFF'}")
        elif key == keys.O:
            self.camera.orbit_enabled = not self.camera.orbit_enabled
            if self.camera.orbit_enabled:
                self.camera._user_idle = 0.0
                print("[gui3] orbit: ON")
            else:
                print("[gui3] orbit: OFF")
        elif self.post_effect_on:
            self._post.on_key(key, action, keys)

    def on_resize(self, width: int, height: int):
        self._post.resize(width, height)


if __name__ == "__main__":
    mglw.run_window_config(CircleAxisGUI)
