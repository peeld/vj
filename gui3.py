"""
gui3.py - CircleAxisGUI
Random-sized circles centred on the X axis (in YZ-planes perpendicular to it),
with diagonal traversal lines woven between them.  Each circle also has 64
turbine-blade squares at its edge that fan-pitch from 0->360 degrees and spin
continuously like a jet-engine fan.

Controls:
  R          -- regenerate with a new seed
  P          -- toggle post-effect on / off
  O          -- toggle orbit camera on / off
  Mouse drag -- orbit camera
  Scroll     -- zoom
  ESC        -- quit

Post-effect tweaks (post-effect must be ON):
  Z/X  -- scene blend down/up   D/F  -- decay down/up
  Q/W  -- rotation down/up      A/S  -- zoom down/up
  H/J  -- hue shift down/up     C/V  -- chromatic aberration down/up
  B/N  -- saturation down/up    K/L  -- smear down/up
  M    -- cycle smear pattern   G    -- cycle blend mode
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

# Turbine blade constants
N_BLADES          = 64      # squares per circle
BLADE_SIZE_FACTOR = 0.125   # blade side = radius * this
BLADE_SPIN_SPEED  = 0.2     # radians / second


# ---------------------------------------------------------------------------
# Feedback preset
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    pts:    np.ndarray,
    colors: np.ndarray,
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


# ---------------------------------------------------------------------------
# Geometry builders
# ---------------------------------------------------------------------------

def _circle_pts(cx: float, radius: float, n_seg: int, cy: float = 0.0) -> np.ndarray:
    """Closed circle in the YZ plane at x=cx, centred at (cx, cy, 0)."""
    theta = np.linspace(0.0, 2.0 * np.pi, n_seg + 1, dtype=np.float32)
    pts   = np.zeros((n_seg + 1, 3), dtype=np.float32)
    pts[:, 0] = cx + cy * 10
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


def _blade_positions(cx: float, radius: float, spin_offset: float) -> np.ndarray:
    """
    Returns (N_BLADES * 4, 3) vertex positions for turbine-blade quads.

    Each blade i:
      - sits at angular position  theta_i = spin_offset + 2*pi*i/N_BLADES
      - has blade-pitch angle     phi_i   = 2*pi*i/N_BLADES  (fans 0 to ~360 deg)

    The pitch rotates the blade span axis from the X direction toward the
    circle tangent direction, giving a turbine-blade appearance.

    Vertex order per blade: [+pitch+radial, +pitch-radial, -pitch-radial, -pitch+radial]
    Two triangles per blade: (0,1,2) and (0,2,3).
    """
    n  = N_BLADES
    bi = np.arange(n, dtype=np.float32)

    theta = spin_offset + 2.0 * np.pi * bi / n   # position around circle
    phi   = 2.0 * np.pi * bi / n                  # blade pitch angle

    hs = radius * BLADE_SIZE_FACTOR * 0.5          # half-side length

    # Centre of each blade on the circumference  (n, 3)
    P = np.zeros((n, 3), dtype=np.float32)
    P[:, 0] = cx
    P[:, 1] = radius * np.cos(theta)
    P[:, 2] = radius * np.sin(theta)

    # Radial unit vector (outward from circle centre)  (n, 3)
    R = np.zeros((n, 3), dtype=np.float32)
    R[:, 1] = np.cos(theta)
    R[:, 2] = np.sin(theta)

    # Tangent unit vector (CCW along circle in YZ plane)  (n, 3)
    T = np.zeros((n, 3), dtype=np.float32)
    T[:, 1] = -np.sin(theta)
    T[:, 2] =  np.cos(theta)

    # X-axis direction  (n, 3)
    X_hat = np.zeros((n, 3), dtype=np.float32)
    X_hat[:, 0] = 1.0

    # Pitch direction: X rotated toward T by phi around R
    c = np.cos(phi)[:, None]
    s = np.sin(phi)[:, None]
    pitch_dir = c * X_hat + s * T               # (n, 3)

    # Four corners of each blade quad
    c0 = P + hs * pitch_dir + hs * R
    c1 = P + hs * pitch_dir - hs * R
    c2 = P - hs * pitch_dir - hs * R
    c3 = P - hs * pitch_dir + hs * R

    corners = np.stack([c0, c1, c2, c3], axis=1)   # (n, 4, 3)
    return corners.reshape(n * 4, 3).astype(np.float32)


@dataclass
class CircleAnimMeta:
    cx:            float
    radius:        float
    phase:         float   # random phase offset (radians)
    speed:         float   # angular speed (rad/s)
    amplitude:     float   # max displacement along X
    n_verts:       int     # ribbon vertex count for this circle
    n_blade_verts: int     # = N_BLADES * 4


@dataclass
class TravAnimMeta:
    x1:        float
    x2:        float
    y:         float
    z:         float
    phase:     float
    speed:     float
    amplitude: float


def build_geometry(
    n_circles: int = N_CIRCLES,
    bound:     float = BOUND,
    seed:      int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
    """
    Returns (verts, colors, indices, circle_metas, trav_metas).

    Vertex layout (must match _animated_positions exactly):
      for each circle:  [circle ribbon verts]  [blade quad verts]
      for each trav:    [line ribbon verts]
    """
    rng = np.random.default_rng(seed)
    pseed = int(seed) if seed is not None else 0

    cool = generate_palette(ColorScheme.TRIADIC,       seed=pseed)
    warm = generate_palette(ColorScheme.COMPLEMENTARY, seed=pseed + 13)
    cool_hs = [colorsys.rgb_to_hsv(*rgb)[:2] for rgb in cool]
    warm_hs = [colorsys.rgb_to_hsv(*rgb)[:2] for rgb in warm]

    all_verts:  list[np.ndarray] = []
    all_colors: list[np.ndarray] = []
    all_idx:    list[np.ndarray] = []
    vert_offset = 0

    def add_ribbon(pts: np.ndarray, colors: np.ndarray, half_w: float) -> None:
        nonlocal vert_offset
        if len(pts) < 2:
            return
        v, c, ix = _ribbon_mesh(pts, colors, half_w, vert_offset)
        all_verts.append(v)
        all_colors.append(c)
        all_idx.append(ix)
        vert_offset += v.shape[0]

    # Circles + turbine blades
    circle_metas: list[CircleAnimMeta] = []

    for ci in range(n_circles):
        cx     = float(rng.uniform(-bound * 0.88, bound * 0.88))
        radius = float(rng.uniform(0.04, bound * 0.88))

        phase     = float(rng.uniform(0.0, 2.0 * np.pi))
        speed     = float(rng.uniform(0.08, 0.28))
        amplitude = float(rng.uniform(0.04, 0.22))

        bh, bs = cool_hs[ci % len(cool_hs)]
        bh = (bh + ci * 0.041) % 1.0

        # Circle ribbon
        pts = _circle_pts(cx, radius, N_SEG)
        t   = np.linspace(0.0, 1.0, len(pts), dtype=np.float32)
        cols = np.stack([
            _hsv((bh + ti * 0.10) % 1.0, max(0.55, bs), 0.40 + 0.60 * ti)
            for ti in t
        ])
        add_ribbon(pts, cols, CIRCLE_HALF_W)

        # Turbine blade quads
        blade_verts_init = _blade_positions(cx, radius, 0.0)

        bi_arr  = np.arange(N_BLADES, dtype=np.float32)
        phi_arr = 2.0 * np.pi * bi_arr / N_BLADES
        # alpha slightly lower for edge-on blades (phi near 0 or pi)
        face_alpha = (np.abs(np.sin(phi_arr)) * 0.35 + 0.65).astype(np.float32)

        blade_colors = np.zeros((N_BLADES * 4, 4), dtype=np.float32)
        for bi in range(N_BLADES):
            blade_frac = bi / N_BLADES
            col = _hsv(
                (bh + blade_frac * 0.12) % 1.0,
                min(1.0, bs + 0.15),
                0.75 + 0.25 * blade_frac,
                float(face_alpha[bi] * 0.1),
            )
            blade_colors[bi * 4 : bi * 4 + 4] = col

        # Two triangles per blade: (0,1,2) and (0,2,3)
        blade_idx = np.zeros(N_BLADES * 6, dtype=np.uint32)
        for bi in range(N_BLADES):
            base = vert_offset + bi * 4
            o    = bi * 6
            blade_idx[o]   = base;     blade_idx[o+1] = base + 1
            blade_idx[o+2] = base + 2; blade_idx[o+3] = base
            blade_idx[o+4] = base + 2; blade_idx[o+5] = base + 3

        all_verts.append(blade_verts_init)
        all_colors.append(blade_colors)
        all_idx.append(blade_idx)
        vert_offset += N_BLADES * 4

        circle_metas.append(CircleAnimMeta(
            cx=cx, radius=radius,
            phase=phase, speed=speed, amplitude=amplitude,
            n_verts=(N_SEG + 1) * 2,
            n_blade_verts=N_BLADES * 4,
        ))

    # Traversal lines
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
        add_ribbon(pts, cols, LINE_HALF_W)

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


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

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

        print("[gui3] ready  --  R: regenerate  P: post-effect  O: orbit  ESC: quit")

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
        """
        Rebuild all vertex positions at time t.

        Order must exactly match build_geometry:
          for each circle: [ribbon verts] [blade verts]
          for each trav:   [ribbon verts]
        """
        parts: list[np.ndarray] = []

        for m in self._circle_metas:
            cx  = m.cx + m.amplitude * np.sin(t * m.speed + m.phase) * 10

            # Circle ribbon
            pts = _circle_pts(cx, m.radius, N_SEG)
            parts.append(_ribbon_positions(pts, CIRCLE_HALF_W))

            # Turbine blades -- spin continuously
            spin = t * BLADE_SPIN_SPEED
            parts.append(_blade_positions(cx, m.radius, spin))

        for m in self._trav_metas:
            dx  = m.amplitude * np.sin(t * m.speed + m.phase) * 15
            p1  = np.array([m.x1 + dx, m.y, m.z], dtype=np.float32)
            p2  = np.array([m.x2 + dx, m.y, m.z], dtype=np.float32)
            pts = np.stack([p1, p2])
            parts.append(_ribbon_positions(pts, LINE_HALF_W))

        return np.vstack(parts).astype(np.float32)

    # Per-frame

    def on_render(self, current_time: float, frame_time: float):
        self.camera.tick(frame_time)
        mvp = self.camera.mvp(self.window_size)

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

    # Input

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
