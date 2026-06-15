"""
gui2.py
ZigzagGUI -- procedurally generated zigzag ribbons with feedback post-effect and orbit camera.

Line generation rules (per ribbon):
  1. Pick a primary axis (X, Y, or Z) at random
  2. Start at the negative bound on that axis (random position on the other two)
  3. Move forward a random distance along the axis
  4. Turn ±45° (random) into a random perpendicular direction, travel a random distance
  5. Turn back to face the primary axis direction
  6. Repeat steps 3-5 until the primary-axis bound is exceeded

Each path is extruded into a flat ribbon whose width is perpendicular to the
direction of travel at each point.

Controls:
  R          -- regenerate ribbons with a new seed
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

BOUND            = 1.0    # scene occupies the [-BOUND, BOUND]³ cube
RIBBON_HALF_W    = 0.004  # half-width of each ribbon in world units
N_RIBBONS        = 60

# ── Feedback preset ────────────────────────────────────────────────────────────

@dataclass
class ZigzagParams(FeedbackParams):
    base_zoom        : float = 1.005
    zoom_sensitivity : float = 0.0
    base_rot         : float = 0.0015
    rot_sensitivity  : float = 0.0
    decay            : float = 0.990
    ripple_strength  : float = 0.0
    ripple_freq      : float = 10.0
    hue_shift        : float = 0.010
    chroma_offset    : float = 0.006
    sat_boost        : float = 1.15
    smear_strength   : float = 0.0


# ── Geometry ───────────────────────────────────────────────────────────────────

def _hsv(h: float, s: float, v: float, a: float = 1.0) -> np.ndarray:
    h = h % 1.0
    i = int(h * 6)
    f = h * 6 - i
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r, g, b = [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)][i % 6]
    return np.array([r, g, b, a], dtype=np.float32)


def _zigzag_path(rng, bound: float) -> tuple[np.ndarray, int]:
    """
    One zigzag path following the generation rules.
    Returns (points (N,3), primary_axis_index).
    """
    e    = np.eye(3, dtype=np.float32)
    axis = int(rng.integers(0, 3))
    perp = [i for i in range(3) if i != axis]

    pos = np.zeros(3, dtype=np.float32)
    pos[axis] = -bound
    for p in perp:
        pos[p] = float(rng.uniform(-bound * 0.6, bound * 0.6))

    pts = [pos.copy()]

    while pos[axis] < bound:
        # Move forward along primary axis
        dist = float(rng.uniform(0.06, 0.40)) * bound * 2
        nxt  = pos + e[axis] * dist
        if nxt[axis] >= bound:
            nxt[axis] = bound
            pts.append(nxt)
            break
        pos = nxt
        pts.append(pos.copy())

        # Turn ±45° into a random perpendicular direction and travel
        pi   = int(rng.choice(perp))
        sign = float(rng.choice([-1.0, 1.0]))
        diag = (e[axis] + e[pi] * sign) * 0.70711
        dist = float(rng.uniform(0.04, 0.22)) * bound * 2
        nxt  = pos + diag * dist
        if nxt[axis] >= bound:
            nxt[axis] = bound
            pts.append(nxt)
            break
        pos = nxt
        pts.append(pos.copy())
        # Implicit return to axis direction on next iteration

    return np.array(pts, dtype=np.float32), axis


_REF_VECS = np.array([
    [0.0, 1.0, 0.0],   # primary: up
    [1.0, 0.0, 0.0],   # fallback 1
    [0.0, 0.0, 1.0],   # fallback 2
], dtype=np.float32)


def _width_dirs(tangents: np.ndarray) -> np.ndarray:
    """
    Compute a unit vector perpendicular to each tangent.
    Tries reference vectors in order until cross product is non-degenerate.
    """
    N   = len(tangents)
    out = np.empty((N, 3), dtype=np.float32)
    for ref in _REF_VECS:
        w   = np.cross(tangents, ref)         # (N, 3)
        mag = np.linalg.norm(w, axis=1)       # (N,)
        ok  = mag > 0.1
        w[ok] /= mag[ok, None]
        out[ok] = w[ok]
        if ok.all():
            break
    return out


def _ribbon_mesh(
    pts:       np.ndarray,    # (N, 3) path points
    colors:    np.ndarray,    # (N, 4) per-point colours
    half_w:    float,
    offset:    int,           # global vertex offset for index buffer
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrude a path into a flat ribbon perpendicular to the direction of travel.
    Returns (verts (2N,3), cols (2N,4), indices (6*(N-1),)) with global offset applied.
    """
    N = len(pts)

    # Per-vertex tangents (averaged from adjacent segments)
    seg  = pts[1:] - pts[:-1]                 # (N-1, 3) segment vectors
    tans = np.empty((N, 3), dtype=np.float32)
    tans[0]    = seg[0]
    tans[1:-1] = seg[:-1] + seg[1:]           # sum of prev + next segment
    tans[-1]   = seg[-1]
    mag = np.linalg.norm(tans, axis=1, keepdims=True)
    mag = np.where(mag < 1e-8, 1.0, mag)
    tans /= mag

    # Width vectors: perpendicular to travel direction
    w_dirs = _width_dirs(tans)                 # (N, 3)

    # Left and right edge vertices, interleaved: L0 R0 L1 R1 ...
    left  = pts - w_dirs * half_w
    right = pts + w_dirs * half_w

    verts = np.empty((N * 2, 3), dtype=np.float32)
    verts[0::2] = left
    verts[1::2] = right

    # Both edges share the path colour; edges are slightly darker for depth cue
    cols = np.empty((N * 2, 4), dtype=np.float32)
    cols[0::2] = colors * np.array([0.65, 0.65, 0.65, 1.0])   # left edge (darker)
    cols[1::2] = colors                                         # right edge (full)

    # Triangle indices for each quad strip segment
    n_quads = N - 1
    i   = np.arange(n_quads, dtype=np.uint32)
    v0  = offset + 2 * i        # left[i]
    v1  = offset + 2 * i + 1   # right[i]
    v2  = offset + 2 * i + 2   # left[i+1]
    v3  = offset + 2 * i + 3   # right[i+1]

    idx = np.empty(n_quads * 6, dtype=np.uint32)
    idx[0::6] = v0;  idx[1::6] = v1;  idx[2::6] = v3   # tri 1
    idx[3::6] = v0;  idx[4::6] = v3;  idx[5::6] = v2   # tri 2

    return verts, cols, idx


def build_ribbons(n: int = N_RIBBONS, bound: float = BOUND, seed: int | None = None):
    """
    Generate *n* independent zigzag ribbons.
    Returns (vertices, colors, indices) for RibbonDrawable.setup().
    """
    rng = np.random.default_rng(seed)

    # Triadic palette — one harmonious base color per axis (X, Y, Z).
    # Saturation and lightness come from color_harmony's muted defaults.
    palette_seed = int(seed) if seed is not None else 0
    palette_rgb  = generate_palette(ColorScheme.TRIADIC, seed=palette_seed)
    # Extract hue + saturation from each palette color for use in _hsv below.
    axis_hs = [colorsys.rgb_to_hsv(*rgb)[:2] for rgb in palette_rgb]

    all_verts  : list[np.ndarray] = []
    all_colors : list[np.ndarray] = []
    all_idx    : list[np.ndarray] = []
    vert_offset = 0

    for li in range(n):
        pts, axis = _zigzag_path(rng, bound)
        nv = len(pts)
        if nv < 2:
            continue

        # Colour: hue drifts per line and along the path; brighter at the far end.
        # Base hue and saturation are anchored to the harmony palette for this axis.
        base_hue, base_sat = axis_hs[axis]
        base_hue = (base_hue + li * 0.031) % 1.0
        t        = np.linspace(0.0, 1.0, nv, dtype=np.float32)
        pt_cols  = np.stack([
            _hsv((base_hue + ti * 0.12) % 1.0, base_sat, 0.30 + 0.70 * ti)
            for ti in t
        ])

        verts, cols, idx = _ribbon_mesh(pts, pt_cols, RIBBON_HALF_W, vert_offset)

        all_verts.append(verts)
        all_colors.append(cols)
        all_idx.append(idx)
        vert_offset += verts.shape[0]   # 2 * nv

    return (
        np.vstack(all_verts).astype(np.float32),
        np.vstack(all_colors).astype(np.float32),
        np.concatenate(all_idx).astype(np.uint32),
    )


# ── GUI ────────────────────────────────────────────────────────────────────────

class ZigzagGUI(mglw.WindowConfig):
    title        = "Warp -- Zigzag Ribbons"
    window_size  = (1280, 720)
    gl_version   = (3, 3)
    resizable    = True
    aspect_ratio = None

    SCENE_ALPHA: float = 0.20

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._seed = 0
        self._ribbons: RibbonDrawable | None = None
        self._regen()

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self.camera = OrbitCamera()
        self._drag  = False

        self.post_effect_on = True
        w, h = self.window_size
        self._post = FeedbackPostEffect(
            params        = ZigzagParams(),
            scene_alpha   = self.SCENE_ALPHA,
            smear_pattern = "swirl",
        )
        self._post.setup(self.ctx, w, h)

        print("[gui2] ready  —  R: regenerate  P: post-effect  O: orbit  ESC: quit")

    def _regen(self):
        self._ribbons = RibbonDrawable(self.ctx)
        self._ribbons.setup(*build_ribbons(seed=self._seed))
        print(f"[gui2] ribbons generated  (seed={self._seed})")
        self._seed += 1

    # -- Per-frame ----------------------------------------------------------------

    def on_render(self, current_time: float, frame_time: float):
        self.camera.tick(frame_time)
        mvp = self.camera.mvp(self.window_size)

        if self.post_effect_on:
            self._post.bind_scene_fbo()
            self._ribbons.draw(mvp)
            self._post.process(self._post.fbo, current_time, dt=0.0)
            self._post.blit_to_screen(self.ctx.screen)
        else:
            self.ctx.screen.use()
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.clear(0.04, 0.04, 0.06, 1.0)
            self._ribbons.draw(mvp)

    # -- Input --------------------------------------------------------------------

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
            print(f"[gui2] post-effect: {'ON' if self.post_effect_on else 'OFF'}")
        elif key == keys.O:
            self.camera.orbit_enabled = not self.camera.orbit_enabled
            if self.camera.orbit_enabled:
                self.camera._user_idle = 0.0
                print("[gui2] orbit: ON")
            else:
                print("[gui2] orbit: OFF")
        elif self.post_effect_on:
            self._post.on_key(key, action, keys)

    def on_resize(self, width: int, height: int):
        self._post.resize(width, height)


if __name__ == "__main__":
    mglw.run_window_config(ZigzagGUI)
