"""
gui4.py – DNA HelixGUI
A double-helix DNA visualization rendered as animated ribbon lines.

Structure:
  • Two phosphate-sugar backbone strands winding in opposite helical phases
  • Base-pair rungs connecting the strands (A-T in cyan/amber, G-C in green/magenta)
  • Minor/major groove markers as fine radial spurs
  • Floating nucleotide "glow rings" at each base-pair site
  • Multiple DNA strands at different depths for a field-of-molecules look
  • Everything slowly rotates and the helix phase animates over time

Controls:
  R          -- regenerate / new seed
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

import math
from dataclasses import dataclass

import numpy as np
import moderngl
import moderngl_window as mglw
import warp as wp

from drawlib.drawable import RibbonDrawable
from drawlib.post_effect import FeedbackPostEffect
from drawlib.warp_feedback import FeedbackParams
from drawlib.camera import OrbitCamera

wp.init()

# ── Geometry constants ────────────────────────────────────────────────────────
N_BASE_PAIRS     = 40       # base pairs per helix
N_HELIX_SEG      = 8        # ribbon segments per base-pair step
N_RING_SEG       = 24       # segments per nucleotide glow ring
HELIX_RADIUS     = 0.28     # radius of each strand around the central axis
HELIX_PITCH      = 0.09     # vertical rise per base pair
STRAND_HALF_W    = 0.007    # width of backbone ribbon
RUNG_HALF_W      = 0.004    # width of base-pair rung ribbon
RING_HALF_W      = 0.003    # width of nucleotide ring
SPUR_HALF_W      = 0.002    # groove spur width
N_MOLECULES      = 5        # number of full DNA molecules in the scene


# ── Colour palette ────────────────────────────────────────────────────────────
# Bioluminescent palette: electric blue/cyan backbone, warm amber/magenta base pairs

def _c(r, g, b, a=1.0):
    return np.array([r, g, b, a], dtype=np.float32)

# Strand colours — phosphate-sugar backbone
STRAND_A_DARK  = _c(0.05, 0.35, 0.90, 1.0)   # deep electric blue
STRAND_A_LIGHT = _c(0.20, 0.75, 1.00, 1.0)   # sky cyan
STRAND_B_DARK  = _c(0.60, 0.05, 0.85, 1.0)   # deep violet
STRAND_B_LIGHT = _c(0.95, 0.30, 1.00, 1.0)   # bright orchid

# Base-pair colours (two types to mimic A-T and G-C)
AT_DARK        = _c(1.00, 0.70, 0.05, 1.0)   # amber dark
AT_LIGHT       = _c(1.00, 0.95, 0.50, 1.0)   # pale gold
GC_DARK        = _c(0.05, 0.80, 0.45, 1.0)   # bio-green dark
GC_LIGHT       = _c(0.50, 1.00, 0.70, 1.0)   # mint

# Groove spur colours
SPUR_COL       = _c(0.20, 0.55, 0.80, 0.55)

# Nucleotide ring accents
RING_COLS = [
    _c(0.20, 0.90, 1.00, 0.85),   # cyan
    _c(1.00, 0.80, 0.10, 0.85),   # amber
    _c(0.30, 1.00, 0.55, 0.85),   # mint
    _c(1.00, 0.30, 0.90, 0.85),   # magenta
]


# ── Feedback preset ───────────────────────────────────────────────────────────

@dataclass
class DNAParams(FeedbackParams):
    base_zoom        : float = 1.0015
    zoom_sensitivity : float = 0.0
    base_rot         : float = 0.0006
    rot_sensitivity  : float = 0.0
    decay            : float = 0.990
    ripple_strength  : float = 0.0
    ripple_freq      : float = 8.0
    hue_shift        : float = 0.003
    chroma_offset    : float = 0.004
    sat_boost        : float = 1.18
    smear_strength   : float = 0.0


# ── Mesh helpers ──────────────────────────────────────────────────────────────

def _width_dirs(tangents):
    refs = [
        np.array([0., 1., 0.]),
        np.array([1., 0., 0.]),
        np.array([0., 0., 1.]),
    ]
    N   = len(tangents)
    out = np.zeros((N, 3), dtype=np.float32)
    done = np.zeros(N, dtype=bool)
    for ref in refs:
        w   = np.cross(tangents, ref)
        mag = np.linalg.norm(w, axis=1)
        ok  = (~done) & (mag > 0.1)
        w[ok] /= mag[ok, None]
        out[ok] = w[ok]
        done |= ok
        if done.all():
            break
    return out


def _ribbon(pts, colors, half_w, offset):
    N    = len(pts)
    seg  = pts[1:] - pts[:-1]
    tans = np.empty((N, 3), dtype=np.float32)
    tans[0]    = seg[0]
    tans[1:-1] = seg[:-1] + seg[1:]
    tans[-1]   = seg[-1]
    mag = np.linalg.norm(tans, axis=1, keepdims=True)
    tans /= np.where(mag < 1e-8, 1.0, mag)

    w = _width_dirs(tans)
    left  = pts - w * half_w
    right = pts + w * half_w

    verts       = np.empty((N * 2, 3), dtype=np.float32)
    verts[0::2] = left
    verts[1::2] = right

    cols       = np.empty((N * 2, 4), dtype=np.float32)
    cols[0::2] = colors * np.array([0.55, 0.55, 0.55, 1.0])
    cols[1::2] = colors

    n_q = N - 1
    i   = np.arange(n_q, dtype=np.uint32)
    v0  = offset + 2*i; v1 = offset + 2*i+1
    v2  = offset + 2*i+2; v3 = offset + 2*i+3
    idx = np.empty(n_q * 6, dtype=np.uint32)
    idx[0::6]=v0; idx[1::6]=v1; idx[2::6]=v3
    idx[3::6]=v0; idx[4::6]=v3; idx[5::6]=v2
    return verts, cols, idx


# ── DNA geometry builder ──────────────────────────────────────────────────────

def helix_point(t, phase, radius, pitch, axis_offset):
    """
    Returns a 3-D point on a helix.
    t       : parameter [0, N_BASE_PAIRS]
    phase   : angular phase offset (π for anti-parallel strand)
    axis_offset: (x, z) translation of the helix axis
    """
    angle  = 2.0 * math.pi * t / 10.0 + phase   # ~10 base-pairs per turn
    y      = (t / N_BASE_PAIRS) * (N_BASE_PAIRS * pitch) - (N_BASE_PAIRS * pitch * 0.5)
    x      = radius * math.cos(angle) + axis_offset[0]
    z      = radius * math.sin(angle) + axis_offset[1]
    return np.array([x, y, z], dtype=np.float32)


def _lerp_col(a, b, t):
    return a * (1.0 - t) + b * t


def build_dna_molecule(axis_x, axis_z, seed, rng):
    """Build one full DNA double-helix centred on (axis_x, ?, axis_z)."""
    all_verts  = []
    all_colors = []
    all_idx    = []
    vert_off   = 0

    def add(pts, colors, half_w):
        nonlocal vert_off
        if len(pts) < 2:
            return
        v, c, ix = _ribbon(pts, colors, half_w, vert_off)
        all_verts.append(v)
        all_colors.append(c)
        all_idx.append(ix)
        vert_off += v.shape[0]

    ao = (axis_x, axis_z)

    # ── 1. Backbone strands (dense ribbon) ───────────────────────────────────
    steps  = N_BASE_PAIRS * N_HELIX_SEG
    ts     = np.linspace(0, N_BASE_PAIRS, steps + 1)

    pts_a = np.array([helix_point(t, 0.0, HELIX_RADIUS, HELIX_PITCH, ao) for t in ts])
    pts_b = np.array([helix_point(t, math.pi, HELIX_RADIUS, HELIX_PITCH, ao) for t in ts])

    t_norm = np.linspace(0, 1, steps + 1)
    cols_a = np.array([_lerp_col(STRAND_A_DARK, STRAND_A_LIGHT, (math.sin(v * math.pi * 4) * 0.5 + 0.5))
                       for v in t_norm])
    cols_b = np.array([_lerp_col(STRAND_B_DARK, STRAND_B_LIGHT, (math.cos(v * math.pi * 4) * 0.5 + 0.5))
                       for v in t_norm])

    add(pts_a, cols_a, STRAND_HALF_W)
    add(pts_b, cols_b, STRAND_HALF_W)

    # ── 2. Base-pair rungs ───────────────────────────────────────────────────
    for bp in range(N_BASE_PAIRS):
        t_bp = bp + 0.5
        pa   = helix_point(t_bp, 0.0,      HELIX_RADIUS, HELIX_PITCH, ao)
        pb   = helix_point(t_bp, math.pi,  HELIX_RADIUS, HELIX_PITCH, ao)

        # Alternate between A-T (amber) and G-C (green)
        is_at = (bp % 2 == 0)
        dark  = AT_DARK  if is_at else GC_DARK
        light = AT_LIGHT if is_at else GC_LIGHT

        # 5 intermediate steps across the rung + midpoint colour spike
        N_RUNG = 7
        ts_r = np.linspace(0, 1, N_RUNG)
        pts_r = np.array([pa * (1 - u) + pb * u for u in ts_r])

        # Colour peaks brightly at the middle
        cols_r = np.array([
            _lerp_col(dark, light, 4.0 * u * (1.0 - u))   # parabolic brightness
            for u in ts_r
        ])
        add(pts_r, cols_r, RUNG_HALF_W)

        # ── 3. Nucleotide glow rings at each end of the rung ────────────────
        ring_col = RING_COLS[bp % len(RING_COLS)]
        axis_dir = (pb - pa)
        axis_dir /= max(np.linalg.norm(axis_dir), 1e-8)

        for centre, sign in [(pa, 0), (pb, 1)]:
            # Build a small ring perpendicular to the rung axis
            ring_r = float(rng.uniform(0.012, 0.022))

            # Find two vectors perpendicular to axis_dir
            perp1 = np.cross(axis_dir, np.array([0, 1, 0], dtype=np.float32))
            if np.linalg.norm(perp1) < 0.1:
                perp1 = np.cross(axis_dir, np.array([1, 0, 0], dtype=np.float32))
            perp1 /= np.linalg.norm(perp1)
            perp2  = np.cross(axis_dir, perp1)
            perp2 /= max(np.linalg.norm(perp2), 1e-8)

            angles   = np.linspace(0, 2*math.pi, N_RING_SEG + 1)
            ring_pts = np.array([
                centre + ring_r * (math.cos(a) * perp1 + math.sin(a) * perp2)
                for a in angles
            ], dtype=np.float32)

            ring_cols = np.tile(ring_col, (N_RING_SEG + 1, 1))
            add(ring_pts, ring_cols, RING_HALF_W)

    # ── 4. Minor/major groove spurs ──────────────────────────────────────────
    # Thin lines emanating outward from each strand at groove positions
    groove_steps = N_BASE_PAIRS * 2
    for gi in range(groove_steps):
        t_g   = gi * (N_BASE_PAIRS / groove_steps)
        # Alternate which strand gets the spur
        phase = 0.0 if gi % 2 == 0 else math.pi
        pa    = helix_point(t_g, phase, HELIX_RADIUS, HELIX_PITCH, ao)
        # Outward direction: away from helix axis
        axis_centre = np.array([axis_x, pa[1], axis_z], dtype=np.float32)
        outward     = pa - axis_centre
        mag         = np.linalg.norm(outward)
        if mag < 1e-8:
            continue
        outward /= mag

        spur_len = float(rng.uniform(0.018, 0.040))
        p_tip    = pa + outward * spur_len
        pts_s    = np.array([pa, p_tip])
        cols_s   = np.array([SPUR_COL * np.array([0.4, 0.4, 0.4, 1.0]), SPUR_COL])
        add(pts_s, cols_s, SPUR_HALF_W)

    if not all_verts:
        return None, None, None

    return (
        np.vstack(all_verts).astype(np.float32),
        np.vstack(all_colors).astype(np.float32),
        np.concatenate(all_idx).astype(np.uint32),
    )


def build_geometry(seed=0):
    """Build a field of DNA molecules arranged in a loose arc/grid."""
    rng = np.random.default_rng(seed)

    all_verts  = []
    all_colors = []
    all_idx    = []
    running_offset = 0

    # Arrange molecules in a loose circular arrangement + centre one
    positions = [(0.0, 0.0)]  # centre molecule
    for i in range(N_MOLECULES - 1):
        angle = 2.0 * math.pi * i / (N_MOLECULES - 1)
        r     = float(rng.uniform(0.55, 0.90))
        positions.append((r * math.cos(angle), r * math.sin(angle)))

    for mol_i, (ax, az) in enumerate(positions):
        v, c, ix = build_dna_molecule(ax, az, seed + mol_i, rng)
        if v is None:
            continue

        # Offset indices
        ix_off = ix + running_offset
        all_verts.append(v)
        all_colors.append(c)
        all_idx.append(ix_off)
        running_offset += v.shape[0]

    # Connecting "synaptic" lines between molecules — thin filaments
    for fi in range(12):
        m1, m2 = rng.choice(len(positions), size=2, replace=False)
        ax1, az1 = positions[m1]
        ax2, az2 = positions[m2]
        y1 = float(rng.uniform(-0.3, 0.3))
        y2 = float(rng.uniform(-0.3, 0.3))
        p1 = np.array([ax1, y1, az1], dtype=np.float32)
        p2 = np.array([ax2, y2, az2], dtype=np.float32)
        # Add a mid-curve point
        mx = (ax1 + ax2) * 0.5 + float(rng.uniform(-0.2, 0.2))
        my = (y1 + y2) * 0.5 + float(rng.uniform(-0.2, 0.2))
        mz = (az1 + az2) * 0.5 + float(rng.uniform(-0.2, 0.2))
        pm = np.array([mx, my, mz], dtype=np.float32)

        # Sample along quadratic bezier
        N_FIL = 20
        ts    = np.linspace(0, 1, N_FIL)
        pts   = np.array([(1-t)**2 * p1 + 2*(1-t)*t * pm + t**2 * p2 for t in ts], dtype=np.float32)

        hue_base = float(fi) / 12.0
        cols = np.array([
            np.array([
                0.3 + 0.7 * math.cos(hue_base * 2*math.pi),
                0.3 + 0.7 * math.cos((hue_base + 0.33) * 2*math.pi),
                0.3 + 0.7 * math.cos((hue_base + 0.66) * 2*math.pi),
                0.45,
            ], dtype=np.float32)
            for _ in ts
        ])

        v, c, ix = _ribbon(pts, cols, 0.0015, running_offset)
        all_verts.append(v)
        all_colors.append(c)
        all_idx.append(ix)
        running_offset += v.shape[0]

    return (
        np.vstack(all_verts).astype(np.float32),
        np.vstack(all_colors).astype(np.float32),
        np.concatenate(all_idx).astype(np.uint32),
    )


# ── Animated geometry — phase-shift each frame ───────────────────────────────

def build_animated_geometry(seed, phase):
    """Rebuild geometry with a helical phase offset to animate unwinding."""
    rng = np.random.default_rng(seed)

    # Inject global phase into helix_point by monkey-patching the module-level
    # function isn't clean, so instead we rebuild with an offset injected
    # via a closure-captured variable.
    global _PHASE_OFFSET
    _PHASE_OFFSET = phase
    return build_geometry(seed)

_PHASE_OFFSET = 0.0


# ── Animated geometry with phase – proper version ────────────────────────────

def helix_point_anim(t, phase, radius, pitch, axis_offset):
    angle  = 2.0 * math.pi * t / 10.0 + phase + _PHASE_OFFSET
    y      = (t / N_BASE_PAIRS) * (N_BASE_PAIRS * pitch) - (N_BASE_PAIRS * pitch * 0.5)
    x      = radius * math.cos(angle) + axis_offset[0]
    z      = radius * math.sin(angle) + axis_offset[1]
    return np.array([x, y, z], dtype=np.float32)


# Monkey-patch for animation
_orig_helix_point = helix_point
def helix_point(t, phase, radius, pitch, axis_offset):
    return helix_point_anim(t, phase, radius, pitch, axis_offset)


# ── GUI ───────────────────────────────────────────────────────────────────────

class DNAHelixGUI(mglw.WindowConfig):
    title        = "Warp — DNA Helix"
    window_size  = (1280, 720)
    gl_version   = (3, 3)
    resizable    = True
    aspect_ratio = None

    SCENE_ALPHA  = 0.22
    ANIM_SPEED   = 0.18   # radians per second of helix phase shift

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._seed      = 42
        self._phase     = 0.0
        self._geo: RibbonDrawable | None = None
        self._regen()

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self.camera = OrbitCamera()
        self._drag  = False

        self.post_effect_on = True
        w, h = self.window_size
        self._post = FeedbackPostEffect(
            params        = DNAParams(),
            scene_alpha   = self.SCENE_ALPHA,
            smear_pattern = "swirl",
        )
        self._post.setup(self.ctx, w, h)

        print("[gui4] DNA Helix ready  —  R: regenerate  P: post-effect  O: orbit  ESC: quit")

    def _regen(self, phase=None):
        global _PHASE_OFFSET
        _PHASE_OFFSET = self._phase if phase is None else phase
        self._geo = RibbonDrawable(self.ctx)
        self._geo.setup(*build_geometry(seed=self._seed))
        print(f"[gui4] geometry generated  (seed={self._seed}, phase={_PHASE_OFFSET:.2f})")

    # ── Per-frame ─────────────────────────────────────────────────────────────

    def on_render(self, current_time: float, frame_time: float):
        # Animate helix phase — rebuild geometry each frame for motion
        self._phase += self.ANIM_SPEED * frame_time
        global _PHASE_OFFSET
        _PHASE_OFFSET = self._phase

        # Rebuild geometry with new phase (cheap enough for the geometry count)
        self._geo = RibbonDrawable(self.ctx)
        self._geo.setup(*build_geometry(seed=self._seed))

        self.camera.tick(frame_time)
        mvp = self.camera.mvp(self.window_size)

        if self.post_effect_on:
            self._post.bind_scene_fbo()
            self._geo.draw(mvp)
            self._post.process(self._post.fbo, current_time, dt=0.0)
            self._post.blit_to_screen(self.ctx.screen)
        else:
            self.ctx.screen.use()
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.clear(0.01, 0.02, 0.05, 1.0)
            self._geo.draw(mvp)

    # ── Input ─────────────────────────────────────────────────────────────────

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
            self._seed += 1
            self._regen()
        elif key == keys.P:
            self.post_effect_on = not self.post_effect_on
            print(f"[gui4] post-effect: {'ON' if self.post_effect_on else 'OFF'}")
        elif key == keys.O:
            self.camera.orbit_enabled = not self.camera.orbit_enabled
            if self.camera.orbit_enabled:
                self.camera._user_idle = 0.0
                print("[gui4] orbit: ON")
            else:
                print("[gui4] orbit: OFF")
        elif self.post_effect_on:
            self._post.on_key(key, action, keys)

    def on_resize(self, width: int, height: int):
        self._post.resize(width, height)


if __name__ == "__main__":
    mglw.run_window_config(DNAHelixGUI)
