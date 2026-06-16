"""
elements/circleaxis.py - CircleAxisDrawing
Geometry, animation, and draw logic for the circle-axis scene.

Exports:
  build_geometry(n_circles, bound, seed) -> (verts, colors, indices, circle_metas, trav_metas)
  CircleAxisDrawing(ctx)                 -> drawable scene object
"""

import colorsys
from dataclasses import dataclass

import numpy as np
import moderngl

from drawlib.drawable import RibbonDrawable
from color_harmony import ColorScheme, generate_palette

import audio_metrics

# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------

BOUND          = 1.0
N_CIRCLES      = 24
N_SEG          = 90    # segments per circle arc
N_TRAV_LINES   = 35    # diagonal traversal lines
CIRCLE_HALF_W  = 0.0045
LINE_HALF_W    = 0.0022

# Turbine blade constants
N_BLADES          = 64      # squares per circle
BLADE_SIZE_FACTOR = 0.125   # blade side = radius * this
BLADE_SPIN_SPEED  = 0.2     # radians / second

# Emitted circle constants
MAX_EMIT_CIRCLES = 24       # pool size
EMIT_LIFETIME    = 3.5      # seconds before an emitted circle is gone
EMIT_FADE_IN     = 0.25     # seconds to fade in
EMIT_FADE_OUT    = 1.8      # seconds to dissolve out
EMIT_INTERVAL    = 0.30     # seconds between spawns when active
EMIT_RADIUS_MIN  = 0.04
EMIT_RADIUS_MAX  = 0.75
EMIT_HALF_W      = 0.006    # ribbon half-width for emitted circles


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def _hsv(h: float, s: float, v: float, a: float = 1.0) -> np.ndarray:
    h = h % 1.0
    i = int(h * 6)
    f = h * 6 - i
    p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
    r, g, b = [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)][i % 6]
    return np.array([r, g, b, a], dtype=np.float32)


# ---------------------------------------------------------------------------
# Ribbon mesh helpers
# ---------------------------------------------------------------------------

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


def _blade_positions(
    cx: float,
    radius: float,
    spin_offset: float,
    n_blades: int = N_BLADES,
    blade_size_factor: float = BLADE_SIZE_FACTOR,
) -> np.ndarray:
    """
    Returns (n_blades * 4, 3) vertex positions for turbine-blade quads.

    Each blade i:
      - sits at angular position  theta_i = spin_offset + 2*pi*i/n_blades
      - has blade-pitch angle     phi_i   = 2*pi*i/n_blades  (fans 0 to ~360 deg)

    Vertex order per blade: [+pitch+radial, +pitch-radial, -pitch-radial, -pitch+radial]
    Two triangles per blade: (0,1,2) and (0,2,3).
    """
    n  = n_blades
    bi = np.arange(n, dtype=np.float32)

    theta = spin_offset + 2.0 * np.pi * bi / n
    phi   = 2.0 * np.pi * bi / n

    hs = radius * blade_size_factor * 0.5

    P = np.zeros((n, 3), dtype=np.float32)
    P[:, 0] = cx
    P[:, 1] = radius * np.cos(theta)
    P[:, 2] = radius * np.sin(theta)

    R = np.zeros((n, 3), dtype=np.float32)
    R[:, 1] = np.cos(theta)
    R[:, 2] = np.sin(theta)

    T = np.zeros((n, 3), dtype=np.float32)
    T[:, 1] = -np.sin(theta)
    T[:, 2] =  np.cos(theta)

    X_hat = np.zeros((n, 3), dtype=np.float32)
    X_hat[:, 0] = 1.0

    c = np.cos(phi)[:, None]
    s = np.sin(phi)[:, None]
    pitch_dir = c * X_hat + s * T

    c0 = P + hs * pitch_dir + hs * R
    c1 = P + hs * pitch_dir - hs * R
    c2 = P - hs * pitch_dir - hs * R
    c3 = P - hs * pitch_dir + hs * R

    corners = np.stack([c0, c1, c2, c3], axis=1)
    return corners.reshape(n * 4, 3).astype(np.float32)


# ---------------------------------------------------------------------------
# Animation metadata
# ---------------------------------------------------------------------------

@dataclass
class CircleAnimMeta:
    cx:            float
    radius:        float
    phase:         float
    speed:         float
    amplitude:     float
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


@dataclass
class EmitCircle:
    cx:     float
    radius: float
    birth:  float   # absolute time of spawn
    hue:    float   # HSV hue for colour


# ---------------------------------------------------------------------------
# Geometry builder (public)
# ---------------------------------------------------------------------------

def build_geometry(
    n_circles        : int        = N_CIRCLES,
    n_trav_lines     : int        = N_TRAV_LINES,
    n_blades         : int        = N_BLADES,
    blade_size_factor: float      = BLADE_SIZE_FACTOR,
    bound            : float      = BOUND,
    seed             : int | None = None,
    palette          : list | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
    """
    Returns (verts, colors, indices, circle_metas, trav_metas).

    Vertex layout (must match CircleAxisDrawing._animated_positions exactly):
      for each circle:  [circle ribbon verts]  [blade quad verts]
      for each trav:    [line ribbon verts]
    """
    rng = np.random.default_rng(seed)
    pseed = int(seed) if seed is not None else 0

    if palette and len(palette) >= 1:
        mid  = max(1, len(palette) // 2)
        cool = palette[:mid] if mid > 0 else palette
        warm = palette[mid:] if palette[mid:] else palette
    else:
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

    circle_metas: list[CircleAnimMeta] = []

    for ci in range(n_circles):
        cx     = float(rng.uniform(-bound * 0.88, bound * 0.88))
        radius = float(rng.uniform(0.04, bound * 0.88))

        phase     = float(rng.uniform(0.0, 2.0 * np.pi))
        speed     = float(rng.uniform(0.08, 0.28))
        amplitude = float(rng.uniform(0.04, 0.22))

        bh, bs = cool_hs[ci % len(cool_hs)]
        bh = (bh + ci * 0.041) % 1.0

        pts = _circle_pts(cx, radius, N_SEG)
        t   = np.linspace(0.0, 1.0, len(pts), dtype=np.float32)
        cols = np.stack([
            _hsv((bh + ti * 0.10) % 1.0, max(0.55, bs), 0.40 + 0.60 * ti)
            for ti in t
        ])
        add_ribbon(pts, cols, CIRCLE_HALF_W)

        blade_verts_init = _blade_positions(cx, radius, 0.0,
                                              n_blades=n_blades,
                                              blade_size_factor=blade_size_factor)

        bi_arr  = np.arange(n_blades, dtype=np.float32)
        phi_arr = 2.0 * np.pi * bi_arr / n_blades
        face_alpha = (np.abs(np.sin(phi_arr)) * 0.35 + 0.65).astype(np.float32)

        blade_colors = np.zeros((n_blades * 4, 4), dtype=np.float32)
        for bi in range(n_blades):
            blade_frac = bi / n_blades
            col = _hsv(
                (bh + blade_frac * 0.12) % 1.0,
                min(1.0, bs + 0.15),
                0.75 + 0.25 * blade_frac,
                float(face_alpha[bi] * 0.1),
            )
            blade_colors[bi * 4 : bi * 4 + 4] = col

        blade_idx = np.zeros(n_blades * 6, dtype=np.uint32)
        for bi in range(n_blades):
            base = vert_offset + bi * 4
            o    = bi * 6
            blade_idx[o]   = base;     blade_idx[o+1] = base + 1
            blade_idx[o+2] = base + 2; blade_idx[o+3] = base
            blade_idx[o+4] = base + 2; blade_idx[o+5] = base + 3

        all_verts.append(blade_verts_init)
        all_colors.append(blade_colors)
        all_idx.append(blade_idx)
        vert_offset += n_blades * 4

        circle_metas.append(CircleAnimMeta(
            cx=cx, radius=radius,
            phase=phase, speed=speed, amplitude=amplitude,
            n_verts=(N_SEG + 1) * 2,
            n_blade_verts=n_blades * 4,
        ))

    trav_metas: list[TravAnimMeta] = []

    for li in range(n_trav_lines):
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

        lv = float(li / max(1, n_trav_lines))
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
# Drawing class (public)
# ---------------------------------------------------------------------------

class CircleAxisDrawing:
    """
    Encapsulates all geometry, animation, and draw calls for the circle-axis scene.

    Usage:
        drawing = CircleAxisDrawing(ctx)
        drawing.update_audio(metrics)          # call from audio callback
        drawing.draw(mvp, current_time)        # call each frame
        drawing.on_key(key, action, keys)      # returns True if key was consumed
    """

    def __init__(self, ctx: moderngl.Context):
        self._ctx = ctx
        self._seed = 0
        self._geo: RibbonDrawable | None = None
        self._circle_metas: list[CircleAnimMeta] = []
        self._trav_metas:   list[TravAnimMeta]   = []
        self._audio_data:   audio_metrics.AudioMetrics | None = None

        # Tuneable instance attributes (PropertyManager binds to these;
        # n_circles / n_trav_lines / n_blades / blade_size_factor take effect
        # on the next regen(); blade_spin_speed is applied every frame).
        self.n_circles        = N_CIRCLES
        self.n_trav_lines     = N_TRAV_LINES
        self.n_blades         = N_BLADES
        self.blade_size_factor = BLADE_SIZE_FACTOR
        self.blade_spin_speed  = BLADE_SPIN_SPEED

        # Current colour palette — set via set_palette(); used by regen() and _spawn_emit()
        self._palette: list = []

        # Emitted circle pool
        self._rng          = np.random.default_rng()
        self._emit_pool:   list[EmitCircle | None] = [None] * MAX_EMIT_CIRCLES
        self._spawn_timer  = 0.0
        self._emit_geo     = RibbonDrawable(ctx)
        self._emit_geo_ready = False
        self._init_emit_geo()

        self.regen()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_palette(self, palette: list) -> None:
        """Store the colour palette; takes effect on the next regen() call."""
        self._palette = list(palette)

    def update_audio(self, m: audio_metrics.AudioMetrics) -> None:
        """Receive a fresh AudioMetrics value to drive animation parameters."""
        self._audio_data = m

    def regen(self) -> None:
        """Rebuild geometry with the next seed, using current instance settings."""
        if self._geo is None:
            self._geo = RibbonDrawable(self._ctx)
        verts, colors, idx, circle_metas, trav_metas = build_geometry(
            n_circles   = self.n_circles,
            n_trav_lines= self.n_trav_lines,
            n_blades    = self.n_blades,
            blade_size_factor = self.blade_size_factor,
            seed        = self._seed,
            palette     = self._palette or None,
        )
        self._geo.setup(verts, colors, idx)
        self._circle_metas = circle_metas
        self._trav_metas   = trav_metas
        print(f"[circleaxis] geometry generated  (seed={self._seed}  "
              f"circles={self.n_circles}  trav={self.n_trav_lines}  "
              f"blades={self.n_blades})")
        self._seed += 1

    def on_key(self, key, action, keys) -> bool:
        """
        Handle key events relevant to drawing.
        Returns True if the key was consumed.
        """
        if action != keys.ACTION_PRESS:
            return False
        if key == keys.R:
            self.regen()
            return True
        return False

    def step(self, dt: float, t: float, enabled: bool) -> None:
        """Advance emitted circle pool: age out dead circles, spawn new ones when active."""
        # Age out expired circles
        for i, ec in enumerate(self._emit_pool):
            if ec is not None and (t - ec.birth) >= EMIT_LIFETIME:
                self._emit_pool[i] = None

        # Spawn new circles when active
        if enabled:
            self._spawn_timer += dt
            while self._spawn_timer >= EMIT_INTERVAL:
                self._spawn_timer -= EMIT_INTERVAL
                self._spawn_emit(t)
        else:
            self._spawn_timer = 0.0

    def draw(self, mvp: np.ndarray, t: float) -> None:
        """Update animated positions and issue the draw call."""
        # Base geometry is kept alive for animation state but not drawn;
        # only emitted circles render so the element is empty when inactive.
        new_verts = self._animated_positions(t)
        self._geo.update(vertices=new_verts)

        # Draw emitted circles (always, so they dissolve even after deactivation)
        self._update_emit_geo(t)
        self._ctx.blend_func = moderngl.ONE, moderngl.ONE
        self._emit_geo.draw(mvp)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    # ------------------------------------------------------------------
    # Emitted circle helpers
    # ------------------------------------------------------------------

    def _init_emit_geo(self) -> None:
        """Pre-allocate fixed-size GPU buffers for the emitted circle pool."""
        verts_per_circle = (N_SEG + 1) * 2
        idx_per_circle   = N_SEG * 6

        n_verts = MAX_EMIT_CIRCLES * verts_per_circle
        n_idx   = MAX_EMIT_CIRCLES * idx_per_circle

        verts  = np.zeros((n_verts, 3), dtype=np.float32)
        colors = np.zeros((n_verts, 4), dtype=np.float32)

        # Build indices once — they never change, only positions/colors update
        all_idx = np.zeros(n_idx, dtype=np.uint32)
        seg_i = np.arange(N_SEG, dtype=np.uint32)
        for slot in range(MAX_EMIT_CIRCLES):
            v_base = slot * verts_per_circle
            i_base = slot * idx_per_circle
            v0 = v_base + 2 * seg_i
            v1 = v_base + 2 * seg_i + 1
            v2 = v_base + 2 * seg_i + 2
            v3 = v_base + 2 * seg_i + 3
            slot_idx = np.empty(idx_per_circle, dtype=np.uint32)
            slot_idx[0::6] = v0;  slot_idx[1::6] = v1;  slot_idx[2::6] = v3
            slot_idx[3::6] = v0;  slot_idx[4::6] = v3;  slot_idx[5::6] = v2
            all_idx[i_base:i_base + idx_per_circle] = slot_idx

        self._emit_geo.setup(verts, colors, all_idx)
        self._emit_geo_ready = True

    def _spawn_emit(self, t: float) -> None:
        """Place a new emitted circle into a free pool slot (or oldest)."""
        # Find a free slot
        slot = next((i for i, ec in enumerate(self._emit_pool) if ec is None), None)
        if slot is None:
            # Evict the oldest
            oldest_age = -1.0
            slot = 0
            for i, ec in enumerate(self._emit_pool):
                if ec is not None:
                    age = t - ec.birth
                    if age > oldest_age:
                        oldest_age = age
                        slot = i

        cx     = float(self._rng.uniform(-BOUND * 0.85, BOUND * 0.85))
        radius = float(self._rng.uniform(EMIT_RADIUS_MIN, EMIT_RADIUS_MAX))
        if self._palette:
            rgb = self._palette[int(self._rng.integers(0, len(self._palette)))]
            hue = colorsys.rgb_to_hsv(*rgb)[0]
        else:
            hue = float(self._rng.uniform(0.0, 1.0))
        self._emit_pool[slot] = EmitCircle(cx=cx, radius=radius, birth=t, hue=hue)

    @staticmethod
    def _emit_alpha(age: float) -> float:
        """Envelope: fade in → hold → dissolve."""
        if age >= EMIT_LIFETIME:
            return 0.0
        if age < EMIT_FADE_IN:
            return age / EMIT_FADE_IN
        if age > EMIT_LIFETIME - EMIT_FADE_OUT:
            return (EMIT_LIFETIME - age) / EMIT_FADE_OUT
        return 1.0

    def _update_emit_geo(self, t: float) -> None:
        """Rebuild positions + colors for the emitted circle pool and upload."""
        if not self._emit_geo_ready:
            return

        verts_per_circle = (N_SEG + 1) * 2
        n_verts = MAX_EMIT_CIRCLES * verts_per_circle

        all_pos = np.zeros((n_verts, 3), dtype=np.float32)
        all_col = np.zeros((n_verts, 4), dtype=np.float32)

        FAR = np.array([1e6, 1e6, 1e6], dtype=np.float32)

        for slot, ec in enumerate(self._emit_pool):
            v_base = slot * verts_per_circle
            v_end  = v_base + verts_per_circle

            if ec is None:
                all_pos[v_base:v_end] = FAR
                continue

            age   = t - ec.birth
            alpha = self._emit_alpha(age)

            pts  = _circle_pts(ec.cx, ec.radius, N_SEG)
            vpos = _ribbon_positions(pts, EMIT_HALF_W)
            all_pos[v_base:v_end] = vpos

            # Colour: hue shifts gently with age, bright saturated
            hue_shift = (ec.hue + age * 0.04) % 1.0
            t_arr     = np.linspace(0.0, 1.0, verts_per_circle, dtype=np.float32)
            cols      = np.stack([
                _hsv((hue_shift + ti * 0.08) % 1.0, 0.80, 0.90 + 0.10 * ti, alpha)
                for ti in t_arr
            ])
            all_col[v_base:v_end] = cols

        self._emit_geo.update(vertices=all_pos, colors=all_col)

    # ------------------------------------------------------------------
    # Internal animation
    # ------------------------------------------------------------------

    def _animated_positions(self, t: float) -> np.ndarray:
        """
        Rebuild all vertex positions at time t.

        Order must exactly match build_geometry:
          for each circle: [ribbon verts] [blade verts]
          for each trav:   [ribbon verts]
        """
        parts: list[np.ndarray] = []

        for m in self._circle_metas:
            aa = m.amplitude
            if self._audio_data is not None:
                aa = self._audio_data.energy * 0.4

            cx = m.cx + aa * np.sin(aa * t * m.speed + m.phase)

            pts = _circle_pts(cx, m.radius, N_SEG)
            parts.append(_ribbon_positions(pts, CIRCLE_HALF_W))

            n_blades_this = m.n_blade_verts // 4
            spin = t * self.blade_spin_speed
            parts.append(_blade_positions(cx, m.radius, spin,
                                          n_blades=n_blades_this,
                                          blade_size_factor=self.blade_size_factor))

        for m in self._trav_metas:
            dx  = m.amplitude * np.sin(t * m.speed + m.phase) * 15
            p1  = np.array([m.x1 + dx, m.y, m.z], dtype=np.float32)
            p2  = np.array([m.x2 + dx, m.y, m.z], dtype=np.float32)
            pts = np.stack([p1, p2])
            parts.append(_ribbon_positions(pts, LINE_HALF_W))

        return np.vstack(parts).astype(np.float32)
