"""
post/warp_feedback.py — reusable GPU feedback-loop pixel modifier.

The FeedbackLoop class wraps a ping-pong RGBA buffer pair and a
rotation/zoom/ripple/fisheye feedback kernel.  To use it:

    from post.warp_feedback import FeedbackLoop, FeedbackParams

    loop = FeedbackLoop(width=1280, height=720)

    # Each frame:
    loop.step(time_val)                       # feedback: prev → curr
    my_draw_kernel(loop.curr, ...)           # draw anything into curr
    frame_bgr = loop.to_bgr()               # uint8 BGR for OpenCV / display
    loop.advance()                           # curr becomes prev for next frame

Screen-space smear vectors
--------------------------
A coarse vector field (default 16×16) drives a per-pixel UV offset applied
*after* zoom/rotate/fisheye, so trails smear in the field's direction rather
than radiating from the centre.

    loop.set_smear_pattern("swirl")       # built-in presets
    loop.set_smear_vectors(my_array)      # (grid_h, grid_w, 2) float32
    params.smear_strength = 0.015         # UV units per frame (try 0.005–0.03)

Built-in patterns: "swirl", "swirl_cw", "spiral", "rightward", "leftward",
                   "upward", "downward", "inward", "outward",
                   "diagonal", "cross", "turbulence"

New parameters
--------------
  fisheye_strength : float  (default 0.0)
      Barrel (>0) or pincushion (<0) lens distortion applied to the UV
      lookup each feedback step.  Try ±0.3 – ±1.5 for visible effect.
      Positive values push pixels outward (wide-angle look); negative
      values pull them inward (telephoto compression).
"""

from dataclasses import dataclass, field
from typing import ClassVar
import numpy as np
import warp as wp

from prop import Node, Prop


# ═══════════════════════════════════════════════
#  Parameters dataclass
# ═══════════════════════════════════════════════

@dataclass
class FeedbackParams(Node, section="feedback"):
    # Zoom: >1 pulls toward centre, <1 pushes outward, 1.0 = no movement.
    zoom: float = 1.0                # zoom factor applied each step (1.0 = none)

    # Rotation around centre.
    rotation: float = 0.0004         # radians of rotation per step

    # Brightness decay — keeps energy from accumulating forever.
    decay: float = 0.97             # multiply RGB by this each frame (0.95–0.999)

    # Radial ripple.
    ripple_strength: float = 0.0    # max displacement in pixels (0 = off)
    ripple_freq: float = 12.0       # spatial frequency (rings per radius)

    # Hue rotation applied to the prev-frame sample each feedback step.
    hue_shift: float = 0.0          # radians per frame (try 0.005 – 0.02)

    # Chromatic aberration: R/B channels are sampled at radially offset
    # positions, creating colour halos around bright edges.
    chroma_offset: float = 0.0      # pixel offset as fraction of width (try 0.008–0.02)

    # Saturation pump: boost colour saturation on the prev-frame sample so
    # trails stay vivid as they decay rather than fading to grey.
    sat_boost: float = 1.0          # 1.0 = none, 1.1 = punchy, 1.3 = lurid

    # Screen-space smear: each pixel is sampled from prev at a UV position
    # shifted by (smear_field_vector * smear_strength).
    smear_strength: float = 0.0     # UV offset per frame (try 0.005 – 0.03)

    # Fisheye / barrel-pincushion lens distortion applied to the UV lookup.
    #   > 0  barrel (wide-angle push-out),  try 0.3 – 1.5
    #   < 0  pincushion (pull-in),          try -0.3 – -1.5
    #   0.0  no distortion (default)
    fisheye_strength: float = 0.0

    _zoom_prop:             ClassVar[Prop] = Prop("Zoom", float, 1.0, 0.950, 1.050, 0.001, attr="zoom", description="Zoom factor each feedback step (1.0 = none, >1 = inward, <1 = outward)")
    _rotation_prop:         ClassVar[Prop] = Prop("Rotation", float, 0.0008, 0.0, 0.01, 0.0001, attr="rotation", description="Radians of rotation per feedback step")
    _decay_prop:            ClassVar[Prop] = Prop("Decay", float, 0.993, 0.80, 0.999, 0.010, attr="decay", description="Per-step brightness multiplier (lower = shorter trails)")
    _ripple_strength_prop:  ClassVar[Prop] = Prop("Ripple Strength", float, 0.0, 0.0, 50.0, 0.5, attr="ripple_strength", description="Max pixel displacement of the radial ripple")
    _ripple_freq_prop:      ClassVar[Prop] = Prop("Ripple Frequency", float, 10.0, 1.0, 30.0, 0.5, attr="ripple_freq", description="Spatial frequency of ripple rings per radius unit")
    _hue_shift_prop:        ClassVar[Prop] = Prop("Hue Shift", float, 0.005, 0.0, 0.05, 0.002, attr="hue_shift", description="Hue rotation applied to each feedback sample (radians/step)")
    _chroma_offset_prop:    ClassVar[Prop] = Prop("Chromatic Aberration", float, 0.005, 0.0, 0.05, 0.002, attr="chroma_offset", description="Radial R/B channel offset as fraction of width")
    _sat_boost_prop:        ClassVar[Prop] = Prop("Saturation Boost", float, 1.12, 1.0, 2.0, 0.05, attr="sat_boost", description="Saturation multiplier on feedback sample (1.0 = flat)")
    _smear_strength_prop:   ClassVar[Prop] = Prop("Smear Strength", float, 0.0, 0.0, 0.10, 0.005, attr="smear_strength", description="UV offset per step along the smear field direction")
    _fisheye_strength_prop: ClassVar[Prop] = Prop("Fisheye", float, 0.0, -2.0, 2.0, 0.05, attr="fisheye_strength", description=">0 barrel (wide), <0 pincushion (telephoto), 0 = none")


# ═══════════════════════════════════════════════
#  Warp helper functions
# ═══════════════════════════════════════════════

@wp.func
def boost_saturation(r: float, g: float, b: float, factor: float) -> wp.vec3:
    """
    Push saturation up (factor > 1) so trails stay vivid through decay.
    Uses a standard luma-weighted approach; clamp happens at write time.
    """
    lum = r * 0.299 + g * 0.587 + b * 0.114
    return wp.vec3(
        lum + (r - lum) * factor,
        lum + (g - lum) * factor,
        lum + (b - lum) * factor,
    )


@wp.func
def rotate_hue(r: float, g: float, b: float, angle: float) -> wp.vec3:
    """Rotate the hue of an RGB triplet by ``angle`` radians."""
    cos_a = wp.cos(angle)
    sin_a = wp.sin(angle)
    k  = (1.0 - cos_a) / 3.0
    sq = sin_a * 0.5773502691896258   # sin / sqrt(3)

    nr = r * (cos_a + k) + g * (k - sq) + b * (k + sq)
    ng = r * (k + sq)   + g * (cos_a + k) + b * (k - sq)
    nb = r * (k - sq)   + g * (k + sq)   + b * (cos_a + k)

    return wp.vec3(nr, ng, nb)


@wp.func
def bilinear_sample(
    buf: wp.array(dtype=wp.float32),
    u: float, v: float,
    w: int, h: int,
) -> wp.vec4:
    """
    Bilinear sample of a flat RGBA float32 buffer at normalised UV (0-1).
    Clamps to edge.
    """
    px = wp.clamp(u * float(w - 1), 0.0, float(w - 1))
    py = wp.clamp(v * float(h - 1), 0.0, float(h - 1))

    x0 = int(wp.floor(px));  y0 = int(wp.floor(py))
    x1 = wp.min(x0 + 1, w - 1)
    y1 = wp.min(y0 + 1, h - 1)

    tx = px - float(x0);  ty = py - float(y0)

    b00 = (y0 * w + x0) * 4;  b01 = (y0 * w + x1) * 4
    b10 = (y1 * w + x0) * 4;  b11 = (y1 * w + x1) * 4

    r = (buf[b00]     * (1.0 - tx) + buf[b01]     * tx) * (1.0 - ty) \
      + (buf[b10]     * (1.0 - tx) + buf[b11]     * tx) * ty
    g = (buf[b00 + 1] * (1.0 - tx) + buf[b01 + 1] * tx) * (1.0 - ty) \
      + (buf[b10 + 1] * (1.0 - tx) + buf[b11 + 1] * tx) * ty
    b = (buf[b00 + 2] * (1.0 - tx) + buf[b01 + 2] * tx) * (1.0 - ty) \
      + (buf[b10 + 2] * (1.0 - tx) + buf[b11 + 2] * tx) * ty
    a = (buf[b00 + 3] * (1.0 - tx) + buf[b01 + 3] * tx) * (1.0 - ty) \
      + (buf[b10 + 3] * (1.0 - tx) + buf[b11 + 3] * tx) * ty

    return wp.vec4(r, g, b, a)


@wp.func
def sample_smear_field(
    field: wp.array(dtype=wp.vec2),
    u: float, v: float,
    gw: int, gh: int,
) -> wp.vec2:
    """
    Bilinearly sample a (gh × gw) vec2 grid at normalised UV (0–1).
    Returns the interpolated 2D smear vector at that screen position.
    """
    px = wp.clamp(u * float(gw - 1), 0.0, float(gw - 1))
    py = wp.clamp(v * float(gh - 1), 0.0, float(gh - 1))

    x0 = int(wp.floor(px));  y0 = int(wp.floor(py))
    x1 = wp.min(x0 + 1, gw - 1)
    y1 = wp.min(y0 + 1, gh - 1)

    tx = px - float(x0);  ty = py - float(y0)

    v00 = field[y0 * gw + x0]
    v01 = field[y0 * gw + x1]
    v10 = field[y1 * gw + x0]
    v11 = field[y1 * gw + x1]

    vx = (v00[0] * (1.0 - tx) + v01[0] * tx) * (1.0 - ty) \
       + (v10[0] * (1.0 - tx) + v11[0] * tx) * ty
    vy = (v00[1] * (1.0 - tx) + v01[1] * tx) * (1.0 - ty) \
       + (v10[1] * (1.0 - tx) + v11[1] * tx) * ty

    return wp.vec2(vx, vy)


# ═══════════════════════════════════════════════
#  Feedback kernel
# ═══════════════════════════════════════════════

@wp.kernel
def _feedback_kernel(
    prev:  wp.array(dtype=wp.float32),
    curr:  wp.array(dtype=wp.float32),
    w: int, h: int,
    zoom: float,
    rotation: float,
    decay: float,
    ripple_strength: float, ripple_freq: float,
    time_val: float,
    hue_shift: float,
    chroma_offset: float,
    sat_boost: float,
    # smear
    smear_field: wp.array(dtype=wp.vec2),
    smear_gw: int, smear_gh: int,
    smear_strength: float,
    # fisheye
    fisheye_strength: float,
):
    tid = wp.tid()
    if tid >= w * h:
        return

    px = tid % w
    py = tid // w

    u = float(px) / float(w - 1)
    v = float(py) / float(h - 1)

    # Centred coordinates
    cx = u - 0.5
    cy = v - 0.5

    # ── Zoom toward centre ────────────────────────────────────────────────
    cx2  = cx * zoom
    cy2  = cy * zoom

    # ── Rotation ──────────────────────────────────────────────────────────
    cos_a = wp.cos(rotation);  sin_a = wp.sin(rotation)
    cx3   = cx2 * cos_a - cy2 * sin_a
    cy3   = cx2 * sin_a + cy2 * cos_a

    # ── Radial ripple ─────────────────────────────────────────────────────
    dist  = wp.sqrt(cx3 * cx3 + cy3 * cy3) + 1e-6
    phase = dist * ripple_freq - time_val * 0.2
    ripple = ripple_strength * wp.sin(phase)
    nx = cx3 + (cx3 / dist) * ripple / float(w)
    ny = cy3 + (cy3 / dist) * ripple / float(h)

    # ── Fisheye / barrel-pincushion lens distortion ───────────────────────
    # Applied after zoom/rotate so it warps the sampling position, giving
    # a lens-like bending of the accumulated trails.
    #   fisheye_strength > 0  →  barrel  (wide-angle push-out)
    #   fisheye_strength < 0  →  pincushion (pull-in)
    if fisheye_strength != 0.0:
        r2 = nx * nx + ny * ny
        scale = 1.0 + fisheye_strength * r2
        nx = nx * scale
        ny = ny * scale

    # ── Screen-space smear ────────────────────────────────────────────────
    # Subtract sv so the pattern direction matches visual flow direction:
    # "outward" vectors → sample from closer to centre → content moves outward.
    if smear_strength > 0.0:
        sv = sample_smear_field(smear_field, nx + 0.5, ny + 0.5, smear_gw, smear_gh)
        nx = nx - sv[0] * smear_strength * smear_strength
        ny = ny - sv[1] * smear_strength * smear_strength

    # ── Chromatic aberration: sample R further from centre, B closer ──────
    cdx = (cx3 / (dist + 1e-6)) * chroma_offset
    cdy = (cy3 / (dist + 1e-6)) * chroma_offset * (float(w) / float(h))

    s_r = bilinear_sample(prev, (nx + cdx) + 0.5, (ny + cdy) + 0.5, w, h)
    s_g = bilinear_sample(prev,  nx        + 0.5,  ny        + 0.5, w, h)
    s_b = bilinear_sample(prev, (nx - cdx) + 0.5, (ny - cdy) + 0.5, w, h)

    raw_r = s_r[0];  raw_g = s_g[1];  raw_b = s_b[2]
    alpha_val = s_g[3]

    sat = boost_saturation(raw_r, raw_g, raw_b, sat_boost)
    rgb = rotate_hue(sat[0], sat[1], sat[2], hue_shift)

    base = tid * 4
    curr[base]     = wp.clamp(rgb[0] * decay, 0.0, 1.0)
    curr[base + 1] = wp.clamp(rgb[1] * decay, 0.0, 1.0)
    curr[base + 2] = wp.clamp(rgb[2] * decay, 0.0, 1.0)
    curr[base + 3] = alpha_val * decay


# ═══════════════════════════════════════════════
#  Smear pattern generators  (CPU, numpy)
# ═══════════════════════════════════════════════

def _make_smear_pattern(name: str, gw: int, gh: int) -> "np.ndarray":
    xs = np.linspace(-1.0, 1.0, gw, dtype=np.float32)
    ys = np.linspace(-1.0, 1.0, gh, dtype=np.float32)
    XX, YY = np.meshgrid(xs, ys)

    name = name.lower()

    if name == "swirl":
        vx, vy = -YY, XX
    elif name == "swirl_cw":
        vx, vy = YY, -XX
    elif name == "spiral":
        # Outward swirl: tangential (swirl) + radial (outward) components.
        # Creates expanding corkscrew trails — combine with zoom for tunnel effect.
        r  = np.sqrt(XX ** 2 + YY ** 2) + 1e-8
        vx = XX / r - YY
        vy = YY / r + XX
    elif name == "rightward":
        vx, vy = np.ones_like(XX), np.zeros_like(XX)
    elif name == "leftward":
        vx, vy = -np.ones_like(XX), np.zeros_like(XX)
    elif name == "upward":
        vx, vy = np.zeros_like(XX), -np.ones_like(XX)
    elif name == "downward":
        vx, vy = np.zeros_like(XX), np.ones_like(XX)
    elif name == "inward":
        vx, vy = -XX, -YY
    elif name == "outward":
        vx, vy = XX, YY
    elif name == "diagonal":
        vx = np.ones_like(XX)
        vy = -np.ones_like(XX)
    elif name == "cross":
        vx = np.where(YY < 0, 1.0, 0.0).astype(np.float32)
        vy = np.where(YY >= 0, -1.0, 0.0).astype(np.float32)
    elif name == "turbulence":
        rng = np.random.default_rng(42)
        angles = rng.uniform(0, 2 * np.pi, (gh, gw)).astype(np.float32)
        vx = np.cos(angles)
        vy = np.sin(angles)
    else:
        raise ValueError(
            f"Unknown smear pattern '{name}'. "
            "Valid: swirl, swirl_cw, spiral, rightward, leftward, upward, downward, "
            "inward, outward, diagonal, cross, turbulence"
        )

    mag = np.sqrt(vx ** 2 + vy ** 2) + 1e-8
    return np.stack([(vx / mag).astype(np.float32),
                     (vy / mag).astype(np.float32)], axis=-1)  # (gh, gw, 2)


SMEAR_PATTERNS: list[str] = [
    "swirl", "swirl_cw", "spiral",
    "rightward", "leftward", "upward", "downward",
    "inward", "outward", "diagonal", "cross", "turbulence",
]


# ═══════════════════════════════════════════════
#  Public class
# ═══════════════════════════════════════════════

class FeedbackLoop:
    """
    Manages two ping-pong RGBA float32 GPU buffers and a feedback kernel.

    Buffer layout: flat 1-D array of length (height * width * 4).
    Pixel (x, y) -> index (y * width + x) * 4, channels R G B A.

    Smear field
    -----------
    A (smear_grid_h x smear_grid_w) grid of vec2 unit vectors is
    bilinearly interpolated per pixel to offset each UV sample from prev,
    creating directional smear/streak trails independent of the centre.

        loop.set_smear_pattern("spiral")
        params.smear_strength = 0.015   # UV units (0.01 ~ 1% of width)

    Fisheye
    -------
    Set ``params.fisheye_strength`` to a non-zero value to apply barrel
    (positive) or pincushion (negative) lens distortion each step.
    """

    def __init__(
        self,
        width:        int = 1280,
        height:       int = 720,
        device:       str = "cuda",
        params:       "FeedbackParams | None" = None,
        smear_grid_w: int = 16,
        smear_grid_h: int = 16,
    ):
        self.width  = width
        self.height = height
        self.device = device
        self.params = params or FeedbackParams()
        self.smear_grid_w = smear_grid_w
        self.smear_grid_h = smear_grid_h

        n = height * width * 4
        self._prev = wp.zeros(n, dtype=wp.float32, device=device)
        self._curr = wp.zeros(n, dtype=wp.float32, device=device)

        # Zero field = no smear until set_smear_pattern() is called
        self._smear_field = wp.zeros(
            smear_grid_w * smear_grid_h, dtype=wp.vec2, device=device
        )

    @property
    def prev(self) -> "wp.array":
        return self._prev

    @property
    def curr(self) -> "wp.array":
        return self._curr

    @property
    def smear_field(self) -> "wp.array":
        """GPU vec2 array (gh x gw). Use helpers or write directly."""
        return self._smear_field

    def write_rgba(self, arr: "np.ndarray", into: str = "curr") -> None:
        """Copy a (H, W, 4) or (H*W*4,) float32 array into a buffer."""
        flat = np.ascontiguousarray(arr, dtype=np.float32).flatten()
        dst  = self._curr if into == "curr" else self._prev
        wp.copy(dst, wp.array(flat, dtype=wp.float32, device=self.device))

    def set_smear_vectors(self, vectors: "np.ndarray") -> None:
        """
        Upload a (smear_grid_h, smear_grid_w, 2) or (N, 2) float32 array.
        Normalise to unit length for predictable results.
        """
        flat = np.ascontiguousarray(vectors, dtype=np.float32).reshape(-1, 2)
        expected = self.smear_grid_w * self.smear_grid_h
        if flat.shape[0] != expected:
            raise ValueError(
                f"Expected {expected} vectors ({self.smear_grid_h}x{self.smear_grid_w}), "
                f"got {flat.shape[0]}"
            )
        wp.copy(self._smear_field, wp.array(flat, dtype=wp.vec2, device=self.device))

    def set_smear_pattern(self, name: str) -> None:
        """Set the smear field to a named preset (see SMEAR_PATTERNS list)."""
        self.set_smear_vectors(_make_smear_pattern(name, self.smear_grid_w, self.smear_grid_h))

    def clear_smear(self) -> None:
        """Reset the smear field to zero (no smear)."""
        self._smear_field = wp.zeros(
            self.smear_grid_w * self.smear_grid_h, dtype=wp.vec2, device=self.device
        )

    def step(
        self,
        time_val: float = 0.0,
        params:   "FeedbackParams | None" = None,
    ) -> None:
        """Run the feedback kernel: reads prev, writes curr."""
        p = params or self.params
        wp.launch(
            _feedback_kernel,
            dim=self.width * self.height,
            inputs=[
                self._prev, self._curr,
                self.width, self.height,
                float(p.zoom),
                float(p.rotation),
                float(p.decay),
                float(p.ripple_strength), float(p.ripple_freq),
                float(time_val),
                float(p.hue_shift),
                float(p.chroma_offset),
                float(p.sat_boost),
                self._smear_field,
                self.smear_grid_w, self.smear_grid_h,
                float(p.smear_strength),
                float(p.fisheye_strength),
            ],
            device=self.device,
        )

    def advance(self) -> None:
        """Swap curr <-> prev. Call at the end of each frame."""
        self._prev, self._curr = self._curr, self._prev

    def to_bgr(self) -> "np.ndarray":
        """Return current buffer as a contiguous uint8 (H, W, 3) BGR array."""
        arr = self._curr.numpy().reshape(self.height, self.width, 4)
        arr = np.clip(arr, 0.0, 1.0)
        rgb = (arr[:, :, :3] * 255).astype(np.uint8)
        return np.ascontiguousarray(rgb[:, :, ::-1])

    def to_rgba_f32(self) -> "np.ndarray":
        """Return current buffer as a float32 (H, W, 4) RGBA array."""
        return self._curr.numpy().reshape(self.height, self.width, 4).copy()
