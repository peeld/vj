"""
post_effect.py
PostEffect ABC and built-in concrete units.

A PostEffect is given the off-screen FBO that contains the rendered 3-D scene
and returns an OpenGL texture containing the processed result.  Viewport3D
blits that texture to the screen via a fullscreen quad — the effect knows
nothing about the window or camera.

Built-in effects
----------------
  FeedbackPostEffect  — GPU feedback loop: zoom · rotation · ripple · trails
                        (wraps warp_feedback.FeedbackLoop)

Writing your own
----------------
Subclass PostEffect and implement the three abstract methods:

    class GlitchEffect(PostEffect):
        def setup(self, ctx, w, h):
            self._tex = ctx.texture((w, h), 3)
            ...

        def process(self, scene_fbo, t, dt):
            # read scene_fbo, apply effect, write to self._tex
            return self._tex

        def resize(self, w, h):
            self._tex = self._ctx.texture((w, h), 3)

Then assign it to your scene:
    class MyScene(Scene):
        post_effect = GlitchEffect()
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import moderngl
import warp as wp

from drawlib.warp_feedback import FeedbackLoop, FeedbackParams, SMEAR_PATTERNS


# ── Warp device ────────────────────────────────────────────────────────────────
wp.init()
DEVICE = "cuda" if wp.get_cuda_device_count() > 0 else "cpu"


# ── Scene-inject kernel ────────────────────────────────────────────────────────
# Blends a freshly rendered 3-D frame into the feedback buffer.
# Keeping this here (not in warp_feedback.py) so FeedbackLoop stays generic.

BLEND_MODES = ["lerp", "additive", "screen", "lighten", "color_dodge", "difference", "overlay"]

# mode indices (must match order above)
_BLEND_LERP       = 0
_BLEND_ADDITIVE   = 1
_BLEND_SCREEN     = 2
_BLEND_LIGHTEN    = 3
_BLEND_DODGE      = 4
_BLEND_DIFFERENCE = 5
_BLEND_OVERLAY    = 6


@wp.func
def _blend_channel(c: float, s: float, alpha: float, mode: int) -> float:
    """
    Blend feedback channel c with scene channel s at weight alpha.
      0 lerp        standard crossfade
      1 additive    accumulate light; clamps at 1
      2 screen      soft additive; asymptotes to white
      3 lighten      max(c, s*alpha)  — trails only keep brighter pixels
      4 color_dodge  c / (1 - s*alpha); brightens where scene is bright
      5 difference   |c - s*alpha|; interference/inversion patterns
      6 overlay      multiply darks, screen lights
    """
    sa = s * alpha
    if mode == _BLEND_ADDITIVE:
        return wp.clamp(c + sa, 0.0, 1.0)
    elif mode == _BLEND_SCREEN:
        return 1.0 - (1.0 - c) * (1.0 - sa)
    elif mode == _BLEND_LIGHTEN:
        return wp.max(c, sa)
    elif mode == _BLEND_DODGE:
        return wp.clamp(c / (1.0 - sa + 1e-4), 0.0, 1.0)
    elif mode == _BLEND_DIFFERENCE:
        return wp.abs(c - sa)
    elif mode == _BLEND_OVERLAY:
        return wp.where(c < 0.5, 2.0 * c * sa, 1.0 - 2.0 * (1.0 - c) * (1.0 - sa))
    else:  # lerp
        return c * (1.0 - alpha) + sa


@wp.kernel
def _decode_rgba_u8_flip(
    src: wp.array(dtype=wp.uint8),    # raw RGBA uint8, row-0 = bottom (GL convention)
    dst: wp.array(dtype=wp.float32),  # flat RGBA float32, row-0 = top (image-space)
    w:   int,
    h:   int,
):
    """Flip Y and convert RGBA uint8 → float32 in one GPU pass, avoiding CPU decode."""
    tid     = wp.tid()          # destination pixel index (top-down)
    row     = tid // w
    col     = tid % w
    src_row = h - 1 - row       # flip Y: GL row 0 is bottom
    src_idx = (src_row * w + col) * 4
    dst_idx = tid * 4
    dst[dst_idx]     = float(src[src_idx])     / 255.0
    dst[dst_idx + 1] = float(src[src_idx + 1]) / 255.0
    dst[dst_idx + 2] = float(src[src_idx + 2]) / 255.0
    dst[dst_idx + 3] = float(src[src_idx + 3]) / 255.0


@wp.kernel
def _pack_rgb_u8(
    src: wp.array(dtype=wp.float32),  # flat RGBA float32 (feedback curr buffer)
    dst: wp.array(dtype=wp.uint8),    # flat RGB uint8 (alpha dropped)
):
    """Clip, scale, and pack float32 RGBA → uint8 RGB on GPU, replacing to_bgr() + flip."""
    tid      = wp.tid()
    src_base = tid * 4
    dst_base = tid * 3
    dst[dst_base]     = wp.uint8(int(wp.clamp(src[src_base],     0.0, 1.0) * 255.0))
    dst[dst_base + 1] = wp.uint8(int(wp.clamp(src[src_base + 1], 0.0, 1.0) * 255.0))
    dst[dst_base + 2] = wp.uint8(int(wp.clamp(src[src_base + 2], 0.0, 1.0) * 255.0))


@wp.kernel
def _inject_scene(
    curr:  wp.array(dtype=wp.float32),   # feedback curr buffer  (in/out)
    scene: wp.array(dtype=wp.float32),   # flat float32 RGBA scene pixels
    alpha: float,                        # blend weight for scene
    mode:  int,                          # BLEND_MODES index
):
    tid  = wp.tid()
    base = tid * 4
    curr[base]     = _blend_channel(curr[base],     scene[base],     alpha, mode)
    curr[base + 1] = _blend_channel(curr[base + 1], scene[base + 1], alpha, mode)
    curr[base + 2] = _blend_channel(curr[base + 2], scene[base + 2], alpha, mode)
    curr[base + 3] = 1.0


# ══════════════════════════════════════════════════════════════════════════════
#  Abstract base
# ══════════════════════════════════════════════════════════════════════════════

class PostEffect(ABC):
    """
    Interface for a post-processing unit.

    Viewport3D calls these in order:
      1. ``setup(ctx, w, h)``  — once at startup
      2. ``process(fbo, t, dt)`` — every frame (when post-effect is enabled)
      3. ``resize(w, h)``       — on window resize
      4. ``on_key(...)``        — forwarded key events (optional)
    """

    @abstractmethod
    def setup(self, ctx: moderngl.Context, w: int, h: int) -> None:
        """Allocate GPU resources sized to the viewport."""

    @abstractmethod
    def process(
        self,
        scene_fbo: moderngl.Framebuffer,
        t:  float,
        dt: float,
    ) -> moderngl.Texture:
        """
        Process the rendered scene and return a texture to blit to screen.
        ``scene_fbo`` contains the 3-D scene rendered at the current MVP.
        """

    def resize(self, w: int, h: int) -> None:
        """Recreate any size-dependent resources when the window is resized."""

    def on_key(self, key, action, keys) -> None:
        """Optional: handle keyboard events forwarded from the viewport."""


# ══════════════════════════════════════════════════════════════════════════════
#  FeedbackPostEffect
# ══════════════════════════════════════════════════════════════════════════════

class FeedbackPostEffect(PostEffect):
    """
    GPU feedback loop: zoom, rotation, ripple, hue-shift, chromatic
    aberration, saturation boost.  Wraps warp_feedback.FeedbackLoop.

    Parameters
    ----------
    params      : FeedbackParams controlling the effect (mutate at runtime).
    scene_alpha : How strongly the fresh 3-D frame bleeds in each tick.
                  0 = pure echo chamber / infinite trails.
                  1 = no trails at all (equivalent to no post-effect).
                  Good range: 0.05 – 0.30.

    Key bindings (forwarded automatically by Viewport3D)
    ----------------------------------------------------
      Z / X  — scene_alpha ↓ / ↑   (trail length)
      D / F  — decay ↓ / ↑
      Q / W  — rotation speed ↓ / ↑
      A / S  — zoom ↓ / ↑
      H / J  — hue shift ↓ / ↑
      C / V  — chromatic aberration ↓ / ↑
      B / N  — saturation boost ↓ / ↑
      K / L  — smear strength ↓ / ↑
      M      — cycle smear pattern
      G      — cycle scene blend mode (lerp → additive → screen → …)
    """

    def __init__(
        self,
        params:        FeedbackParams = None,
        scene_alpha:   float          = 0.13,
        smear_pattern: str            = "swirl",
        blend_mode:    str            = "lerp",
    ):
        self.params        = params or FeedbackParams()
        self.scene_alpha   = float(scene_alpha)
        self._smear_pattern_name = smear_pattern
        self._smear_pattern_idx  = SMEAR_PATTERNS.index(smear_pattern) \
                                   if smear_pattern in SMEAR_PATTERNS else 0
        self._blend_mode_idx = BLEND_MODES.index(blend_mode) \
                               if blend_mode in BLEND_MODES else 0

        self._ctx:         moderngl.Context  | None = None
        self._loop:        FeedbackLoop      | None = None
        self._display_tex: moderngl.Texture  | None = None
        self._scene_gpu:   wp.array          | None = None  # float32 RGBA, top-down
        self._raw_gpu:     wp.array          | None = None  # uint8  RGBA, bottom-up (GL)
        self._result_u8:   wp.array          | None = None  # uint8  RGB,  no alpha

    # ── PostEffect interface ──────────────────────────────────────────────────

    def setup(self, ctx: moderngl.Context, w: int, h: int) -> None:
        self._ctx  = ctx
        self._loop = FeedbackLoop(w, h, device=DEVICE, params=self.params)
        self._display_tex = ctx.texture((w, h), 3)
        self._loop.set_smear_pattern(self._smear_pattern_name)
        self._alloc_scratch(w, h)

    def _alloc_scratch(self, w: int, h: int) -> None:
        """Pre-allocate per-frame scratch buffers so process() does zero allocation."""
        n = w * h
        self._raw_gpu   = wp.zeros(n * 4, dtype=wp.uint8,   device=DEVICE)
        self._scene_gpu = wp.zeros(n * 4, dtype=wp.float32, device=DEVICE)
        self._result_u8 = wp.zeros(n * 3, dtype=wp.uint8,   device=DEVICE)

    def process(
        self,
        scene_fbo: moderngl.Framebuffer,
        t:  float,
        dt: float,
    ) -> moderngl.Texture:
        w = scene_fbo.width
        h = scene_fbo.height

        # ── FBO → Warp float32 array ──────────────────────────────────────
        # ModernGL returns pixels row-0 = bottom (GL convention).
        # Upload raw bytes as uint8 then flip+decode entirely on GPU.
        raw = scene_fbo.color_attachments[0].read()        # RGBA uint8 bytes
        # Zero-copy CPU wrapper → single H2D transfer into the pre-allocated buffer.
        raw_cpu = wp.array(np.frombuffer(raw, dtype=np.uint8),
                           dtype=wp.uint8, copy=False, device="cpu")
        wp.copy(self._raw_gpu, raw_cpu)
        wp.launch(_decode_rgba_u8_flip,
                  dim=w * h,
                  inputs=[self._raw_gpu, self._scene_gpu, w, h],
                  device=DEVICE)

        # ── Feedback: prev → curr ─────────────────────────────────────────
        self._loop.step(time_val=t, params=self.params)

        # ── Inject fresh scene into feedback curr buffer ──────────────────
        wp.launch(
            _inject_scene,
            dim=w * h,
            inputs=[self._loop.curr, self._scene_gpu,
                    float(self.scene_alpha), self._blend_mode_idx],
            device=DEVICE,
        )

        # ── Copy result to OpenGL texture ─────────────────────────────────
        # Pack float32 RGBA → uint8 RGB on GPU; single .numpy() readback, no flip.
        wp.launch(_pack_rgb_u8,
                  dim=w * h,
                  inputs=[self._loop.curr, self._result_u8],
                  device=DEVICE)
        self._display_tex.write(self._result_u8.numpy().tobytes())

        self._loop.advance()   # swap ping-pong buffers for next frame
        return self._display_tex

    def resize(self, w: int, h: int) -> None:
        self._loop = FeedbackLoop(w, h, device=DEVICE, params=self.params)
        self._loop.set_smear_pattern(self._smear_pattern_name)
        if self._ctx is not None:
            self._display_tex = self._ctx.texture((w, h), 3)
        self._alloc_scratch(w, h)

    def on_key(self, key, action, keys) -> None:
        if action != keys.ACTION_PRESS:
            return
        p = self.params

        # scene_alpha (trail length)
        if   key == keys.Z:
            self.scene_alpha = round(max(0.02, self.scene_alpha - 0.05), 3)
            print(f"[post] scene_alpha: {self.scene_alpha:.3f}")
        elif key == keys.X:
            self.scene_alpha = round(min(1.00, self.scene_alpha + 0.05), 3)
            print(f"[post] scene_alpha: {self.scene_alpha:.3f}")

        # decay
        elif key == keys.D:
            p.decay = round(max(0.80,  p.decay - 0.010), 4)
            print(f"[post] decay: {p.decay:.4f}")
        elif key == keys.F:
            p.decay = round(min(0.999, p.decay + 0.010), 4)
            print(f"[post] decay: {p.decay:.4f}")

        # rotation
        elif key == keys.Q:
            p.base_rot = round(max(0.0,  p.base_rot - 0.0001), 5)
            print(f"[post] base_rot: {p.base_rot:.5f}")
        elif key == keys.W:
            p.base_rot = round(min(0.01, p.base_rot + 0.0001), 5)
            print(f"[post] base_rot: {p.base_rot:.5f}")

        # zoom
        elif key == keys.A:
            p.base_zoom = round(max(1.000, p.base_zoom - 0.001), 4)
            print(f"[post] base_zoom: {p.base_zoom:.4f}")
        elif key == keys.S:
            p.base_zoom = round(min(1.050, p.base_zoom + 0.001), 4)
            print(f"[post] base_zoom: {p.base_zoom:.4f}")

        # hue shift
        elif key == keys.H:
            p.hue_shift = round(max(0.0,  p.hue_shift - 0.002), 4)
            print(f"[post] hue_shift: {p.hue_shift:.4f}")
        elif key == keys.J:
            p.hue_shift = round(min(0.05, p.hue_shift + 0.002), 4)
            print(f"[post] hue_shift: {p.hue_shift:.4f}")

        # chromatic aberration
        elif key == keys.C:
            p.chroma_offset = round(max(0.0,  p.chroma_offset - 0.002), 4)
            print(f"[post] chroma_offset: {p.chroma_offset:.4f}")
        elif key == keys.V:
            p.chroma_offset = round(min(0.05, p.chroma_offset + 0.002), 4)
            print(f"[post] chroma_offset: {p.chroma_offset:.4f}")

        # saturation boost
        elif key == keys.B:
            p.sat_boost = round(max(1.0, p.sat_boost - 0.05), 3)
            print(f"[post] sat_boost: {p.sat_boost:.3f}")
        elif key == keys.N:
            p.sat_boost = round(min(2.0, p.sat_boost + 0.05), 3)
            print(f"[post] sat_boost: {p.sat_boost:.3f}")

        # smear strength
        elif key == keys.K:
            p.smear_strength = round(max(0.0, p.smear_strength - 0.005), 4)
            print(f"[post] smear_strength: {p.smear_strength:.4f}")
        elif key == keys.L:
            p.smear_strength = round(min(0.10, p.smear_strength + 0.005), 4)
            print(f"[post] smear_strength: {p.smear_strength:.4f}")

        # cycle smear pattern
        elif key == keys.M:
            self._smear_pattern_idx = (self._smear_pattern_idx + 1) % len(SMEAR_PATTERNS)
            self._smear_pattern_name = SMEAR_PATTERNS[self._smear_pattern_idx]
            if self._loop is not None:
                self._loop.set_smear_pattern(self._smear_pattern_name)
            print(f"[post] smear_pattern: {self._smear_pattern_name}")

        # cycle blend mode
        elif key == keys.G:
            self._blend_mode_idx = (self._blend_mode_idx + 1) % len(BLEND_MODES)
            print(f"[post] blend_mode: {BLEND_MODES[self._blend_mode_idx]}")
