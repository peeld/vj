"""
post/feedback.py — FeedbackPostEffect

GPU feedback loop: zoom · rotation · ripple · fisheye · trails.
Wraps post.warp_feedback.FeedbackLoop.

Key bindings (forwarded by gui_merged when this effect is active)
-----------------------------------------------------------------
  Z / X  — scene_alpha ↓ / ↑      (trail length)
  D / F  — decay ↓ / ↑
  Q / W  — rotation speed ↓ / ↑
  A / S  — zoom ↓ / ↑
  H / J  — hue shift ↓ / ↑
  C / V  — chromatic aberration ↓ / ↑
  B / N  — saturation boost ↓ / ↑
  K / L  — smear strength ↓ / ↑
  I / U  — fisheye ↓ / ↑          (pincushion → barrel)
  M      — cycle smear pattern
  G      — cycle blend mode
  T      — cycle named preset
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import moderngl
import warp as wp

from post.base import PostEffect, DEVICE, _QUAD_VERT, _QUAD_FRAG, _QUAD_VERTS
from post.warp_feedback import FeedbackLoop, FeedbackParams, SMEAR_PATTERNS


# ══════════════════════════════════════════════════════════════════════════════
#  Blend modes
# ══════════════════════════════════════════════════════════════════════════════

BLEND_MODES: list[str] = [
    "lerp", "additive", "screen", "lighten",
    "color_dodge", "difference", "overlay",
]

_BLEND_LERP       = 0
_BLEND_ADDITIVE   = 1
_BLEND_SCREEN     = 2
_BLEND_LIGHTEN    = 3
_BLEND_DODGE      = 4
_BLEND_DIFFERENCE = 5
_BLEND_OVERLAY    = 6


# ══════════════════════════════════════════════════════════════════════════════
#  GPU kernels
# ══════════════════════════════════════════════════════════════════════════════

@wp.func
def _blend_channel(c: float, s: float, alpha: float, mode: int) -> float:
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
    src: wp.array(dtype=wp.uint8),
    dst: wp.array(dtype=wp.float32),
    w:   int,
    h:   int,
):
    tid     = wp.tid()
    row     = tid // w
    col     = tid % w
    src_row = h - 1 - row
    src_idx = (src_row * w + col) * 4
    dst_idx = tid * 4
    dst[dst_idx]     = float(src[src_idx])     / 255.0
    dst[dst_idx + 1] = float(src[src_idx + 1]) / 255.0
    dst[dst_idx + 2] = float(src[src_idx + 2]) / 255.0
    dst[dst_idx + 3] = float(src[src_idx + 3]) / 255.0


@wp.kernel
def _pack_rgb_u8(
    src: wp.array(dtype=wp.float32),
    dst: wp.array(dtype=wp.uint8),
):
    tid      = wp.tid()
    src_base = tid * 4
    dst_base = tid * 3
    dst[dst_base]     = wp.uint8(int(wp.clamp(src[src_base],     0.0, 1.0) * 255.0))
    dst[dst_base + 1] = wp.uint8(int(wp.clamp(src[src_base + 1], 0.0, 1.0) * 255.0))
    dst[dst_base + 2] = wp.uint8(int(wp.clamp(src[src_base + 2], 0.0, 1.0) * 255.0))


@wp.kernel
def _inject_scene(
    curr:  wp.array(dtype=wp.float32),
    scene: wp.array(dtype=wp.float32),
    alpha: float,
    mode:  int,
):
    tid  = wp.tid()
    base = tid * 4
    curr[base]     = _blend_channel(curr[base],     scene[base],     alpha, mode)
    curr[base + 1] = _blend_channel(curr[base + 1], scene[base + 1], alpha, mode)
    curr[base + 2] = _blend_channel(curr[base + 2], scene[base + 2], alpha, mode)
    curr[base + 3] = 1.0


# ══════════════════════════════════════════════════════════════════════════════
#  Named presets
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class EffectPreset:
    """A complete snapshot of FeedbackPostEffect settings."""
    name:          str
    params:        FeedbackParams
    scene_alpha:   float = 0.13
    smear_pattern: str   = "swirl"
    blend_mode:    str   = "lerp"


PRESETS: list[EffectPreset] = [
    # gentle — slow atmospheric drift; the MergedGUI default feel.
    EffectPreset(
        name="gentle",
        params=FeedbackParams(
            base_zoom=1.002, zoom_sensitivity=0.0,
            base_rot=0.0008, rot_sensitivity=0.0,
            decay=0.993, ripple_strength=0.0, ripple_freq=10.0,
            hue_shift=0.005, chroma_offset=0.005,
            sat_boost=1.12, smear_strength=0.0, fisheye_strength=0.0,
        ),
        scene_alpha=0.18, smear_pattern="outward", blend_mode="lerp",
    ),
    # tunnel — zoom-in vortex with corkscrew smear and barrel lens.
    EffectPreset(
        name="tunnel",
        params=FeedbackParams(
            base_zoom=1.008, zoom_sensitivity=0.0,
            base_rot=0.001,  rot_sensitivity=0.0,
            decay=0.97, ripple_strength=0.0, ripple_freq=10.0,
            hue_shift=0.003, chroma_offset=0.008,
            sat_boost=1.2, smear_strength=0.015, fisheye_strength=0.6,
        ),
        scene_alpha=0.12, smear_pattern="spiral", blend_mode="screen",
    ),
    # slow_burn — very long additive trails; energy pools and saturates.
    EffectPreset(
        name="slow_burn",
        params=FeedbackParams(
            base_zoom=1.002, zoom_sensitivity=0.0,
            base_rot=0.0002, rot_sensitivity=0.0,
            decay=0.997, ripple_strength=0.0, ripple_freq=10.0,
            hue_shift=0.001, chroma_offset=0.0,
            sat_boost=1.5, smear_strength=0.020, fisheye_strength=0.0,
        ),
        scene_alpha=0.05, smear_pattern="outward", blend_mode="additive",
    ),
    # deep_sea — turbulent organic flow, chroma halos, pincushion pull-in.
    EffectPreset(
        name="deep_sea",
        params=FeedbackParams(
            base_zoom=1.001, zoom_sensitivity=0.0,
            base_rot=0.0,    rot_sensitivity=0.0,
            decay=0.97, ripple_strength=0.0, ripple_freq=10.0,
            hue_shift=0.008, chroma_offset=0.020,
            sat_boost=1.3, smear_strength=0.018, fisheye_strength=-0.4,
        ),
        scene_alpha=0.10, smear_pattern="turbulence", blend_mode="screen",
    ),
    # acid — difference blend creates interference fringes; fast hue cycles them.
    EffectPreset(
        name="acid",
        params=FeedbackParams(
            base_zoom=1.005, zoom_sensitivity=0.0,
            base_rot=0.002,  rot_sensitivity=0.0,
            decay=0.985, ripple_strength=0.0, ripple_freq=10.0,
            hue_shift=0.015, chroma_offset=0.005,
            sat_boost=1.6, smear_strength=0.010, fisheye_strength=0.0,
        ),
        scene_alpha=0.15, smear_pattern="swirl", blend_mode="difference",
    ),
    # aurora — horizontal cross-flow streaks, screen blend, slow colour drift.
    EffectPreset(
        name="aurora",
        params=FeedbackParams(
            base_zoom=1.001, zoom_sensitivity=0.0,
            base_rot=0.0003, rot_sensitivity=0.0,
            decay=0.992, ripple_strength=0.0, ripple_freq=10.0,
            hue_shift=0.006, chroma_offset=0.010,
            sat_boost=1.7, smear_strength=0.025, fisheye_strength=0.0,
        ),
        scene_alpha=0.08, smear_pattern="cross", blend_mode="screen",
    ),
]

_PRESET_NAMES = [p.name for p in PRESETS]


# ══════════════════════════════════════════════════════════════════════════════
#  FeedbackPostEffect
# ══════════════════════════════════════════════════════════════════════════════

class FeedbackPostEffect(PostEffect):
    """
    GPU feedback loop with zoom, rotation, ripple, fisheye, hue-shift,
    chromatic aberration, and saturation boost.

    Parameters
    ----------
    params        : FeedbackParams controlling the effect (mutate at runtime).
    scene_alpha   : How strongly the fresh 3-D frame bleeds in each tick.
    smear_pattern : Starting smear pattern name (see SMEAR_PATTERNS).
    blend_mode    : Starting blend mode name (see BLEND_MODES).
    preset_idx    : Index into PRESETS to load at startup (None = use the
                    individual defaults above instead).
    """

    name = "feedback"

    def __init__(
        self,
        params:        FeedbackParams | None = None,
        scene_alpha:   float                 = 0.13,
        smear_pattern: str                   = "swirl",
        blend_mode:    str                   = "lerp",
        preset_idx:    int | None            = None,
    ):
        self.params      = params or FeedbackParams()
        self.scene_alpha = float(scene_alpha)

        self._smear_pattern_name = smear_pattern
        self._smear_pattern_idx  = (SMEAR_PATTERNS.index(smear_pattern)
                                    if smear_pattern in SMEAR_PATTERNS else 0)
        self._blend_mode_idx = (BLEND_MODES.index(blend_mode)
                                if blend_mode in BLEND_MODES else 0)
        self._preset_idx = preset_idx if preset_idx is not None else -1

        self._ctx:         moderngl.Context     | None = None
        self._loop:        FeedbackLoop         | None = None
        self._display_tex: moderngl.Texture     | None = None
        self._fbo:         moderngl.Framebuffer | None = None
        self._quad_prog:   moderngl.Program     | None = None
        self._quad_vao:    moderngl.VertexArray | None = None
        self._scene_gpu:   wp.array             | None = None
        self._raw_gpu:     wp.array             | None = None
        self._result_u8:   wp.array             | None = None

        # Apply startup preset if requested (params only — no GL yet)
        if preset_idx is not None and 0 <= preset_idx < len(PRESETS):
            self._apply_preset_params(PRESETS[preset_idx])

    # ── PostEffect abstract interface ─────────────────────────────────────────

    def setup(self, ctx: moderngl.Context, w: int, h: int) -> None:
        self._ctx  = ctx
        self._loop = FeedbackLoop(w, h, device=DEVICE, params=self.params)
        self._display_tex = ctx.texture((w, h), 3)
        self._loop.set_smear_pattern(self._smear_pattern_name)
        self._alloc_scratch(w, h)
        self._build_fbo(w, h)
        self._build_quad(ctx)

    @property
    def fbo(self) -> moderngl.Framebuffer:
        return self._fbo

    def bind_scene_fbo(self) -> None:
        self._fbo.use()
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.clear(0.04, 0.04, 0.06, 1.0)

    def blit_to_screen(self, screen: moderngl.Framebuffer) -> None:
        screen.use()
        self._ctx.disable(moderngl.DEPTH_TEST)
        self._ctx.clear(0.0, 0.0, 0.0, 1.0)
        self._display_tex.use(0)
        self._quad_prog["tex"].value = 0
        self._quad_vao.render(moderngl.TRIANGLE_STRIP)

    def process(
        self,
        scene_fbo: moderngl.Framebuffer,
        t:  float,
        dt: float,
    ) -> moderngl.Texture:
        w = scene_fbo.width
        h = scene_fbo.height

        raw     = scene_fbo.color_attachments[0].read()
        raw_cpu = wp.array(np.frombuffer(raw, dtype=np.uint8),
                           dtype=wp.uint8, copy=False, device="cpu")
        wp.copy(self._raw_gpu, raw_cpu)
        wp.launch(_decode_rgba_u8_flip,
                  dim=w * h,
                  inputs=[self._raw_gpu, self._scene_gpu, w, h],
                  device=DEVICE)

        self._loop.step(time_val=t, params=self.params)

        wp.launch(
            _inject_scene,
            dim=w * h,
            inputs=[self._loop.curr, self._scene_gpu,
                    float(self.scene_alpha), self._blend_mode_idx],
            device=DEVICE,
        )

        wp.launch(_pack_rgb_u8,
                  dim=w * h,
                  inputs=[self._loop.curr, self._result_u8],
                  device=DEVICE)
        self._display_tex.write(self._result_u8.numpy().tobytes())

        self._loop.advance()
        return self._display_tex

    def resize(self, w: int, h: int) -> None:
        self._loop = FeedbackLoop(w, h, device=DEVICE, params=self.params)
        self._loop.set_smear_pattern(self._smear_pattern_name)
        if self._ctx is not None:
            self._display_tex = self._ctx.texture((w, h), 3)
            self._build_fbo(w, h)
        self._alloc_scratch(w, h)

    def on_key(self, key, action, keys) -> None:
        if action != keys.ACTION_PRESS:
            return
        p = self.params

        if   key == keys.Z:
            self.scene_alpha = round(max(0.02, self.scene_alpha - 0.05), 3)
            print(f"[post] scene_alpha: {self.scene_alpha:.3f}")
        elif key == keys.X:
            self.scene_alpha = round(min(1.00, self.scene_alpha + 0.05), 3)
            print(f"[post] scene_alpha: {self.scene_alpha:.3f}")
        elif key == keys.D:
            p.decay = round(max(0.80,  p.decay - 0.010), 4)
            print(f"[post] decay: {p.decay:.4f}")
        elif key == keys.F:
            p.decay = round(min(0.999, p.decay + 0.010), 4)
            print(f"[post] decay: {p.decay:.4f}")
        elif key == keys.Q:
            p.base_rot = round(max(0.0,  p.base_rot - 0.0001), 5)
            print(f"[post] base_rot: {p.base_rot:.5f}")
        elif key == keys.W:
            p.base_rot = round(min(0.01, p.base_rot + 0.0001), 5)
            print(f"[post] base_rot: {p.base_rot:.5f}")
        elif key == keys.A:
            p.base_zoom = round(max(1.000, p.base_zoom - 0.001), 4)
            print(f"[post] base_zoom: {p.base_zoom:.4f}")
        elif key == keys.S:
            p.base_zoom = round(min(1.050, p.base_zoom + 0.001), 4)
            print(f"[post] base_zoom: {p.base_zoom:.4f}")
        elif key == keys.H:
            p.hue_shift = round(max(0.0,  p.hue_shift - 0.002), 4)
            print(f"[post] hue_shift: {p.hue_shift:.4f}")
        elif key == keys.J:
            p.hue_shift = round(min(0.05, p.hue_shift + 0.002), 4)
            print(f"[post] hue_shift: {p.hue_shift:.4f}")
        elif key == keys.C:
            p.chroma_offset = round(max(0.0,  p.chroma_offset - 0.002), 4)
            print(f"[post] chroma_offset: {p.chroma_offset:.4f}")
        elif key == keys.V:
            p.chroma_offset = round(min(0.05, p.chroma_offset + 0.002), 4)
            print(f"[post] chroma_offset: {p.chroma_offset:.4f}")
        elif key == keys.B:
            p.sat_boost = round(max(1.0, p.sat_boost - 0.05), 3)
            print(f"[post] sat_boost: {p.sat_boost:.3f}")
        elif key == keys.N:
            p.sat_boost = round(min(2.0, p.sat_boost + 0.05), 3)
            print(f"[post] sat_boost: {p.sat_boost:.3f}")
        elif key == keys.K:
            p.smear_strength = round(max(0.0,  p.smear_strength - 0.005), 4)
            print(f"[post] smear_strength: {p.smear_strength:.4f}")
        elif key == keys.L:
            p.smear_strength = round(min(0.10, p.smear_strength + 0.005), 4)
            print(f"[post] smear_strength: {p.smear_strength:.4f}")
        elif key == keys.I:
            p.fisheye_strength = round(max(-2.0, p.fisheye_strength - 0.05), 3)
            print(f"[post] fisheye_strength: {p.fisheye_strength:.3f}")
        elif key == keys.U:
            p.fisheye_strength = round(min(2.0,  p.fisheye_strength + 0.05), 3)
            print(f"[post] fisheye_strength: {p.fisheye_strength:.3f}")
        elif key == keys.M:
            self._smear_pattern_idx = (self._smear_pattern_idx + 1) % len(SMEAR_PATTERNS)
            self._smear_pattern_name = SMEAR_PATTERNS[self._smear_pattern_idx]
            if self._loop is not None:
                self._loop.set_smear_pattern(self._smear_pattern_name)
            print(f"[post] smear_pattern: {self._smear_pattern_name}")
        elif key == keys.G:
            self._blend_mode_idx = (self._blend_mode_idx + 1) % len(BLEND_MODES)
            print(f"[post] blend_mode: {BLEND_MODES[self._blend_mode_idx]}")
        elif key == keys.T:
            self._preset_idx = (self._preset_idx + 1) % len(PRESETS)
            self.load_preset(PRESETS[self._preset_idx])

    # ── Preset support ────────────────────────────────────────────────────────

    def load_preset(self, preset: EffectPreset) -> None:
        """
        Apply a named preset, updating params in-place so any external
        reference to ``self.params`` (e.g. param_dialog) sees the new values.
        """
        self._apply_preset_params(preset)
        if self._loop is not None:
            self._loop.set_smear_pattern(self._smear_pattern_name)
        print(f"[post] preset → {preset.name}  "
              f"(smear={preset.smear_pattern}  blend={preset.blend_mode})")

    def _apply_preset_params(self, preset: EffectPreset) -> None:
        """Copy preset values into the live params object (no GL side-effects)."""
        src = preset.params
        for f in src.__dataclass_fields__:
            setattr(self.params, f, getattr(src, f))
        self.scene_alpha         = preset.scene_alpha
        self._smear_pattern_name = preset.smear_pattern
        self._smear_pattern_idx  = (SMEAR_PATTERNS.index(preset.smear_pattern)
                                    if preset.smear_pattern in SMEAR_PATTERNS else 0)
        self._blend_mode_idx     = (BLEND_MODES.index(preset.blend_mode)
                                    if preset.blend_mode in BLEND_MODES else 0)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _alloc_scratch(self, w: int, h: int) -> None:
        n = w * h
        self._raw_gpu   = wp.zeros(n * 4, dtype=wp.uint8,   device=DEVICE)
        self._scene_gpu = wp.zeros(n * 4, dtype=wp.float32, device=DEVICE)
        self._result_u8 = wp.zeros(n * 3, dtype=wp.uint8,   device=DEVICE)

    def _build_fbo(self, w: int, h: int) -> None:
        self._fbo = self._ctx.framebuffer(
            color_attachments=[self._ctx.texture((w, h), 4)],
            depth_attachment=self._ctx.depth_texture((w, h)),
        )

    def _build_quad(self, ctx: moderngl.Context) -> None:
        vbo = ctx.buffer(_QUAD_VERTS.tobytes())
        self._quad_prog = ctx.program(vertex_shader=_QUAD_VERT, fragment_shader=_QUAD_FRAG)
        self._quad_vao  = ctx.vertex_array(self._quad_prog, [(vbo, "2f", "in_pos")])
