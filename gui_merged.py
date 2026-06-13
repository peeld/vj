"""
gui_merged.py - MergedGUI
Combines the cloud/ball/NN-graph scene (gui.py) with the circle-axis scene
(gui3.py) into a single window with per-element visibility toggles.

Element toggles:
  1  -- cloud data + ball data
  2  -- NN graph
  3  -- circle axis
  4  -- laser ribbons

Other controls:
  R          -- regenerate all elements
  P          -- toggle post pipeline on / off (raw scene render)
  O          -- toggle orbit camera on / off
  Tab        -- cycle effect type  (feedback -> pass_through -> ...)
  Mouse drag -- orbit camera
  Scroll     -- zoom
  ESC        -- quit

Post-effect tweaks (effect must be ON, feedback effect active):
  T    -- cycle preset  (gentle -> tunnel -> slow_burn -> deep_sea -> acid -> aurora)
  Z/X  -- scene blend down/up     D/F  -- decay down/up
  Q/W  -- rotation down/up        A/S  -- zoom down/up
  H/J  -- hue shift down/up       C/V  -- chromatic aberration down/up
  B/N  -- saturation down/up      K/L  -- smear strength down/up
  I/U  -- fisheye down/up         M    -- cycle smear pattern
  G    -- cycle blend mode
"""

import random
from dataclasses import dataclass

import moderngl
import moderngl_window as mglw
import warp as wp

from post import (
    FeedbackPostEffect, PassThroughEffect, GlitchEffect,
    FeedbackParams, BLEND_MODES, SMEAR_PATTERNS, PRESETS,
)
from drawlib.camera import OrbitCamera

from elements.cloud import CloudElement
from elements.nn_graph import NNGraph
from elements.circleaxis import CircleAxisDrawing
from elements.laser_ribbons import LaserRibbons

import audio_metrics

wp.init()
DEVICE = "cuda" if wp.get_cuda_device_count() > 0 else "cpu"
print(f"[warp] device: {DEVICE}")


# ---------------------------------------------------------------------------
# Shared state -- param_dialog reads/writes these directly
# ---------------------------------------------------------------------------

# FeedbackParams initialised with the "gentle" preset values.
# Stored at module level so param_dialog can hold a live reference.
_params = FeedbackParams(
    base_zoom=1.002, zoom_sensitivity=0.0,
    base_rot=0.0008, rot_sensitivity=0.0,
    decay=0.993,
    ripple_strength=0.0, ripple_freq=10.0,
    hue_shift=0.005, chroma_offset=0.005,
    sat_boost=1.12, smear_strength=0.0,
    fisheye_strength=0.0,
)


@dataclass
class SceneControls:
    show_cloud    : bool  = True
    show_nn       : bool  = True
    show_circles  : bool  = True
    show_lasers   : bool  = True
    scene_alpha   : float = 0.18
    blend_mode    : str   = "lerp"
    smear_pattern : str   = "outward"


_controls = SceneControls()


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

class MergedGUI(mglw.WindowConfig):
    title        = "Warp -- Merged Scene"
    window_size  = (1280, 720)
    gl_version   = (3, 3)
    resizable    = True
    aspect_ratio = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # -- Scene elements ---------------------------------------------------
        self._cloud   = CloudElement(self.ctx)
        self.nn_graph = NNGraph(self.ctx, device=DEVICE)
        self._circles = CircleAxisDrawing(self.ctx)
        self._lasers  = LaserRibbons(self.ctx)

        # -- GL state ---------------------------------------------------------
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # -- Camera -----------------------------------------------------------
        self.camera = OrbitCamera()
        self._drag  = False
        self.time   = 0.0

        # -- Post-effect list -------------------------------------------------
        # All effects are set up at startup; Tab cycles between them at runtime.
        # _params is shared with param_dialog so live edits are reflected.
        self.post_effect_on = True
        w, h = self.window_size

        self._effects = [
            FeedbackPostEffect(
                params        = _params,
                scene_alpha   = _controls.scene_alpha,
                smear_pattern = _controls.smear_pattern,
                blend_mode    = _controls.blend_mode,
                preset_idx    = 0,          # start on "gentle"
            ),
            PassThroughEffect(),
            GlitchEffect(),
        ]
        self._effect_idx = 0

        for eff in self._effects:
            eff.setup(self.ctx, w, h)

        # -- Audio ------------------------------------------------------------
        self._audio = audio_metrics.AudioAnalyzer(
           # device   = "CABLE Output (VB-Audio Virtual Cable), Windows DirectSound",
            on_frame = self._on_audio_frame,
        )
        self._audio.start()

        effect_names = "  |  ".join(e.name for e in self._effects)
        print(f"[merged] effects: {effect_names}")
        print("[merged] 1=cloud  2=nn  3=circles  4=lasers  R=regen  P=post  O=orbit  Tab=effect  ESC=quit")

    # -- Convenience ----------------------------------------------------------

    @property
    def _active_effect(self):
        return self._effects[self._effect_idx]

    # -- Audio ----------------------------------------------------------------

    def _on_audio_frame(self, m: audio_metrics.AudioMetrics) -> None:
        self._circles.update_audio(m)

    # -- Regenerate -----------------------------------------------------------

    def _regen_all(self) -> None:
        self._circles.regen()
        self._cloud.randomize()
        nn_seed = random.randint(0, 2**31 - 1)
        self.nn_graph.randomize(nn_seed)
        print(f"[merged] regenerated  (nn_seed={nn_seed})")

    # -- Per-frame ------------------------------------------------------------

    def on_render(self, current_time: float, frame_time: float):
        self.time += frame_time

        if _controls.show_cloud:
            self._cloud.step(self.time, frame_time)

        if _controls.show_nn:
            self.nn_graph.step(self.time)
            self.nn_graph.upload()

        # Sync param_dialog controls -> active feedback effect each frame.
        eff = self._active_effect
        if isinstance(eff, FeedbackPostEffect):
            eff.scene_alpha = _controls.scene_alpha
            if _controls.blend_mode in BLEND_MODES:
                eff._blend_mode_idx = BLEND_MODES.index(_controls.blend_mode)
            if (_controls.smear_pattern != eff._smear_pattern_name
                    and _controls.smear_pattern in SMEAR_PATTERNS):
                eff._smear_pattern_name = _controls.smear_pattern
                eff._smear_pattern_idx  = SMEAR_PATTERNS.index(_controls.smear_pattern)
                if eff._loop is not None:
                    eff._loop.set_smear_pattern(_controls.smear_pattern)

        self.camera.tick(frame_time)
        mvp = self.camera.mvp(self.window_size)

        if _controls.show_lasers:
            cam_eye, cam_fwd, cam_right, cam_up = self.camera.position_and_axes()
            self._lasers.step(frame_time, cam_eye, cam_fwd, cam_right, cam_up)

        if self.post_effect_on:
            eff.bind_scene_fbo()
            self._draw_scene(mvp, current_time)
            eff.process(eff.fbo, current_time, dt=0.0)
            eff.blit_to_screen(self.ctx.screen)
        else:
            self.ctx.screen.use()
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.clear(0.04, 0.04, 0.06, 1.0)
            self._draw_scene(mvp, current_time)

    def _draw_scene(self, mvp, t: float) -> None:
        if _controls.show_cloud:
            self._cloud.draw(mvp)
        if _controls.show_nn:
            self.nn_graph.draw(mvp)
        if _controls.show_circles:
            self._circles.draw(mvp, t)
        if _controls.show_lasers:
            self._lasers.draw(mvp)

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

        # -- Scene toggles (always active) ------------------------------------
        if key == keys.NUMBER_1:
            _controls.show_cloud = not _controls.show_cloud
            print(f"[merged] cloud/ball: {'ON' if _controls.show_cloud else 'OFF'}")
        elif key == keys.NUMBER_2:
            _controls.show_nn = not _controls.show_nn
            print(f"[merged] nn-graph: {'ON' if _controls.show_nn else 'OFF'}")
        elif key == keys.NUMBER_3:
            _controls.show_circles = not _controls.show_circles
            print(f"[merged] circles: {'ON' if _controls.show_circles else 'OFF'}")
        elif key == keys.NUMBER_4:
            _controls.show_lasers = not _controls.show_lasers
            print(f"[merged] lasers: {'ON' if _controls.show_lasers else 'OFF'}")
        elif key == keys.R:
            self._regen_all()
        elif key == keys.ESCAPE:
            self.wnd.close()

        # -- Post pipeline toggle ---------------------------------------------
        elif key == keys.P:
            self.post_effect_on = not self.post_effect_on
            print(f"[merged] post pipeline: {'ON' if self.post_effect_on else 'OFF'}")

        # -- Camera -----------------------------------------------------------
        elif key == keys.O:
            self.camera.orbit_enabled = not self.camera.orbit_enabled
            if self.camera.orbit_enabled:
                self.camera._user_idle = 0.0
                print("[merged] orbit: ON")
            else:
                print("[merged] orbit: OFF")

        # -- Effect switcher (Tab) and per-effect keys ------------------------
        elif self.post_effect_on:
            if key == keys.TAB:
                self._effect_idx = (self._effect_idx + 1) % len(self._effects)
                print(f"[merged] effect -> {self._active_effect.name}")
            else:
                # Forward to active effect; sync any changes back to _controls
                eff = self._active_effect
                eff.on_key(key, action, keys)
                if isinstance(eff, FeedbackPostEffect):
                    _controls.scene_alpha   = eff.scene_alpha
                    _controls.blend_mode    = BLEND_MODES[eff._blend_mode_idx]
                    _controls.smear_pattern = eff._smear_pattern_name

    def on_resize(self, width: int, height: int):
        for eff in self._effects:
            eff.resize(width, height)


if __name__ == "__main__":
    from param_dialog import start_param_dialog
    start_param_dialog(
        ("Post-FX", _params),
        ("Scene",   _controls, {
            "blend_mode":    ("combo", BLEND_MODES),
            "smear_pattern": ("combo", SMEAR_PATTERNS),
        }),
        title="MergedGUI Params",
    )
    mglw.run_window_config(MergedGUI)
