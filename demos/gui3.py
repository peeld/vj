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

from dataclasses import dataclass

import moderngl
import moderngl_window as mglw
import warp as wp

from drawlib.post_effect import FeedbackPostEffect
from drawlib.warp_feedback import FeedbackParams
from drawlib.camera import OrbitCamera

import audio_metrics
from elements.circleaxis import CircleAxisDrawing

wp.init()


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

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Camera
        self.camera = OrbitCamera()
        self._drag  = False

        # Drawing
        self._drawing = CircleAxisDrawing(self.ctx)

        # Post-effect
        self.post_effect_on = True
        w, h = self.window_size
        self._post = FeedbackPostEffect(
            params        = CircleParams(),
            scene_alpha   = self.SCENE_ALPHA,
            smear_pattern = "outward",
        )
        self._post.setup(self.ctx, w, h)

        # Audio — wires into drawing and post-effect via callback
        self._audio = audio_metrics.AudioAnalyzer(
            device   = "CABLE Output (VB-Audio Virtual Cable), Windows DirectSound",
            on_frame = self._on_audio_frame,
        )
        self._audio.start()

        print("[gui3] ready  --  R: regenerate  P: post-effect  O: orbit  ESC: quit")

    # ------------------------------------------------------------------
    # Audio callback — links audio → drawing + post-effect
    # ------------------------------------------------------------------

    def _on_audio_frame(self, m: audio_metrics.AudioMetrics) -> None:
        self._drawing.amplitude = m.energy * 0.4
        self._post.params.sat_boost = m.bass * 4

    # ------------------------------------------------------------------
    # Per-frame
    # ------------------------------------------------------------------

    def on_render(self, current_time: float, frame_time: float):
        self.camera.tick(frame_time)
        mvp = self.camera.mvp(self.window_size)

        if self.post_effect_on:
            self._post.bind_scene_fbo()
            self._drawing.draw(mvp, current_time)
            self._post.process(self._post.fbo, current_time, dt=0.0)
            self._post.blit_to_screen(self.ctx.screen)
        else:
            self.ctx.screen.use()
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.clear(0.04, 0.04, 0.06, 1.0)
            self._drawing.draw(mvp, current_time)

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

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

        # Drawing class gets first pick (handles R)
        if self._drawing.on_key(key, action, keys):
            return

        if key == keys.ESCAPE:
            self.wnd.close()
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
