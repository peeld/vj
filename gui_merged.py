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


from param_dialog import start_param_dialog, apply_fullscreen_to_monitor

_pending_monitor: int | None = None

from post import (
    FeedbackPostEffect, PassThroughEffect, GlitchEffect, BokehEffect,
    FeedbackParams, BLEND_MODES, SMEAR_PATTERNS, PRESETS,
)
from drawlib.camera import OrbitCamera

from elements.cloud import CloudElement
from elements.circleaxis import CircleAxisDrawing
from elements.laser_ribbons import LaserRibbons
from elements.nn_graph import NNGraph

from property_manager import PropertyManager, build_default_manager

# Late-bound audio callbacks registered by MergedGUI after it constructs its
# scene elements.  AudioPanel calls each entry from the audio thread so that
# the panel is the single audio stream owner.
_circles_audio_fns: list = []

# Module-level PM (params + controls only; element props added in MergedGUI.__init__)
# Shared with the Qt panels so MidiPanel can show all scene/feedback params
# before the GL window starts.
_pm: PropertyManager | None = None

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


_EFFECT_NAMES = ["feedback", "pass_through", "glitch", "bokeh"]


@dataclass
class SceneControls:
    show_cloud    : bool  = True
    show_nn       : bool  = True
    show_circles  : bool  = True
    show_lasers   : bool  = True
    scene_alpha   : float = 0.18
    blend_mode    : str   = "screen"   # screen keeps trail persistence = decay only
    smear_pattern : str   = "outward"
    active_effect : str   = "feedback"


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
        self._cloud     = CloudElement(self.ctx)
        self.nn_graph   = NNGraph(self.ctx, device=DEVICE)
        self._circles   = CircleAxisDrawing(self.ctx)
        self._lasers    = LaserRibbons(self.ctx)

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
            BokehEffect(),
        ]
        self._effect_idx = 0

        for eff in self._effects:
            eff.setup(self.ctx, w, h)

        # -- Audio ------------------------------------------------------------
        # AudioPanel (Qt thread) owns the audio stream; register this scene's
        # callback so the panel drives circles alongside PM bindings.
        _circles_audio_fns.append(self._circles.update_audio)

        # -- Property manager -------------------------------------------------
        # Extend the shared module-level PM with element-specific properties now
        # that the GL elements exist.  If _pm was already created by the Qt thread
        # (see __main__), properties that were registered earlier are skipped
        # (idempotent); only element sections are added here.
        global _pm
        self.pm = build_default_manager(
            _params, _controls,
            self.nn_graph, self._lasers, self._circles,
            pm=_pm,
        )
        _pm = self.pm   # keep module ref in sync

        effect_names = "  |  ".join(e.name for e in self._effects)
        print(f"[merged] effects: {effect_names}")
        print("[merged] 1=cloud  2=nn  3=circles  4=lasers  5=nn  R=regen  P=post  O=orbit  Tab=effect  ESC=quit")
        print("[merged] pm.describe() to list all properties  |  pm.save_preset('name') to snapshot")

    # -- Convenience ----------------------------------------------------------

    @property
    def _active_effect(self):
        return self._effects[self._effect_idx]

    # -- Regenerate -----------------------------------------------------------

    def _regen_all(self) -> None:
        self._circles.regen()
        self._cloud.randomize()
        nn_seed = random.randint(0, 2**31 - 1)
        self.nn_graph.randomize(nn_seed)
        self._nn.randomize(nn_seed)
        print(f"[merged] regenerated  (nn_seed={nn_seed})")

    # -- Per-frame ------------------------------------------------------------

    def on_render(self, current_time: float, frame_time: float):

        global _pending_monitor
        if _pending_monitor is not None:
            apply_fullscreen_to_monitor(self.wnd, _pending_monitor)
            _pending_monitor = None

        self.time += frame_time

        self._cloud.step(self.time, frame_time, _controls.show_cloud)

        if _controls.show_nn:
            self.nn_graph.step(self.time)
            self.nn_graph.upload()

        # Sync active_effect selection from param_dialog -> _effect_idx.
        desired = _controls.active_effect
        effect_names = [e.name for e in self._effects]
        if desired in effect_names:
            new_idx = effect_names.index(desired)
            if new_idx != self._effect_idx:
                self._effect_idx = new_idx
                print(f"[merged] effect -> {self._active_effect.name}")
        else:
            _controls.active_effect = self._active_effect.name

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

        cam_eye, cam_fwd, cam_right, cam_up = self.camera.position_and_axes()
        self._lasers.step(frame_time, cam_eye, cam_fwd, cam_right, cam_up, _controls.show_lasers)
        self._circles.step(frame_time, current_time, _controls.show_circles)

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
        self._cloud.draw(mvp)
        if _controls.show_nn:
            self.nn_graph.draw(mvp)
        self._circles.draw(mvp, t)
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

        # -- App-level actions (not delegated to PropertyManager) -------------
        if key == keys.R:
            self._regen_all()
            return
        if key == keys.ESCAPE:
            self.wnd.close()
            return
        if key == keys.P:
            self.post_effect_on = not self.post_effect_on
            print(f"[merged] post pipeline: {'ON' if self.post_effect_on else 'OFF'}")
            return
        if key == keys.O:
            self.camera.orbit_enabled = not self.camera.orbit_enabled
            self.camera._user_idle = 0.0 if self.camera.orbit_enabled else self.camera._user_idle
            print(f"[merged] orbit: {'ON' if self.camera.orbit_enabled else 'OFF'}")
            return

        if key == keys.T and self.post_effect_on:
            eff = self._active_effect
            if isinstance(eff, FeedbackPostEffect):
                eff.on_key(key, action, keys)
                # Sync preset changes back to _params (already in-place) and _controls
                _controls.scene_alpha   = eff.scene_alpha
                _controls.blend_mode    = BLEND_MODES[eff._blend_mode_idx]
                _controls.smear_pattern = eff._smear_pattern_name
            return

        # -- All other keys: delegate to PropertyManager ----------------------
        # Map moderngl key constants to the string names registered in the PM.
        _KEY_MAP = {
            keys.NUMBER_1: "NUMBER_1", keys.NUMBER_2: "NUMBER_2",
            keys.NUMBER_3: "NUMBER_3", keys.NUMBER_4: "NUMBER_4",
            keys.TAB: "TAB",
            keys.Z: "Z", keys.X: "X",
            keys.D: "D", keys.F: "F",
            keys.Q: "Q", keys.W: "W",
            keys.A: "A", keys.S: "S",
            keys.H: "H", keys.J: "J",
            keys.C: "C", keys.V: "V",
            keys.B: "B", keys.N: "N",
            keys.K: "K", keys.L: "L",
            keys.I: "I", keys.U: "U",
            keys.G: "G", keys.M: "M",
        }
        key_name = _KEY_MAP.get(key)
        if key_name and self.pm.apply_key_action(key_name):
            # Sync any scene_alpha / blend_mode / smear_pattern changes to the
            # active FeedbackPostEffect so the GPU effect picks them up.
            eff = self._active_effect
            if isinstance(eff, FeedbackPostEffect):
                eff.scene_alpha = _controls.scene_alpha
                if _controls.blend_mode in BLEND_MODES:
                    eff._blend_mode_idx = BLEND_MODES.index(_controls.blend_mode)
                if _controls.smear_pattern != eff._smear_pattern_name:
                    eff._smear_pattern_name = _controls.smear_pattern
                    eff._smear_pattern_idx  = SMEAR_PATTERNS.index(_controls.smear_pattern)
                    if eff._loop is not None:
                        eff._loop.set_smear_pattern(_controls.smear_pattern)
            return

    def on_resize(self, width: int, height: int):
        for eff in self._effects:
            eff.resize(width, height)


if __name__ == "__main__":
    import threading
    from PySide6.QtWidgets import QApplication
    from param_dialog import ParamDialog
    from midi_input   import get_router
    from midi_panel   import MidiPanel
    from audio_panel  import AudioPanel

    _router = get_router()

    # Build base PM now (feedback + scene props only; elements added in GL __init__)
    _pm = build_default_manager(_params, _controls, None, None, None)

    def _run_qt() -> None:
        """Single Qt thread — one QApplication, both panels."""
        app = QApplication.instance() or QApplication([])

        dlg = ParamDialog(
            [
                ("Post-FX", _params, {}),
                ("Scene",   _controls, {
                    "blend_mode":    ("combo", BLEND_MODES),
                    "smear_pattern": ("combo", SMEAR_PATTERNS),
                    "active_effect": ("combo", _EFFECT_NAMES),
                }),
            ],
            title="MergedGUI Params",
            on_monitor_change=lambda idx: globals().__setitem__('_pending_monitor', idx),
        )
        dlg.show()

        # MidiPanel now uses PropertyManager directly.
        # _pm initially has feedback + scene; element sections (nn_graph, lasers,
        # circles) are added to the same object when MergedGUI.__init__ runs.
        midi = MidiPanel(_router, _pm, title="MIDI Assignments")
        midi.show()

        # AudioPanel owns the single audio stream; _circles_audio_fns is
        # populated by MergedGUI.__init__ so circles receive audio even though
        # the GL window starts after this Qt thread.
        audio = AudioPanel(
            pm             = _pm,
            title          = "Audio Input",
            extra_on_frame = lambda m: [fn(m) for fn in _circles_audio_fns],
        )
        audio.show()

        app.exec()

    threading.Thread(target=_run_qt, daemon=True, name="qt-ui").start()

    mglw.run_window_config(MergedGUI)

    # After the GL window closes, the PropertyManager is accessible as:
    #   gui_instance.pm
    # Use pm.save_json("session.json") to persist presets and mappings.
