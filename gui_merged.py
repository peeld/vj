"""

App-level controls (not user-configurable):
  R          -- regenerate all elements
  P          -- toggle post pipeline on / off (raw scene render)
  O          -- toggle orbit camera on / off
  T          -- cycle feedback preset (when feedback effect active)
  Mouse drag -- orbit camera
  Scroll     -- zoom
  ESC        -- quit

All other key mappings are configured via the Link Manager panel (EventLinks).
"""

import random
from dataclasses import dataclass

import moderngl
import moderngl_window as mglw
import warp as wp

from collections.abc import Callable

_pending_monitor: int | None = None

from post import (
    FeedbackPostEffect, PassThroughEffect, GlitchEffect, BokehEffect,
    FeedbackParams, BLEND_MODES, SMEAR_PATTERNS, PRESETS,
)
from drawlib.camera import OrbitCamera

from elements.base import DrawingElement, FrameContext, ELEMENT_TYPES
import elements.cloud, elements.nn_graph, elements.circleaxis, elements.laser_ribbons  # noqa: F401 -- registers cloud/nn_graph/circles/lasers

from property_manager import PropertyManager, build_default_manager
from link_manager import LinkManager, KEY_NAMES

from PySide6.QtGui import QGuiApplication

from PySide6.QtWidgets import (
    QApplication, QWidget, QFormLayout, QLineEdit,
    QLabel, QScrollArea, QVBoxLayout, QFrame,
    QCheckBox, QComboBox, QPushButton,
)
from PySide6.QtCore import QTimer, Qt, QSettings

# Module-level PM (params + controls only; element props added in MergedGUI.__init__)
# Shared with the Qt panels so MidiPanel can show all scene/feedback params
# before the GL window starts.
_pm: PropertyManager | None = None

# Module-level LinkManager — owns the SourceRegistry written by audio/MIDI threads
# and read by the GL thread each frame.
_lm = LinkManager()

# Current palette from ColorPanel — updated by the Qt listener, read by the GL thread
# at element spawn / regen time.  Simple list assignment is thread-safe in CPython.
_current_palette: list = []

# Set to True by the Apply button; consumed in on_render() to push the palette to
# elements without a full regen.  Same pattern as _pending_monitor.
_pending_palette_apply: bool = False

# Read-only snapshot of the live scene-element list, rebuilt once per frame in
# on_render().  ElementsPanel polls this (Qt thread) to mirror GL-thread state
# without touching GL objects directly.  Simple list assignment is thread-safe
# in CPython, same pattern as _current_palette.
_element_snapshot: list[dict] = []


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
    # Per-element visibility lives on each DrawingElement instance (.visible)
    # instead of named booleans here, since the element list is dynamic --
    # see MergedGUI.elements / add_element() / remove_element().
    scene_alpha   : float = 0.18
    blend_mode    : str   = "screen"   # screen keeps trail persistence = decay only
    smear_pattern : str   = "outward"
    active_effect : str   = "feedback"


_controls = SceneControls()


def apply_fullscreen_to_monitor(mglw_wnd, monitor_index: int) -> None:
    """
    Switch an mglw window to fullscreen on the given monitor.

    Must be called from the main/render thread.  Typical usage::

        _pending_monitor: int | None = None

        def render(self, time, frametime):
            global _pending_monitor
            if _pending_monitor is not None:
                apply_fullscreen_to_monitor(self.wnd, _pending_monitor)
                _pending_monitor = None
            ...

        settings = SettingsPanel(
            on_monitor_change=lambda idx: globals().__setitem__('_pending_monitor', idx),
        )

    Supports the pyglet, glfw, and pygame2 mglw backends.

    Args:
        mglw_wnd      : the ``self.wnd`` WindowConfig attribute
        monitor_index : 0-based index into the list of attached monitors
                        (0 = primary / first monitor)
    """
    backend = getattr(mglw_wnd, "name", "")

    # ── pyglet ────────────────────────────────────────────────────────────────
    if backend == "pyglet":
        try:
            # Use the display attached to the existing window — works across
            # all pyglet versions without importing pyglet.canvas / pyglet.display.
            screens = mglw_wnd._window.display.get_screens()
            if not screens:
                print("[param_dialog] apply_fullscreen_to_monitor: no pyglet screens found")
                return
            monitor_index = max(0, min(monitor_index, len(screens) - 1))
            mglw_wnd._window.set_fullscreen(True, screen=screens[monitor_index])
        except Exception as exc:
            print(f"[param_dialog] apply_fullscreen_to_monitor (pyglet) failed: {exc}")
        return

    # ── glfw ──────────────────────────────────────────────────────────────────
    if backend == "glfw":
        try:
            import glfw
            monitors = glfw.get_monitors()
            if not monitors:
                print("[param_dialog] apply_fullscreen_to_monitor: no GLFW monitors found")
                return
            monitor_index = max(0, min(monitor_index, len(monitors) - 1))
            monitor = monitors[monitor_index]
            mode = glfw.get_video_mode(monitor)
            glfw.set_window_monitor(
                mglw_wnd._window, monitor,
                0, 0, mode.size.width, mode.size.height, mode.refresh_rate,
            )
        except Exception as exc:
            print(f"[param_dialog] apply_fullscreen_to_monitor (glfw) failed: {exc}")
        return

    # ── pygame2 / SDL2 ────────────────────────────────────────────────────────
    if backend == "pygame2":
        try:
            import pygame._sdl2.video as sdl2
            # SDL2 display index maps to monitor; recreate window on the target display
            sdl_win = mglw_wnd._sdl_window
            sdl_win.position = sdl2.WINDOWPOS_CENTERED_DISPLAY(monitor_index)
            sdl_win.set_fullscreen(True)
        except Exception as exc:
            print(f"[param_dialog] apply_fullscreen_to_monitor (pygame2) failed: {exc}")
        return

    print(f"[param_dialog] apply_fullscreen_to_monitor: unsupported backend '{backend}'")



# ── settings panel ─────────────────────────────────────────────────────────────

class SettingsPanel(QWidget):
    """
    Application settings, organized into labeled sections.

    on_monitor_change : optional callback(monitor_index: int) invoked when the
                        user picks a monitor and clicks "Go Fullscreen" in the
                        Display section.  Use
                        ``apply_fullscreen_to_monitor(self.wnd, index)`` in the
                        callback or render loop to apply the switch via GLFW.
    """

    def __init__(
        self,
        on_monitor_change: Callable[[int], None] | None = None,
        title: str = "Settings",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        # self.setStyleSheet(_STYLESHEET)
        self.setMinimumWidth(380)

        self._on_monitor_change = on_monitor_change

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        form = QFormLayout()
        form.setContentsMargins(6, 6, 6, 6)
        form.setSpacing(4)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        outer.addLayout(form)
        outer.addStretch()

        self._build_display_section(form)

    # ── Display section ──────────────────────────────────────────────────────

    def _build_display_section(self, form: QFormLayout) -> None:
        """'Display' section: monitor picker and fullscreen button."""
        hdr = QLabel("Display")
        hdr.setObjectName("section_header")
        form.addRow(hdr)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #2e2e38;")
        form.addRow(sep)

        self._monitor_combo = QComboBox()
        for label in self._get_monitor_labels():
            self._monitor_combo.addItem(label)
        saved_idx = int(QSettings("WarpApp", "WarpApp").value("display/monitor_index", 0))
        if 0 <= saved_idx < self._monitor_combo.count():
            self._monitor_combo.setCurrentIndex(saved_idx)
        form.addRow(QLabel("Monitor"), self._monitor_combo)

        btn = QPushButton("Go Fullscreen")
        btn.setStyleSheet(
            "QPushButton {"
            "  background-color: #1e2a40; color: #5eaeff;"
            "  border: 1px solid #38383f; border-radius: 3px; padding: 4px 10px;"
            "}"
            "QPushButton:hover { background-color: #243050; }"
            "QPushButton:pressed { background-color: #2a4070; }"
        )
        btn.clicked.connect(self._emit_monitor_change)
        form.addRow("", btn)

    @staticmethod
    def _get_monitor_labels() -> list[str]:
        app = QGuiApplication.instance()
        screens = app.screens() if app else []
        labels = []
        for i, s in enumerate(screens):
            geo = s.geometry()
            tag = " [primary]" if s == app.primaryScreen() else ""
            labels.append(
                f"Monitor {i}: {s.name()} "
                f"({geo.width()}×{geo.height()} @ {geo.x()},{geo.y()}){tag}"
            )
        if not labels:
            labels = ["Monitor 0 (unknown)"]
        return labels

    def _emit_monitor_change(self) -> None:
        if self._on_monitor_change is not None:
            idx = self._monitor_combo.currentIndex()
            QSettings("WarpApp", "WarpApp").setValue("display/monitor_index", idx)
            self._on_monitor_change(idx)


# ── public entry point ────────────────────────────────────────────────────────


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
        # Dynamic list -- see add_element()/remove_element().  The startup set
        # mirrors the previous hardcoded cloud/nn_graph/circles/lasers members;
        # draw order follows list order, so this also preserves the original
        # back-to-front draw sequence.
        self.elements: list[DrawingElement] = []
        for kind in ("cloud", "nn_graph", "circles", "lasers"):
            self.add_element(kind)

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

        # -- Palette ----------------------------------------------------------
        if _current_palette:
            self._apply_palette()

        # -- Link manager -----------------------------------------------------
        self.lm = _lm
        self._held_keys: set[str] = set()
        self.lm.event_bus.subscribe("regen", lambda _: self._regen_all())
        self.lm.event_bus.subscribe("element.add", lambda kind: self.add_element(kind))
        self.lm.event_bus.subscribe("element.remove", lambda name: self.remove_element(name))
        self.lm.event_bus.subscribe("element.set_visible", self._on_set_visible_event)

        # -- Property manager -------------------------------------------------
        # Extend the shared module-level PM with element-specific properties now
        # that the GL elements exist.  If _pm was already created by the Qt thread
        # (see __main__), properties that were registered earlier are skipped
        # (idempotent); only element sections are added here.
        # NOTE: still bound to the first nn_graph/lasers/circles instance found --
        # build_default_manager moves to a per-element-instance scheme if/when
        # the UI needs to bind properties for more than one instance per kind.
        global _pm
        self.pm = build_default_manager(
            _params, _controls,
            self._first_of_kind("nn_graph"), self._first_of_kind("lasers"), self._first_of_kind("circles"),
            pm=_pm,
        )
        _pm = self.pm   # keep module ref in sync

        effect_names = "  |  ".join(e.name for e in self._effects)
        print(f"[merged] effects: {effect_names}")
        print("[merged] R=regen  P=post  O=orbit  Tab=effect  ESC=quit")
        print("[merged] pm.describe() to list all properties  |  pm.save_preset('name') to snapshot")

    # -- Convenience ----------------------------------------------------------

    @property
    def _active_effect(self):
        return self._effects[self._effect_idx]

    # -- Dynamic element list ---------------------------------------------------

    def add_element(self, kind: str, **kwargs) -> DrawingElement:
        """Construct a new DrawingElement of *kind* and append it to self.elements.

        *kind* must be a key in elements.base.ELEMENT_TYPES.  At most one
        live instance per kind is permitted, since "<kind>.visible" is the
        permanent Channels identity for that kind.  Must be called on the
        GL thread (it touches moderngl_window's ctx).
        """
        global _pm
        factory = ELEMENT_TYPES.get(kind)
        if factory is None:
            raise ValueError(f"unknown element kind '{kind}' (known: {sorted(ELEMENT_TYPES)})")
        if any(el.kind == kind for el in self.elements):
            raise ValueError(f"element kind '{kind}' already has a live instance")
        element = factory(self.ctx, DEVICE, **kwargs)
        if _current_palette:
            element.set_palette(_current_palette)
        self.elements.append(element)
        _pm.bind(f"{kind}.visible", element, "visible")
        print(f"[merged] added element '{element.name}'")
        return element

    def remove_element(self, name: str) -> bool:
        """Remove the element with the given .name. Must be called on the GL thread."""
        global _pm
        for i, el in enumerate(self.elements):
            if el.name == name:
                _pm.unbind(f"{el.kind}.visible")
                del self.elements[i]
                print(f"[merged] removed element '{name}'")
                return True
        return False

    def _first_of_kind(self, kind: str):
        """Return the first element of *kind*, or None. Used to bind
        build_default_manager's element-specific properties to a live
        instance before any per-element-instance UI exists."""
        for el in self.elements:
            if el.kind == kind:
                return el
        return None

    def _on_set_visible_event(self, payload) -> None:
        name, value = payload
        for el in self.elements:
            if el.name == name:
                el.visible = bool(value)
                break

    # -- Palette --------------------------------------------------------------

    def _apply_palette(self) -> None:
        """Push the current module-level palette to all scene elements."""
        if not _current_palette:
            return
        for el in self.elements:
            el.set_palette(_current_palette)

    # -- Regenerate -----------------------------------------------------------

    def _regen_all(self) -> None:
        self._apply_palette()
        for el in self.elements:
            el.regen()
        print("[merged] regenerated")

    # -- Per-frame ------------------------------------------------------------

    def on_render(self, current_time: float, frame_time: float):

        global _pending_monitor, _pending_palette_apply, _element_snapshot
        if _pending_monitor is not None:
            apply_fullscreen_to_monitor(self.wnd, _pending_monitor)
            _pending_monitor = None
        if _pending_palette_apply:
            self._apply_palette()
            _pending_palette_apply = False

        self.time += frame_time

        # ── Source registry: clock + keyboard hold ────────────────────────────
        reg = self.lm.source_registry
        reg.update("clock.t", self.time)
        for _kn in KEY_NAMES:
            reg.update(f"key.{_kn}_hold", 1.0 if _kn in self._held_keys else 0.0)

        # ── Tick envelopes + LFOs + parameters → source registry ─────────────
        self.lm.tick_envelopes(frame_time)
        self.lm.tick_lfos(frame_time)
        self.lm.tick_parameters(frame_time)

        # ── Threshold detectors → EventBus ────────────────────────────────────
        self.lm.tick_thresholds()

        # ── Drain EventBus → envelope triggers + EventLinks (incl. element add/
        # remove/visibility events fired from the Qt-thread Elements panel) ────
        self.lm.evaluate_events(self.pm)

        # ── Evaluate all signal links → write to PM ───────────────────────────
        self.lm.evaluate_links(self.pm, frame_time)

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

        ctx = FrameContext(
            time=self.time, current_time=current_time, frame_time=frame_time,
            cam_eye=cam_eye, cam_fwd=cam_fwd, cam_right=cam_right, cam_up=cam_up,
        )
        for el in self.elements:
            el.step(ctx)

        _element_snapshot = [
            {"name": el.name, "kind": el.kind, "visible": el.visible}
            for el in self.elements
        ]

        if self.post_effect_on:
            eff.bind_scene_fbo()
            self._draw_scene(mvp, ctx)
            eff.process(eff.fbo, current_time, dt=0.0)
            eff.blit_to_screen(self.ctx.screen)
        else:
            self.ctx.screen.use()
            self.ctx.enable(moderngl.DEPTH_TEST)
            self.ctx.clear(0.04, 0.04, 0.06, 1.0)
            self._draw_scene(mvp, ctx)

    def _draw_scene(self, mvp, ctx: FrameContext) -> None:
        for el in self.elements:
            if el.visible:
                el.draw(mvp, ctx)

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
        keys = self.wnd.keys

        # -- Keyboard hold-state tracking (press AND release) -----------------
        # moderngl_window's `keys` attribute names match KEY_NAMES exactly.
        _KEY_MAP = {getattr(keys, name): name for name in KEY_NAMES}
        key_name = _KEY_MAP.get(key)
        if key_name is not None:
            if action == keys.ACTION_PRESS:
                self._held_keys.add(key_name)
                self.lm.event_bus.fire(f"key.{key_name}.press")
            elif action == keys.ACTION_RELEASE:
                self._held_keys.discard(key_name)
                self.lm.event_bus.fire(f"key.{key_name}.release")

        if action != keys.ACTION_PRESS:
            return

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


    def on_resize(self, width: int, height: int):
        for eff in self._effects:
            eff.resize(width, height)


if __name__ == "__main__":
    import threading
    from PySide6.QtWidgets import QApplication
    from midi_input   import get_router
    from midi_panel   import MidiPanel
    from osc_input    import get_router as get_osc_router
    from osc_panel    import OscPanel
    from audio_panel  import AudioPanel

    _router     = get_router()
    _osc_router = get_osc_router()

    # Build base PM now (feedback + scene props only; elements added in GL __init__)
    _pm = build_default_manager(_params, _controls, None, None, None)

    import json, pathlib
    _POS_FILE = pathlib.Path(__file__).with_name("window_positions.json")

    def _load_positions() -> dict:
        try:
            return json.loads(_POS_FILE.read_text())
        except Exception:
            return {}

    def _save_positions(widgets: dict) -> None:
        data = {}
        for name, w in widgets.items():
            g = w.geometry()
            data[name] = {"x": g.x(), "y": g.y(), "w": g.width(), "h": g.height()}
        _POS_FILE.write_text(json.dumps(data, indent=2))

    def _restore_geometry(widget, saved: dict, key: str) -> None:
        if key in saved:
            g = saved[key]
            widget.setGeometry(g["x"], g["y"], g["w"], g["h"])

    def _run_qt() -> None:
        """Single Qt thread — one QApplication, one combined control panel."""
        app = QApplication.instance() or QApplication([])

        saved = _load_positions()

        settings = SettingsPanel(
            on_monitor_change=lambda idx: globals().__setitem__('_pending_monitor', idx),
        )

        midi = MidiPanel(_router, title="MIDI Input",
                         source_registry=_lm.source_registry,
                         event_bus=_lm.event_bus)

        osc = OscPanel(_osc_router, title="OSC Input",
                       source_registry=_lm.source_registry,
                       event_bus=_lm.event_bus)

        audio = AudioPanel(
            title           = "Audio Input",
            source_registry = _lm.source_registry,
        )

        from color_panel import ColorPanel

        def _on_palette_change(palette: list) -> None:
            global _current_palette
            _current_palette = palette

        def _on_palette_apply(palette: list) -> None:
            global _current_palette, _pending_palette_apply
            _current_palette = palette
            _pending_palette_apply = True

        colors = ColorPanel(title="Color Harmony", on_change=_on_palette_change, on_apply=_on_palette_apply)

        from elements_panel import ElementsPanel
        elements_tab = ElementsPanel(
            event_bus    = _lm.event_bus,
            get_snapshot = lambda: _element_snapshot,
            title        = "Scene Elements",
        )

        from link_panel import LinkManagerPanel
        links = LinkManagerPanel(
            _lm, _pm,
            title="Warp Controls",
            extra_tabs=[
                ("Elements", elements_tab),
                ("MIDI",     midi),
                ("OSC",      osc),
                ("Audio",    audio),
                ("Colors",   colors),
                ("Settings", settings),
            ],
        )
        _restore_geometry(links, saved, "ControlPanel")
        links.show()

        app.aboutToQuit.connect(lambda: _save_positions({"ControlPanel": links}))

        app.exec()

    threading.Thread(target=_run_qt, daemon=True, name="qt-ui").start()

    mglw.run_window_config(MergedGUI)

    # After the GL window closes, the PropertyManager is accessible as:
    #   gui_instance.pm
    # Use pm.save_json("session.json") to persist presets and mappings.
