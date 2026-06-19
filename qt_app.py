"""
qt_app.py — Qt event-loop thread: creates QApplication, all panels, and ControlBar.
"""

from __future__ import annotations

import json
import pathlib
import threading

_POS_FILE = pathlib.Path(__file__).with_name("window_positions.json")


def _load_positions() -> dict:
    try:
        return json.loads(_POS_FILE.read_text())
    except Exception:
        return {}


def _save_positions(panels: dict) -> None:
    data = _load_positions()
    for name, widget in panels.items():
        g = widget.geometry()
        data[name] = {"x": g.x(), "y": g.y(), "w": g.width(), "h": g.height()}
    try:
        _POS_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def _restore_geometry(widget, saved: dict, key: str) -> None:
    if key in saved:
        g = saved[key]
        widget.setGeometry(g["x"], g["y"], g["w"], g["h"])


def run_qt(
    lm,
    pm_ref,
    bpm,
    on_monitor_change,
    get_element_snapshot,
    on_palette_change,
    on_palette_apply,
    quit_event: threading.Event,
    set_signaller=None,
    perf_monitor=None,
    get_video_element=None,
) -> None:
    """Run the Qt event loop. Meant to be called from a daemon thread."""
    from midi_input import get_router
    from midi_panel import MidiPanel
    from osc_input import get_router as get_osc_router
    from osc_panel import OscPanel
    from audio_panel import AudioPanel, AudioDeviceSelector
    from color_panel import ColorPanel
    from elements_panel import ElementsPanel
    from link_panel import LinkManagerPanel
    from bpm_panel import BpmPanel
    from control_bar import ControlBar
    from settings_panel import SettingsPanel
    from video_panel import VideoPanel

    from PySide6.QtCore import QObject, QTimer, Qt, Signal
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)

    saved = _load_positions()

    _router = get_router()
    _osc_router = get_osc_router()

    midi = MidiPanel(_router, title="MIDI Input",
                     source_registry=lm.source_registry,
                     event_bus=lm.event_bus)

    osc = OscPanel(_osc_router, title="OSC Input",
                   source_registry=lm.source_registry,
                   event_bus=lm.event_bus)

    audio = AudioPanel(title="Audio Input", source_registry=lm.source_registry)

    audio_device = AudioDeviceSelector(
        title="Audio Device",
        source_registry=lm.source_registry,
        on_metrics=audio.update_metrics,
    )

    settings = SettingsPanel(on_monitor_change=on_monitor_change)

    colors = ColorPanel(
        title="Color Harmony",
        on_change=on_palette_change,
        on_apply=on_palette_apply,
    )

    elements_panel = ElementsPanel(
        event_bus=lm.event_bus,
        get_snapshot=get_element_snapshot,
        title="Scene Elements",
    )

    lm_panel = LinkManagerPanel(lm, pm_ref, title="Warp Controls")

    bpm_panel = BpmPanel(bpm, event_bus=lm.event_bus)

    video_panel = VideoPanel(
        get_video_element=get_video_element or (lambda: None),
        source_registry=lm.source_registry,
        title="Video",
    )

    panels = {
        "Links":    lm_panel,
        "BPM":      bpm_panel,
        "Elements": elements_panel,
        "Audio":    audio,
        "Device":   audio_device,
        "Colors":   colors,
        "Settings": settings,
        "MIDI":     midi,
        "OSC":      osc,
        "Video":    video_panel,
    }

    for key, panel in panels.items():
        _restore_geometry(panel, saved, key)

    control_bar = ControlBar(
        panels, lm, pm_ref,
        midi_router=_router,
        osc_router=_osc_router,
        bpm=bpm,
        get_element_snapshot=get_element_snapshot,
        perf_monitor=perf_monitor,
    )
    control_bar.show()

    class _Signaller(QObject):
        show_panel = Signal()

    signaller = _Signaller()
    signaller.show_panel.connect(control_bar.show_and_raise, Qt.ConnectionType.QueuedConnection)
    if set_signaller is not None:
        set_signaller(signaller)

    app.aboutToQuit.connect(lambda: _save_positions(panels))

    quit_poll = QTimer()
    quit_poll.setInterval(100)

    def _check_quit():
        if quit_event.is_set():
            quit_poll.stop()
            app.quit()

    quit_poll.timeout.connect(_check_quit)
    quit_poll.start()

    app.exec()
