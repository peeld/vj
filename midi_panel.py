"""
midi_panel.py — MIDI input device selector + live event monitor

PySide6 panel for connecting a MIDI device.  On each MIDI event the raw
listener writes values to the SourceRegistry and fires onto the EventBus so
that LinkManager can drive scene parameters.  Use the Link Manager panel to
create MIDI-driven mappings via SignalLinks and EventLinks.

Usage (gui_merged.py __main__ block)::

    from midi_input import get_router
    from midi_panel import MidiPanel

    _router = get_router()
    midi = MidiPanel(_router,
                     source_registry=_lm.source_registry,
                     event_bus=_lm.event_bus)
    midi.show()
"""

from PySide6.QtCore    import QTimer, Qt, QSettings
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton,
    QFrame, QSizePolicy,
)

from midi_input import MidiRouter, get_router


# ── stylesheet ────────────────────────────────────────────────────────────────

_STYLESHEET = """
QWidget {
    background-color: #1a1a22;
    color: #c8c8d0;
    font-family: Consolas, "Courier New", monospace;
    font-size: 12px;
}
QLabel { color: #707078; }
QLabel#hdr  { color: #c8c8d0; font-weight: bold; padding-top: 4px; }
QLabel#act  { color: #ffb347; font-weight: bold; }
QLabel#ok   { color: #7ec87e; }
QLabel#err  { color: #e07070; }
QLabel#info { color: #707078; font-style: italic; }

QComboBox {
    background-color: #111118;
    color: #5eaeff;
    border: 1px solid #38383f;
    border-radius: 3px;
    padding: 2px 6px;
    min-width: 160px;
}
QComboBox:focus { border: 1px solid #5eaeff; }
QComboBox QAbstractItemView {
    background-color: #1a1a22;
    color: #c8c8d0;
    selection-background-color: #2a4070;
    border: 1px solid #38383f;
}
QComboBox::drop-down { border: none; width: 18px; }
QComboBox::down-arrow {
    image: none;
    border-left:  4px solid transparent;
    border-right: 4px solid transparent;
    border-top:   5px solid #707078;
    margin-right: 4px;
}

QPushButton {
    background-color: #2a2a36;
    color: #c8c8d0;
    border: 1px solid #38383f;
    border-radius: 3px;
    padding: 3px 10px;
    min-width: 60px;
}
QPushButton:hover  { background-color: #32323f; border-color: #5eaeff; }
QPushButton:pressed{ background-color: #1a1a28; }

QScrollBar:vertical {
    background: #1a1a22; width: 6px; margin: 0;
}
QScrollBar::handle:vertical { background: #38383f; border-radius: 3px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QFrame#sep {
    color: #2e2e38;
    max-height: 1px;
    background-color: #2e2e38;
}
"""


# ── helpers ───────────────────────────────────────────────────────────────────

def _sep() -> QFrame:
    f = QFrame()
    f.setObjectName("sep")
    f.setFrameShape(QFrame.HLine)
    return f


def _hdr(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("hdr")
    return lbl


def _note_name(n: int) -> str:
    names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    return f"{names[n % 12]}{n // 12 - 1}"


def _restyle(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)


# ── main panel ────────────────────────────────────────────────────────────────

class MidiPanel(QWidget):
    """
    Live MIDI input panel.

    Connects to a MIDI device via MidiRouter and streams events into
    SourceRegistry (CC values) and EventBus (note-on/off).  Use the Link
    Manager panel to create MIDI-driven mappings via SignalLinks and EventLinks.
    """

    def __init__(
        self,
        router          : MidiRouter,
        title           : str = "MIDI Input",
        source_registry = None,   # SourceRegistry | None
        event_bus       = None,   # EventBus | None
    ):
        super().__init__()
        self.router           = router
        self._source_registry = source_registry
        self._event_bus       = event_bus

        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setStyleSheet(_STYLESHEET)
        self.setMinimumWidth(380)
        self.resize(400, 260)

        self._build_ui()
        self._refresh_devices()
        self._restore_settings()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(80)

        self.router.add_listener(self._raw_listener)

    # ── teardown ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self.router.remove_listener(self._raw_listener)
        super().closeEvent(event)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        # ── Device section ────────────────────────────────────────────────────
        root.addWidget(_hdr("Device"))
        root.addWidget(_sep())

        dev_row = QHBoxLayout()
        self._dev_combo = QComboBox()
        self._dev_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        dev_row.addWidget(self._dev_combo)

        self._btn_refresh = QPushButton("↻")
        self._btn_refresh.setFixedWidth(28)
        self._btn_refresh.clicked.connect(self._refresh_devices)
        dev_row.addWidget(self._btn_refresh)

        self._btn_connect = QPushButton("Connect")
        self._btn_connect.clicked.connect(self._on_connect)
        dev_row.addWidget(self._btn_connect)

        self._btn_disconnect = QPushButton("Disconnect")
        self._btn_disconnect.clicked.connect(self._on_disconnect)
        dev_row.addWidget(self._btn_disconnect)
        root.addLayout(dev_row)

        self._lbl_status = QLabel("Not connected")
        self._lbl_status.setObjectName("info")
        root.addWidget(self._lbl_status)

        # ── Last Event ────────────────────────────────────────────────────────
        root.addSpacing(4)
        root.addWidget(_hdr("Last Event"))
        root.addWidget(_sep())
        self._lbl_activity = QLabel("—")
        self._lbl_activity.setObjectName("act")
        root.addWidget(self._lbl_activity)

        root.addStretch(1)

    # ── device management ──────────────────────────────────────────────────────

    def _refresh_devices(self) -> None:
        self._dev_combo.clear()
        names = self.router.available_devices()
        if names:
            self._dev_combo.addItems(names)
        else:
            self._dev_combo.addItem("(no devices found)")

    def _restore_settings(self) -> None:
        qs = QSettings("WarpApp", "WarpApp")
        device = qs.value("midi/device", "")
        if device:
            idx = self._dev_combo.findText(device)
            if idx >= 0:
                self._dev_combo.setCurrentIndex(idx)
                self._on_connect()

    def _on_connect(self) -> None:
        name = self._dev_combo.currentText()
        if name.startswith("("):
            return
        ok = self.router.start(name)
        if ok:
            self._lbl_status.setObjectName("ok")
            self._lbl_status.setText(f"● Connected: {name}")
            QSettings("WarpApp", "WarpApp").setValue("midi/device", name)
        else:
            self._lbl_status.setObjectName("err")
            self._lbl_status.setText(f"✗ Failed to connect: {name}")
        _restyle(self._lbl_status)

    def _on_disconnect(self) -> None:
        self.router.stop()
        self._lbl_status.setObjectName("info")
        self._lbl_status.setText("Not connected")
        _restyle(self._lbl_status)

    # ── MIDI event routing ────────────────────────────────────────────────────

    def _raw_listener(self, evt: dict) -> None:
        """Called from MIDI thread — updates SourceRegistry and fires EventBus."""
        try:
            if evt["type"] == "cc":
                if self._source_registry is not None:
                    self._source_registry.update(
                        f"midi.cc{evt['number']}", evt["value"] / 127.0
                    )
            elif evt["type"] == "note":
                note  = evt["number"]
                value = evt.get("value", 0)
                if self._source_registry is not None:
                    if value > 0:
                        self._source_registry.update(f"midi.note{note}",     1.0)
                        self._source_registry.update(f"midi.note{note}_vel", value / 127.0)
                    else:
                        self._source_registry.update(f"midi.note{note}", 0.0)
                        # _vel kept at last note-on value — not updated on note-off
                if self._event_bus is not None:
                    if value > 0:
                        self._event_bus.fire(f"midi.note{note}.on",  value)
                    else:
                        self._event_bus.fire(f"midi.note{note}.off", 0)
        except Exception:
            pass

    # ── poll timer ─────────────────────────────────────────────────────────────

    def _poll(self) -> None:
        ev = self.router.last_event
        if ev:
            t  = ev.get("type", "?")
            ch = ev.get("channel", 0)
            n  = ev.get("number", 0)
            v  = ev.get("value",  0)
            if t == "cc":
                self._lbl_activity.setText(f"CC#{n}  ch={ch}  val={v}")
            elif t == "note":
                self._lbl_activity.setText(
                    f"{'Note On' if v > 0 else 'Note Off'}  "
                    f"{_note_name(n)}({n})  ch={ch}  vel={v}"
                )
