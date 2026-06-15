"""
audio_panel.py — Audio input device selector + live metrics monitor

PySide6 panel for selecting an audio input device and viewing live AudioMetrics
values.  Audio signal routing to scene parameters is handled by LinkManager
via SignalLinks — use the Link Manager panel to create audio-driven mappings.

Usage::

    from audio_metrics import AudioAnalyzer, AudioMetrics
    from audio_panel   import AudioPanel

    panel = AudioPanel(
        title="Audio Input",
        source_registry=lm.source_registry,
    )
    panel.show()
"""

from __future__ import annotations

import sounddevice as sd

from PySide6.QtCore    import QTimer, Qt
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton,
    QFrame, QSizePolicy, QProgressBar,
)

from audio_metrics import AudioAnalyzer, AudioMetrics


# ── stylesheet ────────────────────────────────────────────────────────────────

_STYLESHEET = """
QWidget {
    background-color: #1a1a22;
    color: #c8c8d0;
    font-family: Consolas, "Courier New", monospace;
    font-size: 12px;
}
QLabel           { color: #707078; }
QLabel#hdr       { color: #c8c8d0; font-weight: bold; padding-top: 4px; }
QLabel#metric    { color: #c8c8d0; min-width: 72px; }
QLabel#value     { color: #5eaeff; min-width: 42px; }
QLabel#ok        { color: #7ec87e; }
QLabel#err       { color: #e07070; }
QLabel#info      { color: #707078; font-style: italic; }

QComboBox {
    background-color: #111118;
    color: #5eaeff;
    border: 1px solid #38383f;
    border-radius: 3px;
    padding: 2px 6px;
    min-width: 120px;
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
QPushButton:hover   { background-color: #32323f; border-color: #5eaeff; }
QPushButton:pressed { background-color: #1a1a28; }

QProgressBar {
    background-color: #111118;
    border: 1px solid #2e2e38;
    border-radius: 2px;
    height: 8px;
    text-align: right;
    color: transparent;
}
QProgressBar::chunk { background-color: #2a5080; border-radius: 1px; }
QProgressBar#pulse  ::chunk { background-color: #ffb347; }

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


def _restyle(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)


# ── metric definitions (attr, display label, is_pulse) ───────────────────────

_METRICS: list[tuple[str, str, bool]] = [
    ("sub_bass",   "Sub Bass",   False),
    ("bass",       "Bass",       False),
    ("low_mid",    "Low Mid",    False),
    ("mid",        "Mid",        False),
    ("high_mid",   "High Mid",   False),
    ("treble",     "Treble",     False),
    ("energy",     "Energy",     False),
    ("brightness", "Brightness", False),
    ("flux",       "Flux",       False),
    ("kick",       "Kick",       True),
    ("onset",      "Onset",      True),
]


# ── panel ─────────────────────────────────────────────────────────────────────

class AudioPanel(QWidget):
    """
    Live audio input panel.

    Connects to a sounddevice input and streams AudioMetrics values into the
    SourceRegistry each frame so LinkManager can drive scene parameters via
    SignalLinks.  Use the Link Manager panel to create audio-driven mappings.

    Parameters
    ----------
    title : str
        Window title.
    extra_on_frame : callable | None
        Optional extra callback called each frame with the latest AudioMetrics.
    source_registry : SourceRegistry | None
        If provided, all AudioMetrics fields are written as audio.<field> each frame.
    """

    def __init__(
        self,
        title           : str = "Audio Input",
        extra_on_frame  = None,
        source_registry = None,   # SourceRegistry | None
    ):
        super().__init__()
        self._extra_on_frame  = extra_on_frame
        self._source_registry = source_registry
        self._analyzer: AudioAnalyzer | None = None
        self._metrics  = AudioMetrics()

        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setStyleSheet(_STYLESHEET)
        self.setMinimumWidth(420)
        self.resize(460, 480)

        self._build_ui()
        self._refresh_devices()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(33)

    # ── teardown ──────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._stop_analyzer()
        super().closeEvent(event)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        # ── Device ────────────────────────────────────────────────────────────
        root.addWidget(_hdr("Device"))
        root.addWidget(_sep())

        dev_row = QHBoxLayout()
        self._dev_combo = QComboBox()
        self._dev_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        dev_row.addWidget(self._dev_combo)

        btn_refresh = QPushButton("↻")
        btn_refresh.setFixedWidth(28)
        btn_refresh.clicked.connect(self._refresh_devices)
        dev_row.addWidget(btn_refresh)

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

        # ── Live Metrics ──────────────────────────────────────────────────────
        root.addSpacing(4)
        root.addWidget(_hdr("Live Metrics"))
        root.addWidget(_sep())

        self._bars:   dict[str, QProgressBar] = {}
        self._values: dict[str, QLabel]       = {}

        for attr, label, is_pulse in _METRICS:
            row = QHBoxLayout()
            row.setSpacing(8)

            lbl = QLabel(label)
            lbl.setObjectName("metric")
            lbl.setFixedWidth(72)
            row.addWidget(lbl)

            bar = QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(0)
            bar.setFixedHeight(8)
            if is_pulse:
                bar.setObjectName("pulse")
            row.addWidget(bar, stretch=1)

            val = QLabel("0.000")
            val.setObjectName("value")
            val.setFixedWidth(42)
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(val)

            self._bars[attr]   = bar
            self._values[attr] = val
            root.addLayout(row)

        root.addStretch(1)

    # ── device management ─────────────────────────────────────────────────────

    def _refresh_devices(self) -> None:
        self._dev_combo.clear()
        try:
            devices = sd.query_devices()
        except Exception:
            self._dev_combo.addItem("(sounddevice unavailable)")
            return

        found = False
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                self._dev_combo.addItem(f"[{i}] {dev['name']}", userData=i)
                found = True

        if not found:
            self._dev_combo.addItem("(no input devices found)")

    def _on_connect(self) -> None:
        idx = self._dev_combo.currentIndex()
        device_index = self._dev_combo.itemData(idx)
        if device_index is None:
            return

        self._stop_analyzer()
        try:
            self._analyzer = AudioAnalyzer(
                device=device_index,
                on_frame=self._on_audio_frame,
            )
            self._analyzer.start()
            name = self._dev_combo.currentText()
            self._lbl_status.setObjectName("ok")
            self._lbl_status.setText(f"● Connected: {name}")
        except Exception as exc:
            self._analyzer = None
            self._lbl_status.setObjectName("err")
            self._lbl_status.setText(f"✗ {exc}")
        _restyle(self._lbl_status)

    def _on_disconnect(self) -> None:
        self._stop_analyzer()
        self._lbl_status.setObjectName("info")
        self._lbl_status.setText("Not connected")
        _restyle(self._lbl_status)
        self._metrics = AudioMetrics()

    def _stop_analyzer(self) -> None:
        if self._analyzer is not None:
            try:
                self._analyzer.stop()
            except Exception:
                pass
            self._analyzer = None

    # ── audio callback ────────────────────────────────────────────────────────

    _REGISTRY_FIELDS = (
        "sub_bass", "bass", "low_mid", "mid", "high_mid",
        "treble", "energy", "brightness", "flux", "kick", "onset",
    )

    def _on_audio_frame(self, metrics: AudioMetrics) -> None:
        self._metrics = metrics
        if self._source_registry is not None:
            try:
                for field in self._REGISTRY_FIELDS:
                    self._source_registry.update(
                        f"audio.{field}", getattr(metrics, field, 0.0)
                    )
            except Exception:
                pass
        if self._extra_on_frame is not None:
            try:
                self._extra_on_frame(metrics)
            except Exception:
                pass

    # ── poll timer ────────────────────────────────────────────────────────────

    def _poll(self) -> None:
        m = self._metrics
        for attr, _label, _pulse in _METRICS:
            v = getattr(m, attr, 0.0)
            self._bars[attr].setValue(int(v * 1000))
            self._values[attr].setText(f"{v:.3f}")


# ── standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    app = QApplication.instance() or QApplication(sys.argv)
    panel = AudioPanel()
    panel.show()
    sys.exit(app.exec())
