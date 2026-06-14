"""
audio_panel.py — Audio input device selector + live metrics monitor

PySide6 panel for selecting an audio input device, viewing live AudioMetrics
values, and managing PropertyManager audio bindings.  Designed to sit alongside
property_manager.py and midi_panel.py.

Usage::

    from audio_metrics    import AudioAnalyzer, AudioMetrics
    from audio_panel      import AudioPanel
    from property_manager import PropertyManager, build_default_manager

    pm    = build_default_manager(_params, _controls, None, None, None)
    panel = AudioPanel(pm=pm)
    panel.show()

The panel can also run standalone (no pm required)::

    panel = AudioPanel()
    panel.show()
"""

from __future__ import annotations

import sounddevice as sd

from PySide6.QtCore    import QTimer, Qt
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QScrollArea,
    QFrame, QSizePolicy, QProgressBar,
)

from audio_metrics    import AudioAnalyzer, AudioMetrics
from property_manager import PropertyManager


# ── stylesheet — matches midi_panel.py ────────────────────────────────────────

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
QLabel#binding   { color: #a070d0; }

QComboBox {
    background-color: #111118;
    color: #5eaeff;
    border: 1px solid #38383f;
    border-radius: 3px;
    padding: 2px 6px;
    min-width: 180px;
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

QScrollArea, QScrollArea > QWidget > QWidget {
    background-color: #1a1a22;
    border: none;
}
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


# Ordered metric definitions: (attr, display label, is_pulse)
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

    Parameters
    ----------
    pm : PropertyManager | None
        If provided, ``pm.apply_audio(metrics)`` is called each frame and
        the current audio bindings are shown in the Bindings section.
    title : str
        Window title.
    extra_on_frame : callable | None
        Optional extra callback called from the audio thread each frame with
        the latest ``AudioMetrics``.  Use this to drive scene elements (e.g.
        ``circles.update_audio``) when the panel owns the only audio stream.
    """

    def __init__(
        self,
        pm             : PropertyManager | None = None,
        title          : str = "Audio Input",
        extra_on_frame = None,
    ):
        super().__init__()
        self.pm              = pm
        self._extra_on_frame = extra_on_frame
        self._analyzer: AudioAnalyzer | None = None
        self._metrics  = AudioMetrics()

        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setStyleSheet(_STYLESHEET)
        self.setMinimumWidth(380)
        self.resize(400, 600)

        self._build_ui()
        self._refresh_devices()

        # Poll at ~30 fps
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

        # Build one row per metric: [label] [bar] [value]
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

        # ── Audio Bindings (only when pm is supplied) ─────────────────────────
        if self.pm is not None:
            root.addSpacing(4)
            root.addWidget(_hdr("Audio Bindings"))
            root.addWidget(_sep())

            self._scroll = QScrollArea()
            self._scroll.setWidgetResizable(True)
            self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self._bind_container = QWidget()
            self._bind_layout    = QVBoxLayout(self._bind_container)
            self._bind_layout.setContentsMargins(0, 2, 0, 2)
            self._bind_layout.setSpacing(2)
            self._bind_layout.addStretch()
            self._scroll.setWidget(self._bind_container)
            root.addWidget(self._scroll, stretch=1)

            self._refresh_bindings()

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
            return  # placeholder item

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

    # ── audio callback (audio thread → shared metrics object) ─────────────────

    def _on_audio_frame(self, metrics: AudioMetrics) -> None:
        """Called from the sounddevice audio thread.  Only store the reference."""
        self._metrics = metrics
        # Apply to PM bindings — PropertyManager.set() is not thread-safe but
        # the race risk is negligible (float writes are atomic in CPython).
        if self.pm is not None:
            try:
                self.pm.apply_audio(metrics)
            except Exception:
                pass
        if self._extra_on_frame is not None:
            try:
                self._extra_on_frame(metrics)
            except Exception:
                pass

    # ── poll timer (Qt main thread) ───────────────────────────────────────────

    def _poll(self) -> None:
        m = self._metrics
        for attr, _label, _pulse in _METRICS:
            v = getattr(m, attr, 0.0)
            self._bars[attr].setValue(int(v * 1000))
            self._values[attr].setText(f"{v:.3f}")

    # ── bindings list ─────────────────────────────────────────────────────────

    def _refresh_bindings(self) -> None:
        if self.pm is None:
            return

        # Clear (leave trailing stretch)
        while self._bind_layout.count() > 1:
            item = self._bind_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        bindings = self.pm.audio_bindings()
        if not bindings:
            lbl = QLabel("No audio bindings registered.")
            lbl.setObjectName("info")
            self._bind_layout.insertWidget(0, lbl)
            return

        for i, b in enumerate(bindings):
            text = (
                f"{b.metric_attr:<12}  →  {b.prop_key}"
                f"  [{b.min_val:.4g} – {b.max_val:.4g}]"
            )
            lbl = QLabel(text)
            lbl.setObjectName("binding")
            lbl.setWordWrap(False)
            self._bind_layout.insertWidget(i, lbl)


# ── standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    app = QApplication.instance() or QApplication(sys.argv)
    panel = AudioPanel()
    panel.show()
    sys.exit(app.exec())
