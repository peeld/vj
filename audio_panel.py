"""
audio_panel.py — Audio input device selector + live metrics monitor

PySide6 panel for selecting an audio input device, viewing live AudioMetrics
values, and managing PropertyManager audio mappings.  Each mapping can:
  • "range"    — map the signal to a [min, max] range
  • "add"      — add (signal × scale) to the current property value each frame
  • "multiply" — multiply the current value by (signal × scale) each frame
  • "trigger"  — output 0 or 1 with hysteresis (on/off thresholds)

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
    QFrame, QSizePolicy, QProgressBar, QLineEdit,
    QDoubleSpinBox, QStackedWidget,
)

from audio_metrics    import AudioAnalyzer, AudioMetrics
from property_manager import PropertyManager, AudioBinding


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
QLabel#binding   { color: #a070d0; }
QLabel#sub       { color: #707078; font-size: 11px; }

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
QPushButton#add     { color: #7ec87e; border-color: #3a5a3a; }
QPushButton#add:hover { border-color: #7ec87e; }
QPushButton#remove  { color: #e07070; border-color: #5a2a2a; min-width: 20px;
                       padding: 2px 6px; }
QPushButton#remove:hover { border-color: #e07070; }

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
QFrame#form_box {
    background-color: #12121a;
    border: 1px solid #2e2e38;
    border-radius: 4px;
}
QLineEdit, QDoubleSpinBox {
    background-color: #111118;
    color: #5eaeff;
    border: 1px solid #38383f;
    border-radius: 3px;
    padding: 2px 5px;
}
QLineEdit:focus, QDoubleSpinBox:focus { border: 1px solid #5eaeff; }
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    width: 14px;
    background: #2a2a36;
    border: none;
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


def _dspin(value: float, lo: float, hi: float, step: float = 0.01,
           decimals: int = 3, width: int = 72) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    s.setValue(value)
    s.setFixedWidth(width)
    return s


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

_METRIC_ATTRS = [a for a, _, _ in _METRICS]
_METRIC_LABELS = {a: l for a, l, _ in _METRICS}

_MODES = ["range", "add", "multiply", "trigger"]
_MODE_LABELS = {
    "range":    "Range  (map signal → [min, max])",
    "add":      "Add    (value += signal × scale)",
    "multiply": "Multiply (value *= signal × scale)",
    "trigger":  "Trigger  (0/1 with hysteresis)",
}


# ── mapping form ──────────────────────────────────────────────────────────────

class _AddMappingForm(QFrame):
    """Inline form for composing a new AudioBinding."""

    def __init__(self, pm: PropertyManager, parent: QWidget = None):
        super().__init__(parent)
        self.pm = pm
        self.setObjectName("form_box")
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)

        # ── row 1: metric → property ──────────────────────────────────────────
        r1 = QHBoxLayout()
        r1.setSpacing(6)

        self._metric_combo = QComboBox()
        for attr in _METRIC_ATTRS:
            self._metric_combo.addItem(_METRIC_LABELS[attr], userData=attr)
        self._metric_combo.setFixedWidth(100)
        r1.addWidget(QLabel("Metric"))
        r1.addWidget(self._metric_combo)

        r1.addSpacing(8)
        r1.addWidget(QLabel("Property"))
        self._prop_combo = QComboBox()
        self._prop_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._prop_combo.setMinimumWidth(160)
        self._refresh_props()
        r1.addWidget(self._prop_combo)
        layout.addLayout(r1)

        # ── row 2: mode + scale ───────────────────────────────────────────────
        r2 = QHBoxLayout()
        r2.setSpacing(6)

        r2.addWidget(QLabel("Mode"))
        self._mode_combo = QComboBox()
        for m in _MODES:
            self._mode_combo.addItem(m, userData=m)
        self._mode_combo.setFixedWidth(100)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        r2.addWidget(self._mode_combo)

        r2.addSpacing(8)
        r2.addWidget(QLabel("Scale"))
        self._scale_spin = _dspin(1.0, -100.0, 100.0, step=0.1, decimals=3, width=72)
        r2.addWidget(self._scale_spin)
        r2.addStretch()
        layout.addLayout(r2)

        # ── row 3: mode-dependent params (stacked) ────────────────────────────
        self._stack = QStackedWidget()

        # page 0 — range
        page_range = QWidget()
        rr = QHBoxLayout(page_range)
        rr.setContentsMargins(0, 0, 0, 0)
        rr.setSpacing(6)
        rr.addWidget(QLabel("Min"))
        self._min_spin = _dspin(0.0, -1e6, 1e6, step=0.01, decimals=4, width=80)
        rr.addWidget(self._min_spin)
        rr.addSpacing(6)
        rr.addWidget(QLabel("Max"))
        self._max_spin = _dspin(1.0, -1e6, 1e6, step=0.01, decimals=4, width=80)
        rr.addWidget(self._max_spin)
        rr.addStretch()
        self._stack.addWidget(page_range)     # index 0

        # page 1 — add/multiply (no extra params beyond scale)
        page_addmul = QWidget()
        ra = QHBoxLayout(page_addmul)
        ra.setContentsMargins(0, 0, 0, 0)
        lbl_hint = QLabel("(no extra parameters — scale is applied to signal before operation)")
        lbl_hint.setObjectName("sub")
        ra.addWidget(lbl_hint)
        ra.addStretch()
        self._stack.addWidget(page_addmul)    # index 1

        # page 2 — trigger
        page_trig = QWidget()
        rt = QHBoxLayout(page_trig)
        rt.setContentsMargins(0, 0, 0, 0)
        rt.setSpacing(6)
        rt.addWidget(QLabel("On threshold"))
        self._on_spin  = _dspin(0.5, 0.0, 1.0, step=0.05, decimals=3, width=68)
        rt.addWidget(self._on_spin)
        rt.addSpacing(6)
        rt.addWidget(QLabel("Off threshold"))
        self._off_spin = _dspin(0.3, 0.0, 1.0, step=0.05, decimals=3, width=68)
        rt.addWidget(self._off_spin)
        rt.addStretch()
        self._stack.addWidget(page_trig)     # index 2

        layout.addWidget(self._stack)

        # ── row 4: add button ─────────────────────────────────────────────────
        r4 = QHBoxLayout()
        r4.addStretch()
        btn_add = QPushButton("＋  Add Mapping")
        btn_add.setObjectName("add")
        btn_add.clicked.connect(self._on_add)
        r4.addWidget(btn_add)
        layout.addLayout(r4)

        self._on_mode_changed()

    def _refresh_props(self) -> None:
        self._prop_combo.clear()
        if self.pm is None:
            return
        for prop in self.pm.all_props():
            if prop.type in (float, int, bool):
                self._prop_combo.addItem(
                    f"{prop.key}  ({prop.label})", userData=prop.key
                )

    def _on_mode_changed(self) -> None:
        mode = self._mode_combo.currentData()
        if mode == "range":
            self._stack.setCurrentIndex(0)
        elif mode in ("add", "multiply"):
            self._stack.setCurrentIndex(1)
        elif mode == "trigger":
            self._stack.setCurrentIndex(2)

    def _on_add(self) -> None:
        if self.pm is None:
            return
        metric_attr = self._metric_combo.currentData()
        prop_key    = self._prop_combo.currentData()
        if not metric_attr or not prop_key:
            return

        mode = self._mode_combo.currentData()
        b = AudioBinding(
            metric_attr  =metric_attr,
            prop_key     =prop_key,
            mode         =mode,
            scale        =self._scale_spin.value(),
            min_val      =self._min_spin.value(),
            max_val      =self._max_spin.value(),
            on_threshold =self._on_spin.value(),
            off_threshold=self._off_spin.value(),
        )
        self.pm.bind_audio(b)

        # bubble up to AudioPanel to refresh the list
        panel = self._find_audio_panel()
        if panel is not None:
            panel._refresh_bindings()

    def _find_audio_panel(self) -> "AudioPanel | None":
        w = self.parent()
        while w is not None:
            if isinstance(w, AudioPanel):
                return w
            w = w.parent()
        return None


# ── binding row ───────────────────────────────────────────────────────────────

class _BindingRow(QFrame):
    """One row in the bindings list, showing info + a remove button."""

    def __init__(self, binding: AudioBinding, index: int,
                 pm: PropertyManager, parent: QWidget = None):
        super().__init__(parent)
        self._binding = binding
        self._index   = index
        self._pm      = pm
        self._build()

    def _build(self) -> None:
        row = QHBoxLayout(self)
        row.setContentsMargins(4, 2, 4, 2)
        row.setSpacing(6)

        b = self._binding
        metric_lbl = _METRIC_LABELS.get(b.metric_attr, b.metric_attr)

        if b.mode == "range":
            detail = f"[{b.min_val:.4g} – {b.max_val:.4g}]  scale={b.scale:.3g}"
        elif b.mode in ("add", "multiply"):
            detail = f"scale={b.scale:.3g}"
        elif b.mode == "trigger":
            detail = f"on>{b.on_threshold:.3g}  off<{b.off_threshold:.3g}  scale={b.scale:.3g}"
        else:
            detail = ""

        text = f"{metric_lbl:<12} → {b.prop_key}   [{b.mode}]  {detail}"
        lbl  = QLabel(text)
        lbl.setObjectName("binding")
        lbl.setWordWrap(False)
        row.addWidget(lbl, stretch=1)

        btn = QPushButton("✕")
        btn.setObjectName("remove")
        btn.setFixedWidth(26)
        btn.clicked.connect(self._on_remove)
        row.addWidget(btn)

    def _on_remove(self) -> None:
        try:
            self._pm._audio_bindings.pop(self._index)
        except IndexError:
            pass
        panel = self._find_audio_panel()
        if panel is not None:
            panel._refresh_bindings()

    def _find_audio_panel(self) -> "AudioPanel | None":
        w = self.parent()
        while w is not None:
            if isinstance(w, AudioPanel):
                return w
            w = w.parent()
        return None


# ── panel ─────────────────────────────────────────────────────────────────────

class AudioPanel(QWidget):
    """
    Live audio input panel with mapping management.

    Parameters
    ----------
    pm : PropertyManager | None
        If provided, ``pm.apply_audio(metrics)`` is called each frame and
        mappings are shown and can be added/removed in the panel.
    title : str
        Window title.
    extra_on_frame : callable | None
        Optional extra callback called each frame with the latest AudioMetrics.
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
        self.setMinimumWidth(420)
        self.resize(460, 720)

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

        # ── Audio Mappings (only when pm is supplied) ─────────────────────────
        if self.pm is not None:
            root.addSpacing(4)
            root.addWidget(_hdr("Audio Mappings"))
            root.addWidget(_sep())

            # Add-mapping form
            self._form = _AddMappingForm(pm=self.pm, parent=self)
            root.addWidget(self._form)

            root.addSpacing(4)

            # Scrollable binding list
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

    def _on_audio_frame(self, metrics: AudioMetrics) -> None:
        self._metrics = metrics
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

    # ── poll timer ────────────────────────────────────────────────────────────

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
            lbl = QLabel("No mappings yet — use the form above to add one.")
            lbl.setObjectName("info")
            self._bind_layout.insertWidget(0, lbl)
            return

        for i, b in enumerate(bindings):
            row = _BindingRow(b, i, self.pm, parent=self._bind_container)
            self._bind_layout.insertWidget(i, row)


# ── standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    app = QApplication.instance() or QApplication(sys.argv)
    panel = AudioPanel()
    panel.show()
    sys.exit(app.exec())
