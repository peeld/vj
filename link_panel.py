"""
link_panel.py — Qt panel for the LinkManager signal routing layer.

PySide6 widget with 6 tabs:
  Channels     — expression-based continuous source → sink mappings
  Parameters   — event-driven stateful sources (p.* namespace)
  Events       — discrete event → action links; threshold detectors
  Envelopes    — ADSR envelopes triggered by events
  LFOs         — low-frequency oscillators
  Sources      — live read-only view of the source registry

Auto-saves to link_state.json (5 s debounce) on every change.
Auto-loads link_state.json on startup if the file exists.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer, Signal, QSortFilterProxyModel, QEvent
from PySide6.QtGui import QColor, QKeyEvent, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QLineEdit,
    QTableWidget, QTableWidgetItem, QCheckBox, QHeaderView,
    QTabWidget, QFrame, QDoubleSpinBox, QSpinBox,
    QCompleter, QAbstractItemView, QDialog, QDialogButtonBox, QInputDialog,
    QTableView, QStyledItemDelegate, QAbstractItemDelegate,
    QTreeWidget, QTreeWidgetItem,
)

from link_manager import (
    LinkManager, SignalLink, EnvelopeDef, LFODef, EventLink, ThresholdDef,
    ParameterDef, EVAL_MATH_NS, _flat_to_ns, KEY_NAMES,
)

if TYPE_CHECKING:
    from property_manager import PropertyManager


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
QLabel#value     { color: #5eaeff; }
QLabel#ok        { color: #7ec87e; }
QLabel#err       { color: #e07070; }
QLabel#info      { color: #707078; font-style: italic; }

QLineEdit {
    background-color: #111118;
    color: #5eaeff;
    border: 1px solid #38383f;
    border-radius: 3px;
    padding: 2px 6px;
}
QLineEdit:focus { border: 1px solid #5eaeff; }

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
QPushButton#remove  { color: #e07070; border-color: #5a2a2a; min-width: 20px; padding: 2px 6px; }
QPushButton#remove:hover { border-color: #e07070; }
QPushButton#trigger { color: #ffb347; border-color: #5a3a00; }
QPushButton#trigger:hover { border-color: #ffb347; }
QPushButton#pick { color: #5eaeff; border-color: #1e3050; min-width: 20px; padding: 1px 2px; }
QPushButton#pick:hover { border-color: #5eaeff; }

QTableWidget, QTableView {
    background-color: #111118;
    alternate-background-color: #141420;
    color: #c8c8d0;
    border: 1px solid #38383f;
    gridline-color: #2a2a38;
    selection-background-color: #1e3050;
    selection-color: #c8c8d0;
}
QTableWidget QHeaderView::section, QTableView QHeaderView::section {
    background-color: #1a1a22;
    color: #707078;
    border: none;
    border-bottom: 1px solid #38383f;
    padding: 3px 6px;
}
QTableWidget::item, QTableView::item { padding: 2px 4px; }

QTabWidget::pane {
    border: 1px solid #38383f;
    background-color: #1a1a22;
}
QTabBar::tab {
    background-color: #111118;
    color: #707078;
    border: 1px solid #38383f;
    border-bottom: none;
    padding: 4px 12px;
    margin-right: 2px;
}
QTabBar::tab:selected { background-color: #1a1a22; color: #c8c8d0; }
QTabBar::tab:hover:!selected { background-color: #1e1e2a; }

QScrollBar:vertical {
    background: #1a1a22; width: 6px; margin: 0;
}
QScrollBar::handle:vertical { background: #38383f; border-radius: 3px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

QDoubleSpinBox {
    background-color: #111118;
    color: #5eaeff;
    border: 1px solid #38383f;
    border-radius: 3px;
    padding: 2px 4px;
}
QDoubleSpinBox:focus { border: 1px solid #5eaeff; }
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    background-color: #2a2a36;
    border: none;
    width: 14px;
}

QCheckBox { color: #c8c8d0; spacing: 6px; }
QCheckBox::indicator {
    width: 13px; height: 13px;
    border: 1px solid #38383f;
    border-radius: 2px;
    background: #111118;
}
QCheckBox::indicator:checked { background: #5eaeff; border-color: #5eaeff; }

QFrame#sep {
    color: #2e2e38;
    max-height: 1px;
    background-color: #2e2e38;
}

QDialog { background-color: #1a1a22; }
QDialogButtonBox QPushButton { min-width: 70px; }
"""


# ── constants ─────────────────────────────────────────────────────────────────

_LFO_SHAPES = ("sine", "saw", "square", "tri")

_PARAMETER_KINDS = ("toggle", "gate", "latch", "pulse", "counter")

_SOURCE_GROUP_ORDER = ["audio", "midi", "clock", "lfo", "env", "p"]


# ── helper: completion lists ──────────────────────────────────────────────────

def _event_completions(lm: LinkManager) -> list[str]:
    srcs = ["audio.onset", "clock.beat"]
    for k in KEY_NAMES:
        srcs.append(f"key.{k}.press")
        srcs.append(f"key.{k}.release")
    for defn in lm._threshold_defs:
        srcs.append(f"audio.threshold.{defn.name}")
    for n in range(128):
        srcs.append(f"midi.note{n}.on")
        srcs.append(f"midi.note{n}.off")
    return srcs


def _expr_completions(lm: LinkManager) -> list[str]:
    keys = lm.source_registry.source_keys()
    math = sorted(EVAL_MATH_NS) + ["smooth", "dt", "const"]
    return keys + math


def _action_completions(lm: LinkManager, pm: "PropertyManager") -> list[str]:
    actions: list[str] = ["regen", "preset('')"]
    preset_names = lm.list_link_presets()
    if preset_names:
        for name in preset_names:
            actions.append(f"link_preset('{name}')")
    else:
        actions.append("link_preset('')")
    return actions


def _trigger_action_completions(lm: LinkManager) -> list[str]:
    names = lm.list_link_presets()
    return [f"link_preset('{n}')" for n in names] if names else ["link_preset('')"]


def _make_completer(words: list[str], parent=None) -> QCompleter:
    c = QCompleter(words, parent)
    c.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
    c.setFilterMode(Qt.MatchFlag.MatchContains)
    c.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
    return c


# ── MIDI / key capture helpers ────────────────────────────────────────────────

def _qt_key_attr(name: str) -> str:
    if name.startswith("NUMBER_"):
        return f"Key_{name[len('NUMBER_'):]}"
    if name == "TAB":
        return "Key_Tab"
    return f"Key_{name}"


# Qt key code → KEY_NAMES string, derived from link_manager.KEY_NAMES so this
# stays in sync with the canonical key list without a second list to drift.
_QT_KEY_TO_NAME: dict[int, str] = {
    getattr(Qt.Key, _qt_key_attr(name)): name for name in KEY_NAMES
}


class _KeyCaptureDialog(QDialog):
    """Modal that waits for one mapped key press and returns its event string."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Capture Key")
        self.setStyleSheet(_STYLESHEET)
        self.setFixedSize(300, 120)
        self._result: str | None = None

        lo = QVBoxLayout(self)
        self._lbl = QLabel("Press a key…")
        self._lbl.setObjectName("hdr")
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lo.addWidget(self._lbl)

        self._release_cb = QCheckBox("Capture release event (off)")
        lo.addWidget(self._release_cb)

        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        lo.addWidget(cancel)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        name = _QT_KEY_TO_NAME.get(event.key())
        if name:
            suffix = "release" if self._release_cb.isChecked() else "press"
            self._result = f"key.{name}.{suffix}"
            self.accept()
        elif event.key() == Qt.Key.Key_Escape:
            self.reject()

    def captured_event(self) -> str | None:
        return self._result


class _ChoiceCaptureDialog(QDialog):
    """Modal: pick an enum sink, then one of its valid choice values."""

    def __init__(self, pm: "PropertyManager", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pick Sink + Choice")
        self.setStyleSheet(_STYLESHEET)
        self.setMinimumWidth(360)

        self._enum_defs = [d for d in pm.all_props() if d.choices]

        lo = QVBoxLayout(self)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Sink:"))
        self._sink = QComboBox()
        self._sink.addItems([d.key for d in self._enum_defs])
        self._sink.currentIndexChanged.connect(self._refresh_choices)
        row1.addWidget(self._sink, 1)
        lo.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Choice:"))
        self._choice = QComboBox()
        row2.addWidget(self._choice, 1)
        lo.addLayout(row2)

        self._refresh_choices()

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        lo.addWidget(btns)

    def _refresh_choices(self) -> None:
        self._choice.clear()
        idx = self._sink.currentIndex()
        if 0 <= idx < len(self._enum_defs):
            self._choice.addItems([str(c) for c in self._enum_defs[idx].choices])

    def result(self) -> tuple[str, str] | None:
        idx = self._sink.currentIndex()
        if idx < 0 or idx >= len(self._enum_defs) or self._choice.currentIndex() < 0:
            return None
        return (self._enum_defs[idx].key, self._choice.currentText())


def _eval_preview(expr: str, lm: LinkManager) -> tuple[str, bool]:
    """Return (display_text, is_error)."""
    if not expr.strip():
        return ("", False)
    try:
        snap = lm.source_registry.snapshot()
        ns = {**_flat_to_ns(snap), **EVAL_MATH_NS, "dt": 0.016, "const": lm._const_ns}
        ns.setdefault("smooth", lambda x, tau=0.0: x)
        result = eval(expr, {"__builtins__": {}}, ns)
        if isinstance(result, float):
            return (f"→ {result:.4f}", False)
        return (f"→ {result!r}", False)
    except Exception as exc:
        return (f"error: {exc}", True)


# ── helper: table cell ────────────────────────────────────────────────────────

def _cell(text: str) -> QTableWidgetItem:
    item = QTableWidgetItem(str(text))
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item


def _bool_cell(flag: bool) -> QTableWidgetItem:
    item = _cell("✓" if flag else "✗")
    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
    return item


# ─────────────────────────────────────────────────────────────────────────────
#  Dialogs
# ─────────────────────────────────────────────────────────────────────────────

class _BaseDialog(QDialog):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setStyleSheet(_STYLESHEET)
        self.setMinimumWidth(460)
        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(8)
        self._layout.setContentsMargins(12, 12, 12, 12)

    def _row(self, label: str, widget: QWidget) -> QHBoxLayout:
        r = QHBoxLayout()
        lbl = QLabel(label)
        lbl.setMinimumWidth(110)
        r.addWidget(lbl)
        r.addWidget(widget, 1)
        self._layout.addLayout(r)
        return r

    def _add_buttons(self) -> None:
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        self._layout.addWidget(btns)

    @staticmethod
    def _spinbox(lo: float, hi: float, val: float,
                 decimals: int = 3, step: float = 0.01) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setSingleStep(step)
        sb.setDecimals(decimals)
        sb.setValue(val)
        return sb


# ── Event Link ────────────────────────────────────────────────────────────────

class EventLinkDialog(_BaseDialog):
    def __init__(self, lm: LinkManager, pm: "PropertyManager",
                 link: EventLink | None = None, parent=None,
                 action_completions: list[str] | None = None):
        super().__init__("Event Link" if link is None else "Edit Event Link", parent)

        self._event = QLineEdit()
        self._event.setPlaceholderText("midi.note36.on")
        if link:
            self._event.setText(link.event)
        self._event.setCompleter(_make_completer(_event_completions(lm), self))
        self._row("Event:", self._event)

        self._action = QLineEdit()
        self._action.setPlaceholderText("regen")
        if link:
            self._action.setText(link.action)
        actions = action_completions if action_completions is not None else _action_completions(lm, pm)
        self._action.setCompleter(_make_completer(actions, self))
        self._row("Action:", self._action)

        self._condition = QLineEdit()
        self._condition.setPlaceholderText("(optional)  midi.cc7 > 0.5")
        if link and link.condition:
            self._condition.setText(link.condition)
        self._condition.setCompleter(_make_completer(_expr_completions(lm), self))
        self._row("Condition:", self._condition)

        self._enabled = QCheckBox("Enabled")
        self._enabled.setChecked(link.enabled if link else True)
        self._layout.addWidget(self._enabled)

        self._add_buttons()

    def result_link(self) -> EventLink:
        return EventLink(
            event     = self._event.text().strip(),
            action    = self._action.text().strip(),
            enabled   = self._enabled.isChecked(),
            condition = self._condition.text().strip() or None,
        )


# ── Envelope ──────────────────────────────────────────────────────────────────

class EnvelopeDialog(_BaseDialog):
    def __init__(self, lm: LinkManager, defn: EnvelopeDef | None = None, parent=None):
        super().__init__("Envelope" if defn is None else "Edit Envelope", parent)

        self._name = QLineEdit()
        self._name.setPlaceholderText("kick")
        if defn:
            self._name.setText(defn.name)
        self._row("Name:", self._name)

        self._trigger = QLineEdit()
        self._trigger.setPlaceholderText("audio.onset")
        if defn:
            self._trigger.setText(defn.trigger)
        self._trigger.setCompleter(_make_completer(_event_completions(lm), self))
        self._row("Trigger:", self._trigger)

        self._gate_off = QLineEdit()
        self._gate_off.setPlaceholderText("(optional)  midi.note36.off")
        if defn and defn.gate_off:
            self._gate_off.setText(defn.gate_off)
        self._gate_off.setCompleter(_make_completer(_event_completions(lm), self))
        self._row("Gate Off:", self._gate_off)

        self._attack  = self._spinbox(0.0, 30.0, defn.attack  if defn else 0.01,  step=0.001)
        self._decay   = self._spinbox(0.0, 30.0, defn.decay   if defn else 0.1,   step=0.01)
        self._sustain = self._spinbox(0.0, 1.0,  defn.sustain if defn else 0.0,   step=0.01)
        self._release = self._spinbox(0.0, 30.0, defn.release if defn else 0.2,   step=0.01)
        self._peak    = self._spinbox(0.0, 1.0,  defn.peak    if defn else 1.0,   step=0.01)

        self._row("Attack (s):",  self._attack)
        self._row("Decay (s):",   self._decay)
        self._row("Sustain:",     self._sustain)
        self._row("Release (s):", self._release)
        self._row("Peak:",        self._peak)

        self._add_buttons()

    def result_def(self) -> EnvelopeDef:
        return EnvelopeDef(
            name     = self._name.text().strip(),
            trigger  = self._trigger.text().strip(),
            gate_off = self._gate_off.text().strip() or None,
            attack   = self._attack.value(),
            decay    = self._decay.value(),
            sustain  = self._sustain.value(),
            release  = self._release.value(),
            peak     = self._peak.value(),
        )


# ── LFO ──────────────────────────────────────────────────────────────────────

class LFODialog(_BaseDialog):
    def __init__(self, defn: LFODef | None = None, parent=None):
        super().__init__("LFO" if defn is None else "Edit LFO", parent)
        self.setMinimumWidth(360)

        self._name = QLineEdit()
        self._name.setPlaceholderText("slow_sine")
        if defn:
            self._name.setText(defn.name)
        self._row("Name:", self._name)

        self._shape = QComboBox()
        self._shape.addItems(list(_LFO_SHAPES))
        if defn:
            idx = self._shape.findText(defn.shape)
            if idx >= 0:
                self._shape.setCurrentIndex(idx)
        self._row("Shape:", self._shape)

        self._rate = self._spinbox(0.001, 1000.0, defn.rate_hz if defn else 1.0,
                                   decimals=3, step=0.1)
        self._row("Rate (Hz):", self._rate)

        self._phase = self._spinbox(0.0, 1.0, defn.phase if defn else 0.0,
                                    decimals=3, step=0.01)
        self._row("Phase (0-1):", self._phase)

        self._add_buttons()

    def result_def(self) -> LFODef:
        return LFODef(
            name    = self._name.text().strip(),
            shape   = self._shape.currentText(),
            rate_hz = self._rate.value(),
            phase   = self._phase.value(),
        )


# ── Threshold ─────────────────────────────────────────────────────────────────

class ThresholdDialog(_BaseDialog):
    def __init__(self, lm: LinkManager, defn: ThresholdDef | None = None, parent=None):
        super().__init__("Threshold" if defn is None else "Edit Threshold", parent)
        self.setMinimumWidth(380)

        self._name = QLineEdit()
        self._name.setPlaceholderText("bass_hit")
        if defn:
            self._name.setText(defn.name)
        self._row("Name:", self._name)

        self._source = QComboBox()
        self._source.setEditable(True)
        raw_keys = lm.source_registry.source_keys()
        primary = [k for k in raw_keys
                   if not k.endswith("_smooth") and not k.endswith("_peak")]
        self._source.addItems(primary or raw_keys)
        if defn:
            idx = self._source.findText(defn.source)
            if idx >= 0:
                self._source.setCurrentIndex(idx)
            else:
                self._source.setCurrentText(defn.source)
        self._row("Source:", self._source)

        self._high = self._spinbox(0.0, 1.0, defn.high if defn else 0.7)
        self._row("High:", self._high)

        self._low = self._spinbox(0.0, 1.0, defn.low if defn else 0.3)
        self._row("Low:", self._low)

        self._min_interval = self._spinbox(0.0, 60.0,
                                           defn.min_interval_s if defn else 0.05,
                                           decimals=3, step=0.01)
        self._row("Min Interval (s):", self._min_interval)

        self._add_buttons()

    def result_def(self) -> ThresholdDef:
        return ThresholdDef(
            name           = self._name.text().strip(),
            source         = self._source.currentText().strip(),
            high           = self._high.value(),
            low            = self._low.value(),
            min_interval_s = self._min_interval.value(),
        )


# ── Parameter ─────────────────────────────────────────────────────────────────

class ParameterDialog(_BaseDialog):
    def __init__(self, lm: LinkManager, defn: ParameterDef | None = None, parent=None):
        super().__init__("Parameter" if defn is None else "Edit Parameter", parent)
        self.setMinimumWidth(420)

        self._name = QLineEdit()
        self._name.setPlaceholderText("scene_toggle")
        if defn:
            self._name.setText(defn.name)
        self._row("Name:", self._name)

        self._kind = QComboBox()
        self._kind.addItems(list(_PARAMETER_KINDS))
        if defn:
            idx = self._kind.findText(defn.kind)
            if idx >= 0:
                self._kind.setCurrentIndex(idx)
        self._row("Kind:", self._kind)

        self._trigger = QLineEdit()
        self._trigger.setPlaceholderText("audio.onset")
        if defn:
            self._trigger.setText(defn.trigger)
        self._trigger.setCompleter(_make_completer(_event_completions(lm), self))
        self._row("Trigger:", self._trigger)

        self._off_event = QLineEdit()
        self._off_event.setPlaceholderText("midi.note36.off")
        if defn and defn.off_event:
            self._off_event.setText(defn.off_event)
        self._off_event.setCompleter(_make_completer(_event_completions(lm), self))
        self._off_event_widget = self._cond_row("Off Event:", self._off_event)

        self._wrap_at = QSpinBox()
        self._wrap_at.setRange(2, 64)
        self._wrap_at.setValue(defn.wrap_at if defn else 8)
        self._wrap_at_widget = self._cond_row("Wrap At:", self._wrap_at)

        self._pulse_ms = QDoubleSpinBox()
        self._pulse_ms.setRange(1.0, 10000.0)
        self._pulse_ms.setSingleStep(10.0)
        self._pulse_ms.setDecimals(1)
        self._pulse_ms.setValue(defn.pulse_ms if defn else 100.0)
        self._pulse_ms_widget = self._cond_row("Pulse (ms):", self._pulse_ms)

        self._snap_n = QSpinBox()
        self._snap_n.setRange(0, 16)
        self._snap_n.setValue(defn.snap_n if defn else 0)
        self._snap_n_widget = self._cond_row("Snap N:", self._snap_n)

        self._add_buttons()

        self._kind.currentTextChanged.connect(self._update_visibility)
        self._update_visibility(self._kind.currentText())

    def _cond_row(self, label: str, widget: QWidget) -> QWidget:
        container = QWidget()
        r = QHBoxLayout(container)
        r.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label)
        lbl.setMinimumWidth(110)
        r.addWidget(lbl)
        r.addWidget(widget, 1)
        self._layout.addWidget(container)
        return container

    def _update_visibility(self, kind: str) -> None:
        self._off_event_widget.setVisible(kind in ("gate", "latch"))
        self._wrap_at_widget.setVisible(kind == "counter")
        self._pulse_ms_widget.setVisible(kind == "pulse")
        self._snap_n_widget.setVisible(kind == "counter")

    def result_def(self) -> ParameterDef:
        return ParameterDef(
            name      = self._name.text().strip(),
            kind      = self._kind.currentText(),
            trigger   = self._trigger.text().strip(),
            off_event = self._off_event.text().strip() or None,
            wrap_at   = self._wrap_at.value(),
            pulse_ms  = self._pulse_ms.value(),
            snap_n    = self._snap_n.value(),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Source picker popup
# ─────────────────────────────────────────────────────────────────────────────

class _SourcePickerPopup(QFrame):
    """Popup for picking a source key to insert as a channel expression."""

    def __init__(self, snapshot: dict, callback, parent=None):
        super().__init__(parent, Qt.WindowType.Popup)
        self._callback = callback

        self.setStyleSheet(_STYLESHEET)
        self.setFrameShape(QFrame.Shape.StyledPanel)

        lo = QVBoxLayout(self)
        lo.setContentsMargins(4, 4, 4, 4)
        lo.setSpacing(4)

        self._filter = QLineEdit()
        self._filter.setPlaceholderText("filter…")
        lo.addWidget(self._filter)

        self._tree = QTreeWidget()
        self._tree.setHeaderHidden(True)
        self._tree.setRootIsDecorated(True)
        lo.addWidget(self._tree)

        self._build_tree(snapshot)

        self._filter.textChanged.connect(self._apply_filter)
        self._tree.itemClicked.connect(self._on_item_clicked)

    def _build_tree(self, snapshot: dict) -> None:
        groups: dict[str, list[str]] = {}
        for key in sorted(snapshot.keys()):
            if key.endswith("_smooth") or key.endswith("_peak"):
                continue
            prefix = key.split(".")[0] if "." in key else "other"
            groups.setdefault(prefix, []).append(key)

        ordered = [g for g in _SOURCE_GROUP_ORDER if g in groups]
        rest = sorted(k for k in groups if k not in _SOURCE_GROUP_ORDER and k != "other")
        if "other" in groups:
            rest.append("other")
        ordered.extend(rest)

        for group_name in ordered:
            group_item = QTreeWidgetItem(self._tree, [group_name])
            group_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            for key in groups[group_name]:
                short = key[len(group_name) + 1:] if key.startswith(group_name + ".") else key
                val = snapshot.get(key, 0.0)
                child = QTreeWidgetItem(group_item, [f"{short}  {val:.4f}"])
                child.setData(0, Qt.ItemDataRole.UserRole, key)

        self._tree.expandAll()

    def _apply_filter(self, text: str) -> None:
        text = text.lower()
        for gi in range(self._tree.topLevelItemCount()):
            group_item = self._tree.topLevelItem(gi)
            any_visible = False
            for ci in range(group_item.childCount()):
                child = group_item.child(ci)
                full_key = child.data(0, Qt.ItemDataRole.UserRole) or ""
                visible = not text or text in full_key.lower()
                child.setHidden(not visible)
                if visible:
                    any_visible = True
            group_item.setHidden(not any_visible)

    def _on_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        key = item.data(0, Qt.ItemDataRole.UserRole)
        if key:
            self._callback(key)
            self.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Tab widgets
# ─────────────────────────────────────────────────────────────────────────────

def _make_table(col_labels: list[str]) -> QTableWidget:
    t = QTableWidget(0, len(col_labels))
    t.setHorizontalHeaderLabels(col_labels)
    t.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    t.setAlternatingRowColors(True)
    t.verticalHeader().hide()
    t.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    t.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
    return t


# ── Sources ───────────────────────────────────────────────────────────────────

class SourcesTab(QWidget):
    def __init__(self, lm: LinkManager, parent=None):
        super().__init__(parent)
        self._lm = lm

        lo = QVBoxLayout(self)
        lo.setContentsMargins(4, 4, 4, 4)

        info = QLabel("Live source registry — read-only  (~10 fps)")
        info.setObjectName("info")
        lo.addWidget(info)

        self._table = _make_table(["Key", "Value"])
        self._table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents)
        lo.addWidget(self._table)

        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    def _refresh(self) -> None:
        snap = self._lm.source_registry.snapshot()
        keys = sorted(snap)
        self._table.setRowCount(len(keys))
        for r, k in enumerate(keys):
            self._table.setItem(r, 0, _cell(k))
            self._table.setItem(r, 1, _cell(f"{snap[k]:.4f}"))


# ── Channels model ───────────────────────────────────────────────────────────

# Live Value / Expression foreground colors, matching the QLabel#value/#info/#err
# stylesheet colors (link_panel.py:53-56) and the default QWidget text color.
_FG_DEFAULT = QColor("#c8c8d0")
_FG_BLUE    = QColor("#5eaeff")
_FG_GREY    = QColor("#707078")
_FG_RED     = QColor("#e07070")


class ChannelsModel(QStandardItemModel):
    """QStandardItemModel backing the property-bay Channels table.

    Columns: 0=On  1=Channel  2=Default  3=Expression  4=Live Value
    """

    COL_ON   = 0
    COL_KEY  = 1
    COL_DEF  = 2
    COL_EXPR = 3
    COL_LIVE = 4
    COL_PICK = 5

    def __init__(self, parent=None):
        super().__init__(0, 6, parent)
        self.setHorizontalHeaderLabels(["✓", "Channel", "Default", "Expression", "Live Value", ""])
        self.horizontalHeaderItem(0).setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.row_for_key: dict[str, int] = {}

    def populate(self, lm: "LinkManager", pm: "PropertyManager") -> None:
        self.setRowCount(0)
        self.row_for_key.clear()
        link_map = {l.sink_key: l for l in lm._signal_links}

        for row, prop in enumerate(pm.all_props()):
            key  = prop.key
            link = link_map.get(key)
            expr = link.expression if link else ""

            # Col 0: On checkbox — UserCheckable only when an expression exists
            on_item = QStandardItem()
            on_item.setCheckState(
                Qt.CheckState.Checked
                if (link and link.enabled and expr)
                else Qt.CheckState.Unchecked
            )
            base_flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            on_item.setFlags(
                base_flags | Qt.ItemFlag.ItemIsUserCheckable if expr else base_flags
            )

            # Col 1: Key — read-only
            key_item = QStandardItem(key)
            key_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)

            # Col 2: Default — editable, seeded from baseline or property default
            stored = lm.get_baseline(key, prop.default)
            if isinstance(stored, bool):
                def_text = str(stored)
            else:
                try:
                    def_text = f"{float(stored):.4f}"
                except (TypeError, ValueError):
                    def_text = str(stored)
            def_item = QStandardItem(def_text)
            def_item.setForeground(QColor("#5eaeff"))
            def_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEditable
            )

            # Col 3: Expression — editable, seeded from active link or ""
            expr_item = QStandardItem(expr)
            expr_item.setFlags(
                Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsEditable
            )

            # Col 4: Live Value — read-only, populated by the refresh timer
            live_item = QStandardItem("")
            live_item.setForeground(QColor("#5eaeff"))
            live_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)

            # Col 5: Pick — placeholder; button set via setIndexWidget
            pick_item = QStandardItem("")
            pick_item.setFlags(Qt.ItemFlag.ItemIsEnabled)

            self.appendRow([on_item, key_item, def_item, expr_item, live_item, pick_item])
            self.row_for_key[key] = row


def _make_channels_proxy(model: ChannelsModel) -> QSortFilterProxyModel:
    """Create a QSortFilterProxyModel over model, filtering on the Key column."""
    proxy = QSortFilterProxyModel()
    proxy.setSourceModel(model)
    proxy.setFilterKeyColumn(ChannelsModel.COL_KEY)
    proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
    return proxy


# ── Channels delegate ────────────────────────────────────────────────────────

class _ChannelsDelegate(QStyledItemDelegate):
    """Custom delegate: persistent QCheckBox for bool Default cells,
    persistent QComboBox for choice-type Default cells."""

    previewChanged = Signal(str, bool)
    previewEnded = Signal()

    def __init__(self, model: ChannelsModel, proxy: "QSortFilterProxyModel",
                 lm: LinkManager, pm: "PropertyManager", parent=None):
        super().__init__(parent)
        self._model = model
        self._proxy = proxy
        self._lm    = lm
        self._pm    = pm

    def _prop_defn_for(self, proxy_index):
        src = self._proxy.mapToSource(proxy_index)
        if src.column() != ChannelsModel.COL_DEF:
            return None
        key_item = self._model.item(src.row(), ChannelsModel.COL_KEY)
        if key_item is None:
            return None
        return self._pm._defs.get(key_item.text())

    def _choices_for(self, proxy_index) -> list | None:
        defn = self._prop_defn_for(proxy_index)
        return [str(c) for c in defn.choices] if (defn and defn.choices) else None

    def _is_bool_for(self, proxy_index) -> bool:
        defn = self._prop_defn_for(proxy_index)
        return defn is not None and defn.type is bool

    def createEditor(self, parent, option, index):
        src = self._proxy.mapToSource(index)
        if src.column() == ChannelsModel.COL_EXPR:
            editor = QLineEdit(parent)
            editor.setCompleter(_make_completer(_expr_completions(self._lm), editor))
            editor.textChanged.connect(self._emit_preview)
            editor.destroyed.connect(lambda *_: self.previewEnded.emit())
            return editor
        if self._is_bool_for(index):
            cb = QCheckBox(parent)
            cb.clicked.connect(lambda _checked: self.commitData.emit(cb))
            return cb
        choices = self._choices_for(index)
        if choices is not None:
            combo = QComboBox(parent)
            combo.addItems(choices)
            combo.activated.connect(lambda _: self.commitData.emit(combo))
            return combo
        return super().createEditor(parent, option, index)

    def setEditorData(self, editor, index):
        if isinstance(editor, QCheckBox):
            src  = self._proxy.mapToSource(index)
            item = self._model.itemFromIndex(src)
            if item:
                editor.setChecked(item.text().lower() in ("true", "1", "yes", "on"))
        elif isinstance(editor, QComboBox):
            src  = self._proxy.mapToSource(index)
            item = self._model.itemFromIndex(src)
            if item:
                i = editor.findText(item.text())
                editor.setCurrentIndex(max(i, 0))
        else:
            super().setEditorData(editor, index)

    def setModelData(self, editor, model, index):
        if isinstance(editor, QCheckBox):
            model.setData(index, str(editor.isChecked()), Qt.ItemDataRole.EditRole)
        elif isinstance(editor, QComboBox):
            model.setData(index, editor.currentText(), Qt.ItemDataRole.EditRole)
        else:
            super().setModelData(editor, model, index)

    def _emit_preview(self, text: str) -> None:
        display, is_error = _eval_preview(text, self._lm)
        self.previewChanged.emit(display, is_error)


# ── Channels ─────────────────────────────────────────────────────────────────

class ChannelsTab(QWidget):
    changed = Signal()

    def __init__(self, lm: LinkManager, pm: "PropertyManager", parent=None):
        super().__init__(parent)
        self._lm = lm
        self._pm = pm
        self._updating = False  # re-entrancy guard for _on_item_changed

        lo = QVBoxLayout(self)
        lo.setContentsMargins(4, 4, 4, 4)

        self._model = ChannelsModel(self)
        self._proxy = _make_channels_proxy(self._model)

        self._view = QTableView()
        self._view.setModel(self._proxy)
        self._view.setAlternatingRowColors(True)
        self._view.verticalHeader().hide()
        self._view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._view.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self._delegate = _ChannelsDelegate(self._model, self._proxy, lm, pm, self._view)
        self._delegate.previewChanged.connect(self._on_expr_preview_changed)
        self._delegate.previewEnded.connect(self._on_expr_preview_ended)
        self._view.setItemDelegate(self._delegate)
        hh = self._view.horizontalHeader()
        hh.setSectionResizeMode(ChannelsModel.COL_ON,   QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(ChannelsModel.COL_KEY,  QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(ChannelsModel.COL_DEF,  QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(ChannelsModel.COL_EXPR, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(ChannelsModel.COL_LIVE, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(ChannelsModel.COL_PICK, QHeaderView.ResizeMode.Fixed)
        hh.resizeSection(ChannelsModel.COL_PICK, 26)

        lo.addLayout(self._build_filter_bar())
        lo.addWidget(self._view)

        self._hint_default_text = (
            "Double-click Expression to edit · ⊕ to assign a source · "
            "Bool/choice defaults update on click. · "
            "Live Value: red = eval error · grey = disabled-row preview"
        )
        self._hint = QLabel(self._hint_default_text)
        self._hint.setObjectName("info")
        lo.addWidget(self._hint)

        self._live_timer = QTimer(self)
        self._live_timer.setInterval(100)
        self._live_timer.timeout.connect(self._refresh_live_values)
        self._live_timer.start()

        self._model.itemChanged.connect(self._on_item_changed)

        self._rebuild()

    def _rebuild(self) -> None:
        self._close_persistent_editors()
        self._updating = True
        try:
            self._model.populate(self._lm, self._pm)
        finally:
            self._updating = False
        self._open_persistent_editors()

    def _close_persistent_editors(self) -> None:
        for row in range(self._model.rowCount()):
            proxy_def = self._proxy.mapFromSource(
                self._model.index(row, ChannelsModel.COL_DEF)
            )
            if proxy_def.isValid() and self._view.isPersistentEditorOpen(proxy_def):
                self._view.closePersistentEditor(proxy_def)
            proxy_pick = self._proxy.mapFromSource(
                self._model.index(row, ChannelsModel.COL_PICK)
            )
            if proxy_pick.isValid():
                self._view.setIndexWidget(proxy_pick, None)

    def _open_persistent_editors(self) -> None:
        for key, row in self._model.row_for_key.items():
            defn = self._pm._defs.get(key)
            if defn is not None and (defn.type is bool or defn.choices):
                src_index = self._model.index(row, ChannelsModel.COL_DEF)
                proxy_index = self._proxy.mapFromSource(src_index)
                if proxy_index.isValid():
                    self._view.openPersistentEditor(proxy_index)
            btn = QPushButton("⊕")
            btn.setObjectName("pick")
            btn.setFixedSize(22, 20)
            btn.clicked.connect(lambda checked=False, sk=key, b=btn: self._open_picker(b, sk))
            src_pick = self._model.index(row, ChannelsModel.COL_PICK)
            proxy_pick = self._proxy.mapFromSource(src_pick)
            if proxy_pick.isValid():
                self._view.setIndexWidget(proxy_pick, btn)

    def _open_picker(self, anchor: QPushButton, sink_key: str) -> None:
        snap = self._lm.source_registry.snapshot()

        def on_select(source_key: str) -> None:
            row = self._model.row_for_key.get(sink_key)
            if row is None:
                return
            expr_item = self._model.item(row, ChannelsModel.COL_EXPR)
            if expr_item is not None:
                expr_item.setText(source_key)

        popup = _SourcePickerPopup(snap, on_select)
        pos = anchor.mapToGlobal(anchor.rect().bottomLeft())
        popup.move(pos)
        popup.resize(280, 320)
        popup.show()

    def _refresh_live_values(self) -> None:
        """Update the Live Value column; skips unchanged cells to suppress repaints."""
        if not hasattr(self, "_model"):
            return
        self._updating = True
        try:
            for key, row in self._model.row_for_key.items():
                expr_item = self._model.item(row, ChannelsModel.COL_EXPR)
                live_item = self._model.item(row, ChannelsModel.COL_LIVE)
                expr_text = expr_item.text().strip() if expr_item is not None else ""

                if not expr_text:
                    try:
                        new_text = f"{self._pm.get(key):.4f}"
                    except (TypeError, ValueError):
                        new_text = str(self._pm.get(key))
                    self._set_cell_style(live_item, new_text, _FG_BLUE, "")
                    self._set_cell_style(expr_item, None, _FG_DEFAULT, "")
                    continue

                display, is_error = _eval_preview(expr_text, self._lm)
                if is_error:
                    self._set_cell_style(live_item, "⚠ error", _FG_RED, display)
                    self._set_cell_style(expr_item, None, _FG_RED, display)
                    continue

                value_text = display[2:] if display.startswith("→ ") else display
                on_item = self._model.item(row, ChannelsModel.COL_ON)
                enabled = on_item is not None and on_item.checkState() == Qt.CheckState.Checked
                if enabled:
                    self._set_cell_style(live_item, value_text, _FG_BLUE, "")
                else:
                    self._set_cell_style(
                        live_item, value_text, _FG_GREY,
                        "Disabled — preview of what this link would evaluate to if enabled.",
                    )
                self._set_cell_style(expr_item, None, _FG_DEFAULT, "")
        finally:
            self._updating = False

    @staticmethod
    def _set_cell_style(item: "QStandardItem | None", text: "str | None", color: QColor, tooltip: str) -> None:
        """Apply text/foreground/tooltip to item, skipping writes that wouldn't change anything
        (text=None leaves the item's text untouched -- used for the Expression cell, which this
        is only ever tinting, not retexting)."""
        if item is None:
            return
        if text is not None and item.text() != text:
            item.setText(text)
        if item.foreground().color() != color:
            item.setForeground(color)
        if item.toolTip() != tooltip:
            item.setToolTip(tooltip)

    def _on_expr_preview_changed(self, display: str, is_error: bool) -> None:
        self._hint.setText(display or self._hint_default_text)
        self._hint.setObjectName("err" if is_error else "info")
        self._hint.style().unpolish(self._hint)
        self._hint.style().polish(self._hint)

    def _on_expr_preview_ended(self) -> None:
        self._hint.setText(self._hint_default_text)
        self._hint.setObjectName("info")
        self._hint.style().unpolish(self._hint)
        self._hint.style().polish(self._hint)

    # ── Filter bar (layout inserted by Stage 6) ───────────────────────────────

    def _build_filter_bar(self) -> "QHBoxLayout":
        """Create the filter QLineEdit + ✕ button; returns the row layout for Stage 6."""
        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("filter properties…")
        self._filter_edit.textChanged.connect(self._apply_filter)
        self._filter_edit.installEventFilter(self)

        clear_btn = QPushButton("✕")
        clear_btn.setFixedWidth(24)
        clear_btn.setObjectName("remove")
        clear_btn.setToolTip("Clear filter")
        clear_btn.clicked.connect(self._filter_edit.clear)

        row = QHBoxLayout()
        row.setSpacing(4)
        row.addWidget(self._filter_edit, 1)
        row.addWidget(clear_btn)
        return row

    def _apply_filter(self, text: str) -> None:
        if hasattr(self, "_proxy"):
            self._proxy.setFilterFixedString(text)

    def eventFilter(self, obj, event) -> bool:
        if hasattr(self, "_filter_edit") and obj is self._filter_edit:
            if (event.type() == QEvent.Type.KeyPress
                    and event.key() == Qt.Key.Key_Escape):
                self._filter_edit.clear()
                return True
        return super().eventFilter(obj, event)

    # ── itemChanged handler (wired to self._model in Stage 6) ────────────────

    def _on_item_changed(self, item: "QStandardItem") -> None:
        """Commit inline edits from the property-bay table to lm / pm."""
        if self._updating:
            return

        col = item.column()
        row = item.row()
        key_item = self._model.item(row, ChannelsModel.COL_KEY)
        if key_item is None:
            return
        key = key_item.text()
        link_map = {l.sink_key: l for l in self._lm._signal_links}

        if col == ChannelsModel.COL_EXPR:
            new_expr = item.text().strip()
            link     = link_map.get(key)
            on_item  = self._model.item(row, ChannelsModel.COL_ON)

            if new_expr:
                if link:
                    if new_expr != link.expression and hasattr(link, "smooth_helper"):
                        link.smooth_helper._states.clear()
                    link.expression = new_expr
                else:
                    self._lm.add_link(
                        SignalLink(sink_key=key, expression=new_expr, enabled=True)
                    )
                self._updating = True
                on_item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsUserCheckable
                )
                if on_item.checkState() == Qt.CheckState.Unchecked:
                    on_item.setCheckState(Qt.CheckState.Checked)
                self._updating = False
            else:
                self._lm.remove_link(key)
                self._lm.apply_baseline(key, self._pm)
                self._updating = True
                on_item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                )
                on_item.setCheckState(Qt.CheckState.Unchecked)
                self._updating = False

            self.changed.emit()

        elif col == ChannelsModel.COL_DEF:
            text = item.text().strip()
            defn = self._pm._defs.get(key)

            val = None
            parsed = False

            if defn is not None and defn.type is bool:
                if text.lower() in ("true", "1", "yes", "on"):
                    val, parsed = True, True
                elif text.lower() in ("false", "0", "no", "off"):
                    val, parsed = False, True
            elif defn is not None and defn.choices and text in [str(c) for c in defn.choices]:
                val, parsed = text, True

            if not parsed:
                try:
                    val = float(text)
                    parsed = True
                except ValueError:
                    pass

            if not parsed:
                self._updating = True
                try:
                    fallback = defn.default if defn is not None else 0.0
                    stored = self._lm.get_baseline(key, fallback)
                    if isinstance(stored, bool):
                        item.setText(str(stored))
                    else:
                        try:
                            item.setText(f"{float(stored):.4f}")
                        except (TypeError, ValueError):
                            item.setText(str(stored))
                finally:
                    self._updating = False
                return

            self._lm.set_baseline(key, val)
            link    = link_map.get(key)
            on_item = self._model.item(row, ChannelsModel.COL_ON)
            if not (link and link.expression) or \
                    on_item.checkState() == Qt.CheckState.Unchecked:
                self._pm.set(key, val)
            self.changed.emit()

        elif col == ChannelsModel.COL_ON:
            link = link_map.get(key)
            if link is None:
                return
            if item.checkState() == Qt.CheckState.Checked:
                link.enabled = True
            else:
                link.enabled = False
                self._lm.apply_baseline(key, self._pm)
            self.changed.emit()


# ── Events (EventLinks + Thresholds) ─────────────────────────────────────────

class EventsTab(QWidget):
    changed = Signal()

    def __init__(self, lm: LinkManager, pm: "PropertyManager", parent=None):
        super().__init__(parent)
        self._lm = lm
        self._pm = pm

        lo = QVBoxLayout(self)
        lo.setContentsMargins(4, 4, 4, 4)
        lo.setSpacing(4)

        # ── Event Links section ───────────────────────────────────────────────
        hdr1 = QLabel("Event Links")
        hdr1.setObjectName("hdr")
        lo.addWidget(hdr1)

        tb1 = QHBoxLayout()
        add_el = QPushButton("+ Add Link")
        add_el.setObjectName("add")
        add_el.clicked.connect(self._add_event_link)
        edit_el = QPushButton("Edit")
        edit_el.clicked.connect(self._edit_event_link)
        rm_el = QPushButton("Remove")
        rm_el.setObjectName("remove")
        rm_el.clicked.connect(self._remove_event_link)
        midi_btn = QPushButton("+ Midi")
        midi_btn.setObjectName("add")
        midi_btn.setToolTip("Wait for the next MIDI note, then open a new event link pre-filled with that event.")
        midi_btn.clicked.connect(self._capture_midi)
        key_btn = QPushButton("+ Key")
        key_btn.setObjectName("add")
        key_btn.setToolTip("Press a key to create an event link pre-filled with that key event.")
        key_btn.clicked.connect(self._capture_key)
        choice_btn = QPushButton("+ Choice")
        choice_btn.setObjectName("add")
        choice_btn.setToolTip("Pick an enum sink and one of its choices, then press a key to "
                               "create a key → choice event link.")
        choice_btn.clicked.connect(self._capture_choice)
        tb1.addWidget(add_el)
        tb1.addWidget(edit_el)
        tb1.addWidget(rm_el)
        tb1.addWidget(midi_btn)
        tb1.addWidget(key_btn)
        tb1.addWidget(choice_btn)
        tb1.addStretch()

        self._capture_status = QLabel("")
        self._capture_status.setObjectName("ok")
        tb1.addWidget(self._capture_status)

        lo.addLayout(tb1)

        self._el_table = _make_table(["On", "Event", "Action", "Condition"])
        el_hh = self._el_table.horizontalHeader()
        el_hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        el_hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        el_hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        el_hh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._el_table.doubleClicked.connect(self._edit_event_link)
        lo.addWidget(self._el_table)

        el_hint = QLabel("Double-click a row to edit.")
        el_hint.setObjectName("info")
        lo.addWidget(el_hint)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("sep")
        lo.addWidget(sep)

        # ── Thresholds section ────────────────────────────────────────────────
        hdr2 = QLabel("Threshold Detectors  →  fire  audio.threshold.<name>")
        hdr2.setObjectName("hdr")
        lo.addWidget(hdr2)

        tb2 = QHBoxLayout()
        add_th = QPushButton("+ Add Threshold")
        add_th.setObjectName("add")
        add_th.clicked.connect(self._add_threshold)
        edit_th = QPushButton("Edit")
        edit_th.clicked.connect(self._edit_threshold)
        rm_th = QPushButton("Remove")
        rm_th.setObjectName("remove")
        rm_th.clicked.connect(self._remove_threshold)
        tb2.addWidget(add_th)
        tb2.addWidget(edit_th)
        tb2.addWidget(rm_th)
        tb2.addStretch()
        lo.addLayout(tb2)

        self._th_table = _make_table(["Name", "Source", "High", "Low", "Min Interval (s)"])
        th_hh = self._th_table.horizontalHeader()
        th_hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        th_hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for i in range(2, 5):
            th_hh.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        self._th_table.doubleClicked.connect(self._edit_threshold)
        lo.addWidget(self._th_table)

        th_hint = QLabel("Double-click a row to edit.")
        th_hint.setObjectName("info")
        lo.addWidget(th_hint)

        self._rebuild()

    # ── MIDI capture ──────────────────────────────────────────────────────────

    def _capture_midi(self) -> None:
        try:
            from midi_input import get_router
            router = get_router()
        except Exception:
            self._capture_status.setText("no MIDI router")
            return

        self._capture_status.setText("waiting for MIDI…")
        self._capture_status.setObjectName("info")
        self._capture_status.style().unpolish(self._capture_status)
        self._capture_status.style().polish(self._capture_status)

        def _on_event(evt: dict) -> None:
            if evt.get("type") != "note":
                return
            router.remove_listener(_on_event)
            note = evt["number"]
            vel  = evt.get("value", 0)
            event_str = f"midi.note{note}.{'on' if vel > 0 else 'off'}"
            # Marshal back to Qt thread
            QTimer.singleShot(0, lambda: self._open_with_event(event_str))

        router.add_listener(_on_event)

    def _capture_key(self) -> None:
        dlg = _KeyCaptureDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            event_str = dlg.captured_event()
            if event_str:
                self._open_with_event(event_str)

    def _capture_choice(self) -> None:
        if not any(d.choices for d in self._pm.all_props()):
            self._capture_status.setText("no enum sinks")
            return
        choice_dlg = _ChoiceCaptureDialog(self._pm, self)
        if choice_dlg.exec() != QDialog.DialogCode.Accepted:
            return
        picked = choice_dlg.result()
        if picked is None:
            return
        sink_key, choice = picked

        key_dlg = _KeyCaptureDialog(self)
        if key_dlg.exec() == QDialog.DialogCode.Accepted:
            event_str = key_dlg.captured_event()
            if event_str:
                self._open_with_event(event_str, f"set({sink_key}, {choice!r})")

    def _open_with_event(self, event_str: str, action_str: str = "") -> None:
        self._capture_status.setText("")
        prefilled = EventLink(event=event_str, action=action_str, enabled=True)
        dlg = EventLinkDialog(self._lm, self._pm, link=prefilled, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            link = dlg.result_link()
            if link.event and link.action:
                self._lm.add_event_link(link)
                self._rebuild()
                self.changed.emit()

    # ── rebuild ───────────────────────────────────────────────────────────────

    def _rebuild(self) -> None:
        links = self._lm._event_links
        self._el_table.setRowCount(len(links))
        for r, link in enumerate(links):
            self._el_table.setItem(r, 0, _bool_cell(link.enabled))
            self._el_table.setItem(r, 1, _cell(link.event))
            self._el_table.setItem(r, 2, _cell(link.action))
            self._el_table.setItem(r, 3, _cell(link.condition or ""))

        defs = self._lm._threshold_defs
        self._th_table.setRowCount(len(defs))
        for r, defn in enumerate(defs):
            self._th_table.setItem(r, 0, _cell(defn.name))
            self._th_table.setItem(r, 1, _cell(defn.source))
            self._th_table.setItem(r, 2, _cell(f"{defn.high:.3f}"))
            self._th_table.setItem(r, 3, _cell(f"{defn.low:.3f}"))
            self._th_table.setItem(r, 4, _cell(f"{defn.min_interval_s:.3f}"))

    # EventLink CRUD

    def _add_event_link(self) -> None:
        dlg = EventLinkDialog(self._lm, self._pm, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            link = dlg.result_link()
            if link.event and link.action:
                self._lm.add_event_link(link)
                self._rebuild()
                self.changed.emit()

    def _edit_event_link(self) -> None:
        row = self._el_table.currentRow()
        if row < 0 or row >= len(self._lm._event_links):
            return
        old = self._lm._event_links[row]
        dlg = EventLinkDialog(self._lm, self._pm, link=old, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._lm._event_links[row] = dlg.result_link()
            self._rebuild()
            self.changed.emit()

    def _remove_event_link(self) -> None:
        row = self._el_table.currentRow()
        if row < 0 or row >= len(self._lm._event_links):
            return
        link = self._lm._event_links[row]
        self._lm.remove_event_link(link.event, link.action)
        self._rebuild()
        self.changed.emit()

    # Threshold CRUD

    def _add_threshold(self) -> None:
        dlg = ThresholdDialog(self._lm, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            defn = dlg.result_def()
            if defn.name:
                self._lm.add_threshold(defn)
                self._rebuild()
                self.changed.emit()

    def _edit_threshold(self) -> None:
        row = self._th_table.currentRow()
        if row < 0 or row >= len(self._lm._threshold_defs):
            return
        old = self._lm._threshold_defs[row]
        dlg = ThresholdDialog(self._lm, defn=old, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_def = dlg.result_def()
            self._lm.remove_threshold(old.name)
            self._lm._threshold_defs.insert(row, new_def)
            self._rebuild()
            self.changed.emit()

    def _remove_threshold(self) -> None:
        row = self._th_table.currentRow()
        if row < 0 or row >= len(self._lm._threshold_defs):
            return
        defn = self._lm._threshold_defs[row]
        self._lm.remove_threshold(defn.name)
        self._rebuild()
        self.changed.emit()


# ── Envelopes ─────────────────────────────────────────────────────────────────

class EnvelopesTab(QWidget):
    changed = Signal()

    def __init__(self, lm: LinkManager, parent=None):
        super().__init__(parent)
        self._lm = lm

        lo = QVBoxLayout(self)
        lo.setContentsMargins(4, 4, 4, 4)

        tb = QHBoxLayout()
        add_btn = QPushButton("+ Add")
        add_btn.setObjectName("add")
        add_btn.clicked.connect(self._add)
        rm_btn = QPushButton("Remove")
        rm_btn.setObjectName("remove")
        rm_btn.clicked.connect(self._remove)
        trig_btn = QPushButton("Trigger")
        trig_btn.setObjectName("trigger")
        trig_btn.setToolTip("Manually trigger the selected envelope (for testing).")
        trig_btn.clicked.connect(self._manual_trigger)
        tb.addWidget(add_btn)
        tb.addWidget(rm_btn)
        tb.addWidget(trig_btn)
        tb.addStretch()
        lo.addLayout(tb)

        self._table = _make_table(["Name", "Trigger", "Gate Off", "A", "D", "S", "R", "Peak"])
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        for i in range(3, 8):
            hh.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        self._table.doubleClicked.connect(self._edit)
        lo.addWidget(self._table)

        info = QLabel("env.<name> values visible in Sources tab.")
        info.setObjectName("info")
        lo.addWidget(info)

        self._rebuild()

    def _rebuild(self) -> None:
        defs = self._lm._envelope_defs
        self._table.setRowCount(len(defs))
        for r, d in enumerate(defs):
            self._table.setItem(r, 0, _cell(d.name))
            self._table.setItem(r, 1, _cell(d.trigger))
            self._table.setItem(r, 2, _cell(d.gate_off or ""))
            self._table.setItem(r, 3, _cell(f"{d.attack:.3f}"))
            self._table.setItem(r, 4, _cell(f"{d.decay:.3f}"))
            self._table.setItem(r, 5, _cell(f"{d.sustain:.3f}"))
            self._table.setItem(r, 6, _cell(f"{d.release:.3f}"))
            self._table.setItem(r, 7, _cell(f"{d.peak:.3f}"))

    def _add(self) -> None:
        dlg = EnvelopeDialog(self._lm, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            defn = dlg.result_def()
            if defn.name and defn.trigger:
                self._lm.add_envelope(defn)
                self._rebuild()
                self.changed.emit()

    def _edit(self) -> None:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._lm._envelope_defs):
            return
        old = self._lm._envelope_defs[row]
        dlg = EnvelopeDialog(self._lm, defn=old, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_def = dlg.result_def()
            # remove_envelope pops from _envelopes and _envelope_defs
            self._lm.remove_envelope(old.name)
            # add_envelope re-creates the runtime object and wires event subscriptions
            self._lm.add_envelope(new_def)
            self._rebuild()
            self.changed.emit()

    def _remove(self) -> None:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._lm._envelope_defs):
            return
        defn = self._lm._envelope_defs[row]
        self._lm.remove_envelope(defn.name)
        self._rebuild()
        self.changed.emit()

    def _manual_trigger(self) -> None:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._lm._envelope_defs):
            return
        name = self._lm._envelope_defs[row].name
        env = self._lm._envelopes.get(name)
        if env:
            env.trigger()


# ── LFOs ──────────────────────────────────────────────────────────────────────

class LFOsTab(QWidget):
    changed = Signal()

    def __init__(self, lm: LinkManager, parent=None):
        super().__init__(parent)
        self._lm = lm

        lo = QVBoxLayout(self)
        lo.setContentsMargins(4, 4, 4, 4)

        tb = QHBoxLayout()
        add_btn = QPushButton("+ Add")
        add_btn.setObjectName("add")
        add_btn.clicked.connect(self._add)
        rm_btn = QPushButton("Remove")
        rm_btn.setObjectName("remove")
        rm_btn.clicked.connect(self._remove)
        tb.addWidget(add_btn)
        tb.addWidget(rm_btn)
        tb.addStretch()
        lo.addLayout(tb)

        self._table = _make_table(["Name", "Shape", "Rate (Hz)", "Phase"])
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.doubleClicked.connect(self._edit)
        lo.addWidget(self._table)

        info = QLabel("lfo.<name> values visible in Sources tab.")
        info.setObjectName("info")
        lo.addWidget(info)

        self._rebuild()

    def _rebuild(self) -> None:
        defs = self._lm._lfo_defs
        self._table.setRowCount(len(defs))
        for r, d in enumerate(defs):
            self._table.setItem(r, 0, _cell(d.name))
            self._table.setItem(r, 1, _cell(d.shape))
            self._table.setItem(r, 2, _cell(f"{d.rate_hz:.3f}"))
            self._table.setItem(r, 3, _cell(f"{d.phase:.3f}"))

    def _add(self) -> None:
        dlg = LFODialog(parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            defn = dlg.result_def()
            if defn.name:
                self._lm.add_lfo(defn)
                self._rebuild()
                self.changed.emit()

    def _edit(self) -> None:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._lm._lfo_defs):
            return
        old = self._lm._lfo_defs[row]
        dlg = LFODialog(defn=old, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_def = dlg.result_def()
            self._lm.remove_lfo(old.name)
            self._lm.add_lfo(new_def)
            self._rebuild()
            self.changed.emit()

    def _remove(self) -> None:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._lm._lfo_defs):
            return
        defn = self._lm._lfo_defs[row]
        self._lm.remove_lfo(defn.name)
        self._rebuild()
        self.changed.emit()


# ── Parameters ────────────────────────────────────────────────────────────────

class ParametersTab(QWidget):
    changed = Signal()

    def __init__(self, lm: LinkManager, parent=None):
        super().__init__(parent)
        self._lm = lm

        lo = QVBoxLayout(self)
        lo.setContentsMargins(4, 4, 4, 4)

        tb = QHBoxLayout()
        add_btn = QPushButton("+ Add")
        add_btn.setObjectName("add")
        add_btn.clicked.connect(self._add)
        edit_btn = QPushButton("Edit")
        edit_btn.clicked.connect(self._edit)
        rm_btn = QPushButton("Remove")
        rm_btn.setObjectName("remove")
        rm_btn.clicked.connect(self._remove)
        tb.addWidget(add_btn)
        tb.addWidget(edit_btn)
        tb.addWidget(rm_btn)
        tb.addStretch()
        lo.addLayout(tb)

        self._table = _make_table(["Name", "Kind", "Trigger", "Off Event", "State"])
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self._table.doubleClicked.connect(self._edit)
        lo.addWidget(self._table)

        info = QLabel("p.<name> values visible in Sources tab.")
        info.setObjectName("info")
        lo.addWidget(info)

        self._live_timer = QTimer(self)
        self._live_timer.setInterval(100)
        self._live_timer.timeout.connect(self._refresh_state_column)
        self._live_timer.start()

        self._rebuild()

    def _rebuild(self) -> None:
        defs = self._lm._parameter_defs
        self._table.setRowCount(len(defs))
        for r, d in enumerate(defs):
            self._table.setItem(r, 0, _cell(d.name))
            self._table.setItem(r, 1, _cell(d.kind))
            self._table.setItem(r, 2, _cell(d.trigger))
            self._table.setItem(r, 3, _cell(d.off_event or ""))
            self._table.setItem(r, 4, _cell(""))

    def _refresh_state_column(self) -> None:
        snap = self._lm.source_registry.snapshot()
        defs = self._lm._parameter_defs
        for r, d in enumerate(defs):
            val = snap.get(f"p.{d.name}", 0.0)
            new_text = f"{val:.4f}"
            item = self._table.item(r, 4)
            if item is not None and item.text() != new_text:
                item.setText(new_text)

    def _add(self) -> None:
        dlg = ParameterDialog(self._lm, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            defn = dlg.result_def()
            if defn.name and defn.trigger:
                self._lm.add_parameter(defn)
                self._rebuild()
                self.changed.emit()

    def _edit(self) -> None:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._lm._parameter_defs):
            return
        old = self._lm._parameter_defs[row]
        dlg = ParameterDialog(self._lm, defn=old, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new_def = dlg.result_def()
            self._lm.remove_parameter(old.name)
            self._lm.add_parameter(new_def)
            self._rebuild()
            self.changed.emit()

    def _remove(self) -> None:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._lm._parameter_defs):
            return
        defn = self._lm._parameter_defs[row]
        self._lm.remove_parameter(defn.name)
        self._rebuild()
        self.changed.emit()


# ── Presets ───────────────────────────────────────────────────────────────────

class PresetsTab(QWidget):
    changed      = Signal()   # routing or trigger list changed → schedule save
    needs_rebuild = Signal()  # preset was loaded → other tabs must rebuild

    def __init__(self, lm: LinkManager, pm: "PropertyManager", parent=None):
        super().__init__(parent)
        self._lm = lm
        self._pm = pm

        lo = QVBoxLayout(self)
        lo.setContentsMargins(4, 4, 4, 4)
        lo.setSpacing(4)

        # ── Preset list ───────────────────────────────────────────────────────
        hdr1 = QLabel("Link Presets")
        hdr1.setObjectName("hdr")
        lo.addWidget(hdr1)

        tb1 = QHBoxLayout()
        save_btn = QPushButton("Save Current As…")
        save_btn.setObjectName("add")
        save_btn.clicked.connect(self._save_preset)
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._load_preset)
        del_btn = QPushButton("Delete")
        del_btn.setObjectName("remove")
        del_btn.clicked.connect(self._delete_preset)
        tb1.addWidget(save_btn)
        tb1.addWidget(load_btn)
        tb1.addWidget(del_btn)
        tb1.addStretch()
        lo.addLayout(tb1)

        self._preset_table = _make_table(["Name"])
        self._preset_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch)
        self._preset_table.doubleClicked.connect(self._load_preset)
        lo.addWidget(self._preset_table)

        info1 = QLabel("Double-click a preset to load it immediately.")
        info1.setObjectName("info")
        lo.addWidget(info1)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("sep")
        lo.addWidget(sep)

        # ── Preset triggers ───────────────────────────────────────────────────
        hdr2 = QLabel("Preset Triggers  —  survive preset switches")
        hdr2.setObjectName("hdr")
        lo.addWidget(hdr2)

        tb2 = QHBoxLayout()
        add_btn = QPushButton("+ Add Trigger")
        add_btn.setObjectName("add")
        add_btn.clicked.connect(self._add_trigger)
        rm_btn = QPushButton("Remove")
        rm_btn.setObjectName("remove")
        rm_btn.clicked.connect(self._remove_trigger)
        tb2.addWidget(add_btn)
        tb2.addWidget(rm_btn)
        tb2.addStretch()
        lo.addLayout(tb2)

        self._trigger_table = _make_table(["On", "Event", "Action", "Condition"])
        th = self._trigger_table.horizontalHeader()
        th.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        th.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        th.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        th.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._trigger_table.doubleClicked.connect(self._edit_trigger)
        lo.addWidget(self._trigger_table)

        info2 = QLabel("Action syntax:  link_preset('name')")
        info2.setObjectName("info")
        lo.addWidget(info2)

        self._rebuild()

    def _rebuild(self) -> None:
        names = self._lm.list_link_presets()
        self._preset_table.setRowCount(len(names))
        for r, name in enumerate(names):
            self._preset_table.setItem(r, 0, _cell(name))

        triggers = self._lm._preset_triggers
        self._trigger_table.setRowCount(len(triggers))
        for r, link in enumerate(triggers):
            self._trigger_table.setItem(r, 0, _bool_cell(link.enabled))
            self._trigger_table.setItem(r, 1, _cell(link.event))
            self._trigger_table.setItem(r, 2, _cell(link.action))
            self._trigger_table.setItem(r, 3, _cell(link.condition or ""))

    # Preset CRUD

    def _save_preset(self) -> None:
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if ok and name.strip():
            self._lm.save_link_preset(name.strip())
            self._rebuild()
            self.changed.emit()

    def _load_preset(self) -> None:
        row = self._preset_table.currentRow()
        if row < 0:
            return
        item = self._preset_table.item(row, 0)
        if item:
            self._lm.load_link_preset(item.text())
            self.needs_rebuild.emit()
            self.changed.emit()

    def _delete_preset(self) -> None:
        row = self._preset_table.currentRow()
        if row < 0:
            return
        item = self._preset_table.item(row, 0)
        if item:
            self._lm.delete_link_preset(item.text())
            self._rebuild()
            self.changed.emit()

    # Preset trigger CRUD

    def _add_trigger(self) -> None:
        dlg = EventLinkDialog(self._lm, self._pm, parent=self,
                              action_completions=_trigger_action_completions(self._lm))
        if dlg.exec() == QDialog.DialogCode.Accepted:
            link = dlg.result_link()
            if link.event and link.action:
                self._lm.add_preset_trigger(link)
                self._rebuild()
                self.changed.emit()

    def _edit_trigger(self) -> None:
        row = self._trigger_table.currentRow()
        if row < 0 or row >= len(self._lm._preset_triggers):
            return
        old = self._lm._preset_triggers[row]
        dlg = EventLinkDialog(self._lm, self._pm, link=old, parent=self,
                              action_completions=_trigger_action_completions(self._lm))
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._lm._preset_triggers[row] = dlg.result_link()
            self._rebuild()
            self.changed.emit()

    def _remove_trigger(self) -> None:
        row = self._trigger_table.currentRow()
        if row < 0 or row >= len(self._lm._preset_triggers):
            return
        link = self._lm._preset_triggers[row]
        self._lm.remove_preset_trigger(link.event, link.action)
        self._rebuild()
        self.changed.emit()


# ─────────────────────────────────────────────────────────────────────────────
#  Main panel
# ─────────────────────────────────────────────────────────────────────────────

class LinkManagerPanel(QWidget):
    """Main panel: tabs for Channels, Parameters, Events, Envelopes, LFOs, Presets, Sources.

    extra_tabs: optional list of (label, widget) pairs appended after the core tabs.
    """

    _STATE_PATH = pathlib.Path(__file__).with_name("link_state.json")

    def __init__(self, lm: LinkManager, pm: "PropertyManager",
                 title: str = "Link Manager", parent=None,
                 extra_tabs: "list[tuple[str, QWidget]] | None" = None):
        super().__init__(parent)
        self._lm = lm
        self._pm = pm
        self.setWindowTitle(title)
        self.setStyleSheet(_STYLESHEET)
        self.resize(820, 600)

        lo = QVBoxLayout(self)
        lo.setContentsMargins(6, 6, 6, 6)
        lo.setSpacing(4)

        tabs = QTabWidget()
        self._channels_tab = ChannelsTab(lm, pm)
        self._param_tab    = ParametersTab(lm)
        self._evt_tab      = EventsTab(lm, pm)
        self._env_tab      = EnvelopesTab(lm)
        self._lfo_tab      = LFOsTab(lm)
        self._preset_tab   = PresetsTab(lm, pm)
        self._src_tab      = SourcesTab(lm)

        tabs.addTab(self._channels_tab, "Channels")
        tabs.addTab(self._param_tab,    "Parameters")
        tabs.addTab(self._evt_tab,      "Events")
        tabs.addTab(self._env_tab,      "Envelopes")
        tabs.addTab(self._lfo_tab,      "LFOs")
        tabs.addTab(self._preset_tab,   "Presets")
        tabs.addTab(self._src_tab,      "Sources")

        for label, widget in (extra_tabs or []):
            tabs.addTab(widget, label)

        lo.addWidget(tabs)

        # Footer
        footer = QHBoxLayout()
        save_btn = QPushButton("Save Now")
        save_btn.clicked.connect(self._save_now)
        footer.addWidget(save_btn)
        footer.addStretch()
        self._status = QLabel("")
        self._status.setObjectName("info")
        footer.addWidget(self._status)
        lo.addLayout(footer)

        # Debounced auto-save timer
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(5000)
        self._save_timer.timeout.connect(self._save_now)

        for tab in (self._channels_tab, self._param_tab, self._evt_tab,
                    self._env_tab, self._lfo_tab, self._preset_tab):
            tab.changed.connect(self._on_changed)

        self._preset_tab.needs_rebuild.connect(self._rebuild_routing_tabs)

        self._auto_load()

        # nn_graph.* / lasers.* / circles.* properties are registered later by
        # MergedGUI.__init__ on the GL thread, after this panel is already built
        # on the Qt thread — so the routing tabs above can miss them. Poll for
        # newly-registered properties and rebuild once they show up.
        self._known_prop_count = len(pm.all_props())
        self._sync_timer = QTimer(self)
        self._sync_timer.setInterval(250)
        self._sync_timer.timeout.connect(self._sync_new_props)
        self._sync_timer.start()

    def _sync_new_props(self) -> None:
        count = len(self._pm.all_props())
        if count != self._known_prop_count:
            self._known_prop_count = count
            self._rebuild_routing_tabs()

    def _on_changed(self) -> None:
        self._save_timer.start()
        self._status.setText("unsaved changes")

    def _rebuild_routing_tabs(self) -> None:
        """Rebuild all routing tabs after a preset load changes the routing state."""
        for tab in (self._channels_tab, self._param_tab, self._evt_tab,
                    self._env_tab, self._lfo_tab):
            tab._rebuild()

    def _save_now(self) -> None:
        try:
            self._lm.save_state(self._STATE_PATH)
            self._status.setText(f"saved  ({self._STATE_PATH.name})")
        except Exception as exc:
            self._status.setText(f"save error: {exc}")

    def _auto_load(self) -> None:
        if not self._STATE_PATH.exists():
            return
        try:
            self._lm.load_state(self._STATE_PATH)
            for tab in (self._channels_tab, self._param_tab, self._evt_tab,
                        self._env_tab, self._lfo_tab, self._preset_tab):
                tab._rebuild()
            # Push saved baselines into PM for every property that has no active link.
            # Properties WITH an active enabled link will be overridden by evaluate_links
            # on the first frame anyway, so it is safe to write all of them here.
            active_keys = {
                l.sink_key for l in self._lm._signal_links if l.enabled and l.expression
            }
            for prop in self._pm.all_props():
                if prop.key not in active_keys:
                    self._lm.apply_baseline(prop.key, self._pm)
            self._status.setText(f"loaded  ({self._STATE_PATH.name})")
        except Exception as exc:
            self._status.setText(f"auto-load error: {exc}")
