"""
link_panel.py — Qt panel for the LinkManager signal routing layer.

PySide6 widget with 5 tabs:
  Signal Links — expression-based continuous source → sink mappings
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

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QLineEdit,
    QTableWidget, QTableWidgetItem, QCheckBox, QHeaderView,
    QTabWidget, QFrame, QDoubleSpinBox,
    QCompleter, QAbstractItemView, QDialog, QDialogButtonBox,
)

from link_manager import (
    LinkManager, SignalLink, EnvelopeDef, LFODef, EventLink, ThresholdDef,
    EVAL_MATH_NS, _flat_to_ns,
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

QTableWidget {
    background-color: #111118;
    alternate-background-color: #141420;
    color: #c8c8d0;
    border: 1px solid #38383f;
    gridline-color: #2a2a38;
    selection-background-color: #1e3050;
    selection-color: #c8c8d0;
}
QTableWidget QHeaderView::section {
    background-color: #1a1a22;
    color: #707078;
    border: none;
    border-bottom: 1px solid #38383f;
    padding: 3px 6px;
}
QTableWidget::item { padding: 2px 4px; }

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

_KEY_NAMES = (
    "NUMBER_1", "NUMBER_2", "NUMBER_3", "NUMBER_4",
    "TAB", "Z", "X", "D", "F", "Q", "W", "A", "S",
    "H", "J", "C", "V", "B", "N", "K", "L", "I", "U", "G", "M",
)

_LFO_SHAPES = ("sine", "saw", "square", "tri")


# ── helper: completion lists ──────────────────────────────────────────────────

def _event_completions(lm: LinkManager) -> list[str]:
    srcs = ["audio.onset", "clock.beat"]
    for k in _KEY_NAMES:
        srcs.append(f"key.{k}.press")
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


def _action_completions(pm: "PropertyManager") -> list[str]:
    actions: list[str] = ["regen", "preset('')"]
    for prop in pm.all_props():
        k = prop.key
        if prop.type is bool:
            actions.append(f"toggle({k})")
        if prop.choices:
            actions.append(f"cycle({k})")
            actions.append(f"cycle_back({k})")
        actions.append(f"set({k}, )")
    return actions


def _make_completer(words: list[str], parent=None) -> QCompleter:
    c = QCompleter(words, parent)
    c.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
    c.setFilterMode(Qt.MatchFlag.MatchContains)
    c.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
    return c


def _eval_preview(expr: str, lm: LinkManager) -> tuple[str, bool]:
    """Return (display_text, is_error)."""
    if not expr.strip():
        return ("", False)
    try:
        snap = lm.source_registry.snapshot()
        ns = {**_flat_to_ns(snap), **EVAL_MATH_NS, "dt": 0.016, "const": lm._const_ns}
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


# ── Signal Link ───────────────────────────────────────────────────────────────

class SignalLinkDialog(_BaseDialog):
    def __init__(self, lm: LinkManager, pm: "PropertyManager",
                 link: SignalLink | None = None, parent=None):
        super().__init__("Signal Link" if link is None else "Edit Signal Link", parent)
        self._lm = lm

        self._sink = QComboBox()
        sink_keys = [p.key for p in pm.all_props()]
        self._sink.addItems(sink_keys)
        if link and link.sink_key in sink_keys:
            self._sink.setCurrentIndex(sink_keys.index(link.sink_key))
        self._row("Sink:", self._sink)

        self._expr = QLineEdit()
        self._expr.setPlaceholderText("lerp(0.98, 0.999, audio.bass)")
        if link:
            self._expr.setText(link.expression)
        self._expr.setCompleter(_make_completer(_expr_completions(lm), self))
        self._row("Expression:", self._expr)

        self._preview = QLabel("")
        self._preview.setObjectName("value")
        self._layout.addWidget(self._preview)

        self._enabled = QCheckBox("Enabled")
        self._enabled.setChecked(link.enabled if link else True)
        self._layout.addWidget(self._enabled)

        self._add_buttons()

        self._timer = QTimer(self)
        self._timer.setInterval(200)
        self._timer.timeout.connect(self._refresh_preview)
        self._timer.start()
        self._expr.textChanged.connect(self._refresh_preview)
        self._refresh_preview()

    def _refresh_preview(self) -> None:
        text, is_err = _eval_preview(self._expr.text(), self._lm)
        self._preview.setObjectName("err" if is_err else "value")
        self._preview.setText(text)
        self._preview.style().unpolish(self._preview)
        self._preview.style().polish(self._preview)

    def result_link(self) -> SignalLink:
        return SignalLink(
            sink_key   = self._sink.currentText(),
            expression = self._expr.text().strip(),
            enabled    = self._enabled.isChecked(),
        )


# ── Event Link ────────────────────────────────────────────────────────────────

class EventLinkDialog(_BaseDialog):
    def __init__(self, lm: LinkManager, pm: "PropertyManager",
                 link: EventLink | None = None, parent=None):
        super().__init__("Event Link" if link is None else "Edit Event Link", parent)

        self._event = QLineEdit()
        self._event.setPlaceholderText("midi.note36.on")
        if link:
            self._event.setText(link.event)
        self._event.setCompleter(_make_completer(_event_completions(lm), self))
        self._row("Event:", self._event)

        self._action = QLineEdit()
        self._action.setPlaceholderText("toggle(scene.show_cloud)")
        if link:
            self._action.setText(link.action)
        self._action.setCompleter(_make_completer(_action_completions(pm), self))
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


# ── Signal Links ──────────────────────────────────────────────────────────────

class SignalLinksTab(QWidget):
    changed = Signal()

    def __init__(self, lm: LinkManager, pm: "PropertyManager", parent=None):
        super().__init__(parent)
        self._lm = lm
        self._pm = pm

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

        self._table = _make_table(["On", "Sink", "Expression", "Value"])
        hh = self._table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._table.doubleClicked.connect(self._edit)
        lo.addWidget(self._table)

        hint = QLabel("Double-click a row to edit its expression.")
        hint.setObjectName("info")
        lo.addWidget(hint)

        self._live_timer = QTimer(self)
        self._live_timer.setInterval(100)
        self._live_timer.timeout.connect(self._update_live_values)
        self._live_timer.start()

        self._rebuild()

    def _rebuild(self) -> None:
        links = self._lm._signal_links
        self._table.setRowCount(len(links))
        for r, link in enumerate(links):
            self._table.setItem(r, 0, _bool_cell(link.enabled))
            self._table.setItem(r, 1, _cell(link.sink_key))
            self._table.setItem(r, 2, _cell(link.expression))
            self._table.setItem(r, 3, _cell(""))

    def _update_live_values(self) -> None:
        if not self._lm._signal_links:
            return
        snap = self._lm.source_registry.snapshot()
        ns = {**_flat_to_ns(snap), **EVAL_MATH_NS, "dt": 0.016, "const": self._lm._const_ns}
        for r, link in enumerate(self._lm._signal_links):
            item = self._table.item(r, 3)
            if item is None:
                continue
            if not link.enabled:
                item.setText("—")
                continue
            try:
                v = eval(link.expression, {"__builtins__": {}}, ns)
                item.setText(f"{v:.4f}" if isinstance(v, float) else str(v))
            except Exception:
                item.setText("err")

    def _add(self) -> None:
        dlg = SignalLinkDialog(self._lm, self._pm, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            link = dlg.result_link()
            if link.sink_key and link.expression:
                self._lm.add_link(link)
                self._rebuild()
                self.changed.emit()

    def _edit(self) -> None:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._lm._signal_links):
            return
        old = self._lm._signal_links[row]
        dlg = SignalLinkDialog(self._lm, self._pm, link=old, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            new = dlg.result_link()
            if new.expression != old.expression and hasattr(old, "smooth_helper"):
                old.smooth_helper._states.clear()
            old.sink_key   = new.sink_key
            old.expression = new.expression
            old.enabled    = new.enabled
            self._rebuild()
            self.changed.emit()

    def _remove(self) -> None:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._lm._signal_links):
            return
        link = self._lm._signal_links[row]
        self._lm.remove_link(link.sink_key)
        self._rebuild()
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
        rm_el = QPushButton("Remove")
        rm_el.setObjectName("remove")
        rm_el.clicked.connect(self._remove_event_link)
        tb1.addWidget(add_el)
        tb1.addWidget(rm_el)
        tb1.addStretch()
        lo.addLayout(tb1)

        self._el_table = _make_table(["On", "Event", "Action", "Condition"])
        el_hh = self._el_table.horizontalHeader()
        el_hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        el_hh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        el_hh.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        el_hh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self._el_table.doubleClicked.connect(self._edit_event_link)
        lo.addWidget(self._el_table)

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
        rm_th = QPushButton("Remove")
        rm_th.setObjectName("remove")
        rm_th.clicked.connect(self._remove_threshold)
        tb2.addWidget(add_th)
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

        self._rebuild()

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


# ─────────────────────────────────────────────────────────────────────────────
#  Main panel
# ─────────────────────────────────────────────────────────────────────────────

class LinkManagerPanel(QWidget):
    """Main panel: tabs for Signal Links, Events, Envelopes, LFOs, Sources."""

    _STATE_PATH = pathlib.Path(__file__).with_name("link_state.json")

    def __init__(self, lm: LinkManager, pm: "PropertyManager",
                 title: str = "Link Manager", parent=None):
        super().__init__(parent)
        self._lm = lm
        self._pm = pm
        self.setWindowTitle(title)
        self.setStyleSheet(_STYLESHEET)
        self.resize(740, 540)

        lo = QVBoxLayout(self)
        lo.setContentsMargins(6, 6, 6, 6)
        lo.setSpacing(4)

        tabs = QTabWidget()
        self._sig_tab = SignalLinksTab(lm, pm)
        self._evt_tab = EventsTab(lm, pm)
        self._env_tab = EnvelopesTab(lm)
        self._lfo_tab = LFOsTab(lm)
        self._src_tab = SourcesTab(lm)

        tabs.addTab(self._sig_tab, "Signal Links")
        tabs.addTab(self._evt_tab, "Events")
        tabs.addTab(self._env_tab, "Envelopes")
        tabs.addTab(self._lfo_tab, "LFOs")
        tabs.addTab(self._src_tab, "Sources")
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

        for tab in (self._sig_tab, self._evt_tab, self._env_tab, self._lfo_tab):
            tab.changed.connect(self._on_changed)

        self._auto_load()

    def _on_changed(self) -> None:
        self._save_timer.start()
        self._status.setText("unsaved changes")

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
            for tab in (self._sig_tab, self._evt_tab, self._env_tab, self._lfo_tab):
                tab._rebuild()
            self._status.setText(f"loaded  ({self._STATE_PATH.name})")
        except Exception as exc:
            self._status.setText(f"auto-load error: {exc}")
