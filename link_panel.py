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

from PySide6.QtCore import Qt, QTimer, Signal, QSortFilterProxyModel, QEvent
from PySide6.QtGui import QColor, QKeyEvent, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QLineEdit,
    QTableWidget, QTableWidgetItem, QCheckBox, QHeaderView,
    QTabWidget, QFrame, QDoubleSpinBox,
    QCompleter, QAbstractItemView, QDialog, QDialogButtonBox, QInputDialog,
    QTableView, QStyledItemDelegate, QAbstractItemDelegate,
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


def _action_completions(lm: LinkManager, pm: "PropertyManager") -> list[str]:
    actions: list[str] = ["regen", "preset('')"]
    preset_names = lm.list_link_presets()
    if preset_names:
        for name in preset_names:
            actions.append(f"link_preset('{name}')")
    else:
        actions.append("link_preset('')")
    for prop in pm.all_props():
        k = prop.key
        if prop.type is bool:
            actions.append(f"toggle({k})")
        if prop.choices:
            actions.append(f"cycle({k})")
            actions.append(f"cycle_back({k})")
        actions.append(f"set({k}, )")
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

# Qt key code → _KEY_NAMES string (matches gui_merged.py _KEY_MAP)
_QT_KEY_TO_NAME: dict[int, str] = {
    Qt.Key.Key_1: "NUMBER_1", Qt.Key.Key_2: "NUMBER_2",
    Qt.Key.Key_3: "NUMBER_3", Qt.Key.Key_4: "NUMBER_4",
    Qt.Key.Key_Tab: "TAB",
    Qt.Key.Key_Z: "Z", Qt.Key.Key_X: "X",
    Qt.Key.Key_D: "D", Qt.Key.Key_F: "F",
    Qt.Key.Key_Q: "Q", Qt.Key.Key_W: "W",
    Qt.Key.Key_A: "A", Qt.Key.Key_S: "S",
    Qt.Key.Key_H: "H", Qt.Key.Key_J: "J",
    Qt.Key.Key_C: "C", Qt.Key.Key_V: "V",
    Qt.Key.Key_B: "B", Qt.Key.Key_N: "N",
    Qt.Key.Key_K: "K", Qt.Key.Key_L: "L",
    Qt.Key.Key_I: "I", Qt.Key.Key_U: "U",
    Qt.Key.Key_G: "G", Qt.Key.Key_M: "M",
}


class _KeyCaptureDialog(QDialog):
    """Modal that waits for one mapped key press and returns its event string."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Capture Key")
        self.setStyleSheet(_STYLESHEET)
        self.setFixedSize(300, 100)
        self._result: str | None = None

        lo = QVBoxLayout(self)
        self._lbl = QLabel("Press a key…")
        self._lbl.setObjectName("hdr")
        self._lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lo.addWidget(self._lbl)

        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        lo.addWidget(cancel)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        name = _QT_KEY_TO_NAME.get(event.key())
        if name:
            self._result = f"key.{name}.press"
            self.accept()
        elif event.key() == Qt.Key.Key_Escape:
            self.reject()

    def captured_event(self) -> str | None:
        return self._result


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
        self._action.setPlaceholderText("toggle(scene.show_cloud)")
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


# ── Signal Links model ────────────────────────────────────────────────────────

class SignalLinksModel(QStandardItemModel):
    """QStandardItemModel backing the property-bay Signal Links table.

    Columns: 0=On  1=Key  2=Default  3=Expression  4=Live Value
    """

    COL_ON   = 0
    COL_KEY  = 1
    COL_DEF  = 2
    COL_EXPR = 3
    COL_LIVE = 4

    def __init__(self, parent=None):
        super().__init__(0, 5, parent)
        self.setHorizontalHeaderLabels(["✓", "Key", "Default", "Expression", "Live Value"])
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

            self.appendRow([on_item, key_item, def_item, expr_item, live_item])
            self.row_for_key[key] = row


def _make_signal_links_proxy(model: SignalLinksModel) -> QSortFilterProxyModel:
    """Create a QSortFilterProxyModel over model, filtering on the Key column."""
    proxy = QSortFilterProxyModel()
    proxy.setSourceModel(model)
    proxy.setFilterKeyColumn(SignalLinksModel.COL_KEY)
    proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
    return proxy


# ── Signal Links delegate ─────────────────────────────────────────────────────

class _SignalLinksDelegate(QStyledItemDelegate):
    """Custom delegate: persistent QCheckBox for bool Default cells,
    persistent QComboBox for choice-type Default cells."""

    def __init__(self, model: SignalLinksModel, proxy: "QSortFilterProxyModel",
                 pm: "PropertyManager", parent=None):
        super().__init__(parent)
        self._model = model
        self._proxy = proxy
        self._pm    = pm

    def _prop_defn_for(self, proxy_index):
        src = self._proxy.mapToSource(proxy_index)
        if src.column() != SignalLinksModel.COL_DEF:
            return None
        key_item = self._model.item(src.row(), SignalLinksModel.COL_KEY)
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


# ── Signal Links ──────────────────────────────────────────────────────────────

class SignalLinksTab(QWidget):
    changed = Signal()

    def __init__(self, lm: LinkManager, pm: "PropertyManager", parent=None):
        super().__init__(parent)
        self._lm = lm
        self._pm = pm
        self._updating = False  # re-entrancy guard for _on_item_changed

        lo = QVBoxLayout(self)
        lo.setContentsMargins(4, 4, 4, 4)

        self._model = SignalLinksModel(self)
        self._proxy = _make_signal_links_proxy(self._model)

        self._view = QTableView()
        self._view.setModel(self._proxy)
        self._view.setAlternatingRowColors(True)
        self._view.verticalHeader().hide()
        self._view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._view.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        self._view.setItemDelegate(
            _SignalLinksDelegate(self._model, self._proxy, pm, self._view)
        )
        hh = self._view.horizontalHeader()
        hh.setSectionResizeMode(SignalLinksModel.COL_ON,   QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(SignalLinksModel.COL_KEY,  QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(SignalLinksModel.COL_DEF,  QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(SignalLinksModel.COL_EXPR, QHeaderView.ResizeMode.Stretch)
        hh.setSectionResizeMode(SignalLinksModel.COL_LIVE, QHeaderView.ResizeMode.ResizeToContents)

        lo.addLayout(self._build_filter_bar())
        lo.addWidget(self._view)

        hint = QLabel("Double-click Expression to edit · Bool/choice defaults update on click.")
        hint.setObjectName("info")
        lo.addWidget(hint)

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
            proxy_index = self._proxy.mapFromSource(
                self._model.index(row, SignalLinksModel.COL_DEF)
            )
            if proxy_index.isValid() and self._view.isPersistentEditorOpen(proxy_index):
                self._view.closePersistentEditor(proxy_index)

    def _open_persistent_editors(self) -> None:
        for key, row in self._model.row_for_key.items():
            defn = self._pm._defs.get(key)
            if defn is None:
                continue
            if defn.type is bool or defn.choices:
                src_index = self._model.index(row, SignalLinksModel.COL_DEF)
                proxy_index = self._proxy.mapFromSource(src_index)
                if proxy_index.isValid():
                    self._view.openPersistentEditor(proxy_index)

    def _refresh_live_values(self) -> None:
        """Update the Live Value column; skips unchanged cells to suppress repaints."""
        if not hasattr(self, "_model"):
            return
        self._updating = True
        try:
            for key, row in self._model.row_for_key.items():
                try:
                    new_text = f"{self._pm.get(key):.4f}"
                except (TypeError, ValueError):
                    new_text = str(self._pm.get(key))
                live_item = self._model.item(row, SignalLinksModel.COL_LIVE)
                if live_item is not None and live_item.text() != new_text:
                    live_item.setText(new_text)
        finally:
            self._updating = False

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
        key_item = self._model.item(row, SignalLinksModel.COL_KEY)
        if key_item is None:
            return
        key = key_item.text()
        link_map = {l.sink_key: l for l in self._lm._signal_links}

        if col == SignalLinksModel.COL_EXPR:
            new_expr = item.text().strip()
            link     = link_map.get(key)
            on_item  = self._model.item(row, SignalLinksModel.COL_ON)

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

        elif col == SignalLinksModel.COL_DEF:
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
            on_item = self._model.item(row, SignalLinksModel.COL_ON)
            if not (link and link.expression) or \
                    on_item.checkState() == Qt.CheckState.Unchecked:
                self._pm.set(key, val)
            self.changed.emit()

        elif col == SignalLinksModel.COL_ON:
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
        tb1.addWidget(add_el)
        tb1.addWidget(edit_el)
        tb1.addWidget(rm_el)
        tb1.addWidget(midi_btn)
        tb1.addWidget(key_btn)
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

    def _open_with_event(self, event_str: str) -> None:
        self._capture_status.setText("")
        prefilled = EventLink(event=event_str, action="", enabled=True)
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
    """Main panel: tabs for Signal Links, Events, Envelopes, LFOs, Presets, Sources.

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
        self._sig_tab     = SignalLinksTab(lm, pm)
        self._evt_tab     = EventsTab(lm, pm)
        self._env_tab     = EnvelopesTab(lm)
        self._lfo_tab     = LFOsTab(lm)
        self._preset_tab  = PresetsTab(lm, pm)
        self._src_tab     = SourcesTab(lm)

        tabs.addTab(self._sig_tab,    "Signal Links")
        tabs.addTab(self._evt_tab,    "Events")
        tabs.addTab(self._env_tab,    "Envelopes")
        tabs.addTab(self._lfo_tab,    "LFOs")
        tabs.addTab(self._preset_tab, "Presets")
        tabs.addTab(self._src_tab,    "Sources")

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

        for tab in (self._sig_tab, self._evt_tab, self._env_tab,
                    self._lfo_tab, self._preset_tab):
            tab.changed.connect(self._on_changed)

        self._preset_tab.needs_rebuild.connect(self._rebuild_routing_tabs)

        self._auto_load()

    def _on_changed(self) -> None:
        self._save_timer.start()
        self._status.setText("unsaved changes")

    def _rebuild_routing_tabs(self) -> None:
        """Rebuild all routing tabs after a preset load changes the routing state."""
        for tab in (self._sig_tab, self._evt_tab, self._env_tab, self._lfo_tab):
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
            for tab in (self._sig_tab, self._evt_tab, self._env_tab,
                        self._lfo_tab, self._preset_tab):
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
