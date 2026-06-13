"""
param_dialog.py — live parameter inspector / editor.

Runs a PySide6 QWidget in a background daemon thread (Windows-safe).
Polls watched dataclass instances every POLL_MS milliseconds and reflects
live values.  Edits are applied to the source object immediately via setattr.

Widget types
------------
- bool fields       → QCheckBox  (auto-detected)
- ("combo", [...])  → QComboBox  (explicit hint)
- everything else   → QLineEdit

Usage
-----
    from param_dialog import start_param_dialog

    start_param_dialog(
        ("Post-FX", params_obj),
        ("Scene",   controls_obj, {"blend_mode": ("combo", ["lerp", "additive"])}),
        title="My App Params",
    )

Each positional arg can be:
    ("Label", obj)             — no widget hints
    ("Label", obj, hints_dict) — per-field hints
    obj                        — class name as label, no hints
"""

import threading
import dataclasses
from typing import Any

from PySide6.QtWidgets import (
    QApplication, QWidget, QFormLayout, QLineEdit,
    QLabel, QScrollArea, QVBoxLayout, QFrame,
    QCheckBox, QComboBox,
)
from PySide6.QtCore import QTimer, Qt


POLL_MS = 100

_STYLESHEET = """
QWidget {
    background-color: #1a1a22;
    color: #c8c8d0;
    font-family: Consolas, "Courier New", monospace;
    font-size: 12px;
}
QLineEdit {
    background-color: #111118;
    color: #5eaeff;
    border: 1px solid #38383f;
    border-radius: 3px;
    padding: 2px 6px;
    selection-background-color: #2a4070;
}
QLineEdit:focus {
    border: 1px solid #5eaeff;
}
QCheckBox {
    color: #c8c8d0;
    spacing: 6px;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #38383f;
    border-radius: 3px;
    background-color: #111118;
}
QCheckBox::indicator:checked {
    background-color: #5eaeff;
    border-color: #5eaeff;
}
QComboBox {
    background-color: #111118;
    color: #5eaeff;
    border: 1px solid #38383f;
    border-radius: 3px;
    padding: 2px 6px;
    min-width: 120px;
}
QComboBox:focus {
    border: 1px solid #5eaeff;
}
QComboBox QAbstractItemView {
    background-color: #1a1a22;
    color: #c8c8d0;
    selection-background-color: #2a4070;
    border: 1px solid #38383f;
}
QComboBox::drop-down {
    border: none;
    width: 18px;
}
QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid #707078;
    margin-right: 4px;
}
QLabel {
    color: #707078;
}
QLabel#section_header {
    color: #c8c8d0;
    font-weight: bold;
    padding-top: 6px;
}
QScrollArea, QScrollArea > QWidget > QWidget {
    background-color: #1a1a22;
    border: none;
}
QScrollBar:vertical {
    background: #1a1a22;
    width: 6px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #38383f;
    border-radius: 3px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
"""


# ── dialog ────────────────────────────────────────────────────────────────────

class ParamDialog(QWidget):
    """
    Live name=value editor for one or more dataclass instances.

    targets : list of (section_label, obj, hints_dict) triples
              hints_dict maps field_name -> hint, where hint is:
                  "combo", [choices]  →  QComboBox
                  (auto) bool field   →  QCheckBox
                  anything else       →  QLineEdit
    """

    def __init__(self, targets: list[tuple[str, Any, dict]], title: str = "Parameters"):
        super().__init__()
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setStyleSheet(_STYLESHEET)
        self.setMinimumWidth(380)

        # { id(obj): { field_name: (widget, ftype, hint) } }
        self._rows: dict[int, dict[str, tuple[Any, type, Any]]] = {}
        self._targets = targets

        self._build_ui()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(POLL_MS)

    # ── build ─────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        form = QFormLayout(container)
        form.setContentsMargins(6, 6, 6, 6)
        form.setSpacing(4)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        for section, obj, hints in self._targets:
            obj_id = id(obj)
            self._rows[obj_id] = {}

            hdr = QLabel(section)
            hdr.setObjectName("section_header")
            form.addRow(hdr)

            sep = QFrame()
            sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet("color: #2e2e38;")
            form.addRow(sep)

            for fname, ftype, fval in self._iter_fields(obj):
                hint = hints.get(fname)
                # auto-detect booleans
                if hint is None and ftype is bool:
                    hint = "check"

                widget = self._make_widget(obj, obj_id, fname, ftype, fval, hint)
                lbl = QLabel(fname)
                form.addRow(lbl, widget)
                self._rows[obj_id][fname] = (widget, ftype, hint)

        scroll.setWidget(container)
        outer.addWidget(scroll)

        n_rows = sum(len(r) for r in self._rows.values())
        self.resize(400, min(80 + 26 * n_rows, 720))

    # ── widget factory ────────────────────────────────────────────────────────

    def _make_widget(self, obj, obj_id, fname, ftype, fval, hint) -> QWidget:
        if hint == "check":
            w = QCheckBox()
            w.setChecked(bool(fval))
            setter = self._make_check_setter(obj, obj_id, fname)
            w.toggled.connect(setter)
            return w

        if isinstance(hint, (list, tuple)) and len(hint) == 2 and hint[0] == "combo":
            choices = hint[1]
            w = QComboBox()
            w.addItems([str(c) for c in choices])
            cur = str(fval)
            if cur in choices:
                w.setCurrentText(cur)
            setter = self._make_combo_setter(obj, obj_id, fname)
            w.currentTextChanged.connect(setter)
            return w

        # default: QLineEdit
        w = QLineEdit(self._fmt(fval))
        w.setFixedWidth(170)
        setter = self._make_line_setter(obj, obj_id, fname, ftype)
        w.editingFinished.connect(setter)
        return w

    # ── setters ───────────────────────────────────────────────────────────────

    def _make_check_setter(self, obj, obj_id, fname):
        def _apply(checked: bool):
            setattr(obj, fname, checked)
        return _apply

    def _make_combo_setter(self, obj, obj_id, fname):
        def _apply(text: str):
            setattr(obj, fname, text)
        return _apply

    def _make_line_setter(self, obj, obj_id, fname, ftype):
        def _apply():
            widget, _, _ = self._rows[id(obj)][fname]
            text = widget.text().strip()
            try:
                if ftype is int:
                    val = int(float(text))
                else:
                    val = float(text)
                setattr(obj, fname, val)
            except (ValueError, TypeError):
                widget.setText(self._fmt(getattr(obj, fname)))
        return _apply

    # ── poll ──────────────────────────────────────────────────────────────────

    def _poll(self) -> None:
        for _, obj, _ in self._targets:
            obj_id = id(obj)
            for fname, (widget, ftype, hint) in self._rows.get(obj_id, {}).items():
                if widget.hasFocus():
                    continue
                live = getattr(obj, fname, None)

                if hint == "check":
                    checked = bool(live)
                    if widget.isChecked() != checked:
                        widget.blockSignals(True)
                        widget.setChecked(checked)
                        widget.blockSignals(False)

                elif isinstance(hint, (list, tuple)) and hint[0] == "combo":
                    text = str(live)
                    if widget.currentText() != text:
                        widget.blockSignals(True)
                        widget.setCurrentText(text)
                        widget.blockSignals(False)

                else:
                    text = self._fmt(live)
                    if widget.text() != text:
                        widget.setText(text)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _iter_fields(obj: Any):
        if dataclasses.is_dataclass(obj):
            for f in dataclasses.fields(obj):
                val = getattr(obj, f.name)
                t = f.type if isinstance(f.type, type) else type(val)
                yield f.name, t, val
        else:
            for name, ann in getattr(obj, "__annotations__", {}).items():
                if not name.startswith("_"):
                    val = getattr(obj, name, None)
                    t = ann if isinstance(ann, type) else type(val)
                    yield name, t, val

    @staticmethod
    def _fmt(val: Any) -> str:
        if isinstance(val, float):
            return f"{val:.6g}"
        return str(val)


# ── public entry point ────────────────────────────────────────────────────────

def start_param_dialog(
    *targets,
    title: str = "Parameters",
    x: int | None = None,
    y: int | None = None,
) -> threading.Thread:
    """
    Launch a ParamDialog in a background daemon thread and return the thread.
    Call this before starting your main rendering loop.

    Each positional arg is one of:
        ("Label", obj)              — no widget hints
        ("Label", obj, hints_dict)  — per-field widget hints:
                                        "field": "check"
                                        "field": ("combo", ["a", "b", ...])
        obj                         — class name as label, no hints
    """
    pairs: list[tuple[str, Any, dict]] = []
    for t in targets:
        if isinstance(t, tuple):
            if len(t) == 3 and isinstance(t[0], str):
                pairs.append((t[0], t[1], t[2]))
            elif len(t) == 2 and isinstance(t[0], str):
                pairs.append((t[0], t[1], {}))
            else:
                pairs.append((type(t[0]).__name__, t[0], {}))
        else:
            pairs.append((type(t).__name__, t, {}))

    def _run() -> None:
        app = QApplication.instance() or QApplication([])
        dlg = ParamDialog(pairs, title=title)
        if x is not None and y is not None:
            dlg.move(x, y)
        dlg.show()
        app.exec()

    t = threading.Thread(target=_run, daemon=True, name="param-dialog")
    t.start()
    return t
