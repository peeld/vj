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
from typing import Any, Callable

from PySide6.QtWidgets import (
    QApplication, QWidget, QFormLayout, QLineEdit,
    QLabel, QScrollArea, QVBoxLayout, QFrame,
    QCheckBox, QComboBox, QPushButton,
)
from PySide6.QtCore import QTimer, Qt, QSettings
from PySide6.QtGui import QGuiApplication


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

    on_monitor_change : optional callback(monitor_index: int) called when the
                        user selects a monitor in the Display section.  When
                        provided, a "Display" section is appended with a combo
                        box listing all attached screens and a "Go Fullscreen"
                        button.
    """

    def __init__(
        self,
        targets: list[tuple[str, Any, dict]],
        title: str = "Parameters",
        on_monitor_change: Callable[[int], None] | None = None,
    ):
        super().__init__()
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setStyleSheet(_STYLESHEET)
        self.setMinimumWidth(380)

        self._on_monitor_change = on_monitor_change

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

        if self._on_monitor_change is not None:
            self._build_monitor_section(form)

        scroll.setWidget(container)
        outer.addWidget(scroll)

        n_rows = sum(len(r) for r in self._rows.values())
        extra = 60 if self._on_monitor_change is not None else 0
        self.resize(400, min(80 + 26 * n_rows + extra, 720))

    # ── monitor section ───────────────────────────────────────────────────────

    def _build_monitor_section(self, form: QFormLayout) -> None:
        """Append a 'Display' section with a monitor picker and fullscreen button."""
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
    on_monitor_change: Callable[[int], None] | None = None,
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

    Keyword args:
        on_monitor_change : callable(monitor_index: int) | None
            When provided, a "Display" section is appended with a monitor
            combo box and a "Go Fullscreen" button.  Clicking the button
            invokes the callback with the selected monitor index (0-based).
            Use ``apply_fullscreen_to_monitor(self.wnd, index)`` in the
            callback or render loop to apply the switch via GLFW.
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
        dlg = ParamDialog(pairs, title=title, on_monitor_change=on_monitor_change)
        if x is not None and y is not None:
            dlg.move(x, y)
        dlg.show()
        # Auto-apply saved fullscreen monitor on startup
        if on_monitor_change is not None:
            qs = QSettings("WarpApp", "WarpApp")
            if qs.contains("display/monitor_index"):
                QTimer.singleShot(0, dlg._emit_monitor_change)
        app.exec()

    t = threading.Thread(target=_run, daemon=True, name="param-dialog")
    t.start()
    return t


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

        start_param_dialog(
            ...,
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
