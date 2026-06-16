"""
elements_panel.py — Scene Elements manager panel

PySide6 panel that lets the user inspect, toggle visibility on, remove, and
add scene drawing elements (MergedGUI.elements) at runtime.

The GL thread owns the actual element list; this panel never touches GL
objects directly. It polls a read-only snapshot (a plain list of dicts,
refreshed once per frame by MergedGUI.on_render) and sends commands back via
LinkManager's EventBus -- the same thread-safe, GL-thread-drained queue used
for MIDI/audio-triggered events elsewhere in this project.

Usage::

    from elements_panel import ElementsPanel

    panel = ElementsPanel(
        event_bus    = lm.event_bus,
        get_snapshot = lambda: _element_snapshot,
    )
    panel.show()
"""

from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore    import Qt, Signal, QTimer
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QScrollArea,
    QFrame, QSizePolicy, QCheckBox,
)

from elements.base import ELEMENT_TYPES
import elements.cloud, elements.nn_graph, elements.circleaxis, elements.laser_ribbons  # noqa: F401 -- registers cloud/nn_graph/circles/lasers


# ── stylesheet (matches color_panel.py / audio_panel.py dark theme) ───────────

_STYLESHEET = """
QWidget {
    background-color: #1a1a22;
    color: #c8c8d0;
    font-family: Consolas, "Courier New", monospace;
    font-size: 12px;
}
QLabel           { color: #707078; }
QLabel#hdr       { color: #c8c8d0; font-weight: bold; padding-top: 4px; }
QLabel#name      { color: #c8c8d0; }
QLabel#kind      { color: #5eaeff; min-width: 64px; }
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
QPushButton#remove  { color: #e07070; border-color: #5a3a3a; min-width: 28px; padding: 3px 6px; }
QPushButton#remove:hover { border-color: #e07070; }

QCheckBox { color: #c8c8d0; spacing: 6px; }
QCheckBox::indicator {
    width: 13px; height: 13px;
    border: 1px solid #38383f;
    border-radius: 2px;
    background: #111118;
}
QCheckBox::indicator:checked { background: #5eaeff; border-color: #5eaeff; }

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
QFrame#row {
    background-color: #12121a;
    border: 1px solid #2e2e38;
    border-radius: 4px;
}
"""


def _sep() -> QFrame:
    f = QFrame()
    f.setObjectName("sep")
    f.setFrameShape(QFrame.HLine)
    return f


def _hdr(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("hdr")
    return lbl


# ── single element row ────────────────────────────────────────────────────────

class _ElementRow(QFrame):
    """One row in the element list: visible checkbox | name | kind | remove."""

    visibility_changed = Signal(str, bool)   # (name, value)
    remove_clicked      = Signal(str)        # (name)

    def __init__(self, name: str, kind: str, visible: bool, parent: QWidget = None):
        super().__init__(parent)
        self.name = name
        self.setObjectName("row")

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 5, 8, 5)
        row.setSpacing(8)

        self._check = QCheckBox()
        self._check.setChecked(visible)
        self._check.toggled.connect(lambda checked: self.visibility_changed.emit(self.name, checked))
        row.addWidget(self._check)

        name_lbl = QLabel(name)
        name_lbl.setObjectName("name")
        row.addWidget(name_lbl, stretch=1)

        kind_lbl = QLabel(kind)
        kind_lbl.setObjectName("kind")
        row.addWidget(kind_lbl)

        remove_btn = QPushButton("✕")
        remove_btn.setObjectName("remove")
        remove_btn.setToolTip(f"Remove {name}")
        remove_btn.clicked.connect(lambda: self.remove_clicked.emit(self.name))
        row.addWidget(remove_btn)

    def set_visible_checked(self, visible: bool) -> None:
        """Sync the checkbox without re-emitting visibility_changed."""
        if self._check.isChecked() != visible:
            self._check.blockSignals(True)
            self._check.setChecked(visible)
            self._check.blockSignals(False)


# ── main panel ────────────────────────────────────────────────────────────────

class ElementsPanel(QWidget):
    """
    Scene elements manager: list current DrawingElement instances, toggle
    visibility, remove, and spawn new ones from the type registry.

    Parameters
    ----------
    event_bus : LinkManager.event_bus
        Commands (add/remove/set_visible) are fired here; the GL thread
        drains and applies them once per frame via MergedGUI's subscriptions.
    get_snapshot : callable
        Returns the current read-only list[dict] of {"name","kind","visible"}
        (e.g. ``lambda: gui_merged._element_snapshot``).
    """

    def __init__(
        self,
        event_bus,
        get_snapshot: Callable[[], list[dict]],
        title: str = "Scene Elements",
        parent=None,
    ):
        super().__init__(parent)
        self._event_bus    = event_bus
        self._get_snapshot = get_snapshot
        self._rows: dict[str, _ElementRow] = {}

        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setStyleSheet(_STYLESHEET)
        self.setMinimumWidth(360)
        self.resize(420, 480)

        self._build_ui()

        self._timer = QTimer(self)
        self._timer.setInterval(150)
        self._timer.timeout.connect(self._refresh)
        self._timer.start()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        root.addWidget(_hdr("Elements"))
        root.addWidget(_sep())

        self._row_container = QWidget()
        self._row_layout     = QVBoxLayout(self._row_container)
        self._row_layout.setContentsMargins(0, 2, 0, 2)
        self._row_layout.setSpacing(4)
        self._row_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self._row_container)
        root.addWidget(scroll, stretch=1)

        root.addSpacing(4)
        root.addWidget(_hdr("Add"))
        root.addWidget(_sep())

        add_row = QHBoxLayout()
        self._type_combo = QComboBox()
        self._type_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        add_row.addWidget(self._type_combo)

        add_btn = QPushButton("+ Add")
        add_btn.setObjectName("add")
        add_btn.clicked.connect(self._on_add_clicked)
        add_row.addWidget(add_btn)
        root.addLayout(add_row)

        self._add_btn = add_btn
        self._refresh_type_combo(set())

        self._status_lbl = QLabel("")
        self._status_lbl.setObjectName("info")
        root.addWidget(self._status_lbl)

    def _refresh_type_combo(self, present_kinds: set[str]) -> None:
        """Repopulate the "+ Add" combo with kinds not already live.

        Only one instance per kind is ever permitted (see MergedGUI.add_element),
        so kinds already present are excluded here as the normal UI path's
        dedup guard.
        """
        current = self._type_combo.currentText()
        available = [k for k in sorted(ELEMENT_TYPES) if k not in present_kinds]
        self._type_combo.clear()
        self._type_combo.addItems(available)
        idx = self._type_combo.findText(current)
        if idx >= 0:
            self._type_combo.setCurrentIndex(idx)
        self._add_btn.setEnabled(bool(available))

    # ── row management ────────────────────────────────────────────────────────

    def _refresh(self) -> None:
        snap = self._get_snapshot()
        names = [item["name"] for item in snap]

        if list(self._rows) != names:
            self._rebuild_rows(snap)
            return

        for item in snap:
            row = self._rows.get(item["name"])
            if row is not None:
                value = item["visible"]
                if isinstance(value, float):
                    value = value > 0.5
                row.set_visible_checked(value)

    def _rebuild_rows(self, snap: list[dict]) -> None:
        while self._row_layout.count() > 1:
            item = self._row_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._rows.clear()

        for i, entry in enumerate(snap):
            visible = entry["visible"]
            if isinstance(visible, float):
                visible = visible > 0.5
            row = _ElementRow(entry["name"], entry["kind"], visible, parent=self._row_container)
            row.visibility_changed.connect(self._on_row_visibility_changed)
            row.remove_clicked.connect(self._on_row_remove)
            self._rows[entry["name"]] = row
            self._row_layout.insertWidget(i, row)

        self._refresh_type_combo({entry["kind"] for entry in snap})
        self._status_lbl.setText(f"{len(snap)} element(s)")

    # ── commands → EventBus ──────────────────────────────────────────────────

    def _on_row_visibility_changed(self, name: str, value: bool) -> None:
        self._event_bus.fire("element.set_visible", (name, value))

    def _on_row_remove(self, name: str) -> None:
        self._event_bus.fire("element.remove", name)

    def _on_add_clicked(self) -> None:
        kind = self._type_combo.currentText()
        if kind:
            self._event_bus.fire("element.add", kind)


# ── standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    class _FakeBus:
        def fire(self, event_id, payload=None):
            print(f"[elements_panel demo] fire({event_id!r}, {payload!r})")

    _demo_snapshot = [
        {"name": "cloud_1", "kind": "cloud", "visible": True},
        {"name": "nn_graph_1", "kind": "nn_graph", "visible": True},
    ]

    app = QApplication.instance() or QApplication(sys.argv)
    panel = ElementsPanel(event_bus=_FakeBus(), get_snapshot=lambda: _demo_snapshot)
    panel.show()
    sys.exit(app.exec())
