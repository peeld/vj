"""
control_bar.py — Narrow always-on-top bar with one toggle button per panel.

Each button is checkable; checked == panel is visible.
Closing a panel unchecks its button via an event filter.
The bar saves/restores its own position in window_positions.json.
"""

from __future__ import annotations

import json
import pathlib

from PySide6.QtCore import Qt, QEvent
from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QFrame

_STYLESHEET = """
QWidget {
    background-color: #1a1a22;
    color: #c8c8d0;
    font-family: Consolas, "Courier New", monospace;
    font-size: 12px;
}
QPushButton {
    background-color: #2a2a36;
    color: #c8c8d0;
    border: 1px solid #38383f;
    border-radius: 3px;
    padding: 4px 12px;
    min-width: 50px;
}
QPushButton:hover   { background-color: #32323f; border-color: #5eaeff; }
QPushButton:pressed { background-color: #1a1a28; }
QPushButton:checked {
    background-color: #1e2a40;
    color: #5eaeff;
    border-color: #5eaeff;
}
"""

_POS_FILE = pathlib.Path(__file__).with_name("window_positions.json")


class ControlBar(QWidget):
    """
    Narrow horizontal bar with one QPushButton per panel.
    panels: ordered dict of { display_label: QWidget }
    """

    def __init__(self, panels: dict[str, QWidget], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Warp")
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setStyleSheet(_STYLESHEET)

        self._panels = panels
        self._buttons: dict[str, QPushButton] = {}

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        for label, panel in panels.items():
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(False)
            btn.toggled.connect(panel.setVisible)
            self._buttons[label] = btn
            layout.addWidget(btn)
            panel.installEventFilter(self)

        self.adjustSize()
        self._restore_position()

    def show_and_raise(self) -> None:
        print("FOO")
        self.showNormal()
        self.raise_()
        self.activateWindow()

    # ── event filter: uncheck button when a panel is closed ──────────────────

    def eventFilter(self, obj: QWidget, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Close:
            for label, panel in self._panels.items():
                if panel is obj:
                    self._buttons[label].setChecked(False)
                    break
        return super().eventFilter(obj, event)

    # ── position persistence ──────────────────────────────────────────────────

    def _load_positions(self) -> dict:
        try:
            return json.loads(_POS_FILE.read_text())
        except Exception:
            return {}

    def _restore_position(self) -> None:
        data = self._load_positions()
        if "ControlBar" in data:
            g = data["ControlBar"]
            self.setGeometry(g["x"], g["y"], g["w"], g["h"])

    def save_position(self) -> None:
        data = self._load_positions()
        g = self.geometry()
        data["ControlBar"] = {"x": g.x(), "y": g.y(), "w": g.width(), "h": g.height()}
        try:
            _POS_FILE.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def closeEvent(self, event) -> None:
        self.save_position()
        super().closeEvent(event)

    def moveEvent(self, event) -> None:
        self.save_position()
        super().moveEvent(event)
