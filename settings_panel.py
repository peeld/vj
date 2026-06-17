"""
settings_panel.py — Application settings panel (extracted from gui_merged.py).

SettingsPanel only contains the Display section (monitor picker / fullscreen).
MIDI, OSC, and Audio device configuration are separate panels shown via ControlBar.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QLabel, QScrollArea,
    QFrame, QComboBox, QPushButton,
)
from PySide6.QtCore import Qt, QSettings


class SettingsPanel(QWidget):
    """
    Application settings — Display section only.

    on_monitor_change : optional callback(monitor_index: int) invoked when the
                        user picks a monitor and clicks "Go Fullscreen".
    """

    def __init__(
        self,
        on_monitor_change: Callable[[int], None] | None = None,
        title: str = "Settings",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setMinimumWidth(380)
        self.resize(420, 300)

        self._on_monitor_change = on_monitor_change

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)

        form = QFormLayout(content)
        form.setContentsMargins(6, 6, 6, 6)
        form.setSpacing(4)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self._build_display_section(form)

    def _build_display_section(self, form: QFormLayout) -> None:
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
