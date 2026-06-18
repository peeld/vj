"""
control_bar.py — Narrow always-on-top bar with one toggle button per panel.

Each button is checkable; checked == panel is visible.
Closing a panel unchecks its button via an event filter.
The bar saves/restores its own position in window_positions.json.
"""

from __future__ import annotations

import json
import pathlib
from typing import Callable

from PySide6.QtCore import Qt, QEvent, QTimer
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel, QFrame, QComboBox, QMenu,
)

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
QPushButton[connected=true] {
    border-color: #44ee77;
    color: #44ee77;
}
QPushButton[connected=true]:checked {
    background-color: #1a3028;
    border-color: #44ee77;
    color: #44ee77;
}
QComboBox {
    background-color: #2a2a36;
    color: #c8c8d0;
    border: 1px solid #38383f;
    border-radius: 3px;
    padding: 2px 6px;
    min-width: 80px;
}
QComboBox:hover { border-color: #5eaeff; }
QComboBox QAbstractItemView {
    background-color: #2a2a36;
    color: #c8c8d0;
    selection-background-color: #1e2a40;
    border: 1px solid #38383f;
}
QLabel#bpm {
    color: #c8c8d0;
    min-width: 62px;
    padding: 2px 6px;
    border: 1px solid #38383f;
    border-radius: 3px;
    background-color: #2a2a36;
}
QLabel#elements {
    color: #707078;
    padding: 1px 2px;
}
QLabel#perf {
    font-size: 11px;
    padding: 1px 2px;
}
"""

_KIND_ABBR: dict[str, str] = {
    "cloud":         "cloud",
    "nn_graph":      "nn",
    "circleaxis":    "axis",
    "laser_ribbons": "lasers",
    "falling_discs": "discs",
    "tree_graph":    "tree",
}

_POS_FILE = pathlib.Path(__file__).with_name("window_positions.json")


class ControlBar(QWidget):
    """
    Three-row bar:
      Row 1 — panel toggle buttons (MIDI/OSC tinted green when connected)
      Row 2 — BPM label + preset combobox + hamburger menu
      Row 3 — abbreviated names of currently-visible drawing elements
    """

    def __init__(
        self,
        panels: dict[str, QWidget],
        lm,
        pm,
        midi_router=None,
        osc_router=None,
        bpm=None,
        get_element_snapshot: Callable[[], list[dict]] | None = None,
        perf_monitor=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Warp")
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setStyleSheet(_STYLESHEET)

        self._panels = panels
        self._buttons: dict[str, QPushButton] = {}
        self._lm = lm
        self._pm = pm
        self._midi_router = midi_router
        self._osc_router = osc_router
        self._bpm = bpm
        self._get_element_snapshot = get_element_snapshot
        self._perf = perf_monitor

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(4)

        # Row 1: panel toggle buttons
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        btn_row.setSpacing(4)
        for label, panel in panels.items():
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(False)
            btn.toggled.connect(panel.setVisible)
            self._buttons[label] = btn
            btn_row.addWidget(btn)
            panel.installEventFilter(self)
        layout.addLayout(btn_row)

        # Row 2: BPM label + preset combobox + hamburger
        self._bpm_label = QLabel("♩ —")
        self._bpm_label.setObjectName("bpm")

        self._preset_combo = QComboBox()
        self._preset_combo.setPlaceholderText("— no preset —")
        self._preset_combo.activated.connect(self._on_combo_activated)

        menu_btn = QPushButton("☰")
        menu_btn.setFixedWidth(28)
        menu_btn.clicked.connect(self._show_preset_menu)

        preset_row = QHBoxLayout()
        preset_row.setContentsMargins(0, 0, 0, 0)
        preset_row.setSpacing(4)
        preset_row.addWidget(self._bpm_label)
        preset_row.addWidget(self._preset_combo, stretch=1)
        preset_row.addWidget(menu_btn)
        layout.addLayout(preset_row)

        # Row 3: visible elements
        self._elements_label = QLabel("")
        self._elements_label.setObjectName("elements")
        elements_row = QHBoxLayout()
        elements_row.setContentsMargins(0, 0, 0, 0)
        elements_row.addWidget(self._elements_label)
        elements_row.addStretch()
        layout.addLayout(elements_row)

        # Row 4: performance metrics (hidden when no perf_monitor attached)
        self._perf_label = QLabel("")
        self._perf_label.setObjectName("perf")
        self._perf_label.setVisible(perf_monitor is not None)
        perf_row = QHBoxLayout()
        perf_row.setContentsMargins(0, 0, 0, 0)
        perf_row.addWidget(self._perf_label)
        perf_row.addStretch()
        layout.addLayout(perf_row)

        self.adjustSize()
        self._restore_position()
        self._rebuild_presets()
        self._lm._on_preset_loaded.append(
            lambda _name: QTimer.singleShot(0, self._rebuild_presets)
        )

        self._status_timer = QTimer(self)
        self._status_timer.setInterval(500)
        self._status_timer.timeout.connect(self._tick)
        self._status_timer.start()
        self._tick()

    def show_and_raise(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    # ── status polling ────────────────────────────────────────────────────────

    def _tick(self) -> None:
        self._update_connection_status()
        self._update_bpm_label()
        self._update_elements_row()
        self._update_perf_row()

    def _update_connection_status(self) -> None:
        for label, router in (("MIDI", self._midi_router), ("OSC", self._osc_router)):
            btn = self._buttons.get(label)
            if btn is None or router is None:
                continue
            connected = bool(router.is_connected)
            if btn.property("connected") != connected:
                btn.setProperty("connected", connected)
                btn.style().unpolish(btn)
                btn.style().polish(btn)

    def _update_bpm_label(self) -> None:
        if self._bpm is None:
            return
        self._bpm_label.setText(f"♩ {self._bpm.bpm:.1f}")

    def _update_elements_row(self) -> None:
        if self._get_element_snapshot is None:
            return
        try:
            snapshot = self._get_element_snapshot()
        except Exception:
            return
        parts = [
            _KIND_ABBR.get(el["kind"], el["kind"])
            for el in snapshot
            if el.get("visible")
        ]
        self._elements_label.setText("  ".join(parts))

    def _update_perf_row(self) -> None:
        if self._perf is None:
            return
        snap = self._perf.snapshot()
        if snap is None:
            self._perf_label.setText("fps: —")
            return

        fps      = snap["fps"]
        render   = snap["render_ms"]
        step     = snap["step_ms"]
        scene    = snap["scene_ms"]
        post     = snap["post_ms"]
        headroom = snap["headroom_ms"]
        pct      = snap["budget_pct"]

        text = (
            f"fps:{fps:.0f}  {render:.1f}ms"
            f"  el:{step:.1f} sc:{scene:.1f} po:{post:.1f}"
            f"  +{headroom:.1f}ms"
        )

        if pct < 80:
            color = "#44ee77"   # green — plenty of headroom
        elif pct < 95:
            color = "#ffcc44"   # yellow — getting tight
        else:
            color = "#ff5544"   # red — at or over budget

        self._perf_label.setStyleSheet(f"color: {color};")
        self._perf_label.setText(text)

    # ── event filter: uncheck button when a panel is closed ──────────────────

    def eventFilter(self, obj: QWidget, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Close:
            for label, panel in self._panels.items():
                if panel is obj:
                    self._buttons[label].setChecked(False)
                    break
        return super().eventFilter(obj, event)

    # ── preset combobox ───────────────────────────────────────────────────────

    def _rebuild_presets(self) -> None:
        self._preset_combo.blockSignals(True)
        self._preset_combo.clear()
        names = self._lm.list_link_presets()
        self._preset_combo.addItems(names)
        active = self._lm._active_preset
        if active and active in names:
            self._preset_combo.setCurrentText(active)
        else:
            self._preset_combo.setCurrentIndex(-1)
        self._preset_combo.blockSignals(False)

    def _on_combo_activated(self, index: int) -> None:
        name = self._preset_combo.itemText(index)
        if name:
            self._lm.load_link_preset(name, pm=self._pm)

    def _show_preset_menu(self) -> None:
        from PySide6.QtWidgets import QInputDialog, QMenu
        menu = QMenu(self)

        act_save = menu.addAction("Save Current As…")
        act_save.triggered.connect(self._save_preset)

        current = self._lm._active_preset
        act_del = menu.addAction(f"Delete '{current}'" if current else "Delete")
        act_del.setEnabled(bool(current))
        act_del.triggered.connect(self._delete_preset)

        menu.addSeparator()

        act_prev = menu.addAction("← Previous")
        act_prev.triggered.connect(lambda: self._lm.prev_link_preset(pm=self._pm))

        act_next = menu.addAction("Next →")
        act_next.triggered.connect(lambda: self._lm.next_link_preset(pm=self._pm))

        menu.exec(self.sender().mapToGlobal(self.sender().rect().bottomLeft()))

    def _save_preset(self) -> None:
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Save Preset", "Preset name:")
        if ok and name.strip():
            self._lm.save_link_preset(name.strip(), pm_props=self._pm.snapshot_nondefault())
            self._rebuild_presets()

    def _delete_preset(self) -> None:
        name = self._lm._active_preset
        if name:
            self._lm.delete_link_preset(name)
            self._rebuild_presets()

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
