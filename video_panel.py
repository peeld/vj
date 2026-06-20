"""
video_panel.py — Multi-slot video playback panel.

Manages a list of VideoPlayer instances; only one plays at a time.
Switching is instant: pause old, play new, wire VideoElement via set_player().
Publishes video.playing / video.position / video.duration / video.slot / video.count
to the SourceRegistry each poll cycle.

EventLink action:  video.switch(N)   — switch to slot N (0-based)
"""

from __future__ import annotations

import os

from PySide6.QtCore    import QTimer, Qt
from PySide6.QtGui     import QBrush, QColor
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QCheckBox,
    QFrame, QListWidget, QListWidgetItem, QAbstractItemView,
)


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
QLabel#info      { color: #707078; font-style: italic; }

QPushButton {
    background-color: #2a2a36;
    color: #c8c8d0;
    border: 1px solid #38383f;
    border-radius: 3px;
    padding: 3px 10px;
    min-width: 52px;
}
QPushButton:hover   { background-color: #32323f; border-color: #5eaeff; }
QPushButton:pressed { background-color: #1a1a28; }
QPushButton:disabled { color: #484858; border-color: #2e2e38; }

QCheckBox { color: #c8c8d0; spacing: 5px; }
QCheckBox::indicator {
    width: 13px; height: 13px;
    border: 1px solid #38383f;
    border-radius: 2px;
    background: #111118;
}
QCheckBox::indicator:checked { background: #2a5080; border-color: #5eaeff; }

QSlider::groove:horizontal {
    height: 4px; background: #2e2e38; border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 12px; height: 12px; margin: -4px 0;
    background: #5eaeff; border-radius: 6px;
}
QSlider::sub-page:horizontal { background: #2a5080; border-radius: 2px; }

QListWidget {
    background-color: #111118;
    border: 1px solid #38383f;
    color: #707078;
}
QListWidget::item { padding: 3px 6px; }
QListWidget::item:selected { background-color: #1e3050; color: #c8c8d0; }
QListWidget::item:hover { background-color: #1a2030; }

QFrame#sep { color: #2e2e38; max-height: 1px; background-color: #2e2e38; }
"""

_COLOR_ACTIVE   = QColor("#5eaeff")
_COLOR_INACTIVE = QColor("#505060")


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


def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 60}:{s % 60:02d}"


# ── panel ─────────────────────────────────────────────────────────────────────

class VideoPanel(QWidget):
    """
    Multi-slot video panel.

    Parameters
    ----------
    get_video_element : callable() -> VideoElement | None
    source_registry   : SourceRegistry | None
    title             : str
    """

    def __init__(
        self,
        get_video_element = None,
        source_registry   = None,
        title             : str = "Video",
        parent            = None,
    ):
        super().__init__(parent)
        self._get_element = get_video_element or (lambda: None)
        self._registry    = source_registry
        self._slots: list[dict] = []   # {"path": str, "player": VideoPlayer, "name": str}
        self._active_idx: int   = -1

        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setStyleSheet(_STYLESHEET)
        self.setMinimumWidth(380)
        self.resize(440, 340)

        self._build_ui()

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(100)
        self._poll_timer.timeout.connect(self._update_ui)
        self._poll_timer.start()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        root.addWidget(_hdr("Video Slots"))
        root.addWidget(_sep())

        toolbar = QHBoxLayout()
        self._add_btn = QPushButton("Add File…")
        self._add_btn.clicked.connect(self._on_add)
        toolbar.addWidget(self._add_btn)
        self._remove_btn = QPushButton("Remove")
        self._remove_btn.clicked.connect(self._on_remove)
        self._remove_btn.setEnabled(False)
        toolbar.addWidget(self._remove_btn)
        toolbar.addStretch(1)
        root.addLayout(toolbar)

        self._slot_list = QListWidget()
        self._slot_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._slot_list.setMaximumHeight(110)
        self._slot_list.currentRowChanged.connect(self._on_slot_selected)
        root.addWidget(self._slot_list)

        root.addWidget(_sep())
        root.addWidget(_hdr("Playback"))

        transport = QHBoxLayout()
        self._play_btn = QPushButton("Play")
        self._play_btn.clicked.connect(self._on_play)
        self._play_btn.setEnabled(False)
        transport.addWidget(self._play_btn)
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.clicked.connect(self._on_pause)
        self._pause_btn.setEnabled(False)
        transport.addWidget(self._pause_btn)
        self._loop_chk = QCheckBox("Loop")
        self._loop_chk.setChecked(True)
        self._loop_chk.stateChanged.connect(self._on_loop_changed)
        transport.addWidget(self._loop_chk)
        transport.addStretch(1)
        root.addLayout(transport)

        seek_row = QHBoxLayout()
        self._seek = QSlider(Qt.Horizontal)
        self._seek.setRange(0, 1000)
        self._seek.sliderReleased.connect(self._on_seek)
        self._seek.setEnabled(False)
        seek_row.addWidget(self._seek, 1)
        self._time_label = QLabel("0:00 / 0:00")
        self._time_label.setObjectName("value")
        self._time_label.setFixedWidth(80)
        self._time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        seek_row.addWidget(self._time_label)
        root.addLayout(seek_row)

        self._info_label = QLabel("FPS: —   Size: —")
        self._info_label.setObjectName("info")
        root.addWidget(self._info_label)

        root.addStretch(1)

    # ── slot management ───────────────────────────────────────────────────────

    def _active_player(self):
        if 0 <= self._active_idx < len(self._slots):
            return self._slots[self._active_idx]["player"]
        return None

    def switch_to(self, idx: int) -> None:
        """Activate slot idx. Safe to call from Qt thread only."""
        if idx < 0 or idx >= len(self._slots):
            return
        old = self._active_player()
        if old is not None:
            old.pause()
        self._active_idx = idx
        new = self._active_player()
        if new is not None:
            new.play()
            el = self._get_element()
            if el is not None:
                el.set_player(new)
        self._refresh_list()
        self._update_controls()

    def _add_slot(self, path: str) -> None:
        from video_player import VideoPlayer
        player = VideoPlayer(path, loop=self._loop_chk.isChecked())
        player.pause()   # inactive until switched to
        self._slots.append({"path": path, "player": player, "name": os.path.basename(path)})
        self._refresh_list()
        if self._active_idx < 0:
            self.switch_to(len(self._slots) - 1)
        self._remove_btn.setEnabled(True)

    def _remove_selected_slot(self) -> None:
        row = self._slot_list.currentRow()
        if row < 0 or row >= len(self._slots):
            return
        slot = self._slots.pop(row)
        slot["player"].close()

        if not self._slots:
            self._active_idx = -1
            el = self._get_element()
            if el is not None:
                el.set_player(None)
        elif self._active_idx >= len(self._slots):
            self.switch_to(len(self._slots) - 1)
        elif self._active_idx == row:
            self.switch_to(min(row, len(self._slots) - 1))
        elif self._active_idx > row:
            self._active_idx -= 1

        self._refresh_list()
        self._update_controls()
        self._remove_btn.setEnabled(bool(self._slots))

    def _refresh_list(self) -> None:
        self._slot_list.blockSignals(True)
        self._slot_list.clear()
        for i, slot in enumerate(self._slots):
            active = (i == self._active_idx)
            item   = QListWidgetItem(f" {'●' if active else '◌'}  {slot['name']}")
            item.setForeground(QBrush(_COLOR_ACTIVE if active else _COLOR_INACTIVE))
            self._slot_list.addItem(item)
        if 0 <= self._active_idx < self._slot_list.count():
            self._slot_list.setCurrentRow(self._active_idx)
        self._slot_list.blockSignals(False)

    def _update_controls(self) -> None:
        p = self._active_player()
        has = p is not None
        self._play_btn.setEnabled(has)
        self._pause_btn.setEnabled(has)
        self._seek.setEnabled(has)
        if has:
            self._info_label.setText(f"FPS: {p.fps:.2f}   Size: {p.width}×{p.height}")
        else:
            self._info_label.setText("FPS: —   Size: —")
            self._time_label.setText("0:00 / 0:00")

    # ── event handlers ────────────────────────────────────────────────────────

    def _on_add(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video (*.mp4 *.mov *.mkv *.avi *.webm);;All files (*)",
        )
        if path:
            self._add_slot(path)

    def _on_remove(self) -> None:
        self._remove_selected_slot()

    def _on_slot_selected(self, row: int) -> None:
        if row >= 0 and row != self._active_idx:
            self.switch_to(row)

    def _on_play(self) -> None:
        p = self._active_player()
        if p:
            p.play()

    def _on_pause(self) -> None:
        p = self._active_player()
        if p:
            p.pause()

    def _on_loop_changed(self, state: int) -> None:
        loop = bool(state)
        for slot in self._slots:
            slot["player"]._loop = loop

    def _on_seek(self) -> None:
        p = self._active_player()
        if p and p.duration > 0:
            t = (self._seek.value() / 1000.0) * p.duration
            p.seek(t)

    # ── poll timer ────────────────────────────────────────────────────────────

    def _update_ui(self) -> None:
        p = self._active_player()
        if p is None:
            return

        pos      = p.position
        duration = p.duration
        playing  = p.playing

        if self._registry is not None:
            try:
                self._registry.update("video.playing",  1.0 if playing else 0.0)
                self._registry.update("video.position", pos)
                self._registry.update("video.duration", duration)
                self._registry.update("video.slot",     float(self._active_idx))
                self._registry.update("video.count",    float(len(self._slots)))
            except Exception:
                pass

        self._time_label.setText(f"{_fmt_time(pos)} / {_fmt_time(duration)}")

        if duration > 0 and not self._seek.isSliderDown():
            self._seek.blockSignals(True)
            self._seek.setValue(int(pos / duration * 1000))
            self._seek.blockSignals(False)

    # ── cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        for slot in self._slots:
            slot["player"].close()
        super().closeEvent(event)


# ── standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    panel = VideoPanel(title="Video (demo)")
    panel.show()
    sys.exit(app.exec())
