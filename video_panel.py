"""
video_panel.py — Video file picker + playback controls.

VideoPanel owns a VideoPlayer and wires it into a VideoElement via set_player().
It also publishes video.playing / video.position / video.duration to the
SourceRegistry each poll cycle so those values are available as link sources.

Usage::

    from video_panel import VideoPanel

    panel = VideoPanel(
        get_video_element=lambda: ...,  # callable returning live VideoElement | None
        source_registry=lm.source_registry,
        title="Video",
    )
    panel.show()
"""

from __future__ import annotations

import os

from PySide6.QtCore    import QTimer, Qt
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QCheckBox,
    QFrame, QSizePolicy,
)


# ── stylesheet (matches audio_panel dark theme) ────────────────────────────────

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
QLabel#path      { color: #9898a8; font-style: italic; }

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
    height: 4px;
    background: #2e2e38;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 12px; height: 12px;
    margin: -4px 0;
    background: #5eaeff;
    border-radius: 6px;
}
QSlider::sub-page:horizontal { background: #2a5080; border-radius: 2px; }

QFrame#sep {
    color: #2e2e38;
    max-height: 1px;
    background-color: #2e2e38;
}
"""


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
    Video playback controls panel.

    Parameters
    ----------
    get_video_element : callable() -> VideoElement | None
        Called when a file is opened to wire the player into the live element.
        May return None if the VideoElement hasn't been added to the scene yet.
    source_registry : SourceRegistry | None
        Receives video.playing / video.position / video.duration each poll tick.
    title : str
        Window title.
    """

    def __init__(
        self,
        get_video_element = None,   # callable() -> VideoElement | None
        source_registry   = None,
        title             : str = "Video",
        parent            = None,
    ):
        super().__init__(parent)
        self._get_element = get_video_element or (lambda: None)
        self._registry    = source_registry
        self._player      = None   # VideoPlayer | None

        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setStyleSheet(_STYLESHEET)
        self.setMinimumWidth(360)
        self.resize(420, 200)

        self._build_ui()

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(100)
        self._poll_timer.timeout.connect(self._update_ui)
        self._poll_timer.start()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        # File row
        root.addWidget(_hdr("Video File"))
        root.addWidget(_sep())

        file_row = QHBoxLayout()
        self._open_btn = QPushButton("Open File…")
        self._open_btn.clicked.connect(self._on_open)
        file_row.addWidget(self._open_btn)

        self._path_label = QLabel("(no file)")
        self._path_label.setObjectName("path")
        self._path_label.setWordWrap(False)
        self._path_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        file_row.addWidget(self._path_label, 1)
        root.addLayout(file_row)

        # Transport row
        root.addWidget(_sep())
        transport_row = QHBoxLayout()

        self._play_btn = QPushButton("Play")
        self._play_btn.clicked.connect(self._on_play)
        self._play_btn.setEnabled(False)
        transport_row.addWidget(self._play_btn)

        self._pause_btn = QPushButton("Pause")
        self._pause_btn.clicked.connect(self._on_pause)
        self._pause_btn.setEnabled(False)
        transport_row.addWidget(self._pause_btn)

        self._loop_chk = QCheckBox("Loop")
        self._loop_chk.setChecked(True)
        self._loop_chk.stateChanged.connect(self._on_loop_changed)
        transport_row.addWidget(self._loop_chk)
        transport_row.addStretch(1)
        root.addLayout(transport_row)

        # Seek bar + time label
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

        # Info row
        self._info_label = QLabel("FPS: —   Size: —")
        self._info_label.setObjectName("info")
        root.addWidget(self._info_label)

        root.addStretch(1)

    # ── event handlers ────────────────────────────────────────────────────────

    def _on_open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video (*.mp4 *.mov *.mkv *.avi *.webm);;All files (*)",
        )
        if not path:
            return
        self._load(path)

    def _load(self, path: str) -> None:
        from video_player import VideoPlayer
        if self._player is not None:
            self._player.close()

        self._player = VideoPlayer(path, loop=self._loop_chk.isChecked())
        short = os.path.basename(path)
        self._path_label.setText(short)
        self._info_label.setText(
            f"FPS: {self._player.fps:.2f}   "
            f"Size: {self._player.width}×{self._player.height}"
        )

        el = self._get_element()
        if el is not None:
            el.set_player(self._player)

        self._player.play()
        self._play_btn.setEnabled(True)
        self._pause_btn.setEnabled(True)
        self._seek.setEnabled(True)

    def _on_play(self) -> None:
        if self._player:
            self._player.play()

    def _on_pause(self) -> None:
        if self._player:
            self._player.pause()

    def _on_loop_changed(self, state: int) -> None:
        if self._player:
            self._player._loop = bool(state)

    def _on_seek(self) -> None:
        if self._player and self._player.duration > 0:
            t = (self._seek.value() / 1000.0) * self._player.duration
            self._player.seek(t)
            el = self._get_element()
            if el is not None:
                el.set_player(self._player)

    # ── poll timer ────────────────────────────────────────────────────────────

    def _update_ui(self) -> None:
        if self._player is None:
            return

        pos      = self._player.position
        duration = self._player.duration
        playing  = self._player.playing

        if self._registry is not None:
            try:
                self._registry.update("video.playing",  1.0 if playing else 0.0)
                self._registry.update("video.position", pos)
                self._registry.update("video.duration", duration)
            except Exception:
                pass

        self._time_label.setText(
            f"{_fmt_time(pos)} / {_fmt_time(duration)}"
        )

        if duration > 0 and not self._seek.isSliderDown():
            self._seek.blockSignals(True)
            self._seek.setValue(int(pos / duration * 1000))
            self._seek.blockSignals(False)

    # ── cleanup ───────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        if self._player is not None:
            self._player.close()
        super().closeEvent(event)


# ── standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    app = QApplication.instance() or QApplication(sys.argv)
    panel = VideoPanel(title="Video (demo)")
    panel.show()
    sys.exit(app.exec())
