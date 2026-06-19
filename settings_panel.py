"""
settings_panel.py — Application settings panel.

SettingsPanel contains Display (monitor picker / fullscreen) and Audio Device
(input device selector + connect/disconnect) sections.
"""

from __future__ import annotations

from collections.abc import Callable

import sounddevice as sd

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QScrollArea,
    QFrame, QComboBox, QPushButton, QSizePolicy,
)
from PySide6.QtCore import Qt, QSettings

from audio_metrics import AudioAnalyzer, AudioMetrics


def _restyle(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)


class SettingsPanel(QWidget):
    """
    Application settings — Display and Audio Device sections.

    on_monitor_change : optional callback(monitor_index: int) invoked when the
                        user picks a monitor and clicks "Go Fullscreen".
    source_registry   : SourceRegistry; audio metrics written as audio.<field>.
    on_metrics        : callable(AudioMetrics) forwarded each audio frame.
    """

    _REGISTRY_FIELDS = (
        "sub_bass", "bass", "low_mid", "mid", "high_mid",
        "treble", "energy", "brightness", "flux", "kick", "onset",
    )

    def __init__(
        self,
        on_monitor_change: Callable[[int], None] | None = None,
        source_registry=None,
        on_metrics=None,
        title: str = "Settings",
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setMinimumWidth(380)
        self.resize(420, 400)

        self._on_monitor_change = on_monitor_change
        self._source_registry = source_registry
        self._on_metrics = on_metrics
        self._analyzer: AudioAnalyzer | None = None

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
        self._build_audio_section(form)

    def closeEvent(self, event):
        self._stop_analyzer()
        super().closeEvent(event)

    # ── Display ───────────────────────────────────────────────────────────────

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

    # ── Audio Device ──────────────────────────────────────────────────────────

    def _build_audio_section(self, form: QFormLayout) -> None:
        form.addRow(QLabel(""))

        hdr = QLabel("Audio Device")
        hdr.setObjectName("section_header")
        form.addRow(hdr)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #2e2e38;")
        form.addRow(sep)

        dev_widget = QWidget()
        dev_layout = QHBoxLayout(dev_widget)
        dev_layout.setContentsMargins(0, 0, 0, 0)
        dev_layout.setSpacing(4)

        self._dev_combo = QComboBox()
        self._dev_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        dev_layout.addWidget(self._dev_combo)

        btn_refresh = QPushButton("↻")
        btn_refresh.setFixedWidth(28)
        btn_refresh.clicked.connect(self._refresh_devices)
        dev_layout.addWidget(btn_refresh)

        self._btn_connect = QPushButton("Connect")
        self._btn_connect.clicked.connect(self._on_connect)
        dev_layout.addWidget(self._btn_connect)

        self._btn_disconnect = QPushButton("Disconnect")
        self._btn_disconnect.clicked.connect(self._on_disconnect)
        dev_layout.addWidget(self._btn_disconnect)

        form.addRow("Device", dev_widget)

        self._lbl_status = QLabel("Not connected")
        self._lbl_status.setObjectName("info")
        form.addRow("", self._lbl_status)

        self._refresh_devices()

    def _refresh_devices(self) -> None:
        self._dev_combo.clear()
        try:
            devices = sd.query_devices()
        except Exception:
            self._dev_combo.addItem("(sounddevice unavailable)")
            return

        found = False
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                self._dev_combo.addItem(f"[{i}] {dev['name']}", userData=i)
                found = True

        if not found:
            self._dev_combo.addItem("(no input devices found)")

    def _on_connect(self) -> None:
        idx = self._dev_combo.currentIndex()
        device_index = self._dev_combo.itemData(idx)
        if device_index is None:
            return

        self._stop_analyzer()
        try:
            self._analyzer = AudioAnalyzer(
                device=device_index,
                on_frame=self._on_audio_frame,
            )
            self._analyzer.start()
            name = self._dev_combo.currentText()
            self._lbl_status.setObjectName("ok")
            self._lbl_status.setText(f"● Connected: {name}")
        except Exception as exc:
            self._analyzer = None
            self._lbl_status.setObjectName("err")
            self._lbl_status.setText(f"✗ {exc}")
        _restyle(self._lbl_status)

    def _on_disconnect(self) -> None:
        self._stop_analyzer()
        self._lbl_status.setObjectName("info")
        self._lbl_status.setText("Not connected")
        _restyle(self._lbl_status)
        if self._on_metrics is not None:
            self._on_metrics(AudioMetrics())

    def _stop_analyzer(self) -> None:
        if self._analyzer is not None:
            try:
                self._analyzer.stop()
            except Exception:
                pass
            self._analyzer = None

    def _on_audio_frame(self, metrics: AudioMetrics) -> None:
        if self._source_registry is not None:
            try:
                for field in self._REGISTRY_FIELDS:
                    self._source_registry.update(
                        f"audio.{field}", getattr(metrics, field, 0.0)
                    )
            except Exception:
                pass
        if self._on_metrics is not None:
            try:
                self._on_metrics(metrics)
            except Exception:
                pass
