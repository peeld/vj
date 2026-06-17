from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore    import QTimer, Qt
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QDoubleSpinBox, QLineEdit, QFrame,
)

if TYPE_CHECKING:
    from bpm_clock import BPMClock
    from link_manager import EventBus


_STYLESHEET = """
QWidget {
    background-color: #1a1a22;
    color: #c8c8d0;
    font-family: Consolas, "Courier New", monospace;
    font-size: 12px;
}
QLabel         { color: #707078; }
QLabel#value   { color: #5eaeff; }
QLabel#dot_off { color: #2e2e38; font-size: 16px; }
QLabel#dot_on  { color: #5eaeff; font-size: 16px; }

QDoubleSpinBox {
    background-color: #111118;
    color: #5eaeff;
    border: 1px solid #38383f;
    border-radius: 3px;
    padding: 2px 4px;
    min-width: 80px;
}
QDoubleSpinBox:focus { border: 1px solid #5eaeff; }

QLineEdit {
    background-color: #111118;
    color: #c8c8d0;
    border: 1px solid #38383f;
    border-radius: 3px;
    padding: 2px 6px;
}
QLineEdit:focus { border: 1px solid #5eaeff; }

QPushButton {
    background-color: #2a2a36;
    color: #c8c8d0;
    border: 1px solid #38383f;
    border-radius: 3px;
    padding: 3px 10px;
    min-width: 40px;
}
QPushButton:hover   { background-color: #32323f; border-color: #5eaeff; }
QPushButton:pressed { background-color: #1a1a28; }
QPushButton#tap     { min-width: 60px; font-weight: bold; }
QPushButton#tap_lit { min-width: 60px; font-weight: bold;
                      background-color: #1a3a1a; border-color: #7ec87e; color: #7ec87e; }

QFrame#sep {
    color: #2e2e38; max-height: 1px; background-color: #2e2e38;
}
"""


def _sep() -> QFrame:
    f = QFrame()
    f.setObjectName("sep")
    f.setFrameShape(QFrame.Shape.HLine)
    return f


class BpmPanel(QWidget):

    def __init__(
        self,
        clock: "BPMClock",
        event_bus: "EventBus",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._clock    = clock
        self._bus      = event_bus
        self._tap_token: int | None = None

        self.setStyleSheet(_STYLESHEET)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        # ── Row 1: BPM control ─────────────────────────────────────────────
        row1 = QHBoxLayout()
        row1.setSpacing(4)

        lbl_bpm = QLabel("BPM:")
        row1.addWidget(lbl_bpm)

        self._bpm_spin = QDoubleSpinBox()
        self._bpm_spin.setRange(20.0, 300.0)
        self._bpm_spin.setSingleStep(0.5)
        self._bpm_spin.setDecimals(1)
        self._bpm_spin.setValue(clock.bpm)
        row1.addWidget(self._bpm_spin)

        self._tap_btn = QPushButton("TAP")
        self._tap_btn.setObjectName("tap")
        row1.addWidget(self._tap_btn)

        self._nudge_minus = QPushButton("−")
        self._nudge_plus  = QPushButton("+")
        row1.addWidget(self._nudge_minus)
        row1.addWidget(self._nudge_plus)

        row1.addStretch()
        root.addLayout(row1)

        # ── Row 2: Latency ─────────────────────────────────────────────────
        row2 = QHBoxLayout()
        row2.setSpacing(4)

        row2.addWidget(QLabel("Latency:"))

        self._lat_spin = QDoubleSpinBox()
        self._lat_spin.setRange(-500.0, 500.0)
        self._lat_spin.setSingleStep(1.0)
        self._lat_spin.setDecimals(0)
        self._lat_spin.setSuffix(" ms")
        self._lat_spin.setValue(clock._latency * 1000.0)
        row2.addWidget(self._lat_spin)
        row2.addStretch()
        root.addLayout(row2)

        root.addWidget(_sep())

        # ── Row 3: Tap event name ──────────────────────────────────────────
        row3 = QHBoxLayout()
        row3.setSpacing(4)
        row3.addWidget(QLabel("Tap event:"))
        self._tap_event_edit = QLineEdit()
        self._tap_event_edit.setPlaceholderText("event name (optional)")
        row3.addWidget(self._tap_event_edit)
        root.addLayout(row3)

        root.addWidget(_sep())

        # ── Row 4: Beat indicator ─────────────────────────────────────────
        row4 = QHBoxLayout()
        row4.setSpacing(8)
        row4.addWidget(QLabel("Beat:"))
        self._dots: list[QLabel] = []
        for _ in range(4):
            d = QLabel("●")
            d.setObjectName("dot_off")
            row4.addWidget(d)
        row4.addStretch()
        root.addLayout(row4)
        self._dots = [row4.itemAt(i + 1).widget() for i in range(4)]

        root.addStretch()

        # ── Auto-repeat timers for nudge buttons ───────────────────────────
        self._nudge_minus_timer = QTimer(self)
        self._nudge_minus_timer.setInterval(120)
        self._nudge_minus_timer.timeout.connect(lambda: self._clock.nudge(-0.5))

        self._nudge_plus_timer = QTimer(self)
        self._nudge_plus_timer.setInterval(120)
        self._nudge_plus_timer.timeout.connect(lambda: self._clock.nudge(+0.5))

        # ── Poll timer: sync UI ← clock state ─────────────────────────────
        self._poll = QTimer(self)
        self._poll.setInterval(100)
        self._poll.timeout.connect(self._on_poll)
        self._poll.start()

        # ── Tap flash reset timer ──────────────────────────────────────────
        self._tap_flash = QTimer(self)
        self._tap_flash.setSingleShot(True)
        self._tap_flash.setInterval(150)
        self._tap_flash.timeout.connect(self._reset_tap_style)

        # ── Connect signals ────────────────────────────────────────────────
        self._bpm_spin.valueChanged.connect(self._on_bpm_changed)
        self._lat_spin.valueChanged.connect(lambda v: self._clock.set_latency_ms(v))
        self._tap_btn.clicked.connect(self._on_tap)
        self._nudge_minus.clicked.connect(lambda: self._clock.nudge(-0.5))
        self._nudge_plus.clicked.connect(lambda: self._clock.nudge(+0.5))
        self._nudge_minus.pressed.connect(self._nudge_minus_timer.start)
        self._nudge_minus.released.connect(self._nudge_minus_timer.stop)
        self._nudge_plus.pressed.connect(self._nudge_plus_timer.start)
        self._nudge_plus.released.connect(self._nudge_plus_timer.stop)
        self._tap_event_edit.editingFinished.connect(self._on_tap_event_changed)

        # Restore tap event from clock (populated by load_state before panel is created)
        if clock.tap_event_name:
            self._tap_event_edit.setText(clock.tap_event_name)
            self._re_subscribe(clock.tap_event_name)

    # ── Slots ──────────────────────────────────────────────────────────────

    def _on_bpm_changed(self, v: float) -> None:
        self._clock.set_bpm(v)

    def _on_tap(self) -> None:
        self._clock.tap()
        self._tap_btn.setObjectName("tap_lit")
        self._tap_btn.style().unpolish(self._tap_btn)
        self._tap_btn.style().polish(self._tap_btn)
        self._tap_flash.start()

    def _reset_tap_style(self) -> None:
        self._tap_btn.setObjectName("tap")
        self._tap_btn.style().unpolish(self._tap_btn)
        self._tap_btn.style().polish(self._tap_btn)

    def _on_tap_event_changed(self) -> None:
        name = self._tap_event_edit.text().strip()
        self._re_subscribe(name)

    def _re_subscribe(self, name: str) -> None:
        if self._tap_token is not None:
            self._bus.unsubscribe(self._tap_token)
            self._tap_token = None
        self._clock.tap_event_name = name
        if name:
            self._tap_token = self._bus.subscribe(name, self._clock.tap_event)

    def _on_poll(self) -> None:
        # sync BPM spinbox silently
        current = self._clock.bpm
        if abs(self._bpm_spin.value() - current) > 0.05:
            self._bpm_spin.blockSignals(True)
            self._bpm_spin.setValue(current)
            self._bpm_spin.blockSignals(False)

        # update beat dots
        n = self._clock.last_beat_n
        for i, dot in enumerate(self._dots):
            obj = "dot_on" if i == n else "dot_off"
            if dot.objectName() != obj:
                dot.setObjectName(obj)
                dot.style().unpolish(dot)
                dot.style().polish(dot)

