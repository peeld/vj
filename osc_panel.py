"""
osc_panel.py — OSC listener host/port control + live event monitor

PySide6 panel for receiving OSC (TouchOSC, Lemur, Max/MSP, Resolume, QLab,
etc.).  On each OSC message the raw listener writes numeric values to the
SourceRegistry (as osc.<key>) and fires non-numeric/bang messages onto the
EventBus (as osc<address>) so that LinkManager can drive scene parameters.
Use the Link Manager panel to create OSC-driven mappings via SignalLinks
and EventLinks.

Usage (gui_merged.py __main__ block)::

    from osc_input import get_router as get_osc_router
    from osc_panel import OscPanel

    _osc_router = get_osc_router()
    osc = OscPanel(_osc_router,
                   source_registry=_lm.source_registry,
                   event_bus=_lm.event_bus)
    osc.show()
"""

import socket

from PySide6.QtCore    import QTimer, Qt, QSettings
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QSpinBox, QPushButton,
    QFrame,
)

from osc_input import OscRouter, get_router


# ── stylesheet (copied verbatim from midi_panel.py for visual consistency) ──

_STYLESHEET = """
QWidget {
    background-color: #1a1a22;
    color: #c8c8d0;
    font-family: Consolas, "Courier New", monospace;
    font-size: 12px;
}
QLabel { color: #707078; }
QLabel#hdr  { color: #c8c8d0; font-weight: bold; padding-top: 4px; }
QLabel#act  { color: #ffb347; font-weight: bold; }
QLabel#ok   { color: #7ec87e; }
QLabel#err  { color: #e07070; }
QLabel#info { color: #707078; font-style: italic; }

QComboBox {
    background-color: #111118;
    color: #5eaeff;
    border: 1px solid #38383f;
    border-radius: 3px;
    padding: 2px 6px;
    min-width: 160px;
}
QComboBox:focus { border: 1px solid #5eaeff; }
QComboBox QAbstractItemView {
    background-color: #1a1a22;
    color: #c8c8d0;
    selection-background-color: #2a4070;
    border: 1px solid #38383f;
}
QComboBox::drop-down { border: none; width: 18px; }
QComboBox::down-arrow {
    image: none;
    border-left:  4px solid transparent;
    border-right: 4px solid transparent;
    border-top:   5px solid #707078;
    margin-right: 4px;
}

QLineEdit {
    background-color: #111118;
    color: #5eaeff;
    border: 1px solid #38383f;
    border-radius: 3px;
    padding: 2px 6px;
}
QLineEdit:focus { border: 1px solid #5eaeff; }

QSpinBox {
    background-color: #111118;
    color: #5eaeff;
    border: 1px solid #38383f;
    border-radius: 3px;
    padding: 2px 6px;
}
QSpinBox:focus { border: 1px solid #5eaeff; }

QPushButton {
    background-color: #2a2a36;
    color: #c8c8d0;
    border: 1px solid #38383f;
    border-radius: 3px;
    padding: 3px 10px;
    min-width: 60px;
}
QPushButton:hover  { background-color: #32323f; border-color: #5eaeff; }
QPushButton:pressed{ background-color: #1a1a28; }

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
"""


# ── helpers ───────────────────────────────────────────────────────────────

def _sep() -> QFrame:
    f = QFrame()
    f.setObjectName("sep")
    f.setFrameShape(QFrame.HLine)
    return f


def _hdr(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setObjectName("hdr")
    return lbl


def _restyle(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def _local_ip() -> str:
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return ""


# ── main panel ────────────────────────────────────────────────────────────

class OscPanel(QWidget):
    """
    Live OSC input panel.

    Connects to a UDP socket via OscRouter and streams messages into
    SourceRegistry (numeric params) and EventBus (bangs / non-numeric
    events).  Use the Link Manager panel to create OSC-driven mappings via
    SignalLinks and EventLinks.
    """

    def __init__(
        self,
        router          : OscRouter,
        title           : str = "OSC Input",
        source_registry = None,   # SourceRegistry | None
        event_bus       = None,   # EventBus | None
    ):
        super().__init__()
        self.router           = router
        self._source_registry = source_registry
        self._event_bus       = event_bus

        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setStyleSheet(_STYLESHEET)
        self.setMinimumWidth(380)
        self.resize(400, 280)

        self._build_ui()
        self._restore_settings()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(80)

        self.router.add_listener(self._raw_listener)

    # ── teardown ─────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self.router.remove_listener(self._raw_listener)
        super().closeEvent(event)

    # ── UI construction ─────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        # ── Listener section ────────────────────────────────────────────
        root.addWidget(_hdr("Listener"))
        root.addWidget(_sep())

        host_row = QHBoxLayout()
        host_row.addWidget(QLabel("Host:"))
        self._host_edit = QLineEdit("0.0.0.0")
        host_row.addWidget(self._host_edit)
        host_row.addWidget(QLabel("Port:"))
        self._port_spin = QSpinBox()
        self._port_spin.setRange(1, 65535)
        self._port_spin.setValue(9000)
        host_row.addWidget(self._port_spin)
        root.addLayout(host_row)

        btn_row = QHBoxLayout()
        self._btn_connect = QPushButton("Connect")
        self._btn_connect.clicked.connect(self._on_connect)
        btn_row.addWidget(self._btn_connect)

        self._btn_disconnect = QPushButton("Disconnect")
        self._btn_disconnect.clicked.connect(self._on_disconnect)
        btn_row.addWidget(self._btn_disconnect)
        root.addLayout(btn_row)

        self._lbl_status = QLabel("Not connected")
        self._lbl_status.setObjectName("info")
        root.addWidget(self._lbl_status)

        ip = _local_ip()
        self._lbl_local_ip = QLabel(f"Local IP: {ip}" if ip else "Local IP: (unknown)")
        self._lbl_local_ip.setObjectName("info")
        root.addWidget(self._lbl_local_ip)

        # ── Last Event ───────────────────────────────────────────────────
        root.addSpacing(4)
        root.addWidget(_hdr("Last Event"))
        root.addWidget(_sep())
        self._lbl_activity = QLabel("—")
        self._lbl_activity.setObjectName("act")
        self._lbl_activity.setWordWrap(True)
        root.addWidget(self._lbl_activity)

        root.addStretch(1)

    # ── connection management ───────────────────────────────────────────

    def _restore_settings(self) -> None:
        qs = QSettings("WarpApp", "WarpApp")
        host = qs.value("osc/host", "")
        port = qs.value("osc/port", "")
        if host:
            self._host_edit.setText(host)
        if port:
            try:
                self._port_spin.setValue(int(port))
            except (TypeError, ValueError):
                pass
        if host and port:
            self._on_connect()

    def _on_connect(self) -> None:
        host = self._host_edit.text().strip() or "0.0.0.0"
        port = self._port_spin.value()
        ok = self.router.start(host, port)
        if ok:
            self._lbl_status.setObjectName("ok")
            self._lbl_status.setText(f"● Listening on {host}:{port}")
            qs = QSettings("WarpApp", "WarpApp")
            qs.setValue("osc/host", host)
            qs.setValue("osc/port", port)
        else:
            self._lbl_status.setObjectName("err")
            self._lbl_status.setText(f"✗ Failed: could not bind {host}:{port}")
        _restyle(self._lbl_status)

    def _on_disconnect(self) -> None:
        self.router.stop()
        self._lbl_status.setObjectName("info")
        self._lbl_status.setText("Not connected")
        _restyle(self._lbl_status)

    # ── OSC event routing ───────────────────────────────────────────────

    def _raw_listener(self, evt: dict) -> None:
        """Called from OSC thread — updates SourceRegistry and fires EventBus."""
        try:
            if evt["type"] == "param":
                if self._source_registry is not None:
                    if "value" in evt:
                        self._source_registry.update(
                            f"osc.{evt['key']}", evt["value"])
                    else:
                        for i, v in enumerate(evt["values"]):
                            self._source_registry.update(
                                f"osc.{evt['key']}_{i}", v)
            elif evt["type"] == "event":
                if self._event_bus is not None:
                    self._event_bus.fire(f"osc{evt['address']}",
                                          evt.get("args"))
        except Exception:
            pass

    # ── poll timer ───────────────────────────────────────────────────────

    def _poll(self) -> None:
        ev = self.router.last_event
        if not ev:
            return
        t       = ev.get("type", "?")
        address = ev.get("address", "?")
        if t == "param":
            if "value" in ev:
                self._lbl_activity.setText(f"PARAM  {address}  =  {ev['value']:.3f}")
            else:
                vals = ", ".join(f"{v:.2f}" for v in ev.get("values", []))
                self._lbl_activity.setText(f"PARAM  {address}  =  [{vals}]")
        elif t == "event":
            self._lbl_activity.setText(f"EVENT  {address}  args={ev.get('args', [])}")
