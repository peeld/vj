"""
midi_panel.py — MIDI assignment GUI

PySide6 panel for assigning MIDI CC / note events to live parameters at runtime.
Matches the dark style of param_dialog.py.

Usage (add to gui_merged.py __main__ block):
--------------------------------------------
    from midi_input import get_router
    from midi_panel import start_midi_panel

    router = get_router()

    start_midi_panel(
        router,
        ("Post-FX", _params),
        ("Scene",   _controls),
        title="MIDI Assignments",
    )

    router.start()   # or pass a specific device name

Flow
----
1. Select device from dropdown → Connect
2. Pick a parameter (Section / field) from the "Parameter" combo
3. Choose CC or Note
4. For CC: set min / max values
5. Click Learn → touch a knob or key → assignment is created
6. Assignments appear in the list; click ✕ to remove
7. Save / Load buttons persist mappings to midi_mappings.json beside the script
"""

import dataclasses
import json
import threading
from pathlib import Path
from typing import Any

from PySide6.QtCore    import QTimer, Qt, Signal, QObject
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QLineEdit,
    QScrollArea, QFrame, QSizePolicy, QFileDialog,
    QButtonGroup, QRadioButton,
)

from midi_input import MidiRouter, get_router


# ── stylesheet (matches param_dialog.py) ──────────────────────────────────────

_STYLESHEET = """
QWidget {
    background-color: #1a1a22;
    color: #c8c8d0;
    font-family: Consolas, "Courier New", monospace;
    font-size: 12px;
}
QLabel { color: #707078; }
QLabel#hdr  { color: #c8c8d0; font-weight: bold; padding-top: 4px; }
QLabel#mono { color: #5eaeff; }
QLabel#act  { color: #ffb347; font-weight: bold; }
QLabel#ok   { color: #7ec87e; }
QLabel#err  { color: #e07070; }
QLabel#info { color: #707078; font-style: italic; }

QLineEdit {
    background-color: #111118;
    color: #5eaeff;
    border: 1px solid #38383f;
    border-radius: 3px;
    padding: 2px 6px;
}
QLineEdit:focus { border: 1px solid #5eaeff; }

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
QPushButton#learn  { color: #ffb347; border-color: #ffb347; }
QPushButton#learn[active="true"] {
    background-color: #3a2800;
    color: #ffdd99;
    border-color: #ffb347;
}
QPushButton#remove {
    background-color: transparent;
    color: #804040;
    border: none;
    min-width: 22px;
    padding: 0;
    font-size: 14px;
}
QPushButton#remove:hover { color: #e07070; }

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
QRadioButton { color: #c8c8d0; spacing: 6px; }
QRadioButton::indicator {
    width: 12px; height: 12px;
    border: 1px solid #38383f;
    border-radius: 6px;
    background: #111118;
}
QRadioButton::indicator:checked { background: #5eaeff; border-color: #5eaeff; }
"""


# ── Qt bridge for cross-thread event delivery ──────────────────────────────────

class _Bridge(QObject):
    midi_event = Signal(dict)


# ── helpers ───────────────────────────────────────────────────────────────────

def _sep() -> QFrame:
    f = QFrame()
    f.setObjectName("sep")
    f.setFrameShape(QFrame.HLine)
    return f


def _hdr(text: str) -> QLabel:
    l = QLabel(text)
    l.setObjectName("hdr")
    return l


def _note_name(n: int) -> str:
    names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    return f"{names[n % 12]}{n // 12 - 1}"


# ── main panel ────────────────────────────────────────────────────────────────

class MidiPanel(QWidget):
    """
    Live MIDI assignment panel.

    targets : list of (section_label, obj, hints_dict)   # same as param_dialog
    """

    def __init__(
        self,
        router  : MidiRouter,
        targets : list[tuple[str, Any, dict]],
        title   : str = "MIDI Assignments",
        save_path: Path | None = None,
    ):
        super().__init__()
        self.router    = router
        self.targets   = targets
        self.save_path = save_path or Path("midi_mappings.json")
        self._learn_active = False
        self._bridge   = _Bridge()
        self._bridge.midi_event.connect(self._on_midi_event)

        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setStyleSheet(_STYLESHEET)
        self.setMinimumWidth(420)
        self.resize(440, 700)

        self._build_ui()
        self._refresh_devices()

        # Poll timer — updates activity label and assignment list
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(80)

        # Register raw listener to forward events to Qt thread
        self.router.add_listener(self._raw_listener)

    # ── teardown ───────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self.router.remove_listener(self._raw_listener)
        super().closeEvent(event)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        # ── Device section ────────────────────────────────────────────────────
        root.addWidget(_hdr("Device"))
        root.addWidget(_sep())

        dev_row = QHBoxLayout()
        self._dev_combo = QComboBox()
        self._dev_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        dev_row.addWidget(self._dev_combo)

        self._btn_refresh = QPushButton("↻")
        self._btn_refresh.setFixedWidth(28)
        self._btn_refresh.clicked.connect(self._refresh_devices)
        dev_row.addWidget(self._btn_refresh)

        self._btn_connect = QPushButton("Connect")
        self._btn_connect.clicked.connect(self._on_connect)
        dev_row.addWidget(self._btn_connect)

        self._btn_disconnect = QPushButton("Disconnect")
        self._btn_disconnect.clicked.connect(self._on_disconnect)
        dev_row.addWidget(self._btn_disconnect)
        root.addLayout(dev_row)

        self._lbl_status = QLabel("Not connected")
        self._lbl_status.setObjectName("info")
        root.addWidget(self._lbl_status)

        # ── Activity ──────────────────────────────────────────────────────────
        root.addSpacing(4)
        root.addWidget(_hdr("Last Event"))
        root.addWidget(_sep())
        self._lbl_activity = QLabel("—")
        self._lbl_activity.setObjectName("act")
        root.addWidget(self._lbl_activity)

        # ── Assignment builder ────────────────────────────────────────────────
        root.addSpacing(4)
        root.addWidget(_hdr("New Assignment"))
        root.addWidget(_sep())

        # Parameter selector
        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("Parameter:"))
        self._param_combo = QComboBox()
        self._param_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._populate_param_combo()
        param_row.addWidget(self._param_combo)
        root.addLayout(param_row)

        # Type: CC / Note
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        self._type_group = QButtonGroup(self)
        self._rb_cc   = QRadioButton("CC")
        self._rb_note = QRadioButton("Note")
        self._rb_cc.setChecked(True)
        self._type_group.addButton(self._rb_cc,   0)
        self._type_group.addButton(self._rb_note, 1)
        self._rb_cc.toggled.connect(self._on_type_changed)
        type_row.addWidget(self._rb_cc)
        type_row.addWidget(self._rb_note)
        type_row.addStretch()
        root.addLayout(type_row)

        # CC range row (hidden for Note mode)
        self._range_widget = QWidget()
        rr = QHBoxLayout(self._range_widget)
        rr.setContentsMargins(0, 0, 0, 0)
        rr.addWidget(QLabel("Min:"))
        self._edit_min = QLineEdit("0.0")
        self._edit_min.setFixedWidth(70)
        rr.addWidget(self._edit_min)
        rr.addWidget(QLabel("Max:"))
        self._edit_max = QLineEdit("1.0")
        self._edit_max.setFixedWidth(70)
        rr.addWidget(self._edit_max)
        rr.addStretch()
        root.addWidget(self._range_widget)

        # Note on/off values row (hidden for CC mode)
        self._note_val_widget = QWidget()
        nvr = QHBoxLayout(self._note_val_widget)
        nvr.setContentsMargins(0, 0, 0, 0)
        self._lbl_note_hint = QLabel("Bool param → toggle  |  Float/int: set on/off value")
        self._lbl_note_hint.setObjectName("info")
        self._lbl_note_hint.setWordWrap(True)
        nvr.addWidget(self._lbl_note_hint)
        self._note_val_widget.setVisible(False)
        root.addWidget(self._note_val_widget)

        # Learn button
        learn_row = QHBoxLayout()
        self._btn_learn = QPushButton("Learn")
        self._btn_learn.setObjectName("learn")
        self._btn_learn.setProperty("active", "false")
        self._btn_learn.clicked.connect(self._on_learn_clicked)
        learn_row.addWidget(self._btn_learn)
        self._lbl_learn = QLabel("Click Learn, then send a MIDI event.")
        self._lbl_learn.setObjectName("info")
        learn_row.addWidget(self._lbl_learn)
        learn_row.addStretch()
        root.addLayout(learn_row)

        # ── Assignments list ──────────────────────────────────────────────────
        root.addSpacing(4)
        root.addWidget(_hdr("Current Assignments"))
        root.addWidget(_sep())

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._assign_container = QWidget()
        self._assign_layout    = QVBoxLayout(self._assign_container)
        self._assign_layout.setContentsMargins(0, 2, 0, 2)
        self._assign_layout.setSpacing(3)
        self._assign_layout.addStretch()
        self._scroll.setWidget(self._assign_container)
        root.addWidget(self._scroll, stretch=1)

        # ── Save / Load ───────────────────────────────────────────────────────
        root.addWidget(_sep())
        io_row = QHBoxLayout()
        btn_save = QPushButton("Save Mappings")
        btn_save.clicked.connect(self._on_save)
        btn_load = QPushButton("Load Mappings")
        btn_load.clicked.connect(self._on_load)
        io_row.addWidget(btn_save)
        io_row.addWidget(btn_load)
        io_row.addStretch()
        root.addLayout(io_row)

        self._refresh_assignments()

    # ── device management ──────────────────────────────────────────────────────

    def _refresh_devices(self) -> None:
        self._dev_combo.clear()
        names = self.router.available_devices()
        if names:
            self._dev_combo.addItems(names)
        else:
            self._dev_combo.addItem("(no devices found)")

    def _on_connect(self) -> None:
        name = self._dev_combo.currentText()
        if name.startswith("("):
            return
        ok = self.router.start(name)
        if ok:
            self._lbl_status.setObjectName("ok")
            self._lbl_status.setText(f"● Connected: {name}")
        else:
            self._lbl_status.setObjectName("err")
            self._lbl_status.setText(f"✗ Failed to connect: {name}")
        self._lbl_status.setStyleSheet("")  # force re-polish
        self._lbl_status.style().unpolish(self._lbl_status)
        self._lbl_status.style().polish(self._lbl_status)

    def _on_disconnect(self) -> None:
        self.router.stop()
        self._lbl_status.setObjectName("info")
        self._lbl_status.setText("Not connected")
        self._lbl_status.style().unpolish(self._lbl_status)
        self._lbl_status.style().polish(self._lbl_status)

    # ── param combo ───────────────────────────────────────────────────────────

    def _populate_param_combo(self) -> None:
        self._param_items: list[tuple[str, Any, str, type]] = []  # (section, obj, field, ftype)
        self._param_combo.clear()

        for section, obj, _ in self.targets:
            fields = list(self._iter_fields(obj))
            for fname, ftype, _ in fields:
                label = f"{section} / {fname}"
                self._param_combo.addItem(label)
                self._param_items.append((section, obj, fname, ftype))

        self._param_combo.currentIndexChanged.connect(self._on_param_changed)
        self._on_param_changed(0)

    def _on_param_changed(self, idx: int) -> None:
        if not self._param_items or idx < 0:
            return
        _, _, _, ftype = self._param_items[idx]
        if ftype is bool:
            self._edit_min.setText("0")
            self._edit_max.setText("1")
        elif ftype is int:
            self._edit_min.setText("0")
            self._edit_max.setText("127")
        else:
            self._edit_min.setText("0.0")
            self._edit_max.setText("1.0")

    def _on_type_changed(self, checked: bool) -> None:
        is_cc = self._rb_cc.isChecked()
        self._range_widget.setVisible(is_cc)
        self._note_val_widget.setVisible(not is_cc)

    # ── MIDI learn ─────────────────────────────────────────────────────────────

    def _on_learn_clicked(self) -> None:
        if self._learn_active:
            self._cancel_learn()
        else:
            self._start_learn()

    def _start_learn(self) -> None:
        self._learn_active = True
        self._learn_event  = None
        self._btn_learn.setText("Listening…  (click to cancel)")
        self._btn_learn.setProperty("active", "true")
        self._btn_learn.style().unpolish(self._btn_learn)
        self._btn_learn.style().polish(self._btn_learn)
        self._lbl_learn.setText("Waiting for MIDI event…")

    def _cancel_learn(self) -> None:
        self._learn_active = False
        self._learn_event  = None
        self._btn_learn.setText("Learn")
        self._btn_learn.setProperty("active", "false")
        self._btn_learn.style().unpolish(self._btn_learn)
        self._btn_learn.style().polish(self._btn_learn)
        self._lbl_learn.setText("Click Learn, then send a MIDI event.")

    def _commit_learn(self, evt: dict) -> None:
        """Called on the Qt thread when a MIDI event arrives during learn mode."""
        idx = self._param_combo.currentIndex()
        if idx < 0 or idx >= len(self._param_items):
            self._cancel_learn()
            return

        section, obj, field, ftype = self._param_items[idx]
        ev_type = evt["type"]
        number  = evt["number"]
        channel = evt.get("channel", 0)

        is_cc_mode   = self._rb_cc.isChecked()
        is_note_mode = self._rb_note.isChecked()

        if ev_type == "cc" and is_cc_mode:
            try:
                min_v = float(self._edit_min.text())
                max_v = float(self._edit_max.text())
            except ValueError:
                min_v, max_v = 0.0, 1.0
            if ftype is int:
                min_v, max_v = int(min_v), int(max_v)
            self.router.map_cc(number, obj, field, min_val=min_v, max_val=max_v,
                               channel=None, section=section)
            self._lbl_learn.setText(f"✓ CC#{number} → {section}/{field}")

        elif ev_type == "note" and is_note_mode:
            self.router.map_note_to_param(number, obj, field,
                                          channel=None, section=section)
            self._lbl_learn.setText(f"✓ Note {_note_name(number)} → {section}/{field}")

        else:
            # Wrong event type — keep listening
            hint = "CC" if is_cc_mode else "Note"
            self._lbl_learn.setText(f"Waiting for {hint} event…")
            return   # don't cancel learn

        self._learn_active = False
        self._learn_event  = None
        self._btn_learn.setText("Learn")
        self._btn_learn.setProperty("active", "false")
        self._btn_learn.style().unpolish(self._btn_learn)
        self._btn_learn.style().polish(self._btn_learn)
        self._refresh_assignments()

    # ── MIDI event from router thread (via Qt signal) ─────────────────────────

    def _raw_listener(self, evt: dict) -> None:
        """Called from MIDI thread — only emit the signal, never touch widgets."""
        self._bridge.midi_event.emit(evt)

    def _on_midi_event(self, evt: dict) -> None:
        """Qt thread handler for incoming MIDI events."""
        if self._learn_active:
            self._commit_learn(evt)

    # ── poll timer ─────────────────────────────────────────────────────────────

    def _poll(self) -> None:
        ev = self.router.last_event
        if ev:
            t  = ev.get("type", "?")
            ch = ev.get("channel", 0)
            n  = ev.get("number", 0)
            v  = ev.get("value",  0)
            if t == "cc":
                self._lbl_activity.setText(f"CC#{n}  ch={ch}  val={v}")
            elif t == "note":
                self._lbl_activity.setText(
                    f"{'Note On' if v > 0 else 'Note Off'}  {_note_name(n)}({n})  ch={ch}  vel={v}"
                )

    # ── assignment list ────────────────────────────────────────────────────────

    def _refresh_assignments(self) -> None:
        # Clear existing rows (except the stretch at the end)
        while self._assign_layout.count() > 1:
            item = self._assign_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        rows: list[tuple[str, dict, str, int]] = []  # (kind, data, display, idx)

        for i, d in enumerate(self.router.get_cc_mappings()):
            label = (f"CC#{d['cc_num']}  →  {d['section']} / {d['field']}"
                     f"   [{d['min_val']:.4g} – {d['max_val']:.4g}]")
            rows.append(("cc", d, label, i))

        for i, d in enumerate(self.router.get_note_param_mappings()):
            note_str = _note_name(d["note"])
            if d.get("toggle"):
                label = f"Note {note_str}({d['note']})  →  {d['section']} / {d['field']}  [toggle]"
            else:
                label = (f"Note {note_str}({d['note']})  →  {d['section']} / {d['field']}"
                         f"   on={d['on_value']} off={d['off_value']}")
            rows.append(("note_param", d, label, i))

        for i, d in enumerate(self.router.get_note_mappings()):
            label = f"Note {_note_name(d['note'])}({d['note']})  →  {d['section']} / {d['field']}  [callback]"
            rows.append(("note_cb", d, label, i))

        if not rows:
            lbl = QLabel("No assignments yet.")
            lbl.setObjectName("info")
            self._assign_layout.insertWidget(0, lbl)
            return

        for pos, (kind, d, label, idx) in enumerate(rows):
            row_w  = QWidget()
            row_hl = QHBoxLayout(row_w)
            row_hl.setContentsMargins(0, 1, 0, 1)
            row_hl.setSpacing(4)

            lbl = QLabel(label)
            lbl.setObjectName("mono")
            lbl.setWordWrap(False)
            row_hl.addWidget(lbl, stretch=1)

            if kind != "note_cb":   # callbacks registered in code: read-only
                btn_rm = QPushButton("✕")
                btn_rm.setObjectName("remove")
                btn_rm.setToolTip("Remove assignment")
                # Capture loop vars
                btn_rm.clicked.connect(
                    self._make_remove_handler(kind, idx)
                )
                row_hl.addWidget(btn_rm)

            self._assign_layout.insertWidget(pos, row_w)

    def _make_remove_handler(self, kind: str, idx: int):
        def _handler():
            if kind == "cc":
                self.router.unmap_cc_by_index(idx)
            elif kind == "note_param":
                self.router.unmap_note_param_by_index(idx)
            elif kind == "note_cb":
                self.router.unmap_note_by_index(idx)
            self._refresh_assignments()
        return _handler

    # ── save / load ───────────────────────────────────────────────────────────

    def _on_save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save MIDI Mappings", str(self.save_path), "JSON (*.json)"
        )
        if path:
            self.router.save_mappings(path)
            self.save_path = Path(path)

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load MIDI Mappings", str(self.save_path), "JSON (*.json)"
        )
        if not path:
            return

        try:
            data = json.loads(Path(path).read_text())
        except Exception as exc:
            print(f"[midi_panel] load error: {exc}")
            return

        # Build a (section, field) → obj lookup from our targets
        lookup: dict[tuple[str, str], Any] = {}
        for section, obj, _ in self.targets:
            for fname, _, _ in self._iter_fields(obj):
                lookup[(section, fname)] = obj

        def resolve(section: str, field: str) -> Any | None:
            return lookup.get((section, field))

        n_cc   = self.router.load_cc_mappings(data.get("cc", []),   resolve)
        n_np   = self.router.load_note_param_mappings(
                     data.get("note_param", []), resolve)
        print(f"[midi_panel] loaded {n_cc} CC + {n_np} note-param mappings from {path}")
        self._refresh_assignments()

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _iter_fields(obj: Any):
        if dataclasses.is_dataclass(obj):
            for f in dataclasses.fields(obj):
                val = getattr(obj, f.name)
                t   = f.type if isinstance(f.type, type) else type(val)
                yield f.name, t, val
        else:
            for name, ann in getattr(obj, "__annotations__", {}).items():
                if not name.startswith("_"):
                    val = getattr(obj, name, None)
                    t   = ann if isinstance(ann, type) else type(val)
                    yield name, t, val


# ── public entry point ────────────────────────────────────────────────────────

def start_midi_panel(
    router    : MidiRouter | None = None,
    *targets,
    title     : str = "MIDI Assignments",
    save_path : str | Path | None = None,
) -> threading.Thread:
    """
    Launch a MidiPanel in a background daemon thread.

    router   — MidiRouter instance; defaults to get_router() singleton
    targets  — same format as start_param_dialog:
                   ("Section", obj)
                   ("Section", obj, hints_dict)
    title    — window title
    save_path— default path for save/load dialogs

    Returns the daemon thread (usually not needed).
    """
    if router is None:
        router = get_router()

    pairs: list[tuple[str, Any, dict]] = []
    for t in targets:
        if isinstance(t, tuple):
            if len(t) >= 3 and isinstance(t[0], str):
                pairs.append((t[0], t[1], t[2]))
            elif len(t) == 2 and isinstance(t[0], str):
                pairs.append((t[0], t[1], {}))
            else:
                pairs.append((type(t[0]).__name__, t[0], {}))
        else:
            pairs.append((type(t).__name__, t, {}))

    sp = Path(save_path) if save_path else None

    def _run() -> None:
        app  = QApplication.instance() or QApplication([])
        panel = MidiPanel(router, pairs, title=title, save_path=sp)
        panel.show()
        app.exec()

    t = threading.Thread(target=_run, daemon=True, name="midi-panel")
    t.start()
    return t

