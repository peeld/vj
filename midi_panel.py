"""
midi_panel.py — MIDI assignment GUI (PropertyManager edition)

PySide6 panel for assigning MIDI CC / note events to parameters managed by a
PropertyManager.  MidiRouter handles hardware I/O only; all binding logic lives
in PropertyManager.

Usage (gui_merged.py __main__ block)::

    from midi_input       import get_router
    from midi_panel       import MidiPanel
    from property_manager import build_default_manager

    _router = get_router()
    _pm     = build_default_manager(_params, _controls, None, None, None)

    midi = MidiPanel(_router, _pm, title="MIDI Assignments")
    midi.show()

    # When the GL window starts and elements exist, rebuild pm in-place:
    #   build_default_manager(_params, _controls, nn, lasers, circles, pm=_pm)

Flow
----
1. Select device → Connect
2. Pick a parameter (grouped by section)
3. Choose CC or Note
   - CC + numeric → enter Min/Max; Learn → touch a knob
   - CC + enum    → auto-detected; shows "Sweeps N choices: …"; Learn → touch a knob
   - Note + bool  → toggle on note-on; Learn → press a key
   - Note + enum  → pick a Value from the dropdown; Learn → press a key
4. Assignments appear in the list; ✕ removes
5. Save/Load persist the full PropertyManager JSON (presets + all mappings)
"""

import json
import threading
from pathlib import Path
from typing import Any

from PySide6.QtCore    import QTimer, Qt, Signal, QObject, QSettings
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QLineEdit,
    QScrollArea, QFrame, QSizePolicy, QFileDialog,
    QButtonGroup, QRadioButton,
)

from midi_input       import MidiRouter, get_router
from property_manager import PropertyManager, PropDef, MidiCCBinding, MidiNoteBinding


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
QLabel#enum { color: #a070d0; font-style: italic; }

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
    lbl = QLabel(text)
    lbl.setObjectName("hdr")
    return lbl


def _note_name(n: int) -> str:
    names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    return f"{names[n % 12]}{n // 12 - 1}"


def _is_enum(prop: PropDef) -> bool:
    return bool(prop.choices)


def _restyle(widget: QWidget) -> None:
    """Force Qt stylesheet re-polish (needed after objectName changes)."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)


# ── main panel ────────────────────────────────────────────────────────────────

class MidiPanel(QWidget):
    """
    Live MIDI assignment panel backed by a PropertyManager.

    router — MidiRouter instance (hardware I/O only)
    pm     — PropertyManager that owns all parameters and bindings
    """

    def __init__(
        self,
        router   : MidiRouter,
        pm       : PropertyManager,
        title    : str = "MIDI Assignments",
        save_path: Path | None = None,
    ):
        super().__init__()
        self.router    = router
        self.pm        = pm
        self.save_path = save_path or Path("midi_mappings.json")

        self._learn_active = False
        # Per-choice enum note learn state
        self._note_enum_learn_target: str | None = None
        self._note_enum_learn_btn:    QPushButton | None = None
        self._bridge = _Bridge()
        self._bridge.midi_event.connect(self._on_midi_event)

        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setStyleSheet(_STYLESHEET)
        self.setMinimumWidth(440)
        self.resize(460, 720)

        self._build_ui()
        self._refresh_devices()
        self._restore_settings()

        # Poll timer — updates activity label
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(80)

        # Raw MIDI listener: forward events to Qt thread AND dispatch to PM
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
        param_row.addWidget(self._param_combo)
        root.addLayout(param_row)

        # CC / Note radio buttons
        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        self._type_group = QButtonGroup(self)
        self._rb_cc   = QRadioButton("CC")
        self._rb_note = QRadioButton("Note")
        self._rb_cc.setChecked(True)
        self._type_group.addButton(self._rb_cc,   0)
        self._type_group.addButton(self._rb_note, 1)
        self._rb_cc.toggled.connect(self._update_mode_ui)
        type_row.addWidget(self._rb_cc)
        type_row.addWidget(self._rb_note)
        type_row.addStretch()
        root.addLayout(type_row)

        # ── CC sub-widgets ────────────────────────────────────────────────────
        # (a) Numeric: Min / Max fields
        self._cc_range_widget = QWidget()
        rr = QHBoxLayout(self._cc_range_widget)
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
        root.addWidget(self._cc_range_widget)

        # (b) Enum: info label showing the sweep
        self._cc_enum_widget = QWidget()
        er = QHBoxLayout(self._cc_enum_widget)
        er.setContentsMargins(0, 0, 0, 0)
        self._lbl_cc_enum = QLabel("")
        self._lbl_cc_enum.setObjectName("enum")
        self._lbl_cc_enum.setWordWrap(True)
        er.addWidget(self._lbl_cc_enum)
        self._cc_enum_widget.setVisible(False)
        root.addWidget(self._cc_enum_widget)

        # ── Note sub-widgets ──────────────────────────────────────────────────
        # (a) Bool / generic: Toggle vs Press mode + hint label
        self._note_bool_widget = QWidget()
        nbr = QVBoxLayout(self._note_bool_widget)
        nbr.setContentsMargins(0, 0, 0, 0)
        nbr.setSpacing(3)

        note_mode_row = QHBoxLayout()
        note_mode_row.addWidget(QLabel("Mode:"))
        self._note_mode_group = QButtonGroup(self)
        self._rb_toggle = QRadioButton("Toggle")
        self._rb_press  = QRadioButton("Press")
        self._rb_toggle.setChecked(True)
        self._note_mode_group.addButton(self._rb_toggle, 0)
        self._note_mode_group.addButton(self._rb_press,  1)
        self._rb_toggle.toggled.connect(self._update_note_hint)
        note_mode_row.addWidget(self._rb_toggle)
        note_mode_row.addWidget(self._rb_press)
        note_mode_row.addStretch()
        nbr.addLayout(note_mode_row)

        self._lbl_note_hint = QLabel("Note-on toggles value.")
        self._lbl_note_hint.setObjectName("info")
        self._lbl_note_hint.setWordWrap(True)
        nbr.addWidget(self._lbl_note_hint)
        self._note_bool_widget.setVisible(False)
        root.addWidget(self._note_bool_widget)

        # (b) Enum: per-choice note assignment rows
        self._note_enum_widget = QWidget()
        ne_vbox = QVBoxLayout(self._note_enum_widget)
        ne_vbox.setContentsMargins(0, 2, 0, 2)
        ne_vbox.setSpacing(2)
        lbl_ne_hdr = QLabel("Assign a note to each value:")
        lbl_ne_hdr.setObjectName("info")
        ne_vbox.addWidget(lbl_ne_hdr)
        # Inner widget that _build_enum_note_rows() clears and rebuilds
        self._note_enum_rows_widget = QWidget()
        self._note_enum_rows_layout = QVBoxLayout(self._note_enum_rows_widget)
        self._note_enum_rows_layout.setContentsMargins(4, 0, 0, 0)
        self._note_enum_rows_layout.setSpacing(2)
        ne_vbox.addWidget(self._note_enum_rows_widget)
        self._note_enum_widget.setVisible(False)
        root.addWidget(self._note_enum_widget)

        # Learn button row (hidden in Note+Enum mode — per-row buttons take over)
        self._learn_row_widget = QWidget()
        learn_row = QHBoxLayout(self._learn_row_widget)
        learn_row.setContentsMargins(0, 0, 0, 0)
        self._btn_learn = QPushButton("Learn")
        self._btn_learn.setObjectName("learn")
        self._btn_learn.setProperty("active", "false")
        self._btn_learn.clicked.connect(self._on_learn_clicked)
        learn_row.addWidget(self._btn_learn)
        self._lbl_learn = QLabel("Click Learn, then send a MIDI event.")
        self._lbl_learn.setObjectName("info")
        learn_row.addWidget(self._lbl_learn)
        learn_row.addStretch()
        root.addWidget(self._learn_row_widget)

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

        # Initial UI state
        self._populate_param_combo()
        self._update_mode_ui()
        self._refresh_assignments()

    # ── device management ──────────────────────────────────────────────────────

    def _refresh_devices(self) -> None:
        self._dev_combo.clear()
        names = self.router.available_devices()
        if names:
            self._dev_combo.addItems(names)
        else:
            self._dev_combo.addItem("(no devices found)")

    def _restore_settings(self) -> None:
        qs = QSettings("WarpApp", "WarpApp")

        # Restore saved MIDI preset
        preset = qs.value("midi/preset_path", "")
        if preset and Path(preset).exists():
            try:
                self.pm.load_json(preset)
                self.save_path = Path(preset)
                self._refresh_assignments()
                self._lbl_status.setText(f"Auto-loaded: {Path(preset).name}")
            except Exception as exc:
                print(f"[midi_panel] auto-load error: {exc}")

        # Restore saved MIDI device and auto-connect
        device = qs.value("midi/device", "")
        if device:
            idx = self._dev_combo.findText(device)
            if idx >= 0:
                self._dev_combo.setCurrentIndex(idx)
                self._on_connect()

    def _on_connect(self) -> None:
        name = self._dev_combo.currentText()
        if name.startswith("("):
            return
        ok = self.router.start(name)
        if ok:
            self._lbl_status.setObjectName("ok")
            self._lbl_status.setText(f"● Connected: {name}")
            QSettings("WarpApp", "WarpApp").setValue("midi/device", name)
        else:
            self._lbl_status.setObjectName("err")
            self._lbl_status.setText(f"✗ Failed to connect: {name}")
        _restyle(self._lbl_status)

    def _on_disconnect(self) -> None:
        self.router.stop()
        self._lbl_status.setObjectName("info")
        self._lbl_status.setText("Not connected")
        _restyle(self._lbl_status)

    # ── param combo ───────────────────────────────────────────────────────────

    def _populate_param_combo(self) -> None:
        """Fill _param_combo from pm, grouped by section."""
        self._param_keys: list[str] = []   # prop_key per combo index
        self._param_combo.clear()

        for section in self.pm.sections():
            for prop in self.pm.props_in(section):
                self._param_combo.addItem(f"{section} / {prop.label or prop.name}")
                self._param_keys.append(prop.key)

        self._param_combo.currentIndexChanged.connect(self._update_mode_ui)
        self._update_mode_ui()

    def _current_prop(self) -> PropDef | None:
        idx = self._param_combo.currentIndex()
        if idx < 0 or idx >= len(self._param_keys):
            return None
        return self.pm._defs.get(self._param_keys[idx])

    # ── mode-aware sub-widget visibility ──────────────────────────────────────

    def _update_mode_ui(self) -> None:
        """Show/hide the right sub-widgets for the current (type, prop kind)."""
        # Cancel any in-progress enum row learn when the user switches param/type
        self._cancel_enum_learn()

        prop      = self._current_prop()
        is_cc     = self._rb_cc.isChecked()
        is_note   = not is_cc
        is_enum   = _is_enum(prop) if prop else False
        is_bool   = (prop and prop.type is bool and not is_enum)
        note_enum = is_note and is_enum

        # CC sub-widgets
        self._cc_range_widget.setVisible(is_cc and not is_enum)
        self._cc_enum_widget.setVisible(is_cc and is_enum)

        # Note sub-widgets
        self._note_bool_widget.setVisible(is_note and not is_enum)
        self._note_enum_widget.setVisible(note_enum)

        # Main Learn row — hidden for Note+Enum (per-row buttons take over)
        self._learn_row_widget.setVisible(not note_enum)

        if prop:
            # Prefill range fields from prop bounds
            if is_cc and not is_enum:
                lo = prop.min_val if prop.min_val is not None else (0   if prop.type is int else 0.0)
                hi = prop.max_val if prop.max_val is not None else (127 if prop.type is int else 1.0)
                self._edit_min.setText(str(lo))
                self._edit_max.setText(str(hi))

            # Fill enum sweep label
            if is_cc and is_enum:
                choices_str = " | ".join(prop.choices)
                self._lbl_cc_enum.setText(
                    f"Sweeps {len(prop.choices)} values: {choices_str}"
                )

            # Fill note hint for non-enum
            if is_note and not is_enum:
                self._update_note_hint()

            # Build per-choice note assignment rows
            if note_enum:
                self._build_enum_note_rows(prop)

    # ── Note mode hint ─────────────────────────────────────────────────────────

    def _update_note_hint(self) -> None:
        """Refresh the note hint label based on current prop and Toggle/Press mode."""
        prop    = self._current_prop()
        is_bool = prop and prop.type is bool and not _is_enum(prop)
        press   = self._rb_press.isChecked()
        if press:
            if is_bool:
                hint = "Key held → True  |  Key released → False"
            else:
                hint = "Key held → on-value  |  Key released → off-value (0)"
        else:
            if is_bool:
                hint = "Note-on toggles boolean."
            else:
                hint = f"Note-on increments {prop.label or prop.name} by one step." if prop else "Note-on toggles value."
        self._lbl_note_hint.setText(hint)

    # ── MIDI learn ─────────────────────────────────────────────────────────────

    def _on_learn_clicked(self) -> None:
        if self._learn_active:
            self._cancel_learn()
        else:
            self._start_learn()

    def _start_learn(self) -> None:
        self._learn_active = True
        self._btn_learn.setText("Listening…  (click to cancel)")
        self._btn_learn.setProperty("active", "true")
        _restyle(self._btn_learn)
        self._lbl_learn.setText("Waiting for MIDI event…")

    def _cancel_learn(self) -> None:
        self._learn_active = False
        self._btn_learn.setText("Learn")
        self._btn_learn.setProperty("active", "false")
        _restyle(self._btn_learn)
        self._lbl_learn.setText("Click Learn, then send a MIDI event.")

    def _commit_learn(self, evt: dict) -> None:
        """Called on the Qt thread when a MIDI event arrives during learn mode."""
        prop = self._current_prop()
        if prop is None:
            self._cancel_learn()
            return

        ev_type  = evt["type"]
        number   = evt["number"]
        channel  = evt.get("channel", 0)
        is_cc    = self._rb_cc.isChecked()
        is_note  = not is_cc
        is_enum  = _is_enum(prop)

        if ev_type == "cc" and is_cc:
            if is_enum:
                # Sweep all choices via CC
                self.pm.bind_enum_to_cc(prop.key, number, channel)
                choices_str = " | ".join(prop.choices)
                self._lbl_learn.setText(
                    f"✓ CC#{number} → {prop.key}  [enum: {choices_str}]"
                )
            else:
                # Range CC
                try:
                    min_v = float(self._edit_min.text())
                    max_v = float(self._edit_max.text())
                except ValueError:
                    min_v = prop.min_val or 0.0
                    max_v = prop.max_val or 1.0
                if prop.type is int:
                    min_v, max_v = int(min_v), int(max_v)
                self.pm.bind_midi_cc(MidiCCBinding(
                    cc=number, prop_key=prop.key,
                    min_val=min_v, max_val=max_v,
                    channel=channel, mode="range",
                ))
                self._lbl_learn.setText(
                    f"✓ CC#{number} → {prop.key}  [{min_v:.4g} – {max_v:.4g}]"
                )

        elif ev_type == "note" and is_note and not is_enum:
            # Toggle / Press (enum case handled by per-row Learn buttons)
            note_mode = "press" if self._rb_press.isChecked() else "toggle"
            self.pm.bind_midi_note(MidiNoteBinding(
                note=number, prop_key=prop.key, channel=channel, mode=note_mode,
            ))
            self._lbl_learn.setText(
                f"✓ Note {_note_name(number)}({number}) → {prop.key}  [{note_mode}]"
            )

        else:
            # Wrong event type (or Note+Enum — handled by per-row buttons)
            want = "CC" if is_cc else "Note"
            self._lbl_learn.setText(f"Waiting for {want} event…")
            return   # don't cancel learn

        # Success
        self._learn_active = False
        self._btn_learn.setText("Learn")
        self._btn_learn.setProperty("active", "false")
        _restyle(self._btn_learn)
        self._refresh_assignments()

    # ── Enum note: per-choice row table ───────────────────────────────────────

    def _build_enum_note_rows(self, prop: PropDef | None = None) -> None:
        """Clear and rebuild the per-choice note assignment rows."""
        if prop is None:
            prop = self._current_prop()
        layout = self._note_enum_rows_layout

        # Clear existing rows
        while layout.count() > 0:
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        if prop is None or not _is_enum(prop):
            return

        # Build lookup: target_value -> (note, channel)
        assigned: dict[str, tuple[int, int]] = {}
        for b in self.pm.midi_note_bindings():
            if b.prop_key == prop.key and b.target_value is not None:
                assigned[str(b.target_value)] = (b.note, b.channel)

        for choice in prop.choices:
            row_w  = QWidget()
            row_hl = QHBoxLayout(row_w)
            row_hl.setContentsMargins(0, 1, 0, 1)
            row_hl.setSpacing(6)

            lbl_choice = QLabel(choice)
            lbl_choice.setFixedWidth(118)
            lbl_choice.setObjectName("mono")
            row_hl.addWidget(lbl_choice)

            note_ch = assigned.get(choice)
            note_str = f"{_note_name(note_ch[0])}({note_ch[0]})" if note_ch else "—"
            lbl_note = QLabel(note_str)
            lbl_note.setFixedWidth(68)
            lbl_note.setObjectName("ok" if note_ch else "info")
            row_hl.addWidget(lbl_note)

            btn_learn = QPushButton("Learn")
            btn_learn.setObjectName("learn")
            btn_learn.setProperty("active", "false")
            btn_learn.setFixedWidth(52)
            # If this row is already the active listen target, show it
            if self._note_enum_learn_target == choice:
                btn_learn.setText("…")
                btn_learn.setProperty("active", "true")
                _restyle(btn_learn)
                self._note_enum_learn_btn = btn_learn
            btn_learn.clicked.connect(
                self._make_enum_learn_handler(choice, btn_learn, prop)
            )
            row_hl.addWidget(btn_learn)

            if note_ch:
                btn_rm = QPushButton("✕")
                btn_rm.setObjectName("remove")
                btn_rm.setFixedWidth(22)
                btn_rm.clicked.connect(
                    self._make_enum_row_remove_handler(note_ch[0], note_ch[1], prop)
                )
                row_hl.addWidget(btn_rm)

            row_hl.addStretch()
            layout.addWidget(row_w)

    def _make_enum_learn_handler(
        self, choice: str, btn: QPushButton, prop: PropDef
    ):
        def _handler():
            if self._note_enum_learn_target == choice:
                self._cancel_enum_learn()
            else:
                self._cancel_enum_learn()
                self._note_enum_learn_target = choice
                self._note_enum_learn_btn    = btn
                btn.setText("…")
                btn.setProperty("active", "true")
                _restyle(btn)
        return _handler

    def _make_enum_row_remove_handler(self, note: int, channel: int, prop: PropDef):
        def _handler():
            self.pm.remove_midi_note(note, channel)
            self._build_enum_note_rows(prop)
            self._refresh_assignments()
        return _handler

    def _cancel_enum_learn(self) -> None:
        """Reset per-row learn state without committing."""
        self._note_enum_learn_target = None
        if self._note_enum_learn_btn is not None:
            self._note_enum_learn_btn.setText("Learn")
            self._note_enum_learn_btn.setProperty("active", "false")
            _restyle(self._note_enum_learn_btn)
            self._note_enum_learn_btn = None

    def _commit_enum_learn(self, evt: dict) -> None:
        """Called on Qt thread when a MIDI event arrives during per-row learn."""
        if evt["type"] != "note" or evt.get("value", 0) == 0:
            return  # ignore CC and note-off

        prop   = self._current_prop()
        choice = self._note_enum_learn_target
        if prop is None or choice is None:
            self._cancel_enum_learn()
            return

        note    = evt["number"]
        channel = evt.get("channel", 0)
        self.pm.bind_midi_note(MidiNoteBinding(
            note=note, prop_key=prop.key,
            channel=channel, target_value=choice,
        ))

        self._cancel_enum_learn()
        self._build_enum_note_rows(prop)
        self._refresh_assignments()

    # ── MIDI event routing ────────────────────────────────────────────────────

    def _raw_listener(self, evt: dict) -> None:
        """Called from MIDI thread -- emit signal (Qt-safe) and dispatch to PM."""
        try:
            if evt["type"] == "cc":
                self.pm.apply_midi_cc(evt.get("channel", 0), evt["number"], evt["value"])
            elif evt["type"] == "note":
                self.pm.apply_midi_note(evt.get("channel", 0), evt["number"], evt.get("value", 0))
        except Exception:
            pass
        self._bridge.midi_event.emit(evt)

    def _on_midi_event(self, evt: dict) -> None:
        """Qt thread handler -- routes to the active learn path."""
        if self._note_enum_learn_target is not None:
            self._commit_enum_learn(evt)
        elif self._learn_active:
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
                    f"{'Note On' if v > 0 else 'Note Off'}  "
                    f"{_note_name(n)}({n})  ch={ch}  vel={v}"
                )

    # ── assignment list ────────────────────────────────────────────────────────

    def _refresh_assignments(self) -> None:
        # Keep the enum note rows in sync whenever assignments change
        prop = self._current_prop()
        if prop and _is_enum(prop) and not self._rb_cc.isChecked():
            self._build_enum_note_rows(prop)

        # Clear existing rows (except the terminal stretch)
        while self._assign_layout.count() > 1:
            item = self._assign_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        rows = []

        for b in self.pm.midi_cc_bindings():
            p = self.pm._defs.get(b.prop_key)
            if p and _is_enum(p):
                choices_str = " | ".join(p.choices)
                label = f"CC#{b.cc}  ->  {b.prop_key}  [enum: {choices_str}]"
            else:
                lo = b.min_val if b.min_val is not None else "?"
                hi = b.max_val if b.max_val is not None else "?"
                label = (f"CC#{b.cc}  ->  {b.prop_key}  [{lo:.4g} - {hi:.4g}]"
                         if isinstance(lo, float)
                         else f"CC#{b.cc}  ->  {b.prop_key}  [{lo} - {hi}]")
            rows.append(("cc", label, b.cc, b.channel))

        for b in self.pm.midi_note_bindings():
            if b.prop_key is None:
                continue
            note_str = f"{_note_name(b.note)}({b.note})"
            if b.target_value is not None:
                label = f"Note {note_str}  ->  {b.prop_key} = '{b.target_value}'"
            else:
                mode_tag = getattr(b, "mode", "toggle")
                label = f"Note {note_str}  ->  {b.prop_key}  [{mode_tag}]"
            rows.append(("note", label, b.note, b.channel))

        if not rows:
            lbl = QLabel("No assignments yet.")
            lbl.setObjectName("info")
            self._assign_layout.insertWidget(0, lbl)
            return

        for pos, (kind, label, num, ch) in enumerate(rows):
            row_w  = QWidget()
            row_hl = QHBoxLayout(row_w)
            row_hl.setContentsMargins(0, 1, 0, 1)
            row_hl.setSpacing(4)

            lbl = QLabel(label)
            lbl.setObjectName("mono")
            lbl.setWordWrap(False)
            row_hl.addWidget(lbl, stretch=1)

            btn_rm = QPushButton("x")
            btn_rm.setObjectName("remove")
            btn_rm.setToolTip("Remove assignment")
            btn_rm.clicked.connect(self._make_remove_handler(kind, num, ch))
            row_hl.addWidget(btn_rm)

            self._assign_layout.insertWidget(pos, row_w)

    def _make_remove_handler(self, kind: str, num: int, channel: int):
        def _handler():
            if kind == "cc":
                self.pm.remove_midi_cc(num, channel)
            else:
                self.pm.remove_midi_note(num, channel)
            self._refresh_assignments()
        return _handler

    # ── save / load ───────────────────────────────────────────────────────────

    def _on_save(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save MIDI Mappings", str(self.save_path), "JSON (*.json)"
        )
        if path:
            self.pm.save_json(path)
            self.save_path = Path(path)
            QSettings("WarpApp", "WarpApp").setValue("midi/preset_path", path)
            self._lbl_learn.setText(f"Saved: {Path(path).name}")

    def _on_load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load MIDI Mappings", str(self.save_path), "JSON (*.json)"
        )
        if not path:
            return
        try:
            self.pm.load_json(path)
            self.save_path = Path(path)
            QSettings("WarpApp", "WarpApp").setValue("midi/preset_path", path)
            self._lbl_learn.setText(f"Loaded: {Path(path).name}")
            self._refresh_assignments()
        except Exception as exc:
            self._lbl_learn.setText(f"Load error: {exc}")
            print(f"[midi_panel] load error: {exc}")
