"""
color_panel.py — Color harmony palette manager

PySide6 panel that provides a GUI frontend to color_harmony.py and lets
the user select, tweak, and propagate colour palettes to the rest of the app.

Features
--------
* Scheme selector (all ColorScheme variants)
* Seed spinner + "random" button
* Saturation and lightness range sliders
* Live colour swatches — click a swatch to open a QColorDialog override
* Per-slot lock button (locked slots survive regeneration)
* Callback list: register listeners that receive List[RGB] on every change
* Auto-regen mode: regenerate whenever a control changes

Usage::

    from color_panel import ColorPanel

    panel = ColorPanel()
    panel.add_change_listener(lambda colors: print(colors))
    panel.show()

Standalone::

    python color_panel.py
"""

from __future__ import annotations

import random as _random
from typing import Callable, Optional

from PySide6.QtCore    import Qt, Signal, QSize
from PySide6.QtGui     import QColor, QIcon, QPixmap, QPainter
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QScrollArea,
    QFrame, QSizePolicy, QSpinBox, QDoubleSpinBox,
    QCheckBox, QColorDialog, QSlider, QGroupBox,
)

from color_harmony import (
    ColorScheme, generate_palette,
    DEFAULT_SAT, DEFAULT_LIT,
    to_hex, RGB,
)


# ── stylesheet (matches audio_panel.py dark theme) ────────────────────────────

_STYLESHEET = """
QWidget {
    background-color: #1a1a22;
    color: #c8c8d0;
    font-family: Consolas, "Courier New", monospace;
    font-size: 12px;
}
QLabel           { color: #707078; }
QLabel#hdr       { color: #c8c8d0; font-weight: bold; padding-top: 4px; }
QLabel#sub       { color: #707078; font-size: 11px; font-style: italic; }
QLabel#hex       { color: #5eaeff; min-width: 64px; }
QLabel#role      { color: #a070d0; min-width: 72px; }
QLabel#info      { color: #707078; font-style: italic; }

QComboBox {
    background-color: #111118;
    color: #5eaeff;
    border: 1px solid #38383f;
    border-radius: 3px;
    padding: 2px 6px;
    min-width: 120px;
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
QPushButton:hover   { background-color: #32323f; border-color: #5eaeff; }
QPushButton:pressed { background-color: #1a1a28; }
QPushButton#gen     { color: #7ec87e; border-color: #3a5a3a; }
QPushButton#gen:hover { border-color: #7ec87e; }
QPushButton#apply   { color: #5eaeff; border-color: #2a4070; }
QPushButton#apply:hover { border-color: #5eaeff; }
QPushButton#rnd     { color: #ffb347; border-color: #5a4a1a; min-width: 28px; padding: 3px 6px; }
QPushButton#rnd:hover { border-color: #ffb347; }
QPushButton#lock    { min-width: 20px; padding: 2px 5px; color: #707078;
                       border-color: #2e2e38; }
QPushButton#lock[locked="true"]  { color: #ffb347; border-color: #ffb347; }
QPushButton#lock:hover { border-color: #c8c8d0; }

QSpinBox, QDoubleSpinBox {
    background-color: #111118;
    color: #5eaeff;
    border: 1px solid #38383f;
    border-radius: 3px;
    padding: 2px 5px;
}
QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid #5eaeff; }
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    width: 14px; background: #2a2a36; border: none;
}

QSlider::groove:horizontal {
    background: #111118;
    border: 1px solid #2e2e38;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #5eaeff;
    border: 1px solid #38383f;
    width: 12px;
    margin: -4px 0;
    border-radius: 6px;
}
QSlider::sub-page:horizontal { background: #2a4070; border-radius: 3px; }

QCheckBox { color: #c8c8d0; spacing: 6px; }
QCheckBox::indicator {
    width: 13px; height: 13px;
    border: 1px solid #38383f;
    border-radius: 2px;
    background: #111118;
}
QCheckBox::indicator:checked { background: #5eaeff; border-color: #5eaeff; }

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
QFrame#swatch_box {
    background-color: #12121a;
    border: 1px solid #2e2e38;
    border-radius: 4px;
}
"""

_ROLE_NAMES = ["primary", "secondary", "tertiary", "quaternary"]

_SCHEME_LABELS = {
    ColorScheme.COMPLEMENTARY:       "Complementary  (2)",
    ColorScheme.SPLIT_COMPLEMENTARY: "Split-Complementary  (3)",
    ColorScheme.TRIADIC:             "Triadic  (3)",
    ColorScheme.ANALOGOUS:           "Analogous  (4)",
    ColorScheme.TETRADIC:            "Tetradic  (4)",
    ColorScheme.MONOCHROMATIC:       "Monochromatic  (4)",
}


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


def _restyle(widget: QWidget) -> None:
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def _rgb_to_qcolor(r: float, g: float, b: float) -> QColor:
    return QColor(round(r * 255), round(g * 255), round(b * 255))


def _qcolor_to_rgb(c: QColor) -> RGB:
    return (c.redF(), c.greenF(), c.blueF())


def _swatch_icon(color: QColor, size: int = 40) -> QIcon:
    """Create a solid-colour QIcon for the swatch button."""
    pix = QPixmap(size, size)
    p = QPainter(pix)
    p.fillRect(0, 0, size, size, color)
    p.end()
    return QIcon(pix)


# ── single colour slot ────────────────────────────────────────────────────────

class _ColorSlot(QFrame):
    """
    One row in the palette display.

    Shows: role label | coloured swatch button | hex code | lock button
    """
    color_changed = Signal(int, tuple)   # (slot_index, (r, g, b))

    def __init__(self, index: int, color: RGB, parent: QWidget = None):
        super().__init__(parent)
        self._index  = index
        self._color  = color
        self._locked = False
        self.setObjectName("swatch_box")
        self._build()

    # ── public ────────────────────────────────────────────────────────────────

    def set_color(self, color: RGB, emit: bool = False) -> None:
        self._color = color
        self._refresh_visuals()
        if emit:
            self.color_changed.emit(self._index, color)

    def color(self) -> RGB:
        return self._color

    def is_locked(self) -> bool:
        return self._locked

    # ── build ─────────────────────────────────────────────────────────────────

    def _build(self) -> None:
        row = QHBoxLayout(self)
        row.setContentsMargins(8, 5, 8, 5)
        row.setSpacing(8)

        # role label
        role = _ROLE_NAMES[self._index] if self._index < len(_ROLE_NAMES) else f"color {self._index}"
        self._role_lbl = QLabel(role)
        self._role_lbl.setObjectName("role")
        self._role_lbl.setFixedWidth(72)
        row.addWidget(self._role_lbl)

        # swatch (clickable colour button)
        self._swatch_btn = QPushButton()
        self._swatch_btn.setFixedSize(QSize(48, 24))
        self._swatch_btn.setFlat(True)
        self._swatch_btn.setCursor(Qt.PointingHandCursor)
        self._swatch_btn.setToolTip("Click to pick a custom colour")
        self._swatch_btn.clicked.connect(self._on_pick)
        row.addWidget(self._swatch_btn)

        # hex label
        self._hex_lbl = QLabel()
        self._hex_lbl.setObjectName("hex")
        self._hex_lbl.setFixedWidth(64)
        row.addWidget(self._hex_lbl)

        # rgb label
        self._rgb_lbl = QLabel()
        self._rgb_lbl.setObjectName("sub")
        self._rgb_lbl.setMinimumWidth(120)
        row.addWidget(self._rgb_lbl, stretch=1)

        # lock button
        self._lock_btn = QPushButton("🔓")
        self._lock_btn.setObjectName("lock")
        self._lock_btn.setFixedWidth(28)
        self._lock_btn.setCheckable(False)
        self._lock_btn.setToolTip("Lock this slot (won't change on regeneration)")
        self._lock_btn.clicked.connect(self._on_lock)
        row.addWidget(self._lock_btn)

        self._refresh_visuals()

    def _refresh_visuals(self) -> None:
        r, g, b = self._color
        qc = _rgb_to_qcolor(r, g, b)

        # swatch colour via stylesheet background
        hex_str = to_hex(self._color)
        # Pick a contrasting border
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        border = "#ffffff" if luma < 0.5 else "#000000"
        self._swatch_btn.setStyleSheet(
            f"QPushButton {{ background-color: {hex_str}; "
            f"border: 1px solid {border}; border-radius: 2px; }}"
        )

        r8, g8, b8 = round(r * 255), round(g * 255), round(b * 255)
        self._hex_lbl.setText(hex_str)
        self._rgb_lbl.setText(f"rgb({r8:3d}, {g8:3d}, {b8:3d})")

    def _on_pick(self) -> None:
        r, g, b = self._color
        initial = _rgb_to_qcolor(r, g, b)
        chosen  = QColorDialog.getColor(initial, self, f"Pick colour — {_ROLE_NAMES[self._index]}")
        if chosen.isValid():
            self.set_color(_qcolor_to_rgb(chosen), emit=True)
            # Auto-lock after manual pick so regen doesn't overwrite it
            if not self._locked:
                self._set_locked(True)

    def _on_lock(self) -> None:
        self._set_locked(not self._locked)

    def _set_locked(self, locked: bool) -> None:
        self._locked = locked
        self._lock_btn.setText("🔒" if locked else "🔓")
        self._lock_btn.setProperty("locked", "true" if locked else "false")
        _restyle(self._lock_btn)


# ── main panel ────────────────────────────────────────────────────────────────

class ColorPanel(QWidget):
    """
    Palette manager panel.

    Parameters
    ----------
    on_change : callable | None
        Optional shorthand for ``panel.add_change_listener(fn)``.
        Called with ``List[RGB]`` whenever the palette changes.
    title : str
        Window title.
    """

    palette_changed = Signal(list)   # List[RGB]

    def __init__(
        self,
        on_change: Optional[Callable[[list[RGB]], None]] = None,
        on_apply : Optional[Callable[[list[RGB]], None]] = None,
        title    : str = "Color Harmony",
    ):
        super().__init__()
        self._listeners:      list[Callable[[list[RGB]], None]] = []
        self._apply_listener: Optional[Callable[[list[RGB]], None]] = on_apply
        self._palette:        list[RGB] = []
        self._slots:          list[_ColorSlot] = []

        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setStyleSheet(_STYLESHEET)
        self.setMinimumWidth(420)
        self.resize(480, 560)

        self._build_ui()

        if on_change is not None:
            self.add_change_listener(on_change)
        if on_apply is not None:
            self._apply_listener = on_apply

        # Generate initial palette
        self._generate()

    # ── public API ────────────────────────────────────────────────────────────

    def add_change_listener(self, fn: Callable[[list[RGB]], None]) -> None:
        """Register a callback; receives List[RGB] on every palette change."""
        self._listeners.append(fn)

    def remove_change_listener(self, fn: Callable[[list[RGB]], None]) -> None:
        self._listeners = [f for f in self._listeners if f is not fn]

    def current_palette(self) -> list[RGB]:
        """Return the current palette as a list of (r, g, b) float tuples."""
        return list(self._palette)

    def current_palette_hex(self) -> list[str]:
        """Return the current palette as '#RRGGBB' hex strings."""
        return [to_hex(c) for c in self._palette]

    def current_palette_uint8(self) -> list[tuple[int, int, int]]:
        """Return the current palette as (0-255) integer tuples."""
        return [tuple(round(c * 255) for c in color) for color in self._palette]

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(6)

        # ── Scheme ────────────────────────────────────────────────────────────
        root.addWidget(_hdr("Harmony Scheme"))
        root.addWidget(_sep())

        scheme_row = QHBoxLayout()
        self._scheme_combo = QComboBox()
        for scheme in ColorScheme:
            self._scheme_combo.addItem(_SCHEME_LABELS[scheme], userData=scheme)
        self._scheme_combo.setCurrentIndex(2)   # triadic as default
        self._scheme_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._scheme_combo.currentIndexChanged.connect(self._on_control_changed)
        scheme_row.addWidget(self._scheme_combo)
        root.addLayout(scheme_row)

        # ── Seed ──────────────────────────────────────────────────────────────
        root.addSpacing(4)
        root.addWidget(_hdr("Seed"))
        root.addWidget(_sep())

        seed_row = QHBoxLayout()
        seed_row.setSpacing(6)

        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 999999)
        self._seed_spin.setValue(42)
        self._seed_spin.setFixedWidth(80)
        self._seed_spin.valueChanged.connect(self._on_control_changed)
        seed_row.addWidget(QLabel("Seed"))
        seed_row.addWidget(self._seed_spin)

        btn_rnd = QPushButton("⚄")
        btn_rnd.setObjectName("rnd")
        btn_rnd.setToolTip("Pick a random seed")
        btn_rnd.clicked.connect(self._on_random_seed)
        seed_row.addWidget(btn_rnd)

        self._auto_check = QCheckBox("Auto-regen")
        self._auto_check.setChecked(True)
        seed_row.addWidget(self._auto_check)

        seed_row.addStretch()
        root.addLayout(seed_row)

        # ── Saturation range ──────────────────────────────────────────────────
        root.addSpacing(4)
        root.addWidget(_hdr("Saturation Range"))
        root.addWidget(_sep())
        root.addLayout(self._build_range_row("sat", DEFAULT_SAT))

        # ── Lightness range ───────────────────────────────────────────────────
        root.addSpacing(4)
        root.addWidget(_hdr("Lightness Range"))
        root.addWidget(_sep())
        root.addLayout(self._build_range_row("lit", DEFAULT_LIT))

        # ── Generate / Apply buttons ──────────────────────────────────────────
        root.addSpacing(6)
        gen_row = QHBoxLayout()
        gen_row.addStretch()
        self._gen_btn = QPushButton("⟳  Generate Palette")
        self._gen_btn.setObjectName("gen")
        self._gen_btn.clicked.connect(self._generate)
        gen_row.addWidget(self._gen_btn)
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.setObjectName("apply")
        self._apply_btn.setToolTip("Push this palette to scene elements (new spawns only, no regen)")
        self._apply_btn.clicked.connect(self._on_apply_clicked)
        gen_row.addWidget(self._apply_btn)
        root.addLayout(gen_row)

        # ── Palette swatches ──────────────────────────────────────────────────
        root.addSpacing(4)
        root.addWidget(_hdr("Palette"))
        root.addWidget(_sep())

        self._swatch_container = QWidget()
        self._swatch_layout    = QVBoxLayout(self._swatch_container)
        self._swatch_layout.setContentsMargins(0, 2, 0, 2)
        self._swatch_layout.setSpacing(4)
        self._swatch_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(self._swatch_container)
        root.addWidget(scroll, stretch=1)

        # ── Status label ──────────────────────────────────────────────────────
        self._status_lbl = QLabel("")
        self._status_lbl.setObjectName("info")
        root.addWidget(self._status_lbl)

    def _build_range_row(self, name: str, defaults: tuple[float, float]) -> QHBoxLayout:
        """Build a min/max slider pair and store refs as self._<name>_{min,max}."""
        row = QHBoxLayout()
        row.setSpacing(8)

        def _slider(default_val: float) -> QSlider:
            s = QSlider(Qt.Horizontal)
            s.setRange(0, 100)
            s.setValue(round(default_val * 100))
            s.setFixedWidth(90)
            s.valueChanged.connect(self._on_control_changed)
            return s

        s_min = _slider(defaults[0])
        s_max = _slider(defaults[1])

        lbl_min = QLabel(f"{defaults[0]:.2f}")
        lbl_min.setObjectName("hex")
        lbl_min.setFixedWidth(32)

        lbl_max = QLabel(f"{defaults[1]:.2f}")
        lbl_max.setObjectName("hex")
        lbl_max.setFixedWidth(32)

        def _update_min_lbl(v):
            lbl_min.setText(f"{v/100:.2f}")
            # Clamp: min can't exceed max
            if v > s_max.value():
                s_max.setValue(v)

        def _update_max_lbl(v):
            lbl_max.setText(f"{v/100:.2f}")
            if v < s_min.value():
                s_min.setValue(v)

        s_min.valueChanged.connect(_update_min_lbl)
        s_max.valueChanged.connect(_update_max_lbl)

        row.addWidget(QLabel("Min"))
        row.addWidget(s_min)
        row.addWidget(lbl_min)
        row.addSpacing(8)
        row.addWidget(QLabel("Max"))
        row.addWidget(s_max)
        row.addWidget(lbl_max)
        row.addStretch()

        setattr(self, f"_{name}_min", s_min)
        setattr(self, f"_{name}_max", s_max)

        return row

    # ── swatch management ─────────────────────────────────────────────────────

    def _rebuild_slots(self, palette: list[RGB]) -> None:
        """Recreate swatch widgets to match the new palette length."""
        # Remove old
        while self._swatch_layout.count() > 1:
            item = self._swatch_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._slots.clear()

        for i, color in enumerate(palette):
            slot = _ColorSlot(i, color, parent=self._swatch_container)
            slot.color_changed.connect(self._on_slot_color_changed)
            self._slots.append(slot)
            self._swatch_layout.insertWidget(i, slot)

    def _update_slots(self, palette: list[RGB]) -> None:
        """Update existing swatch widgets in-place (preserves lock state)."""
        # If slot count changed, rebuild from scratch
        if len(palette) != len(self._slots):
            self._rebuild_slots(palette)
            return

        for slot, color in zip(self._slots, palette):
            if not slot.is_locked():
                slot.set_color(color)

    # ── generation ────────────────────────────────────────────────────────────

    def _generate(self) -> None:
        scheme   = self._scheme_combo.currentData()
        seed     = self._seed_spin.value()
        sat_min  = self._sat_min.value() / 100.0
        sat_max  = self._sat_max.value() / 100.0
        lit_min  = self._lit_min.value() / 100.0
        lit_max  = self._lit_max.value() / 100.0

        # Ensure valid ranges
        if sat_min > sat_max:
            sat_min, sat_max = sat_max, sat_min
        if lit_min > lit_max:
            lit_min, lit_max = lit_max, lit_min

        new_palette = generate_palette(
            scheme    = scheme,
            seed      = seed,
            sat_range = (sat_min, sat_max),
            lit_range = (lit_min, lit_max),
        )

        # Merge locked slots back in
        if self._slots:
            merged = []
            for i, new_color in enumerate(new_palette):
                if i < len(self._slots) and self._slots[i].is_locked():
                    merged.append(self._slots[i].color())
                else:
                    merged.append(new_color)
            new_palette = merged

        self._palette = new_palette
        self._update_slots(new_palette)

        scheme_name = scheme.value.replace("_", " ")
        hex_list    = "  ".join(to_hex(c) for c in new_palette)
        self._status_lbl.setText(f"{scheme_name}  ·  seed {seed}   {hex_list}")

        self._emit_change()

    def _emit_change(self) -> None:
        self.palette_changed.emit(list(self._palette))
        for fn in self._listeners:
            try:
                fn(list(self._palette))
            except Exception as exc:
                print(f"[color_panel] listener error: {exc}")

    # ── slots ─────────────────────────────────────────────────────────────────

    def _on_apply_clicked(self) -> None:
        if self._apply_listener is not None:
            try:
                self._apply_listener(list(self._palette))
            except Exception as exc:
                print(f"[color_panel] apply error: {exc}")

    def _on_slot_color_changed(self, index: int, color: RGB) -> None:
        if index < len(self._palette):
            self._palette[index] = color
        self._emit_change()

    def _on_control_changed(self) -> None:
        if self._auto_check.isChecked():
            self._generate()

    def _on_random_seed(self) -> None:
        self._seed_spin.setValue(_random.randint(0, 999999))
        # _on_control_changed fires via valueChanged; if auto is off, generate anyway
        if not self._auto_check.isChecked():
            self._generate()


# ── standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    app = QApplication.instance() or QApplication(sys.argv)

    def _on_change(palette):
        hexes = "  ".join(to_hex(c) for c in palette)
        print(f"[palette] {hexes}")

    panel = ColorPanel(on_change=_on_change, title="Color Harmony")
    panel.show()
    sys.exit(app.exec())
