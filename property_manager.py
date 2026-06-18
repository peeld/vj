"""
property_manager.py — Centralised property registry for gui_merged.

Every adjustable parameter across all scene modules is declared here with:
  • a unique dotted key     ("section.name", e.g. "feedback.decay")
  • its type / range / step / choices / default value
  • an optional live-object binding so get/set proxy to the real dataclass

The manager additionally owns:
  • Preset snapshots  — save / load / list  (JSON-persisted)

Signal routing (audio, MIDI, envelopes, LFOs, events) is handled by
LinkManager (link_manager.py), which calls pm.set() from the GL thread.
Link presets (snapshots of routing state) and preset triggers (keyboard/MIDI
events that recall them) are also managed by LinkManager.

Typical usage (in gui_merged.py)
---------------------------------
    from property_manager import PropertyManager, build_default_manager

    pm = build_default_manager(_params, _controls, nn_graph, laser, circles)

    # anywhere — live read/write
    pm.get("feedback.decay")          # -> float
    pm.set("feedback.decay", 0.995)

    # preset round-trip
    pm.save_preset("my_look")
    pm.load_preset("my_look")
    pm.save_json("presets.json")      # persists presets
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
#  Property definition
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PropDef:
    """Declares one adjustable property.

    Attributes
    ----------
    key         : unique dotted identifier, e.g. "feedback.decay"
    section     : logical group,            e.g. "feedback"
    name        : bare field name,          e.g. "decay"
    label       : human-readable label,     e.g. "Decay"
    type        : Python type (float, int, bool, str)
    default     : default value
    min_val     : lower bound for numeric types (None = unconstrained)
    max_val     : upper bound for numeric types (None = unconstrained)
    step        : keyboard increment/decrement quantum
    choices     : ordered list of valid strings (for str / enum props)
    widget_hint : UI hint — "slider", "spinbox", "check", "combo", "line"
    description : one-line docstring shown in param panels
    """
    key         : str
    section     : str
    name        : str
    label       : str
    type        : type
    default     : Any
    min_val     : float | None = None
    max_val     : float | None = None
    step        : float | None = None
    choices     : list[str] | None = None
    widget_hint : str | None = None
    description : str = ""

    # Convenience: the full set of (min, val, max) rounded to step precision
    def clamp(self, value: Any) -> Any:
        if self.type in (float, int):
            # Round to step precision first, then clamp so rounding never
            # pushes the value outside [min_val, max_val].
            if self.step is not None:
                n_decimals = max(0, -int(f"{self.step:e}".split("e")[1]))
                value = round(value, n_decimals)
            if self.min_val is not None:
                value = max(self.min_val, value)
            if self.max_val is not None:
                value = min(self.max_val, value)
        if self.type is str and self.choices and value not in self.choices:
            value = self.default
        return value

    def next_choice(self, current: str) -> str:
        """Return the next item in choices (wraps around). For 'cycle' actions."""
        if not self.choices:
            raise ValueError(f"PropDef '{self.key}' has no choices.")
        idx = self.choices.index(current) if current in self.choices else -1
        return self.choices[(idx + 1) % len(self.choices)]

    def prev_choice(self, current: str) -> str:
        """Return the previous item in choices (wraps around)."""
        if not self.choices:
            raise ValueError(f"PropDef '{self.key}' has no choices.")
        idx = self.choices.index(current) if current in self.choices else 1
        return self.choices[(idx - 1) % len(self.choices)]


# ─────────────────────────────────────────────────────────────────────────────
#  PropertyManager
# ─────────────────────────────────────────────────────────────────────────────

class PropertyManager:
    """Central registry for all scene properties, mappings, and presets.

    Each property is:
      • declared with PropDef  (type, range, default, etc.)
      • optionally bound to a live (object, attribute) pair so that
        get/set proxy directly to the real dataclass field

    If no binding is supplied the value is stored internally and callers are
    responsible for reading it via pm.get() when they need it.

    Signal routing (audio, MIDI, envelopes, LFOs) is handled by LinkManager,
    which writes values here via pm.set() from the GL thread.
    """

    def __init__(self) -> None:
        # prop_key -> PropDef
        self._defs: dict[str, PropDef] = {}

        # prop_key -> (obj, attr_name) live binding
        self._bindings: dict[str, tuple[Any, str]] = {}

        # internal storage for unbound props
        self._values: dict[str, Any] = {}

        # presets:  name -> {prop_key -> value}
        self._presets: dict[str, dict[str, Any]] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def register(
        self,
        prop : PropDef,
        obj  : Any  = None,
        attr : str  = None,
    ) -> "PropertyManager":
        """Register a property definition, optionally binding it to obj.attr.

        Binding means get(key) returns getattr(obj, attr) and
        set(key, v) calls setattr(obj, attr, v) — the PropertyManager does
        NOT hold a separate copy.

        If obj/attr are omitted, the default value is stored internally.
        """
        if prop.key in self._defs:
            raise KeyError(f"Property '{prop.key}' already registered.")
        self._defs[prop.key] = prop

        if obj is not None and attr is not None:
            self._bindings[prop.key] = (obj, attr)
        else:
            # Initialise from default; subsequent set() calls update _values
            self._values[prop.key] = prop.default

        return self  # fluent API

    def bind(self, key: str, obj: Any, attr: str) -> None:
        """Attach a live (obj, attr) binding to an already-registered key.

        If the key currently holds an unbound value (e.g. no instance was
        live yet), that value is latched onto obj.attr before the binding
        takes effect, so the new instance picks up the last-known state.
        """
        self._require(key)
        if key in self._values:
            setattr(obj, attr, self._values.pop(key))
        self._bindings[key] = (obj, attr)

    def unbind(self, key: str) -> None:
        """Detach a key's live binding, preserving its last value internally."""
        if key in self._bindings:
            obj, attr = self._bindings.pop(key)
            self._values[key] = getattr(obj, attr)

    # ── Get / Set ─────────────────────────────────────────────────────────────

    def get(self, key: str) -> Any:
        """Return the current live value of a property."""
        self._require(key)
        if key in self._bindings:
            obj, attr = self._bindings[key]
            return getattr(obj, attr)
        return self._values[key]

    def set(self, key: str, value: Any) -> None:
        """Write a clamped/validated value to a property."""
        self._require(key)
        prop  = self._defs[key]
        value = prop.clamp(value)
        if key in self._bindings:
            obj, attr = self._bindings[key]
            setattr(obj, attr, value)
        else:
            self._values[key] = value

    def reset(self, key: str) -> None:
        """Restore a property to its declared default."""
        self.set(key, self._defs[key].default)

    def reset_all(self) -> None:
        """Restore every property to its declared default."""
        for key in self._defs:
            self.reset(key)

    # ── Introspection ─────────────────────────────────────────────────────────

    def sections(self) -> list[str]:
        """Return unique section names in registration order."""
        seen: dict[str, None] = {}
        for d in self._defs.values():
            seen[d.section] = None
        return list(seen)

    def props_in(self, section: str) -> list[PropDef]:
        """All PropDefs belonging to a section."""
        return [d for d in self._defs.values() if d.section == section]

    def all_props(self) -> list[PropDef]:
        return list(self._defs.values())

    def describe(self, lm=None) -> None:
        """Print a human-readable summary of all registered properties.

        If *lm* (a LinkManager) is provided, sinks with an active SignalLink
        are annotated with [linked].
        """
        linked_keys: set[str] = set()
        if lm is not None:
            linked_keys = {sl.sink_key for sl in lm._signal_links if sl.enabled}
        for section in self.sections():
            print(f"\n[{section}]")
            for d in self.props_in(section):
                live = self.get(d.key)
                rng = ""
                if d.min_val is not None or d.max_val is not None:
                    rng = f"  [{d.min_val} … {d.max_val}]"
                if d.choices:
                    rng = f"  {d.choices}"
                driven = "  [linked]" if d.key in linked_keys else ""
                print(f"  {d.name:<24} = {live!r:<12}  (default={d.default!r}){rng}{driven}")

    # ── Snapshot of current values ────────────────────────────────────────────

    def snapshot(self, keys: list[str] | None = None) -> dict[str, Any]:
        """Return a dict of current values (all keys if keys is None)."""
        ks = keys if keys is not None else list(self._defs)
        return {k: self.get(k) for k in ks if k in self._defs}

    def apply_snapshot(self, snap: dict[str, Any]) -> None:
        """Write a snapshot dict back into live properties."""
        for k, v in snap.items():
            if k in self._defs:
                self.set(k, v)

    # ── Presets ───────────────────────────────────────────────────────────────

    def save_preset(
        self,
        name : str,
        keys : list[str] | None = None,
    ) -> None:
        """Snapshot current values into a named preset.

        If *keys* is supplied only those properties are captured; otherwise
        every registered property is captured.
        """
        self._presets[name] = self.snapshot(keys)
        print(f"[pm] preset saved: '{name}'  ({len(self._presets[name])} props)")

    def load_preset(self, name: str) -> None:
        """Restore a named preset, writing each captured value back live."""
        if name not in self._presets:
            raise KeyError(f"Preset '{name}' not found. Available: {self.list_presets()}")
        self.apply_snapshot(self._presets[name])
        print(f"[pm] preset loaded: '{name}'")

    def delete_preset(self, name: str) -> None:
        self._presets.pop(name, None)

    def list_presets(self) -> list[str]:
        return list(self._presets)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialise presets to a plain dict (JSON-safe)."""
        return {
            "presets": {name: dict(snap) for name, snap in self._presets.items()},
        }

    def from_dict(self, data: dict) -> None:
        """Restore presets from a dict.  Unknown keys are silently ignored."""
        for name, snap in data.get("presets", {}).items():
            self._presets[name] = snap

    def save_json(self, path: str | Path) -> None:
        """Persist presets and mappings to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        print(f"[pm] saved → {path}")

    def load_json(self, path: str | Path) -> None:
        """Load presets and mappings from a JSON file (merges, does not replace)."""
        data = json.loads(Path(path).read_text())
        self.from_dict(data)
        print(f"[pm] loaded ← {path}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _require(self, key: str) -> None:
        if key not in self._defs:
            raise KeyError(f"Unknown property key: '{key}'. "
                           f"Registered: {list(self._defs)}")


# ─────────────────────────────────────────────────────────────────────────────
#  Factory:  build the full registry for gui_merged
# ─────────────────────────────────────────────────────────────────────────────

def build_default_manager(
    params,              # FeedbackParams
    controls,            # SceneControls
    cloud,               # Cloud
    nn_graph,            # NNGraph  (may be None before GL window starts)
    lasers,              # LaserRibbons  (may be None)
    circles,             # CircleAxisDrawing  (may be None)
    camera = None,       # OrbitCamera  (may be None before GL window starts)
    pm: "PropertyManager | None" = None,
) -> "PropertyManager":
    """Construct (or extend) a PropertyManager for gui_merged.

    If *pm* is None a new one is created; otherwise properties are appended
    to the existing instance.  Element sections (nn_graph / lasers / circles)
    are skipped when the corresponding argument is None — call again later
    (passing the same *pm*) once the GL elements exist.

    All properties are bound to live objects so get/set proxy directly.
    """
    from post.feedback import BLEND_MODES, SMEAR_PATTERNS, PRESETS
    from elements.base import ELEMENT_TYPES

    if pm is None:
        pm = PropertyManager()

    def R(prop, obj=None, attr=None):
        """Register, skipping keys already present (idempotent re-calls)."""
        if prop.key in pm._defs:
            # If a live binding is now available but wasn't before, attach it.
            if obj is not None and attr is not None and prop.key not in pm._bindings:
                pm._bindings[prop.key] = (obj, attr)
                pm._values.pop(prop.key, None)
            return pm
        return pm.register(prop, obj, attr)

    # ── per-kind visibility / active (one permanent key per draw kind, drivable
    #    from Channels regardless of whether an instance is currently live) ──────
    for kind in ELEMENT_TYPES:
        R(PropDef(f"{kind}.visible", kind, "visible", "Visible",
                  bool, True, widget_hint="check",
                  description=f"Show/hide {kind} (drivable via Link Manager "
                              f"expressions)"))
        R(PropDef(f"{kind}.active", kind, "active", "Active",
                  bool, True, widget_hint="check",
                  description=f"Enable/disable spawning for {kind}; when False "
                              f"stops generating new elements and lets existing "
                              f"ones die off gracefully"))

    # ── feedback (FeedbackParams) ─────────────────────────────────────────────
    _F = "feedback"

    R(PropDef(f"{_F}.zoom",     _F, "zoom",     "Zoom",
              float, 1.0,    0.950, 1.050, 0.001,
              description="Zoom factor each feedback step (1.0 = none, >1 = inward, <1 = outward)"),
      params, "zoom")

    R(PropDef(f"{_F}.rotation", _F, "rotation", "Rotation",
              float, 0.0008, 0.0,   0.01,  0.0001,
              description="Radians of rotation per feedback step"),
      params, "rotation")

    R(PropDef(f"{_F}.decay",            _F, "decay",            "Decay",
              float, 0.993,  0.80,  0.999, 0.010,
              description="Per-step brightness multiplier (lower = shorter trails)"),
      params, "decay")

    R(PropDef(f"{_F}.ripple_strength",  _F, "ripple_strength",  "Ripple Strength",
              float, 0.0,    0.0,   50.0,  0.5,
              description="Max pixel displacement of the radial ripple"),
      params, "ripple_strength")

    R(PropDef(f"{_F}.ripple_freq",      _F, "ripple_freq",      "Ripple Frequency",
              float, 10.0,   1.0,   30.0,  0.5,
              description="Spatial frequency of ripple rings per radius unit"),
      params, "ripple_freq")

    R(PropDef(f"{_F}.hue_shift",        _F, "hue_shift",        "Hue Shift",
              float, 0.005,  0.0,   0.05,  0.002,
              description="Hue rotation applied to each feedback sample (radians/step)"),
      params, "hue_shift")

    R(PropDef(f"{_F}.chroma_offset",    _F, "chroma_offset",    "Chromatic Aberration",
              float, 0.005,  0.0,   0.05,  0.002,
              description="Radial R/B channel offset as fraction of width"),
      params, "chroma_offset")

    R(PropDef(f"{_F}.sat_boost",        _F, "sat_boost",        "Saturation Boost",
              float, 1.12,   1.0,   2.0,   0.05,
              description="Saturation multiplier on feedback sample (1.0 = flat)"),
      params, "sat_boost")

    R(PropDef(f"{_F}.smear_strength",   _F, "smear_strength",   "Smear Strength",
              float, 0.0,    0.0,   0.10,  0.005,
              description="UV offset per step along the smear field direction"),
      params, "smear_strength")

    R(PropDef(f"{_F}.fisheye_strength", _F, "fisheye_strength", "Fisheye",
              float, 0.0,    -2.0,  2.0,   0.05,
              description=">0 barrel (wide), <0 pincushion (telephoto), 0 = none"),
      params, "fisheye_strength")

    # ── scene (SceneControls) ─────────────────────────────────────────────────
    # Per-element visibility is no longer here -- it lives on each
    # DrawingElement instance (.visible) and is managed via the Elements
    # panel / MergedGUI.add_element()/remove_element(), since the element
    # list is dynamic rather than a fixed cloud/nn/circles/lasers set.
    _S = "scene"

    R(PropDef(f"{_S}.scene_alpha",   _S, "scene_alpha",   "Scene Alpha",
              float, 0.18,   0.02,  1.0,   0.05,
              description="How strongly the current frame bleeds into feedback"),
      controls, "scene_alpha")

    R(PropDef(f"{_S}.blend_mode",    _S, "blend_mode",    "Blend Mode",
              str, "lerp",  choices=BLEND_MODES, widget_hint="combo",
              description="Compositing operator for scene→feedback injection"),
      controls, "blend_mode")

    R(PropDef(f"{_S}.smear_pattern", _S, "smear_pattern", "Smear Pattern",
              str, "outward", choices=SMEAR_PATTERNS, widget_hint="combo",
              description="Named directional smear vector field"),
      controls, "smear_pattern")

    R(PropDef(f"{_S}.active_effect", _S, "active_effect", "Active Effect",
              str, "feedback", choices=["feedback", "pass_through", "glitch", "bokeh"],
              widget_hint="combo",
              description="Which post-effect pipeline is active"),
      controls, "active_effect")

    # ── camera (OrbitCamera) ──────────────────────────────────────────────────
    if camera is not None:
        _CAM = "camera"

        R(PropDef(f"{_CAM}.mode",          _CAM, "mode",          "Mode",
                  str, "auto_orbit", choices=["auto_orbit", "static"],
                  widget_hint="combo",
                  description="auto_orbit: Lissajous-driven path; "
                              "static: hold yaw/pitch/dist at set values"),
          camera, "mode")

        R(PropDef(f"{_CAM}.yaw",           _CAM, "yaw",           "Yaw",
                  float, 35.0,  -180.0, 180.0, 1.0,
                  description="Horizontal orbit angle in degrees — authoritative in "
                              "static mode, readable (live) in auto_orbit"),
          camera, "yaw")

        R(PropDef(f"{_CAM}.pitch",         _CAM, "pitch",         "Pitch",
                  float, -25.0,  -89.0,  89.0, 1.0,
                  description="Vertical tilt in degrees — authoritative in static "
                              "mode, readable (live) in auto_orbit"),
          camera, "pitch")

        R(PropDef(f"{_CAM}.distance",      _CAM, "dist",          "Distance",
                  float, 1.0,    1.0,   12.0,  0.1,
                  description="Camera distance from the origin"),
          camera, "dist")

        R(PropDef(f"{_CAM}.orbit_speed",   _CAM, "orbit_speed",   "Orbit Speed",
                  float, 0.22,  -2.0,   2.0,   0.01,
                  description="Left-right angular speed in rad/s (auto_orbit mode; "
                              "drivable via Link Manager expressions)"),
          camera, "orbit_speed")

        R(PropDef(f"{_CAM}.orbit_a",       _CAM, "orbit_a",       "Orbit Radius",
                  float, 1.5,   0.5,   12.0,   0.1,
                  description="XZ semi-axis — controls left-right distance amplitude "
                              "of the orbit path"),
          camera, "orbit_a")

        R(PropDef(f"{_CAM}.orbit_b",       _CAM, "orbit_b",       "Up-Down Amplitude",
                  float, 0.6,   0.0,    5.0,   0.05,
                  description="Y semi-axis — vertical up-down amplitude of the orbit"),
          camera, "orbit_b")

        R(PropDef(f"{_CAM}.orbit_phi",     _CAM, "orbit_phi",     "Vertical Freq",
                  float, 0.809, 0.1,    5.0,   0.05,
                  description="Up-down frequency multiplier relative to orbit_speed "
                              "(default ≈ 0.809, golden-ratio drift)"),
          camera, "orbit_phi")

        R(PropDef(f"{_CAM}.lerp_duration", _CAM, "lerp_duration", "Lerp Duration",
                  float, 1.0,   0.1,   10.0,   0.1,
                  description="Default duration (s) for camera.lerp_to() transitions"),
          camera, "lerp_duration")

    if cloud is not None:

        _N = "cloud"

        R(PropDef(f"{_N}.ball_size", _N, "ball_size", "Size of the ball",
                  float, 0.1,   0.01,  1.0,  0.05,
                  description="World space radius"),
          cloud, "ball_size")

    # ── nn_graph (NNGraph) ────────────────────────────────────────────────────
    if nn_graph is not None:
        _N = "nn_graph"

        R(PropDef(f"{_N}.amplitude", _N, "amplitude", "Drift Amplitude",
                  float, 0.08,  0.0,  0.5,  0.01,
                  description="Sinusoidal drift radius for each node around its base pos"),
          nn_graph, "amplitude")

    # ── lasers (LaserRibbons) ─────────────────────────────────────────────────
    if lasers is not None:
        _L = "lasers"

        R(PropDef(f"{_L}.ribbon_speed",   _L, "ribbon_speed",   "Ribbon Speed",
                  float, 7.0,   0.5,  20.0,  0.5,
                  description="World-space units per second each ribbon travels"),
          lasers, "ribbon_speed")

        R(PropDef(f"{_L}.ribbon_length",  _L, "ribbon_length",  "Ribbon Length",
                  float, 0.30,  0.05, 2.0,   0.05,
                  description="World-space tail length behind the ribbon head"),
          lasers, "ribbon_length")

        R(PropDef(f"{_L}.half_width",     _L, "half_width",     "Ribbon Width",
                  float, 0.018, 0.002, 0.10, 0.002,
                  description="Half-width of the billboard quad in world space"),
          lasers, "half_width")

        R(PropDef(f"{_L}.spawn_interval", _L, "spawn_interval", "Spawn Interval",
                  float, 0.045, 0.01, 0.5,   0.005,
                  description="Seconds between successive ribbon spawns"),
          lasers, "spawn_interval")

        R(PropDef(f"{_L}.spawn_spread",   _L, "spawn_spread",   "Spawn Spread",
                  float, 0.5,   0.0,  3.0,   0.05,
                  description="Lateral jitter radius at spawn point"),
          lasers, "spawn_spread")

        R(PropDef(f"{_L}.max_dist",       _L, "max_dist",       "Max Travel Distance",
                  float, 20.0,  1.0,  50.0,  1.0,
                  description="Travel distance at which a ribbon is killed"),
          lasers, "max_dist")

    # ── circles (CircleAxisDrawing) ───────────────────────────────────────────
    if circles is not None:
        _C = "circles"

        R(PropDef(f"{_C}.n_circles",         _C, "n_circles",         "Circle Count",
                  int,   24,    4,    64,    1,
                  description="Max simultaneous spawned circle+blade ribbons (requires regen)"),
          circles, "n_circles")

        R(PropDef(f"{_C}.n_trav_lines",      _C, "n_trav_lines",      "Traversal Lines",
                  int,   35,    0,    100,   1,
                  description="Number of diagonal traversal line ribbons (requires regen)"),
          circles, "n_trav_lines")

        R(PropDef(f"{_C}.n_blades",          _C, "n_blades",          "Blade Count",
                  int,   64,    8,    128,   8,
                  description="Turbine-blade quads per circle (requires regen)"),
          circles, "n_blades")

        R(PropDef(f"{_C}.blade_spin_speed",  _C, "blade_spin_speed",  "Blade Spin Speed",
                  float, 0.2,  -4.0,  4.0,   0.05,
                  description="Turbine blade rotation speed in radians/second"),
          circles, "blade_spin_speed")

        R(PropDef(f"{_C}.blade_size_factor", _C, "blade_size_factor", "Blade Size",
                  float, 0.125, 0.02, 0.4,   0.005,
                  description="Blade side length as a fraction of circle radius"),
          circles, "blade_size_factor")

        R(PropDef(f"{_C}.amplitude",         _C, "amplitude",         "Drift Amplitude",
                  float, 1.0,   0.0,  100.0,   0.05,
                  description="Global multiplier on each circle's sinusoidal drift amplitude (drivable via Link Manager expressions)"),
          circles, "amplitude")

    return pm
