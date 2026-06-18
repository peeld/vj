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
    from property_manager import PropertyManager
    from elements.base import DrawingElement, ELEMENT_TYPES

    pm = PropertyManager()
    for kind in ELEMENT_TYPES:
        pm.pre_register_node_class(DrawingElement, kind)
    pm.register_node(_params)       # FeedbackParams (Node)
    pm.register_node(_controls)     # SceneControls  (Node)
    pm.register_node(camera)        # OrbitCamera    (Node, after GL init)

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

    def register_node(self, node: Any) -> Any:
        """Register all Prop declarations on a Node instance.

        For each Prop found via node_props(), builds a PropDef and:
          - key not yet registered: registers + binds to the instance.
          - key registered but unbound (e.g. pre_register_node_class was
            called first): latches the stored value onto the instance, then
            binds (same logic as the R() helper in build_default_manager).
          - key registered and already bound: no-op.

        Returns node so the call can be used as a one-liner:
            params = pm.register_node(FeedbackParams())
        """
        from prop import Prop as _Prop
        section = node._node_section
        if not section:
            raise ValueError(
                f"{type(node).__name__} has no _node_section — "
                f"declare it with section= on the class definition."
            )
        for class_attr, prop in type(node).node_props().items():
            instance_attr = prop.attr if prop.attr is not None else class_attr
            key = f"{section}.{instance_attr}"
            if key not in self._defs:
                pd = PropDef(
                    key=key, section=section, name=instance_attr,
                    label=prop.label, type=prop.type, default=prop.default,
                    min_val=prop.min_val, max_val=prop.max_val, step=prop.step,
                    choices=prop.choices, widget_hint=prop.widget_hint,
                    description=prop.description,
                )
                self.register(pd, node, instance_attr)
            elif key not in self._bindings:
                self.bind(key, node, instance_attr)
            # else: already registered and bound → no-op
        return node

    def pre_register_node_class(self, cls: Any, section: str) -> None:
        """Register Prop declarations from a class without a live instance.

        Keys are registered with their default values but no binding.  When
        a live instance later calls register_node(), the key is already
        present so register_node() upgrades to a live binding instead of
        re-registering — preserving any value that was set in the interim.

        Used at boot time to make element-kind keys (e.g. "cloud.visible")
        addressable by Link Manager expressions before any element is live.
        """
        for class_attr, prop in cls.node_props().items():
            instance_attr = prop.attr if prop.attr is not None else class_attr
            key = f"{section}.{instance_attr}"
            if key not in self._defs:
                pd = PropDef(
                    key=key, section=section, name=instance_attr,
                    label=prop.label, type=prop.type, default=prop.default,
                    min_val=prop.min_val, max_val=prop.max_val, step=prop.step,
                    choices=prop.choices, widget_hint=prop.widget_hint,
                    description=prop.description,
                )
                self.register(pd)  # no instance → stored in _values

    def unregister_node(self, node: Any) -> None:
        """Unbind all keys for node's section, preserving their values.

        Keys remain registered (so Link Manager expressions keep resolving),
        but no longer proxy to any object.  Calling register_node() with a
        new instance of the same kind re-binds without re-registering, and
        the preserved values are latched onto the new instance first.
        """
        section = node._node_section
        for class_attr, prop in type(node).node_props().items():
            instance_attr = prop.attr if prop.attr is not None else class_attr
            key = f"{section}.{instance_attr}"
            self.unbind(key)

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
