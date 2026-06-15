"""
property_manager.py — Centralised property registry for gui_merged.

Every adjustable parameter across all scene modules is declared here with:
  • a unique dotted key     ("section.name", e.g. "feedback.decay")
  • its type / range / step / choices / default value
  • an optional live-object binding so get/set proxy to the real dataclass

The manager additionally owns:
  • Preset snapshots  — save / load / list  (JSON-persisted)
  • Keyboard mappings — key name → increment / decrement / toggle / cycle action
  • MIDI mappings     — CC number → scaled property write
                        note number → toggle / callback
  • Audio mappings    — AudioMetrics attribute → scaled property write

Typical usage (in gui_merged.py)
---------------------------------
    from property_manager import PropertyManager, build_default_manager

    pm = build_default_manager(_params, _controls, nn_graph, laser, circles)

    # anywhere — live read/write
    pm.get("feedback.decay")          # -> float
    pm.set("feedback.decay", 0.995)

    # key handler
    pm.apply_key_action("Z")          # uses registered KeyBinding

    # preset round-trip
    pm.save_preset("my_look")
    pm.load_preset("my_look")
    pm.save_json("presets.json")      # persists all presets + mappings
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable


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
#  Mapping types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KeyBinding:
    """Maps a keyboard key name to a property action.

    action values:
      "increment"  — add step (or 1 for int, not valid for str)
      "decrement"  — subtract step
      "toggle"     — flip bool
      "cycle"      — advance through choices list
      "cycle_back" — retreat through choices list
    """
    key_name  : str          # e.g. "Z", "TAB", "NUMBER_1"
    prop_key  : str          # e.g. "feedback.scene_alpha"
    action    : str          # increment / decrement / toggle / cycle / cycle_back
    amount    : float | None = None   # override step if provided


@dataclass
class MidiCCBinding:
    """Maps a MIDI CC number (0–127) to a property.

    mode
    ----
    "range"  (default for numeric props)
        CC 0–127 is linearly scaled to [min_val, max_val] and written to the
        property.  min_val / max_val must be provided.

    "enum"   (default when prop has choices)
        The 0–127 range is divided into len(choices) equal segments; the CC
        value selects the corresponding choice.  min_val / max_val are ignored.
        Setting mode="enum" explicitly forces this behaviour even if min/max
        are supplied.
    """
    cc        : int
    prop_key  : str
    min_val   : float | None = None
    max_val   : float | None = None
    channel   : int  = 0       # 0 = any channel
    mode      : str  = "auto"  # "auto" | "range" | "enum"


@dataclass
class MidiNoteBinding:
    """Maps a MIDI note to a property action or an arbitrary callback.

    Actions (checked in order):
      • callback      — called with velocity; takes priority over everything else
      • target_value  — on note-on (velocity > 0) sets the property to this
                        exact value; perfect for assigning one note per enum choice
      • prop_key only — on note-on toggles a bool property (original behaviour)

    release_value
        If set, written to the property on note-off (velocity == 0).
        Use this for momentary / "hold" behaviour on enum properties:
        e.g. hold note → "additive", release → "lerp".
        When target_value == release_value (or release_value is None) the
        note-off is a no-op, which is the correct behaviour when the note
        itself maps to the default value (e.g. "lerp").
    """
    note          : int
    prop_key      : str | None = None
    callback      : Callable[[int], None] | None = None
    channel       : int = 0
    target_value  : Any = None   # if set, writes this value instead of toggling
    release_value : Any = None   # if set, writes this value on note-off


@dataclass
class AudioBinding:
    """Maps an AudioMetrics attribute to a property (linear range scaling)."""
    metric_attr : str          # e.g. "energy", "bass", "mid", "treble"
    prop_key    : str
    min_val     : float
    max_val     : float


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
    """

    def __init__(self) -> None:
        # prop_key -> PropDef
        self._defs: dict[str, PropDef] = {}

        # prop_key -> (obj, attr_name) live binding
        self._bindings: dict[str, tuple[Any, str]] = {}

        # internal storage for unbound props
        self._values: dict[str, Any] = {}

        # mappings
        self._key_bindings  : dict[str, KeyBinding]      = {}   # key_name -> binding
        self._midi_cc       : dict[tuple[int,int], MidiCCBinding]   = {}  # (ch, cc)
        self._midi_note     : dict[tuple[int,int], MidiNoteBinding] = {}  # (ch, note)
        self._audio_bindings: list[AudioBinding] = []

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

    def describe(self) -> None:
        """Print a human-readable summary of all registered properties."""
        for section in self.sections():
            print(f"\n[{section}]")
            for d in self.props_in(section):
                live = self.get(d.key)
                rng = ""
                if d.min_val is not None or d.max_val is not None:
                    rng = f"  [{d.min_val} … {d.max_val}]"
                if d.choices:
                    rng = f"  {d.choices}"
                print(f"  {d.name:<24} = {live!r:<12}  (default={d.default!r}){rng}")

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

    # ── Keyboard mappings ─────────────────────────────────────────────────────

    def bind_key(self, binding: KeyBinding) -> None:
        """Register a keyboard binding (overwrites any previous binding for that key)."""
        self._key_bindings[binding.key_name] = binding

    def apply_key_action(self, key_name: str) -> bool:
        """Apply the action for a registered key.  Returns True if consumed."""
        b = self._key_bindings.get(key_name)
        if b is None:
            return False
        prop = self._defs[b.prop_key]
        cur  = self.get(b.prop_key)

        if b.action == "toggle":
            self.set(b.prop_key, not cur)
        elif b.action == "cycle":
            self.set(b.prop_key, prop.next_choice(cur))
        elif b.action == "cycle_back":
            self.set(b.prop_key, prop.prev_choice(cur))
        elif b.action in ("increment", "decrement"):
            delta = b.amount if b.amount is not None else (prop.step or 1)
            if b.action == "decrement":
                delta = -delta
            self.set(b.prop_key, cur + delta)
        else:
            return False

        live = self.get(b.prop_key)
        print(f"[pm] {b.prop_key} {b.action} → {live!r}")
        return True

    def key_bindings(self) -> list[KeyBinding]:
        return list(self._key_bindings.values())

    # ── MIDI mappings ─────────────────────────────────────────────────────────

    def bind_midi_cc(self, binding: MidiCCBinding) -> None:
        """Register a MIDI CC → property binding."""
        self._midi_cc[(binding.channel, binding.cc)] = binding

    def bind_midi_note(self, binding: MidiNoteBinding) -> None:
        """Register a MIDI note → toggle / callback binding."""
        self._midi_note[(binding.channel, binding.note)] = binding

    def apply_midi_cc(self, channel: int, cc: int, value: int) -> bool:
        """Apply a CC event (value 0–127).  Returns True if consumed.

        For numeric properties:
            CC linearly maps to [min_val, max_val].

        For enum/str properties (or when mode="enum"):
            CC range is divided into len(choices) equal buckets; the bucket
            index selects the corresponding choice.
            e.g. with 7 choices and CC=64 → bucket 3 → choices[3]
        """
        b = self._midi_cc.get((channel, cc)) or self._midi_cc.get((0, cc))
        if b is None:
            return False

        prop = self._defs.get(b.prop_key)
        if prop is None:
            return False

        # Determine effective mode
        mode = b.mode
        if mode == "auto":
            mode = "enum" if (prop.choices is not None and len(prop.choices) > 0) else "range"

        if mode == "enum":
            choices = prop.choices
            if not choices:
                return False
            # Divide 0–127 into len(choices) equal slots; clamp last bucket edge
            idx = min(int(value / 128.0 * len(choices)), len(choices) - 1)
            self.set(b.prop_key, choices[idx])
            print(f"[pm] CC{cc}={value} → {b.prop_key} = {choices[idx]!r}  "
                  f"(slot {idx}/{len(choices)})")
        else:
            if b.min_val is None or b.max_val is None:
                raise ValueError(
                    f"MidiCCBinding for '{b.prop_key}' in 'range' mode requires "
                    f"min_val and max_val."
                )
            t      = value / 127.0
            scaled = b.min_val + t * (b.max_val - b.min_val)
            self.set(b.prop_key, scaled)
            print(f"[pm] CC{cc}={value} → {b.prop_key} = {scaled:.4g}")

        return True

    def apply_midi_note(self, channel: int, note: int, velocity: int) -> bool:
        """Apply a note-on/off event.  Returns True if consumed.

        Resolution order:
          1. callback(velocity)              — always fires if set
          2. target_value write on note-on   — sets exact value (enum or any type)
          3. bool toggle on note-on          — original behaviour
        """
        b = self._midi_note.get((channel, note)) or self._midi_note.get((0, note))
        if b is None:
            return False

        if b.callback is not None:
            b.callback(velocity)
        elif b.prop_key is not None:
            if velocity > 0:
                if b.target_value is not None:
                    self.set(b.prop_key, b.target_value)
                    print(f"[pm] note {note} → {b.prop_key} = {b.target_value!r}")
                else:
                    cur = self.get(b.prop_key)
                    self.set(b.prop_key, not cur)
                    print(f"[pm] note {note} → {b.prop_key} toggled → {self.get(b.prop_key)!r}")
            else:
                # note-off: revert to release_value if one is set
                if b.release_value is not None:
                    self.set(b.prop_key, b.release_value)
                    print(f"[pm] note {note} released → {b.prop_key} = {b.release_value!r}")
        return True

    def bind_enum_to_cc(self, prop_key: str, cc: int, channel: int = 0) -> None:
        """Convenience: bind a CC knob/slider to sweep through an enum property.

        Sweeping CC 0 → 127 steps through choices[0] → choices[-1].
        The property must have a choices list.

        Example::
            pm.bind_enum_to_cc("scene.blend_mode", cc=14)
        """
        prop = self._defs.get(prop_key)
        if prop is None:
            raise KeyError(f"Unknown property: '{prop_key}'")
        if not prop.choices:
            raise ValueError(f"Property '{prop_key}' has no choices — cannot bind as enum.")
        self.bind_midi_cc(MidiCCBinding(
            cc=cc, prop_key=prop_key,
            channel=channel, mode="enum",
        ))
        print(f"[pm] CC{cc} → {prop_key} (enum: {prop.choices})")

    def bind_enum_to_notes(
        self,
        prop_key      : str,
        start_note    : int,
        channel       : int = 0,
        release_value : Any = None,
    ) -> dict[str, int]:
        """Convenience: assign one MIDI note per enum choice.

        Notes are assigned consecutively starting from start_note:
            start_note + 0  → choices[0]
            start_note + 1  → choices[1]
            …

        Returns a dict mapping choice_value → note for reference.

        release_value
            Value written to the property on note-off (key release).
            Enables momentary / "hold" behaviour: hold note → choice,
            release → release_value.  When a note's own target_value
            equals release_value the note-off is a no-op (correct for
            the note that maps to the default/rest value itself).

            For blend_mode the default rest mode is "lerp", so pass
            release_value="lerp" to get hold-to-activate behaviour on
            all other blend modes while leaving the "lerp" note as a
            plain latch (press sets lerp, release does nothing).

        Example::
            mapping = pm.bind_enum_to_notes(
                "scene.blend_mode", start_note=36, release_value="lerp"
            )
            # note 36 → "lerp" (latch), 37 → "additive" (hold), …
        """
        prop = self._defs.get(prop_key)
        if prop is None:
            raise KeyError(f"Unknown property: '{prop_key}'")
        if not prop.choices:
            raise ValueError(f"Property '{prop_key}' has no choices.")

        mapping: dict[str, int] = {}
        for i, choice in enumerate(prop.choices):
            note = start_note + i
            # A note whose target equals the release_value acts as a plain
            # latch (release_value=None means note-off is a no-op for it).
            rv = None if (release_value is None or choice == release_value) \
                 else release_value
            self.bind_midi_note(MidiNoteBinding(
                note=note,
                prop_key=prop_key,
                channel=channel,
                target_value=choice,
                release_value=rv,
            ))
            mapping[choice] = note

        lines = "  ".join(f"{n}={c!r}" for c, n in mapping.items())
        print(f"[pm] notes → {prop_key}:  {lines}")
        return mapping

    def remove_midi_cc(self, cc: int, channel: int = 0) -> bool:
        """Remove a CC binding.  Returns True if it existed."""
        key = (channel, cc)
        if key in self._midi_cc:
            del self._midi_cc[key]
            return True
        return False

    def remove_midi_note(self, note: int, channel: int = 0) -> bool:
        """Remove a note binding.  Returns True if it existed."""
        key = (channel, note)
        if key in self._midi_note:
            del self._midi_note[key]
            return True
        return False

    def midi_cc_bindings(self) -> list[MidiCCBinding]:
        return list(self._midi_cc.values())

    def midi_note_bindings(self) -> list[MidiNoteBinding]:
        return list(self._midi_note.values())

    # ── Audio mappings ────────────────────────────────────────────────────────

    def bind_audio(self, binding: AudioBinding) -> None:
        """Register an AudioMetrics attribute → property mapping."""
        self._audio_bindings.append(binding)

    def apply_audio(self, metrics: Any) -> None:
        """Apply all audio bindings given a live AudioMetrics object."""
        for b in self._audio_bindings:
            raw = getattr(metrics, b.metric_attr, None)
            if raw is None:
                continue
            t      = max(0.0, min(1.0, float(raw)))
            scaled = b.min_val + t * (b.max_val - b.min_val)
            self.set(b.prop_key, scaled)

    def audio_bindings(self) -> list[AudioBinding]:
        return list(self._audio_bindings)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def _presets_to_dict(self) -> dict:
        return {name: dict(snap) for name, snap in self._presets.items()}

    def _key_bindings_to_list(self) -> list:
        return [
            {
                "key_name": b.key_name,
                "prop_key": b.prop_key,
                "action"  : b.action,
                "amount"  : b.amount,
            }
            for b in self._key_bindings.values()
        ]

    def _midi_cc_to_list(self) -> list:
        return [
            {
                "cc"      : b.cc,
                "prop_key": b.prop_key,
                "min_val" : b.min_val,
                "max_val" : b.max_val,
                "channel" : b.channel,
                "mode"    : b.mode,
            }
            for b in self._midi_cc.values()
        ]

    def _midi_note_to_list(self) -> list:
        return [
            {
                "note"        : b.note,
                "prop_key"    : b.prop_key,
                "channel"     : b.channel,
                "target_value": b.target_value,
            }
            for b in self._midi_note.values()
            if b.prop_key is not None  # skip runtime-only callbacks
        ]

    def _audio_to_list(self) -> list:
        return [
            {
                "metric_attr": b.metric_attr,
                "prop_key"   : b.prop_key,
                "min_val"    : b.min_val,
                "max_val"    : b.max_val,
            }
            for b in self._audio_bindings
        ]

    def to_dict(self) -> dict:
        """Serialise presets and mappings to a plain dict (JSON-safe)."""
        return {
            "presets"      : self._presets_to_dict(),
            "key_bindings" : self._key_bindings_to_list(),
            "midi_cc"      : self._midi_cc_to_list(),
            "midi_notes"   : self._midi_note_to_list(),
            "audio"        : self._audio_to_list(),
        }

    def from_dict(self, data: dict) -> None:
        """Restore presets and serialisable mappings from a dict."""
        for name, snap in data.get("presets", {}).items():
            self._presets[name] = snap

        for kb in data.get("key_bindings", []):
            self.bind_key(KeyBinding(
                key_name=kb["key_name"],
                prop_key=kb["prop_key"],
                action  =kb["action"],
                amount  =kb.get("amount"),
            ))

        for cb in data.get("midi_cc", []):
            self.bind_midi_cc(MidiCCBinding(
                cc      =cb["cc"],
                prop_key=cb["prop_key"],
                min_val =cb.get("min_val"),
                max_val =cb.get("max_val"),
                channel =cb.get("channel", 0),
                mode    =cb.get("mode", "auto"),
            ))

        for nb in data.get("midi_notes", []):
            self.bind_midi_note(MidiNoteBinding(
                note        =nb["note"],
                prop_key    =nb.get("prop_key"),
                channel     =nb.get("channel", 0),
                target_value=nb.get("target_value"),
            ))

        for ab in data.get("audio", []):
            self.bind_audio(AudioBinding(
                metric_attr=ab["metric_attr"],
                prop_key   =ab["prop_key"],
                min_val    =ab["min_val"],
                max_val    =ab["max_val"],
            ))

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
    nn_graph,            # NNGraph  (may be None before GL window starts)
    lasers,              # LaserRibbons  (may be None)
    circles,             # CircleAxisDrawing  (may be None)
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

    # ── feedback (FeedbackParams) ─────────────────────────────────────────────
    _F = "feedback"

    R(PropDef(f"{_F}.base_zoom",        _F, "base_zoom",        "Base Zoom",
              float, 1.002,  1.000, 1.050, 0.001,
              description="Zoom factor each feedback step (1.0 = none, >1 = inward)"),
      params, "base_zoom")

    R(PropDef(f"{_F}.zoom_sensitivity", _F, "zoom_sensitivity", "Zoom/Bass Sensitivity",
              float, 0.0,    0.0,   0.2,   0.005,
              description="Extra zoom added per unit of bass energy"),
      params, "zoom_sensitivity")

    R(PropDef(f"{_F}.base_rot",         _F, "base_rot",         "Base Rotation",
              float, 0.0008, 0.0,   0.01,  0.0001,
              description="Radians of rotation per feedback step"),
      params, "base_rot")

    R(PropDef(f"{_F}.rot_sensitivity",  _F, "rot_sensitivity",  "Rot/Mid Sensitivity",
              float, 0.0,    0.0,   0.1,   0.005,
              description="Extra rotation per unit of mid energy"),
      params, "rot_sensitivity")

    R(PropDef(f"{_F}.decay",            _F, "decay",            "Decay",
              float, 0.993,  0.80,  0.999, 0.010,
              description="Per-step brightness multiplier (lower = shorter trails)"),
      params, "decay")

    R(PropDef(f"{_F}.ripple_strength",  _F, "ripple_strength",  "Ripple Strength",
              float, 0.0,    0.0,   50.0,  0.5,
              description="Max pixel displacement of treble-driven radial ripple"),
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
    _S = "scene"

    R(PropDef(f"{_S}.show_cloud",    _S, "show_cloud",    "Cloud/Ball",
              bool, True,    widget_hint="check",
              description="Toggle cloud + ball element"),
      controls, "show_cloud")

    R(PropDef(f"{_S}.show_nn",       _S, "show_nn",       "NN Graph",
              bool, False,    widget_hint="check",
              description="Toggle nearest-neighbour graph element"),
      controls, "show_nn")

    R(PropDef(f"{_S}.show_circles",  _S, "show_circles",  "Circle Axis",
              bool, False,    widget_hint="check",
              description="Toggle circle-axis ribbons element"),
      controls, "show_circles")

    R(PropDef(f"{_S}.show_lasers",   _S, "show_lasers",   "Laser Ribbons",
              bool, False,    widget_hint="check",
              description="Toggle laser ribbon element"),
      controls, "show_lasers")

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

    # ── nn_graph (NNGraph) ────────────────────────────────────────────────────
    if nn_graph is not None:
        _N = "nn_graph"

        R(PropDef(f"{_N}.amplitude", _N, "amplitude", "Drift Amplitude",
                  float, 0.08,  0.0,  0.5,  0.01,
                  description="Sinusoidal drift radius for each node around its base pos"),
          nn_graph, "amplitude")

        R(PropDef(f"{_N}.fade_dist", _N, "fade_dist", "Edge Fade Distance",
                  float, 0.5,   0.1,  2.0,  0.05,
                  description="World-space distance at which edge alpha drops to zero"),
          nn_graph, "fade_dist")

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
                  description="Number of animated circle ribbons (requires regen)"),
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

    _register_default_key_bindings(pm)

    return pm


def _register_default_key_bindings(pm: PropertyManager) -> None:
    """Register the keyboard bindings hard-coded in gui_merged.on_key_event.

    Idempotent: if a key is already bound it is left unchanged.
    """
    def _bind(b: KeyBinding) -> None:
        if b.key_name not in pm._key_bindings:
            pm.bind_key(b)
    B = _bind  # reuse name so call sites stay the same

    def _B(*args, **kw) -> KeyBinding:
        return KeyBinding(*args, **kw)

    # Scene visibility toggles
    B(_B("NUMBER_1", "scene.show_cloud",   "toggle"))
    B(_B("NUMBER_2", "scene.show_nn",      "toggle"))
    B(_B("NUMBER_3", "scene.show_circles", "toggle"))
    B(_B("NUMBER_4", "scene.show_lasers",  "toggle"))

    # Active effect / blend / smear cycle
    B(_B("TAB", "scene.active_effect", "cycle"))
    B(_B("G",   "scene.blend_mode",    "cycle"))
    B(_B("M",   "scene.smear_pattern", "cycle"))

    # scene_alpha  Z/X
    B(_B("Z", "scene.scene_alpha", "decrement"))
    B(_B("X", "scene.scene_alpha", "increment"))

    # decay  D/F
    B(_B("D", "feedback.decay", "decrement"))
    B(_B("F", "feedback.decay", "increment"))

    # base_rot  Q/W
    B(_B("Q", "feedback.base_rot", "decrement"))
    B(_B("W", "feedback.base_rot", "increment"))

    # base_zoom  A/S
    B(_B("A", "feedback.base_zoom", "decrement"))
    B(_B("S", "feedback.base_zoom", "increment"))

    # hue_shift  H/J
    B(_B("H", "feedback.hue_shift", "decrement"))
    B(_B("J", "feedback.hue_shift", "increment"))

    # chroma_offset  C/V
    B(_B("C", "feedback.chroma_offset", "decrement"))
    B(_B("V", "feedback.chroma_offset", "increment"))

    # sat_boost  B/N
    B(_B("B", "feedback.sat_boost", "decrement"))
    B(_B("N", "feedback.sat_boost", "increment"))

    # smear_strength  K/L
    B(_B("K", "feedback.smear_strength", "decrement"))
    B(_B("L", "feedback.smear_strength", "increment"))

    # fisheye_strength  I/U
    B(_B("I", "feedback.fisheye_strength", "decrement"))
    B(_B("U", "feedback.fisheye_strength", "increment"))
