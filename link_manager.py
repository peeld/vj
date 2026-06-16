"""
link_manager.py — Signal routing layer for gui_merged.

Connects sources (audio, MIDI, keyboard, clock, envelopes, LFOs) to
PropertyManager sinks via evaluated Python expression strings.

Phase 0: stubs + dataclasses.
Phase 1: SourceRegistry + math eval helpers.
Phase 4: EventBus, threshold detectors, EventLink evaluation.
"""

from __future__ import annotations

import math as _math
import queue
import re
import threading
import time
import types
from dataclasses import dataclass, field
from math import exp as _exp
from typing import Any, Callable


# ─────────────────────────────────────────────────────────────────────────────
#  Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SignalLink:
    """Continuous source → sink mapping via a Python expression string.

    The expression is evaluated per-frame with a restricted namespace containing
    all source values and math helpers.  The result is written to the PM sink.

    Example::
        SignalLink(
            sink_key   = "feedback.decay",
            expression = "lerp(0.98, 0.999, audio.bass)",
        )
    """
    sink_key   : str
    expression : str
    enabled    : bool = True

    def to_dict(self) -> dict:
        return {
            "sink_key"  : self.sink_key,
            "expression": self.expression,
            "enabled"   : self.enabled,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SignalLink":
        return cls(
            sink_key   = d["sink_key"],
            expression = d["expression"],
            enabled    = d.get("enabled", True),
        )


@dataclass
class EnvelopeDef:
    """ADSR envelope triggered by an event; output available as env.<name>.

    Example::
        EnvelopeDef(
            name    = "kick",
            trigger = "audio.onset",
            attack  = 0.01,
            decay   = 0.1,
            sustain = 0.0,
            release = 0.2,
            peak    = 1.0,
        )
    """
    name       : str
    trigger    : str            # e.g. "audio.onset" or "midi.note36.on"
    attack     : float = 0.01   # seconds
    decay      : float = 0.1    # seconds
    sustain    : float = 0.0    # 0-1 level while gate held
    release    : float = 0.2    # seconds
    peak       : float = 1.0    # 0-1 peak at end of attack phase
    gate_off   : str | None = None  # event that releases the gate (sustain phase)

    def to_dict(self) -> dict:
        return {
            "name"    : self.name,
            "trigger" : self.trigger,
            "attack"  : self.attack,
            "decay"   : self.decay,
            "sustain" : self.sustain,
            "release" : self.release,
            "peak"    : self.peak,
            "gate_off": self.gate_off,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EnvelopeDef":
        return cls(
            name     = d["name"],
            trigger  = d["trigger"],
            attack   = d.get("attack",  0.01),
            decay    = d.get("decay",   0.1),
            sustain  = d.get("sustain", 0.0),
            release  = d.get("release", 0.2),
            peak     = d.get("peak",    1.0),
            gate_off = d.get("gate_off"),
        )


@dataclass
class LFODef:
    """Low-frequency oscillator; output available as lfo.<name>.

    Example::
        LFODef(name="slow_sine", shape="sine", rate_hz=0.25, phase=0.0)
    """
    name    : str
    shape   : str   = "sine"  # "sine" | "saw" | "square" | "tri"
    rate_hz : float = 1.0
    phase   : float = 0.0     # 0-1 initial phase offset

    def to_dict(self) -> dict:
        return {
            "name"   : self.name,
            "shape"  : self.shape,
            "rate_hz": self.rate_hz,
            "phase"  : self.phase,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "LFODef":
        return cls(
            name    = d["name"],
            shape   = d.get("shape",   "sine"),
            rate_hz = d.get("rate_hz", 1.0),
            phase   = d.get("phase",   0.0),
        )


@dataclass
class EventLink:
    """Routes a discrete event to an action on a PM sink.

    Example::
        EventLink(
            event     = "midi.note36.on",
            action    = "toggle(scene.show_cloud)",
            condition = None,
        )
    """
    event     : str
    action    : str             # e.g. "toggle(scene.show_cloud)", "preset('my_look')"
    enabled   : bool = True
    condition : str | None = None  # optional guard expression evaluated in source ns

    def to_dict(self) -> dict:
        return {
            "event"    : self.event,
            "action"   : self.action,
            "enabled"  : self.enabled,
            "condition": self.condition,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EventLink":
        return cls(
            event     = d["event"],
            action    = d["action"],
            enabled   = d.get("enabled", True),
            condition = d.get("condition"),
        )


@dataclass
class ThresholdDef:
    """Fires event "audio.threshold.<name>" when a source crosses a level.

    Hysteresis: ON fires when source exceeds `high`; OFF fires when it drops
    below `low`.  min_interval_s prevents rapid re-firing.

    Example::
        ThresholdDef(
            name             = "bass_hit",
            source           = "audio.bass",
            high             = 0.7,
            low              = 0.4,
            min_interval_s   = 0.1,
        )
    """
    name           : str
    source         : str    # source registry key, e.g. "audio.bass"
    high           : float = 0.7   # threshold to fire ON event
    low            : float = 0.3   # threshold to fire OFF event (hysteresis)
    min_interval_s : float = 0.05  # minimum seconds between ON firings

    def to_dict(self) -> dict:
        return {
            "name"          : self.name,
            "source"        : self.source,
            "high"          : self.high,
            "low"           : self.low,
            "min_interval_s": self.min_interval_s,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ThresholdDef":
        return cls(
            name           = d["name"],
            source         = d["source"],
            high           = d.get("high",           0.7),
            low            = d.get("low",            0.3),
            min_interval_s = d.get("min_interval_s", 0.05),
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Source Registry
# ─────────────────────────────────────────────────────────────────────────────

class SourceRegistry:
    """Thread-safe flat dict of normalized 0-1 source values.

    Audio/MIDI/keyboard threads write via update(); the GL thread reads via
    snapshot() once per frame to build the expression eval namespace.

    For each key written via update(), two derived variants are maintained:
      <key>_smooth  --  exponential low-pass, fixed tau = _SMOOTH_TAU seconds
      <key>_peak    --  peak-hold with multiplicative decay of _PEAK_DECAY per
                        update() call (phase 1: both values are fixed)
    """

    _SMOOTH_TAU: float = 0.15   # seconds -- smoothing time constant
    _PEAK_DECAY: float = 0.97   # per-update decay factor for peak-hold

    def __init__(self) -> None:
        self._data        : dict[str, float] = {}
        self._smooth_state: dict[str, float] = {}  # key -> last smoothed value
        self._peak_state  : dict[str, float] = {}  # key -> current peak
        self._last_t      : dict[str, float] = {}  # key -> time of last update
        self._lock = threading.Lock()

    # -- write (any thread) ---------------------------------------------------

    def update(self, key: str, value: float) -> None:
        """Write a 0-1 float; automatically updates <key>_smooth and <key>_peak."""
        now = time.perf_counter()
        with self._lock:
            self._data[key] = value

            # smooth variant -- exponential low-pass
            prev_t            = self._last_t.get(key, now)
            dt                = now - prev_t
            self._last_t[key] = now
            prev_s            = self._smooth_state.get(key, value)
            alpha             = 1.0 - _exp(-dt / max(self._SMOOTH_TAU, 1e-6))
            new_s             = prev_s + (value - prev_s) * alpha
            self._smooth_state[key]     = new_s
            self._data[key + "_smooth"] = new_s

            # peak-hold variant -- track maximum, decay when signal drops
            prev_p             = self._peak_state.get(key, value)
            new_p              = max(value, prev_p * self._PEAK_DECAY)
            self._peak_state[key]      = new_p
            self._data[key + "_peak"]  = new_p

    # -- read (GL thread only) ------------------------------------------------

    def snapshot(self) -> dict:
        """Return a shallow copy of all current values for use as an eval namespace."""
        with self._lock:
            return dict(self._data)

    def source_keys(self) -> list:
        """Sorted list of all registered source keys (raw, _smooth, _peak included)."""
        with self._lock:
            return sorted(self._data)


# ─────────────────────────────────────────────────────────────────────────────
#  EventBus
# ─────────────────────────────────────────────────────────────────────────────

class EventBus:
    """Thread-safe pub/sub event bus backed by a queue.Queue.

    Any thread may call fire().  The GL thread drains the queue once per frame
    via drain(), which dispatches payloads to registered subscribers.
    """

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable]] = {}
        self._queue: queue.Queue = queue.Queue()
        self._next_token: int = 0
        self._token_map: dict[int, tuple[str, Callable]] = {}

    def subscribe(self, event_id: str, callback: Callable) -> int:
        """Register callback(payload) for event_id.  Returns an unsubscribe token."""
        self._subscribers.setdefault(event_id, []).append(callback)
        token = self._next_token
        self._next_token += 1
        self._token_map[token] = (event_id, callback)
        return token

    def unsubscribe(self, token: int) -> None:
        """Remove a subscription by the token returned from subscribe()."""
        entry = self._token_map.pop(token, None)
        if entry is None:
            return
        event_id, callback = entry
        try:
            self._subscribers[event_id].remove(callback)
        except (KeyError, ValueError):
            pass

    def fire(self, event_id: str, payload: Any = None) -> None:
        """Enqueue an event.  Thread-safe; returns immediately."""
        self._queue.put_nowait((event_id, payload))

    def drain(self) -> list[tuple[str, Any]]:
        """Drain the queue, call subscribers, return list of (event_id, payload).

        Called from the GL thread once per frame.  Events newly fired during a
        subscriber callback are queued and held for the next frame.
        """
        fired: list[tuple[str, Any]] = []
        try:
            while True:
                fired.append(self._queue.get_nowait())
        except queue.Empty:
            pass
        for event_id, payload in fired:
            for cb in self._subscribers.get(event_id, []):
                try:
                    cb(payload)
                except Exception:
                    pass
        return fired


# ─────────────────────────────────────────────────────────────────────────────
#  Math helpers -- injected into expression eval namespace in Phase 2
# ─────────────────────────────────────────────────────────────────────────────

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


def _remap(x: float, a: float, b: float, c: float, d: float) -> float:
    """Remap x from [a, b] to [c, d]."""
    span = b - a
    if abs(span) < 1e-12:
        return c
    return c + (d - c) * (x - a) / span


#: Injected into every per-link eval namespace.
#: ``smooth`` is NOT included -- it is a per-link stateful SmoothHelper.
EVAL_MATH_NS: dict = {
    "lerp" : _lerp,
    "clamp": _clamp,
    "remap": _remap,
    "sin"  : _math.sin,
    "cos"  : _math.cos,
    "abs"  : abs,
    "min"  : min,
    "max"  : max,
    "pow"  : pow,
    "sqrt" : _math.sqrt,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Namespace builder
# ─────────────────────────────────────────────────────────────────────────────

def _flat_to_ns(flat: dict) -> dict:
    """Convert {'audio.bass': 0.5, 'midi.cc7': 0.3} to {'audio': NS(bass=0.5), ...}.

    Allows expressions to use attribute syntax: audio.bass, midi.cc7, clock.t, etc.
    Keys without a dot are passed through unchanged.
    """
    groups: dict[str, dict] = {}
    result: dict = {}
    for key, val in flat.items():
        if "." in key:
            prefix, _, attr = key.partition(".")
            groups.setdefault(prefix, {})[attr] = val
        else:
            result[key] = val
    for prefix, attrs in groups.items():
        result[prefix] = types.SimpleNamespace(**attrs)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  SmoothHelper — per-SignalLink stateful smooth() callable
# ─────────────────────────────────────────────────────────────────────────────

class SmoothHelper:
    """Stateful smooth(x, tau) injected into each link's eval namespace.

    Call index auto-assigns state slots so multiple smooth() calls in one
    expression each get independent state.  _idx is reset to 0 before each
    eval so slot assignments are stable across frames.
    """

    def __init__(self) -> None:
        self._states: dict[int, float] = {}
        self._idx:    int   = 0
        self.dt:      float = 0.016   # set by evaluate_links before each reset()

    def reset(self) -> None:
        self._idx = 0

    def __call__(self, x: float, tau: float) -> float:
        slot          = self._idx
        self._idx    += 1
        prev          = self._states.get(slot, x)
        alpha         = 1.0 - _exp(-self.dt / max(tau, 1e-6))
        out           = prev + (x - prev) * alpha
        self._states[slot] = out
        return out


# ─────────────────────────────────────────────────────────────────────────────
#  ADSREnvelope
# ─────────────────────────────────────────────────────────────────────────────

class ADSREnvelope:
    """ADSR envelope with linear attack/decay/release segments.

    State machine: IDLE → ATTACK → DECAY → SUSTAIN → RELEASE → IDLE.
    trigger() restarts from ATTACK at any point.
    gate_off() moves SUSTAIN → RELEASE; ignored in other states.
    tick(dt) advances the state machine and returns the current 0-1 output.
    """

    _IDLE    = 0
    _ATTACK  = 1
    _DECAY   = 2
    _SUSTAIN = 3
    _RELEASE = 4

    def __init__(self, attack: float, decay: float, sustain: float,
                 release: float, peak: float = 1.0) -> None:
        self.attack  = attack
        self.decay   = decay
        self.sustain = sustain
        self.release = release
        self.peak    = peak
        self._state  = self._IDLE
        self._value  = 0.0
        self._t      = 0.0   # elapsed time within current phase

    def trigger(self) -> None:
        self._state = self._ATTACK
        self._t     = 0.0

    def gate_off(self) -> None:
        if self._state == self._SUSTAIN:
            self._state = self._RELEASE
            self._t     = 0.0

    def tick(self, dt: float) -> float:
        self._t += dt

        if self._state == self._IDLE:
            self._value = 0.0

        elif self._state == self._ATTACK:
            self._value = self.peak * (self._t / self.attack) if self.attack > 0 else self.peak
            if self._t >= self.attack:
                self._value = self.peak
                self._state = self._DECAY
                self._t     = 0.0

        elif self._state == self._DECAY:
            if self.decay > 0:
                self._value = self.peak + (self.sustain - self.peak) * (self._t / self.decay)
            else:
                self._value = self.sustain
            if self._t >= self.decay:
                self._value = self.sustain
                self._state = self._SUSTAIN
                self._t     = 0.0

        elif self._state == self._SUSTAIN:
            self._value = self.sustain

        elif self._state == self._RELEASE:
            self._value = self.sustain * (1.0 - self._t / self.release) if self.release > 0 else 0.0
            if self._t >= self.release:
                self._value = 0.0
                self._state = self._IDLE
                self._t     = 0.0

        return max(0.0, min(1.0, self._value))


# ─────────────────────────────────────────────────────────────────────────────
#  LFO
# ─────────────────────────────────────────────────────────────────────────────

class LFO:
    """Low-frequency oscillator with a 0-1 phase accumulator.

    Shapes: sine, saw, square, tri.  Output is always 0-1.
    """

    def __init__(self, shape: str, rate_hz: float, phase: float = 0.0) -> None:
        self.shape   = shape
        self.rate_hz = rate_hz
        self._phase  = phase % 1.0

    def tick(self, dt: float) -> float:
        self._phase = (self._phase + self.rate_hz * dt) % 1.0
        p = self._phase
        if self.shape == "sine":
            return (_math.sin(p * 2.0 * _math.pi) + 1.0) * 0.5
        elif self.shape == "saw":
            return p
        elif self.shape == "square":
            return 1.0 if p < 0.5 else 0.0
        elif self.shape == "tri":
            return 1.0 - abs(p * 2.0 - 1.0)
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  LinkManager
# ─────────────────────────────────────────────────────────────────────────────

class LinkManager:
    """Coordinates all signal routing between sources and PM sinks."""

    def __init__(self) -> None:
        self.source_registry = SourceRegistry()
        self.event_bus       = EventBus()

        self._signal_links  : list[SignalLink]         = []
        self._envelope_defs : list[EnvelopeDef]        = []
        self._envelopes     : dict[str, ADSREnvelope]  = {}
        self._lfo_defs      : list[LFODef]             = []
        self._lfos          : dict[str, LFO]           = {}
        self._event_links   : list[EventLink]          = []
        self._threshold_defs: list[ThresholdDef]       = []
        self._threshold_state : dict[str, bool]        = {}  # name -> currently_above_high
        self._threshold_last_t: dict[str, float]       = {}  # name -> perf_counter of last fire
        self._envelope_sub_tokens: dict[str, list[int]] = {}  # name -> EventBus tokens

        self._const_ns = types.SimpleNamespace()  # user-defined named constants

        # Link presets: named snapshots of routing state
        self._presets         : dict[str, dict]  = {}
        # Preset triggers: EventLinks that survive preset switches
        self._preset_triggers : list[EventLink]  = []

        # Baseline values: persistent per-property floats written to PM when an
        # expression is disabled or cleared.  Intentionally NOT cleared by clear_state().
        self._baselines: dict[str, float] = {}

    # ── SignalLink management ─────────────────────────────────────────────────

    def add_link(self, link: SignalLink) -> None:
        """Append a SignalLink and attach a fresh SmoothHelper to it."""
        link.smooth_helper = SmoothHelper()   # type: ignore[attr-defined]
        self._signal_links.append(link)

    def remove_link(self, sink_key: str) -> None:
        """Remove all links targeting sink_key."""
        self._signal_links = [l for l in self._signal_links if l.sink_key != sink_key]

    def enable_link(self, sink_key: str) -> None:
        for link in self._signal_links:
            if link.sink_key == sink_key:
                link.enabled = True

    def disable_link(self, sink_key: str) -> None:
        for link in self._signal_links:
            if link.sink_key == sink_key:
                link.enabled = False

    # ── Baseline management ───────────────────────────────────────────────────

    def set_baseline(self, key: str, val: Any) -> None:
        self._baselines[key] = val

    def get_baseline(self, key: str, fallback: Any) -> Any:
        return self._baselines.get(key, fallback)

    def apply_baseline(self, key: str, pm: Any) -> None:
        """Write the stored baseline for key to pm; uses pm's property default if none set."""
        defn = pm._defs.get(key)
        prop_default = defn.default if defn is not None else 0.0
        pm.set(key, self.get_baseline(key, prop_default))

    # ── Per-frame evaluation ──────────────────────────────────────────────────

    def evaluate_links(self, pm: Any, dt: float) -> None:
        """Evaluate all enabled SignalLinks and write results to PM.

        Called from the GL thread once per frame, after source registry writes
        (clock, keyboard) and before drawing.
        """
        snap      = self.source_registry.snapshot()
        shared_ns = {**_flat_to_ns(snap), **EVAL_MATH_NS,
                     "dt": dt, "const": self._const_ns}

        for link in self._signal_links:
            if not link.enabled:
                continue
            link.smooth_helper.dt = dt        # type: ignore[attr-defined]
            link.smooth_helper.reset()        # type: ignore[attr-defined]
            per_link_ns = {**shared_ns, "smooth": link.smooth_helper}  # type: ignore[attr-defined]
            try:
                value = eval(link.expression, {"__builtins__": {}}, per_link_ns)
                pm.set(link.sink_key, value)
            except Exception:
                pass

    # ── Envelope management ───────────────────────────────────────────────────

    def add_envelope(self, defn: EnvelopeDef) -> None:
        """Register an EnvelopeDef, create its runtime ADSREnvelope, and wire triggers."""
        env = ADSREnvelope(
            attack  = defn.attack,
            decay   = defn.decay,
            sustain = defn.sustain,
            release = defn.release,
            peak    = defn.peak,
        )
        self._envelopes[defn.name] = env
        self._envelope_defs.append(defn)
        tokens = [self.event_bus.subscribe(defn.trigger, lambda _p, e=env: e.trigger())]
        if defn.gate_off:
            tokens.append(self.event_bus.subscribe(defn.gate_off, lambda _p, e=env: e.gate_off()))
        self._envelope_sub_tokens[defn.name] = tokens

    def remove_envelope(self, name: str) -> None:
        for token in self._envelope_sub_tokens.pop(name, []):
            self.event_bus.unsubscribe(token)
        self._envelopes.pop(name, None)
        self._envelope_defs = [d for d in self._envelope_defs if d.name != name]

    def tick_envelopes(self, dt: float) -> None:
        """Advance all envelopes and write env.<name> into the source registry."""
        for name, env in self._envelopes.items():
            self.source_registry.update(f"env.{name}", env.tick(dt))

    # ── LFO management ────────────────────────────────────────────────────────

    def add_lfo(self, defn: LFODef) -> None:
        """Register an LFODef and create its runtime LFO."""
        lfo = LFO(shape=defn.shape, rate_hz=defn.rate_hz, phase=defn.phase)
        self._lfos[defn.name] = lfo
        self._lfo_defs.append(defn)

    def remove_lfo(self, name: str) -> None:
        self._lfos.pop(name, None)
        self._lfo_defs = [d for d in self._lfo_defs if d.name != name]

    def tick_lfos(self, dt: float) -> None:
        """Advance all LFOs and write lfo.<name> into the source registry."""
        for name, lfo in self._lfos.items():
            self.source_registry.update(f"lfo.{name}", lfo.tick(dt))

    # ── Threshold management ──────────────────────────────────────────────────

    def add_threshold(self, defn: ThresholdDef) -> None:
        """Register a ThresholdDef; fires audio.threshold.<name> on rising edge."""
        self._threshold_defs.append(defn)

    def remove_threshold(self, name: str) -> None:
        self._threshold_defs = [d for d in self._threshold_defs if d.name != name]
        self._threshold_state.pop(name, None)
        self._threshold_last_t.pop(name, None)

    def tick_thresholds(self) -> None:
        """Check all ThresholdDefs against the current registry; fire events on edges.

        Rising edge (src >= high and was below): fires audio.threshold.<name>.
        Falling edge (src < low and was above): resets hysteresis without firing.
        min_interval_s prevents the rising edge from re-firing too quickly.
        """
        snap = self.source_registry.snapshot()
        now  = time.perf_counter()
        for defn in self._threshold_defs:
            val       = snap.get(defn.source, 0.0)
            was_above = self._threshold_state.get(defn.name, False)
            if not was_above and val >= defn.high:
                last_t = self._threshold_last_t.get(defn.name, 0.0)
                if now - last_t >= defn.min_interval_s:
                    self.event_bus.fire(f"audio.threshold.{defn.name}", val)
                    self._threshold_state[defn.name]  = True
                    self._threshold_last_t[defn.name] = now
            elif was_above and val < defn.low:
                self._threshold_state[defn.name] = False

    # ── Persistence ───────────────────────────────────────────────────────────

    def clear_state(self) -> None:
        """Remove all routing state: links, envelopes, LFOs, event links, thresholds.

        Does NOT touch _presets, _preset_triggers, or _baselines.  Baselines are a
        persistent property overlay and must survive preset switches and state reloads.
        """
        self._signal_links.clear()
        for name in list(self._envelopes):
            for token in self._envelope_sub_tokens.pop(name, []):
                self.event_bus.unsubscribe(token)
        self._envelope_defs.clear()
        self._envelopes.clear()
        self._lfo_defs.clear()
        self._lfos.clear()
        self._event_links.clear()
        self._threshold_defs.clear()
        self._threshold_state.clear()
        self._threshold_last_t.clear()

    def save_state(self, path) -> None:
        """Serialise all routing state, presets, and preset triggers to JSON."""
        import json as _json
        from pathlib import Path as _Path
        data = {
            "signal_links"    : [l.to_dict() for l in self._signal_links],
            "envelopes"       : [d.to_dict() for d in self._envelope_defs],
            "lfos"            : [d.to_dict() for d in self._lfo_defs],
            "event_links"     : [l.to_dict() for l in self._event_links],
            "thresholds"      : [d.to_dict() for d in self._threshold_defs],
            "baselines"       : dict(self._baselines),
            "presets"         : self._presets,
            "preset_triggers" : [l.to_dict() for l in self._preset_triggers],
        }
        _Path(path).write_text(_json.dumps(data, indent=2))

    def load_state(self, path, replace: bool = False) -> None:
        """Load state from JSON.  If replace=True, clear existing routing state first.

        Presets and preset_triggers in the file are merged (not replaced) into
        the current instance so navigation links accumulate across multiple loads.
        """
        import json as _json
        from pathlib import Path as _Path
        data = _json.loads(_Path(path).read_text())
        if replace:
            self.clear_state()
        for d in data.get("signal_links", []):
            self.add_link(SignalLink.from_dict(d))
        for d in data.get("envelopes", []):
            self.add_envelope(EnvelopeDef.from_dict(d))
        for d in data.get("lfos", []):
            self.add_lfo(LFODef.from_dict(d))
        for d in data.get("event_links", []):
            self.add_event_link(EventLink.from_dict(d))
        for d in data.get("thresholds", []):
            self.add_threshold(ThresholdDef.from_dict(d))
        self._baselines.update(data.get("baselines", {}))
        for name, snap in data.get("presets", {}).items():
            self._presets[name] = snap
        for d in data.get("preset_triggers", []):
            self.add_preset_trigger(EventLink.from_dict(d))

    # ── Link preset management ────────────────────────────────────────────────

    def save_link_preset(self, name: str) -> None:
        """Snapshot the current routing state under a named preset."""
        self._presets[name] = {
            "signal_links" : [l.to_dict() for l in self._signal_links],
            "envelopes"    : [d.to_dict() for d in self._envelope_defs],
            "lfos"         : [d.to_dict() for d in self._lfo_defs],
            "event_links"  : [l.to_dict() for l in self._event_links],
            "thresholds"   : [d.to_dict() for d in self._threshold_defs],
        }
        print(f"[lm] preset saved: '{name}'")

    def load_link_preset(self, name: str) -> None:
        """Restore a named preset, replacing current routing state.

        _preset_triggers and _presets are left untouched.
        """
        if name not in self._presets:
            print(f"[lm] preset '{name}' not found. Available: {list(self._presets)}")
            return
        snap = self._presets[name]
        self.clear_state()
        for d in snap.get("signal_links", []):
            self.add_link(SignalLink.from_dict(d))
        for d in snap.get("envelopes", []):
            self.add_envelope(EnvelopeDef.from_dict(d))
        for d in snap.get("lfos", []):
            self.add_lfo(LFODef.from_dict(d))
        for d in snap.get("event_links", []):
            self.add_event_link(EventLink.from_dict(d))
        for d in snap.get("thresholds", []):
            self.add_threshold(ThresholdDef.from_dict(d))
        print(f"[lm] preset loaded: '{name}'")

    def delete_link_preset(self, name: str) -> None:
        self._presets.pop(name, None)

    def list_link_presets(self) -> list[str]:
        return list(self._presets)

    # ── Preset trigger management ─────────────────────────────────────────────

    def add_preset_trigger(self, link: EventLink) -> None:
        """Register a navigation EventLink that survives preset switches."""
        self._preset_triggers.append(link)

    def remove_preset_trigger(self, event: str, action: str) -> None:
        self._preset_triggers = [
            l for l in self._preset_triggers
            if not (l.event == event and l.action == action)
        ]

    # ── EventLink management ──────────────────────────────────────────────────

    def add_event_link(self, link: EventLink) -> None:
        self._event_links.append(link)

    def remove_event_link(self, event: str, action: str) -> None:
        self._event_links = [
            l for l in self._event_links
            if not (l.event == event and l.action == action)
        ]

    # ── Action dispatcher ─────────────────────────────────────────────────────

    _RE_TOGGLE      = re.compile(r"^toggle\(([^)]+)\)$")
    _RE_CYCLE       = re.compile(r"^cycle\(([^)]+)\)$")
    _RE_CYCLE_BACK  = re.compile(r"^cycle_back\(([^)]+)\)$")
    _RE_SET         = re.compile(r"^set\(([^,]+),\s*(.+)\)$")
    _RE_PRESET      = re.compile(r"""^preset\(['"]([^'"]+)['"]\)$""")
    _RE_LINK_PRESET = re.compile(r"""^link_preset\(['"]([^'"]+)['"]\)$""")

    def _dispatch_action(self, action: str, pm: Any) -> None:
        action = action.strip()

        m = self._RE_TOGGLE.match(action)
        if m:
            key = m.group(1).strip()
            pm.set(key, not pm.get(key))
            return

        m = self._RE_CYCLE.match(action)
        if m:
            key  = m.group(1).strip()
            defn = pm._defs.get(key)
            if defn and defn.choices:
                pm.set(key, defn.next_choice(pm.get(key)))
            return

        m = self._RE_CYCLE_BACK.match(action)
        if m:
            key  = m.group(1).strip()
            defn = pm._defs.get(key)
            if defn and defn.choices:
                pm.set(key, defn.prev_choice(pm.get(key)))
            return

        m = self._RE_SET.match(action)
        if m:
            key     = m.group(1).strip()
            val_str = m.group(2).strip()
            try:
                val = eval(val_str, {"__builtins__": {}}, {})
            except Exception:
                val = val_str
            pm.set(key, val)
            return

        m = self._RE_PRESET.match(action)
        if m:
            pm.load_preset(m.group(1))
            return

        m = self._RE_LINK_PRESET.match(action)
        if m:
            self.load_link_preset(m.group(1))
            return

        if action == "regen":
            # Fires "regen" as an event; subscribers (e.g. MergedGUI) handle it.
            self.event_bus.fire("regen")
            return

    # ── Event evaluation ──────────────────────────────────────────────────────

    def _fire_links(self, links: list, event_id: str, ns: dict, pm: Any) -> None:
        for link in links:
            if not link.enabled or link.event != event_id:
                continue
            if link.condition:
                try:
                    if not eval(link.condition, {"__builtins__": {}}, ns):
                        continue
                except Exception:
                    continue
            try:
                self._dispatch_action(link.action, pm)
            except Exception:
                pass

    def evaluate_events(self, pm: Any) -> None:
        """Drain the EventBus and dispatch events to subscribers and EventLinks.

        Called from the GL thread once per frame, after tick_thresholds() and
        before evaluate_links().  Envelope triggers are handled via subscriptions
        wired in add_envelope(); EventLinks and preset triggers are processed here.
        Preset triggers are evaluated before routing EventLinks so navigation takes
        effect in the same frame.
        """
        fired = self.event_bus.drain()  # also dispatches to subscribers (envelopes etc.)
        if not fired or not (self._event_links or self._preset_triggers):
            return

        snap = self.source_registry.snapshot()
        ns   = {**_flat_to_ns(snap), **EVAL_MATH_NS}

        for event_id, _payload in fired:
            self._fire_links(self._preset_triggers, event_id, ns, pm)
            self._fire_links(self._event_links,     event_id, ns, pm)
