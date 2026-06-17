from __future__ import annotations

import time as _time
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from link_manager import EventBus, SourceRegistry

_MULT_SOURCES = {
    "bpm.phase_2":  2.0,
    "bpm.phase_4":  4.0,
    "bpm.phase_8":  8.0,
    "bpm.phase_16": 16.0,
}
_DIV_SOURCES = {
    "bpm.phase_h2": 0.5,
    "bpm.phase_h4": 0.25,
    "bpm.phase_h8": 0.125,
}


class BPMClock:
    DEFAULT_BPM    = 120.0
    BPM_MIN        = 20.0
    BPM_MAX        = 300.0
    TAP_WINDOW_S   = 3.0
    TAP_BUFFER_MAX = 8

    def __init__(self, event_bus: "EventBus", source_registry: "SourceRegistry") -> None:
        self._bus    = event_bus
        self._reg    = source_registry
        self.bpm     = self.DEFAULT_BPM
        self._phase  = 0.0
        self._beat_n = 0
        self._latency: float = 0.0
        self._taps:    list[float] = []
        self._beat_pulse = False
        self.last_beat_n: int = 0        # read by BpmPanel Qt timer
        self.tap_event_name: str = ""    # persisted by to_dict; wired by BpmPanel

    # ------------------------------------------------------------------
    # BPM control
    # ------------------------------------------------------------------

    def set_bpm(self, bpm: float) -> None:
        self.bpm = max(self.BPM_MIN, min(self.BPM_MAX, float(bpm)))

    def nudge(self, delta_bpm: float) -> None:
        self.set_bpm(self.bpm + delta_bpm)

    def set_latency_ms(self, ms: float) -> None:
        self._latency = ms / 1000.0

    # ------------------------------------------------------------------
    # Tap tempo
    # ------------------------------------------------------------------

    def tap(self) -> None:
        now = _time.perf_counter()
        self._taps.append(now)
        cutoff = now - self.TAP_WINDOW_S
        self._taps = [t for t in self._taps if t >= cutoff]
        if len(self._taps) > self.TAP_BUFFER_MAX:
            self._taps = self._taps[-self.TAP_BUFFER_MAX:]
        if len(self._taps) >= 2:
            diffs = [self._taps[i] - self._taps[i - 1] for i in range(1, len(self._taps))]
            mean_diff = sum(diffs) / len(diffs)
            if mean_diff > 0:
                self.set_bpm(60.0 / mean_diff)
        self._phase = 0.0

    def tap_event(self, _payload=None) -> None:
        self.tap()

    # ------------------------------------------------------------------
    # Per-frame update
    # ------------------------------------------------------------------

    def tick(self, dt: float) -> None:
        self._phase += dt * (self.bpm / 60.0)

        beat_fired = False
        if self._phase >= 1.0:
            self._phase %= 1.0
            beat_fired = True
            self._bus.fire("bpm.beat", self._beat_n)
            self._beat_n = (self._beat_n + 1) % 4
            self.last_beat_n = self._beat_n

        # effective phase with latency offset
        eff_phase = (self._phase + self._latency * self.bpm / 60.0) % 1.0

        reg = self._reg
        reg.update("bpm.phase",   eff_phase)
        reg.update("bpm.beat",    1.0 if beat_fired else 0.0)
        reg.update("bpm.raw_bpm", self.bpm)
        reg.update("bpm.beat_n",  self._beat_n / 4.0)

        for key, mult in _MULT_SOURCES.items():
            reg.update(key, (eff_phase * mult) % 1.0)
        for key, div in _DIV_SOURCES.items():
            reg.update(key, (eff_phase * div) % 1.0)

    # ------------------------------------------------------------------
    # Helper for BPM-synced LFOs
    # ------------------------------------------------------------------

    def beat_phase(self, mult: float = 1.0) -> float:
        eff_phase = (self._phase + self._latency * self.bpm / 60.0) % 1.0
        return (eff_phase * mult) % 1.0

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "bpm":               self.bpm,
            "latency_offset_ms": self._latency * 1000.0,
            "tap_event":         self.tap_event_name,
        }

    @classmethod
    def from_dict(
        cls,
        d: dict,
        event_bus: "EventBus",
        source_registry: "SourceRegistry",
    ) -> "BPMClock":
        obj = cls(event_bus, source_registry)
        obj.set_bpm(d.get("bpm", cls.DEFAULT_BPM))
        obj.set_latency_ms(d.get("latency_offset_ms", 0.0))
        obj.tap_event_name = d.get("tap_event", "")
        return obj
