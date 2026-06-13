"""
midi_input.py — MIDI event router for warp projects.

Listens for MIDI events in a background thread and dispatches them to
registered mappings:

  • CC  events  → scale 0–127 to [min_val, max_val] and set obj.field
  • Note events → call a callback(velocity) — velocity=0 on note_off

Quick start
-----------
    from midi_input import get_router

    router = get_router()
    print(router.available_devices())
    router.start()                           # auto-picks first device

    # CC #7 → _params.decay, range 0.95–1.0
    router.map_cc(7, _params, "decay", min_val=0.95, max_val=1.0, section="Post-FX")

    # CC #1 → _controls.scene_alpha, range 0–1
    router.map_cc(1, _controls, "scene_alpha", section="Scene")

    # Note 60 → toggle _controls.show_cloud
    router.map_note_to_param(60, _controls, "show_cloud", section="Scene")

    # Note 62 → arbitrary callback
    router.map_note(62, lambda vel: print("hi", vel), field="debug", section="Custom")

Install
-------
    pip install mido python-rtmidi
"""

import json
import time
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


try:
    import mido
    _MIDO_OK = True
except ImportError:
    _MIDO_OK = False
    print("[midi] mido not installed — run: pip install mido python-rtmidi")


# ── mapping types ──────────────────────────────────────────────────────────────

@dataclass
class CCMapping:
    """CC number → object.field with linear range scaling."""
    section : str
    field   : str
    obj     : Any
    min_val : float
    max_val : float
    channel : int | None   # None = any channel

    def apply(self, raw: int) -> None:
        t   = raw / 127.0
        val = self.min_val + t * (self.max_val - self.min_val)
        # preserve existing type (int stays int)
        existing = getattr(self.obj, self.field, None)
        if isinstance(existing, int):
            val = int(round(val))
        setattr(self.obj, self.field, val)

    def to_dict(self, cc_num: int) -> dict:
        return dict(type="cc", cc_num=cc_num, channel=self.channel,
                    section=self.section, field=self.field,
                    min_val=self.min_val, max_val=self.max_val)


@dataclass
class NoteMapping:
    """Note number → arbitrary callback(velocity).  velocity=0 on note_off."""
    section  : str
    field    : str           # display name for GUI
    callback : Callable[[int], None]
    channel  : int | None

    def to_dict(self, note: int) -> dict:
        return dict(type="note", note=note, channel=self.channel,
                    section=self.section, field=self.field)


@dataclass
class NoteParamMapping:
    """Note number → object.field with on/off values (or bool toggle)."""
    section   : str
    field     : str
    obj       : Any
    on_value  : Any     # applied on note_on (vel > 0)
    off_value : Any     # applied on note_off (vel == 0)
    toggle    : bool    # if True, toggle bool on note_on; ignore note_off
    channel   : int | None

    def apply(self, velocity: int) -> None:
        if self.toggle:
            if velocity > 0:
                setattr(self.obj, self.field, not getattr(self.obj, self.field))
        else:
            setattr(self.obj, self.field,
                    self.on_value if velocity > 0 else self.off_value)

    def to_dict(self, note: int) -> dict:
        return dict(type="note_param", note=note, channel=self.channel,
                    section=self.section, field=self.field,
                    on_value=self.on_value, off_value=self.off_value,
                    toggle=self.toggle)


# ── router ────────────────────────────────────────────────────────────────────

class MidiRouter:
    """
    Thread-safe MIDI event dispatcher.

    Internally keyed by (channel | None, number):
      _cc_map        : CC→param        {key: list[CCMapping]}
      _note_map      : note→callback   {key: list[NoteMapping]}
      _note_param_map: note→param      {key: list[NoteParamMapping]}

    channel=None in a mapping matches any incoming channel.
    """

    def __init__(self) -> None:
        self._lock            = threading.Lock()
        self._cc_map          : dict[tuple, list[CCMapping]]        = {}
        self._note_map        : dict[tuple, list[NoteMapping]]      = {}
        self._note_param_map  : dict[tuple, list[NoteParamMapping]] = {}
        self._raw_listeners   : list[Callable[[dict], None]]        = []

        self._thread  : threading.Thread | None = None
        self._port    : Any = None
        self._running : bool = False
        self.device_name: str | None = None

        # Latest event dict — polled by the GUI for the activity display
        self.last_event: dict = {}

    # ── device management ──────────────────────────────────────────────────────

    def available_devices(self) -> list[str]:
        """Return all available MIDI input device names."""
        if not _MIDO_OK:
            return []
        try:
            return mido.get_input_names()
        except Exception:
            return []

    @property
    def is_connected(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, device_name: str | None = None) -> bool:
        """
        Open a MIDI input port and start listening in a background thread.
        Pass device_name=None to auto-pick the first available device.
        Returns True on success.
        """
        if not _MIDO_OK:
            print("[midi] mido not available")
            return False

        names = self.available_devices()
        if not names:
            print("[midi] no MIDI input devices found")
            return False

        name = device_name or names[0]
        if name not in names:
            print(f"[midi] device not found: {name!r}  available: {names}")
            return False

        self.stop()
        self._running    = True
        self.device_name = name
        self._thread     = threading.Thread(
            target=self._listen_loop, args=(name,),
            daemon=True, name="midi-router",
        )
        self._thread.start()
        print(f"[midi] started on '{name}'")
        return True

    def stop(self) -> None:
        """Stop the MIDI listener thread."""
        self._running = False
        if self._port is not None:
            try:
                self._port.close()
            except Exception:
                pass
            self._port = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread     = None
        self.device_name = None

    # ── CC mappings ────────────────────────────────────────────────────────────

    def map_cc(
        self,
        cc_num  : int,
        obj     : Any,
        field   : str,
        min_val : float = 0.0,
        max_val : float = 1.0,
        channel : int | None = None,
        section : str = "",
    ) -> None:
        """Bind CC cc_num → obj.field, scaling 0–127 to [min_val, max_val]."""
        key = (channel, cc_num)
        m   = CCMapping(section=section, field=field, obj=obj,
                        min_val=min_val, max_val=max_val, channel=channel)
        with self._lock:
            self._cc_map.setdefault(key, []).append(m)

    def unmap_cc(self, cc_num: int, obj: Any, field: str,
                 channel: int | None = None) -> None:
        key = (channel, cc_num)
        with self._lock:
            lst = self._cc_map.get(key, [])
            self._cc_map[key] = [
                m for m in lst if not (m.obj is obj and m.field == field)
            ]

    def unmap_cc_by_index(self, idx: int) -> None:
        """Remove a CC mapping identified by its position in get_cc_mappings()."""
        with self._lock:
            flat = []
            for key, lst in self._cc_map.items():
                for m in lst:
                    flat.append((key, m))
            if 0 <= idx < len(flat):
                key, target = flat[idx]
                self._cc_map[key] = [m for m in self._cc_map[key] if m is not target]

    # ── note → callback mappings ───────────────────────────────────────────────

    def map_note(
        self,
        note     : int,
        callback : Callable[[int], None],
        field    : str = "",
        channel  : int | None = None,
        section  : str = "",
    ) -> None:
        """Bind note → callback(velocity). velocity=0 on note_off."""
        key = (channel, note)
        m   = NoteMapping(section=section, field=field,
                          callback=callback, channel=channel)
        with self._lock:
            self._note_map.setdefault(key, []).append(m)

    def unmap_note_by_index(self, idx: int) -> None:
        with self._lock:
            flat = []
            for key, lst in self._note_map.items():
                for m in lst:
                    flat.append((key, m))
            if 0 <= idx < len(flat):
                key, target = flat[idx]
                self._note_map[key] = [m for m in self._note_map[key] if m is not target]

    # ── note → param mappings ──────────────────────────────────────────────────

    def map_note_to_param(
        self,
        note      : int,
        obj       : Any,
        field     : str,
        on_value  : Any = None,
        off_value : Any = None,
        channel   : int | None = None,
        section   : str = "",
    ) -> None:
        """
        Bind note → obj.field.

        For bool fields: toggles on note_on, ignores note_off.
        For float/int fields:
          on_value  applied on note_on  (default: max of range or 1.0)
          off_value applied on note_off (default: 0 / 0.0)
        """
        existing = getattr(obj, field, None)
        toggle   = isinstance(existing, bool)

        if not toggle:
            if on_value  is None: on_value  = 1.0 if isinstance(existing, float) else 1
            if off_value is None: off_value = 0.0 if isinstance(existing, float) else 0

        key = (channel, note)
        m   = NoteParamMapping(
            section=section, field=field, obj=obj,
            on_value=on_value, off_value=off_value,
            toggle=toggle, channel=channel,
        )
        with self._lock:
            self._note_param_map.setdefault(key, []).append(m)

    def unmap_note_param_by_index(self, idx: int) -> None:
        with self._lock:
            flat = []
            for key, lst in self._note_param_map.items():
                for m in lst:
                    flat.append((key, m))
            if 0 <= idx < len(flat):
                key, target = flat[idx]
                self._note_param_map[key] = [
                    m for m in self._note_param_map[key] if m is not target
                ]

    # ── raw listeners ──────────────────────────────────────────────────────────

    def add_listener(self, callback: Callable[[dict], None]) -> None:
        """
        Register a raw event listener called on every MIDI message.
        callback receives a dict: {type, channel, number, value}.
        Called from the MIDI thread — keep it fast.
        """
        with self._lock:
            self._raw_listeners.append(callback)

    def remove_listener(self, callback: Callable[[dict], None]) -> None:
        with self._lock:
            self._raw_listeners = [l for l in self._raw_listeners if l is not callback]

    # ── serialization ──────────────────────────────────────────────────────────

    def get_cc_mappings(self) -> list[dict]:
        """Serializable snapshot of all CC mappings."""
        out = []
        with self._lock:
            for (channel, cc_num), lst in sorted(self._cc_map.items(), key=lambda x: x[0]):
                for m in lst:
                    out.append(m.to_dict(cc_num))
        return out

    def get_note_mappings(self) -> list[dict]:
        """Serializable snapshot of all note→callback mappings."""
        out = []
        with self._lock:
            for (channel, note), lst in sorted(self._note_map.items(), key=lambda x: x[0]):
                for m in lst:
                    out.append(m.to_dict(note))
        return out

    def get_note_param_mappings(self) -> list[dict]:
        """Serializable snapshot of all note→param mappings."""
        out = []
        with self._lock:
            for (channel, note), lst in sorted(self._note_param_map.items(), key=lambda x: x[0]):
                for m in lst:
                    out.append(m.to_dict(note))
        return out

    def save_mappings(self, path: str | Path) -> None:
        """Save all current mappings to a JSON file."""
        data = {
            "cc":         self.get_cc_mappings(),
            "note":       self.get_note_mappings(),
            "note_param": self.get_note_param_mappings(),
        }
        Path(path).write_text(json.dumps(data, indent=2))
        n = len(data["cc"]) + len(data["note"]) + len(data["note_param"])
        print(f"[midi] saved {n} mappings → {path}")

    def load_cc_mappings(
        self,
        data: list[dict],
        resolve: Callable[[str, str], Any | None],
    ) -> int:
        """
        Re-apply saved CC mappings.
        resolve(section, field) must return the live object for that section,
        or None if unavailable.  Returns count of successfully restored mappings.
        """
        count = 0
        for d in data:
            obj = resolve(d["section"], d["field"])
            if obj is None:
                print(f"[midi] load: cannot resolve {d['section']}/{d['field']}")
                continue
            self.map_cc(d["cc_num"], obj, d["field"],
                        min_val=d["min_val"], max_val=d["max_val"],
                        channel=d.get("channel"), section=d["section"])
            count += 1
        return count

    def load_note_param_mappings(
        self,
        data: list[dict],
        resolve: Callable[[str, str], Any | None],
    ) -> int:
        count = 0
        for d in data:
            obj = resolve(d["section"], d["field"])
            if obj is None:
                print(f"[midi] load: cannot resolve {d['section']}/{d['field']}")
                continue
            self.map_note_to_param(
                d["note"], obj, d["field"],
                on_value=d.get("on_value"), off_value=d.get("off_value"),
                channel=d.get("channel"), section=d["section"],
            )
            count += 1
        return count

    # ── clear ──────────────────────────────────────────────────────────────────

    def clear_all_mappings(self) -> None:
        with self._lock:
            self._cc_map.clear()
            self._note_map.clear()
            self._note_param_map.clear()

    # ── internal ───────────────────────────────────────────────────────────────

    def _listen_loop(self, device_name: str) -> None:
        try:
            with mido.open_input(device_name) as port:
                self._port = port
                while self._running:
                    for msg in port.iter_pending():
                        self._dispatch(msg)
                    time.sleep(0.001)
        except Exception as exc:
            print(f"[midi] listener error: {exc}")
        finally:
            self._port = None

    def _dispatch(self, msg) -> None:
        ch = getattr(msg, "channel", 0)

        if msg.type == "control_change":
            evt = dict(type="cc", channel=ch, number=msg.control, value=msg.value)
            self.last_event = evt
            self._fire_raw(evt)

            with self._lock:
                candidates = (
                    self._cc_map.get((None, msg.control), []) +
                    self._cc_map.get((ch,   msg.control), [])
                )
                for m in list(candidates):
                    if m.channel is None or m.channel == ch:
                        m.apply(msg.value)

        elif msg.type in ("note_on", "note_off"):
            vel = msg.velocity if msg.type == "note_on" else 0
            evt = dict(type="note", channel=ch, number=msg.note, value=vel)
            self.last_event = evt
            self._fire_raw(evt)

            with self._lock:
                cb_candidates = (
                    self._note_map.get((None, msg.note), []) +
                    self._note_map.get((ch,   msg.note), [])
                )
                pm_candidates = (
                    self._note_param_map.get((None, msg.note), []) +
                    self._note_param_map.get((ch,   msg.note), [])
                )

            for m in list(cb_candidates):
                if m.channel is None or m.channel == ch:
                    try:
                        m.callback(vel)
                    except Exception as exc:
                        print(f"[midi] note callback error: {exc}")

            for m in list(pm_candidates):
                if m.channel is None or m.channel == ch:
                    m.apply(vel)

    def _fire_raw(self, evt: dict) -> None:
        with self._lock:
            listeners = list(self._raw_listeners)
        for cb in listeners:
            try:
                cb(evt)
            except Exception as exc:
                print(f"[midi] raw listener error: {exc}")


# ── module-level singleton ─────────────────────────────────────────────────────

_router: MidiRouter | None = None


def get_router() -> MidiRouter:
    """Return the shared MidiRouter instance, creating it on first call."""
    global _router
    if _router is None:
        _router = MidiRouter()
    return _router
