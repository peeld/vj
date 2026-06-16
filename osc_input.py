"""
osc_input.py — OSC event router for warp projects.

Listens for OSC messages on a UDP socket in a background thread and
dispatches them to raw listeners, mirroring midi_input.py's MidiRouter:

  • Numeric-only args (int/float/bool)  → "param" event, written to
    SourceRegistry every message, like a MIDI CC.
  • Empty args (a bang), or any non-numeric arg (string, blob)  → "event"
    event, fired once per message via EventBus, like a MIDI note.

Input only — there is no OSC client/sender here, matching midi_input.py
(MIDI output isn't part of this app either).

Quick start
-----------
    from osc_input import get_router

    router = get_router()
    router.start("0.0.0.0", 9000)
    router.add_listener(lambda evt: print(evt))

Install
-------
    pip install python-osc
"""

import re
import threading
from typing import Any, Callable


try:
    from pythonosc.dispatcher import Dispatcher
    from pythonosc.osc_server import BlockingOSCUDPServer
    _OSC_OK = True
except ImportError:
    _OSC_OK = False
    print("[osc] python-osc not installed — run: pip install python-osc")


# ── address sanitizing (property side only) ────────────────────────────────

_ADDR_SAFE_RE = re.compile(r"[^0-9a-zA-Z_]")


def sanitize_address(address: str) -> str:
    """Turn an OSC address into a valid Python identifier fragment.

    "/1/fader1" -> "_1_fader1" -- used only for SourceRegistry keys, which
    get dotted-attribute access in SignalLink expressions.  Event ids keep
    the raw address since they're matched by plain string equality.
    """
    return _ADDR_SAFE_RE.sub("_", address)


# ── router ────────────────────────────────────────────────────────────────

class OscRouter:
    """
    Thread-safe OSC message dispatcher.

    Runs a BlockingOSCUDPServer in a daemon thread (single dispatch thread,
    matching MIDI's model — no per-message thread spawn).  Routing decisions
    (param vs event) happen in _on_message(); subscribers receive the raw
    classified dict via add_listener(), same shape as MidiRouter.
    """

    def __init__(self) -> None:
        self._lock          = threading.Lock()
        self._raw_listeners : list[Callable[[dict], None]] = []
        self._server        : Any = None
        self._thread        : threading.Thread | None = None
        self.host: str | None = None
        self.port: int | None = None

        # Latest event dict — polled by the GUI for the activity display
        self.last_event: dict = {}

    @property
    def is_connected(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, host: str = "0.0.0.0", port: int = 9000) -> bool:
        """Bind a UDP socket and start listening in a background thread."""
        if not _OSC_OK:
            print("[osc] python-osc not available")
            return False

        self.stop()

        dispatcher = Dispatcher()
        dispatcher.set_default_handler(self._on_message)

        try:
            self._server = BlockingOSCUDPServer((host, port), dispatcher)
        except OSError as exc:
            print(f"[osc] failed to bind {host}:{port}: {exc}")
            self._server = None
            return False

        self.host = host
        self.port = port
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True, name="osc-router",
        )
        self._thread.start()
        print(f"[osc] listening on {host}:{port}")
        return True

    def stop(self) -> None:
        """Stop the OSC listener thread."""
        if self._server is not None:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
            self._server = None
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        self.host = None
        self.port = None

    # ── raw listeners ────────────────────────────────────────────────────

    def add_listener(self, callback: Callable[[dict], None]) -> None:
        """
        Register a raw event listener called on every OSC message.
        callback receives a dict, see _on_message().  Called from the OSC
        thread — keep it fast.
        """
        with self._lock:
            self._raw_listeners.append(callback)

    def remove_listener(self, callback: Callable[[dict], None]) -> None:
        with self._lock:
            self._raw_listeners = [l for l in self._raw_listeners if l is not callback]

    # ── internal ─────────────────────────────────────────────────────────

    def _on_message(self, address: str, *args) -> None:
        numeric = [a for a in args if isinstance(a, (int, float))]
        if args and len(numeric) == len(args):
            key = sanitize_address(address)
            if len(args) == 1:
                evt = dict(type="param", address=address, key=key,
                           value=float(args[0]))
            else:
                evt = dict(type="param", address=address, key=key,
                           values=[float(a) for a in args])
        else:
            evt = dict(type="event", address=address, args=list(args))
        self.last_event = evt
        self._fire_raw(evt)

    def _fire_raw(self, evt: dict) -> None:
        with self._lock:
            listeners = list(self._raw_listeners)
        for cb in listeners:
            try:
                cb(evt)
            except Exception as exc:
                print(f"[osc] raw listener error: {exc}")


# ── module-level singleton ──────────────────────────────────────────────────

_router: OscRouter | None = None


def get_router() -> OscRouter:
    """Return the shared OscRouter instance, creating it on first call."""
    global _router
    if _router is None:
        _router = OscRouter()
    return _router
