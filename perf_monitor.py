"""perf_monitor.py — rolling per-stage performance metrics with CSV logging.

Written from the GL thread once per frame via record().
Read from the Qt thread via snapshot() for the ControlBar display.
Both sides are thread-safe via a single lock; the lock is held < 1 µs per call.

Frame budget reference: 16.67 ms at 60 fps.

Stage definitions (mirrors the on_render call sequence in gui_merged.py):
  step_ms   — element .step() calls (CPU: Warp kernel launches, param updates)
  scene_ms  — bind_scene_fbo() + _draw_scene()  (CPU: GL draw-call submission)
  post_ms   — effect.process()  (GPU sync + Warp kernels + PBO transfers)
  blit_ms   — effect.blit_to_screen()  (fullscreen quad GL)
  render_ms — total on_render() wall time (sum of above + link eval / camera)

GPU note: scene_ms measures CPU time to *submit* draw calls, not GPU execution
time.  GPU render latency shows up as a stall in post_ms (the PBO map() or
glReadPixels inside process() waits for the GPU).  render_ms is the accurate
total wall-clock cost per frame.
"""

from __future__ import annotations

import collections
import csv
import pathlib
import threading
import time
from dataclasses import dataclass

TARGET_MS: float = 1000.0 / 60.0   # 16.67 ms — 60 fps frame budget


@dataclass
class _FrameSample:
    fps: float
    render_ms: float
    step_ms: float
    scene_ms: float
    post_ms: float
    blit_ms: float
    effect: str


class PerfMonitor:
    """Rolling performance window. Written from GL thread, read from Qt thread."""

    WINDOW = 60          # frames in rolling average (~1 s at 60 fps)
    LOG_INTERVAL = 5.0   # seconds between CSV rows

    def __init__(self, log_path: pathlib.Path | None = None):
        self._samples: collections.deque[_FrameSample] = collections.deque(maxlen=self.WINDOW)
        self._lock = threading.Lock()
        self._log_path = log_path
        self._next_log = time.monotonic() + self.LOG_INTERVAL
        # Skip header if the file already exists from a previous session.
        self._header_written: bool = (
            log_path is not None and log_path.exists()
        )

    # ── GL-thread write ───────────────────────────────────────────────────────

    def record(
        self,
        fps: float,
        render_ms: float,
        step_ms: float,
        scene_ms: float,
        post_ms: float,
        blit_ms: float,
        effect: str = "",
    ) -> None:
        """Record one frame of timing data. Call from the GL thread each frame."""
        s = _FrameSample(fps, render_ms, step_ms, scene_ms, post_ms, blit_ms, effect)
        with self._lock:
            self._samples.append(s)
        self._maybe_log()

    # ── Qt-thread read ────────────────────────────────────────────────────────

    def snapshot(self) -> dict | None:
        """Return averaged metrics over the rolling window. Safe to call from any thread."""
        with self._lock:
            if not self._samples:
                return None
            samples = list(self._samples)

        n = len(samples)

        def avg(attr: str) -> float:
            return sum(getattr(s, attr) for s in samples) / n

        render = avg("render_ms")
        step   = avg("step_ms")
        scene  = avg("scene_ms")
        post   = avg("post_ms")
        blit   = avg("blit_ms")

        return {
            "fps":         avg("fps"),
            "render_ms":   render,
            "step_ms":     step,
            "scene_ms":    scene,
            "post_ms":     post,
            "blit_ms":     blit,
            "overhead_ms": max(0.0, render - step - scene - post - blit),
            "headroom_ms": max(0.0, TARGET_MS - render),
            "budget_pct":  min(100.0, render / TARGET_MS * 100.0),
            "effect":      samples[-1].effect,
        }

    # ── CSV logging (GL thread, every LOG_INTERVAL seconds) ──────────────────

    def _maybe_log(self) -> None:
        if self._log_path is None:
            return
        now = time.monotonic()
        if now < self._next_log:
            return
        self._next_log = now + self.LOG_INTERVAL

        snap = self.snapshot()
        if snap is None:
            return

        try:
            write_header = not self._header_written
            with open(self._log_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow([
                        "timestamp", "fps",
                        "render_ms", "step_ms", "scene_ms",
                        "post_ms", "blit_ms", "overhead_ms",
                        "headroom_ms", "budget_pct", "effect",
                    ])
                    self._header_written = True
                w.writerow([
                    time.strftime("%Y-%m-%dT%H:%M:%S"),
                    f"{snap['fps']:.1f}",
                    f"{snap['render_ms']:.2f}",
                    f"{snap['step_ms']:.2f}",
                    f"{snap['scene_ms']:.2f}",
                    f"{snap['post_ms']:.2f}",
                    f"{snap['blit_ms']:.2f}",
                    f"{snap['overhead_ms']:.2f}",
                    f"{snap['headroom_ms']:.2f}",
                    f"{snap['budget_pct']:.1f}",
                    snap["effect"],
                ])
        except OSError:
            pass
