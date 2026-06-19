import threading
import queue
import time
import numpy as np
import av


class VideoPlayer:
    def __init__(self, path=None, loop=True):
        self._loop = loop
        self._frame_queue = queue.Queue(maxsize=4)
        self._current_frame = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._paused.set()          # starts playing
        self._thread = None
        self._path = None
        self._width = 0
        self._height = 0
        self._fps = 30.0
        self._duration = 0.0
        self._position = 0.0
        if path:
            self.open(path)

    def open(self, path):
        self.close()
        self._path = path
        with av.open(path) as container:
            stream = container.streams.video[0]
            self._width = stream.codec_context.width
            self._height = stream.codec_context.height
            self._fps = float(stream.average_rate or 30)
            if stream.duration and stream.time_base:
                self._duration = float(stream.duration * stream.time_base)
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._decode_loop, daemon=True)
        self._thread.start()

    def close(self):
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._paused.set()      # unblock if paused
            self._thread.join(timeout=2.0)
        self._thread = None
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break

    def play(self):
        self._paused.set()

    def pause(self):
        self._paused.clear()

    def seek(self, t: float):
        # Reopen file at offset t — simpler and thread-safe vs in-flight seek
        path = self._path
        self.close()
        self._position = t
        self.open(path)

    @property
    def width(self): return self._width

    @property
    def height(self): return self._height

    @property
    def fps(self): return self._fps

    @property
    def duration(self): return self._duration

    @property
    def position(self): return self._position

    @property
    def playing(self): return self._paused.is_set()

    def get_current_frame(self):
        """Drain queue to latest frame. Thread-safe."""
        latest = None
        while True:
            try:
                latest = self._frame_queue.get_nowait()
            except queue.Empty:
                break
        if latest is not None:
            with self._lock:
                self._current_frame = latest
        return self._current_frame

    def _decode_loop(self):
        frame_interval = 1.0 / max(self._fps, 1.0)
        while True:
            next_pts = time.perf_counter()
            try:
                with av.open(self._path) as container:
                    stream = container.streams.video[0]
                    stream.thread_type = "AUTO"
                    for frame in container.decode(video=0):
                        if self._stop_event.is_set():
                            return
                        self._paused.wait()

                        rgba = frame.to_ndarray(format="rgba")
                        try:
                            self._frame_queue.put(rgba, timeout=0.1)
                        except queue.Full:
                            pass

                        if frame.pts is not None and stream.time_base:
                            self._position = float(frame.pts * stream.time_base)

                        now = time.perf_counter()
                        sleep_t = next_pts - now
                        if sleep_t > 0:
                            time.sleep(sleep_t)
                        next_pts += frame_interval

            except Exception:
                pass

            if not self._loop or self._stop_event.is_set():
                return
            self._position = 0.0
