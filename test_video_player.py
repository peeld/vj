"""
Unit tests for VideoPlayer (Step 2 of video_plan.md).
No GL context or real video file required — av.open is mocked.
"""
import queue
import threading
import time
from fractions import Fraction
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

import video_player as vp_module
from video_player import VideoPlayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

W, H, FPS, DUR = 320, 240, 30.0, 10.0


def _make_rgba(val=128, w=W, h=H):
    arr = np.full((h, w, 4), val, dtype=np.uint8)
    return arr


def _make_mock_frame(pts=0, val=128, w=W, h=H):
    f = MagicMock()
    f.to_ndarray.return_value = _make_rgba(val, w, h)
    f.pts = pts
    return f


def _make_mock_container(n_frames=3, width=W, height=H, fps=FPS, duration=DUR,
                          frame_val=128):
    """Return a mock av container usable as a context manager."""
    stream = MagicMock()
    stream.codec_context.width = width
    stream.codec_context.height = height
    stream.average_rate = Fraction(int(fps))
    # duration in time_base units; time_base = 1/90000
    stream.time_base = Fraction(1, 90000)
    stream.duration = int(duration * 90000)

    frames = [_make_mock_frame(pts=i * (90000 // int(fps)), val=frame_val,
                               w=width, h=height)
              for i in range(n_frames)]

    container = MagicMock()
    container.__enter__ = MagicMock(return_value=container)
    container.__exit__ = MagicMock(return_value=False)
    container.streams.video = [stream]
    container.decode.return_value = iter(frames)

    return container, stream


# ---------------------------------------------------------------------------
# Metadata tests (open reads width/height/fps/duration correctly)
# ---------------------------------------------------------------------------

class TestMetadata:
    def test_open_reads_width_height(self):
        container, _ = _make_mock_container(width=1920, height=1080)
        with patch("av.open", return_value=container):
            p = VideoPlayer("fake.mp4", loop=False)
        p.close()
        assert p.width == 1920
        assert p.height == 1080

    def test_open_reads_fps(self):
        container, _ = _make_mock_container(fps=24.0)
        with patch("av.open", return_value=container):
            p = VideoPlayer("fake.mp4", loop=False)
        p.close()
        assert p.fps == pytest.approx(24.0)

    def test_open_reads_duration(self):
        container, _ = _make_mock_container(duration=5.0)
        with patch("av.open", return_value=container):
            p = VideoPlayer("fake.mp4", loop=False)
        p.close()
        assert p.duration == pytest.approx(5.0, abs=0.02)

    def test_defaults_before_open(self):
        p = VideoPlayer(loop=False)
        assert p.width == 0
        assert p.height == 0
        assert p.fps == 30.0
        assert p.duration == 0.0
        assert p.position == 0.0

    def test_fps_fallback_when_average_rate_none(self):
        container, stream = _make_mock_container()
        stream.average_rate = None
        with patch("av.open", return_value=container):
            p = VideoPlayer("fake.mp4", loop=False)
        p.close()
        assert p.fps == 30.0


# ---------------------------------------------------------------------------
# Play / pause tests
# ---------------------------------------------------------------------------

class TestPlayPause:
    def _player_no_decode(self):
        """Player with decode thread that blocks immediately on stop_event."""
        container, _ = _make_mock_container(n_frames=0)
        with patch("av.open", return_value=container):
            p = VideoPlayer("fake.mp4", loop=False)
        return p

    def test_starts_playing(self):
        p = self._player_no_decode()
        assert p.playing is True
        p.close()

    def test_pause_sets_not_playing(self):
        p = self._player_no_decode()
        p.pause()
        assert p.playing is False
        p.play()
        p.close()

    def test_play_after_pause(self):
        p = self._player_no_decode()
        p.pause()
        p.play()
        assert p.playing is True
        p.close()

    def test_pause_is_idempotent(self):
        p = self._player_no_decode()
        p.pause()
        p.pause()
        assert p.playing is False
        p.play()
        p.close()


# ---------------------------------------------------------------------------
# Frame delivery tests
# ---------------------------------------------------------------------------

class TestFrameDelivery:
    def test_get_current_frame_none_before_any_frame(self):
        """No frame has been decoded yet → returns None."""
        # Use an empty frame list so the decode thread produces nothing
        container, _ = _make_mock_container(n_frames=0)
        with patch("av.open", return_value=container), \
             patch("time.sleep"):
            p = VideoPlayer("fake.mp4", loop=False)
            # Give thread no time to run; queue is empty
            assert p.get_current_frame() is None
            p.close()

    def test_get_current_frame_returns_rgba_array(self):
        """After decode thread runs, get_current_frame returns (H,W,4) uint8."""
        container, _ = _make_mock_container(n_frames=3, frame_val=200)
        with patch("av.open", return_value=container), \
             patch("time.sleep"):
            p = VideoPlayer("fake.mp4", loop=False)
            p._thread.join(timeout=1.0)   # wait for decode to finish
            frame = p.get_current_frame()

        assert frame is not None
        assert frame.shape == (H, W, 4)
        assert frame.dtype == np.uint8
        assert frame[0, 0, 0] == 200

    def test_get_current_frame_drains_to_latest(self):
        """Calling get_current_frame returns the most recent frame (drains queue)."""
        # Two frames with different fill values; latest should win
        val_first, val_last = 10, 20
        frames = [_make_mock_frame(pts=i, val=val_first if i == 0 else val_last)
                  for i in range(2)]
        container, stream = _make_mock_container(n_frames=0)
        container.decode.return_value = iter(frames)

        with patch("av.open", return_value=container), \
             patch("time.sleep"):
            p = VideoPlayer("fake.mp4", loop=False)
            p._thread.join(timeout=1.0)
            frame = p.get_current_frame()

        assert frame is not None
        assert frame[0, 0, 0] == val_last

    def test_position_updated_from_pts(self):
        """_position should be updated after a frame with pts is decoded."""
        fps_int = 30
        tb = Fraction(1, 90000)
        pts_val = 3000  # 3000/90000 = 1/30 s
        frame = _make_mock_frame(pts=pts_val)
        container, stream = _make_mock_container(n_frames=0)
        stream.time_base = tb
        container.decode.return_value = iter([frame])

        with patch("av.open", return_value=container), \
             patch("time.sleep"):
            p = VideoPlayer("fake.mp4", loop=False)
            p._thread.join(timeout=1.0)

        assert p.position == pytest.approx(float(pts_val * tb), abs=1e-6)
        p.close()


# ---------------------------------------------------------------------------
# Close / cleanup tests
# ---------------------------------------------------------------------------

class TestClose:
    def test_close_stops_thread(self):
        container, _ = _make_mock_container(n_frames=5)
        with patch("av.open", return_value=container), \
             patch("time.sleep"):
            p = VideoPlayer("fake.mp4", loop=False)
            assert p._thread is not None
            p.close()
        assert p._thread is None

    def test_close_drains_frame_queue(self):
        container, _ = _make_mock_container(n_frames=3)
        with patch("av.open", return_value=container), \
             patch("time.sleep"):
            p = VideoPlayer("fake.mp4", loop=False)
            p._thread.join(timeout=1.0)
            assert not p._frame_queue.empty()
            p.close()
        assert p._frame_queue.empty()

    def test_close_is_idempotent(self):
        container, _ = _make_mock_container(n_frames=0)
        with patch("av.open", return_value=container):
            p = VideoPlayer("fake.mp4", loop=False)
            p.close()
            p.close()   # must not raise

    def test_close_without_open(self):
        p = VideoPlayer(loop=False)
        p.close()   # must not raise
        assert p._thread is None


# ---------------------------------------------------------------------------
# No-loop behavior
# ---------------------------------------------------------------------------

class TestLoop:
    def test_no_loop_does_not_reopen(self):
        container, _ = _make_mock_container(n_frames=1)
        open_calls = []

        def fake_av_open(path, **kw):
            open_calls.append(path)
            return container

        with patch("av.open", side_effect=fake_av_open), \
             patch("time.sleep"):
            p = VideoPlayer("fake.mp4", loop=False)
            p._thread.join(timeout=1.0)
            p.close()

        # Two calls: one for metadata in open(), one inside _decode_loop
        assert len(open_calls) == 2
