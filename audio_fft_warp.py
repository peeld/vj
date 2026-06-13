"""
audio_fft_warp.py — audio-reactive GPU feedback visualizer.

Depends on warp_feedback.py (in the same folder).

Controls:
  Q / ESC  – quit
  Z / X    – zoom sensitivity ↓ / ↑
  R / T    – rotation sensitivity ↓ / ↑
  D / F    – decay ↓ / ↑

Requirements:
  pip install warp-lang sounddevice scipy numpy opencv-python
"""

import time
import numpy as np
import scipy.fft as fft_lib
import cv2
import warp as wp

from drawlib.warp_feedback import FeedbackLoop, FeedbackParams

# ──────────────────────────────────────────────
# sounddevice is optional; falls back to synth.
# ──────────────────────────────────────────────
try:
    import sounddevice as sd
    _AUDIO_AVAILABLE = True
except (OSError, ImportError):
    _AUDIO_AVAILABLE = False
    print("[warn] sounddevice / PortAudio not available — using synthetic audio")


# ═══════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════
WIDTH  = 1280
HEIGHT =  720

AUDIO_SAMPLE_RATE  = 44100
AUDIO_BLOCK_FRAMES = 2048
FFT_BINS           = AUDIO_BLOCK_FRAMES // 2

# FFT band bin ranges
BASS_LO, BASS_HI   =   0,  40
MID_LO,  MID_HI    =  40, 200
TREB_LO, TREB_HI   = 200, 600

# Spectrum bar appearance
BAR_ALPHA = 0.8   # blend factor when drawing bars into the buffer


# ═══════════════════════════════════════════════
#  Warp init
# ═══════════════════════════════════════════════
wp.init()
DEVICE = "cuda" if wp.get_cuda_device_count() > 0 else "cpu"
print(f"[warp] using device: {DEVICE}")

fft_gpu = wp.zeros(FFT_BINS, dtype=wp.float32, device=DEVICE)


# ═══════════════════════════════════════════════
#  Spectrum-bar draw kernel  (image generator)
# ═══════════════════════════════════════════════

@wp.kernel
def spectrum_kernel(
    curr:     wp.array(dtype=wp.float32),
    fft:      wp.array(dtype=wp.float32),
    w: int, h: int,
    num_bins: int,
    bar_alpha: float,
):
    """
    Draw one vertical FFT bar per column into ``curr``.
    Colour hue tracks frequency; brightness tracks amplitude.
    Runs *after* the feedback step so bars are crisp before the
    next frame smears them into the flow.
    """
    tid = wp.tid()
    if tid >= w * h:
        return

    px = tid % w
    py = tid // w

    bin_idx = int(float(px) / float(w) * float(num_bins))
    if bin_idx >= num_bins:
        return

    amp   = fft[bin_idx]
    bar_h = int(amp * float(h))

    if h - 1 - py < bar_h:
        hue = float(bin_idx) / float(num_bins)   # 0..1
        h6  = hue * 6.0
        i   = int(h6)
        f   = h6 - float(i)
        q   = 1.0 - f

        r = float(0.0); g = float(0.0); b = float(0.0)
        if i == 0:
            r = 1.0; g = f;   b = 0.0
        elif i == 1:
            r = q;   g = 1.0; b = 0.0
        elif i == 2:
            r = 0.0; g = 1.0; b = f
        elif i == 3:
            r = 0.0; g = q;   b = 1.0
        elif i == 4:
            r = f;   g = 0.0; b = 1.0
        else:
            r = 1.0; g = 0.0; b = q

        bright = 0.5 + amp * 0.5
        base   = tid * 4
        curr[base]     = curr[base]     * (1.0 - bar_alpha) + r * bright * bar_alpha
        curr[base + 1] = curr[base + 1] * (1.0 - bar_alpha) + g * bright * bar_alpha
        curr[base + 2] = curr[base + 2] * (1.0 - bar_alpha) + b * bright * bar_alpha
        curr[base + 3] = 1.0


# ═══════════════════════════════════════════════
#  Audio helpers
# ═══════════════════════════════════════════════

_audio_ring = np.zeros(AUDIO_BLOCK_FRAMES, dtype=np.float32)
_synth_t    = 0.0

if _AUDIO_AVAILABLE:
    def _audio_callback(indata, frames, time_info, status):
        global _audio_ring
        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()
        n = min(len(mono), AUDIO_BLOCK_FRAMES)
        _audio_ring[:n] = mono[:n]

    _stream = sd.InputStream(
        samplerate=AUDIO_SAMPLE_RATE,
        blocksize=AUDIO_BLOCK_FRAMES,
        channels=1,
        dtype="float32",
        callback=_audio_callback,
    )
    _stream.start()


def get_fft_spectrum() -> np.ndarray:
    """Capture audio (or synthesise), return normalised FFT magnitudes."""
    global _synth_t
    if _AUDIO_AVAILABLE:
        signal = _audio_ring.copy()
    else:
        t      = np.linspace(_synth_t,
                             _synth_t + AUDIO_BLOCK_FRAMES / AUDIO_SAMPLE_RATE,
                             AUDIO_BLOCK_FRAMES, endpoint=False)
        beat   = np.sin(2 * np.pi * 60   * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t))
        mid    = np.sin(2 * np.pi * 440  * t) * 0.3
        hi     = np.sin(2 * np.pi * 4000 * t) * 0.15
        signal = (beat + mid + hi).astype(np.float32)
        _synth_t += AUDIO_BLOCK_FRAMES / AUDIO_SAMPLE_RATE

    window   = np.hanning(len(signal)).astype(np.float32)
    spec     = np.abs(fft_lib.rfft(signal * window))[:FFT_BINS].astype(np.float32)
    spec     = np.log1p(spec)
    peak     = spec.max()
    if peak > 1e-6:
        spec /= peak
    return spec


def band_mean(spec: np.ndarray, lo: int, hi: int) -> float:
    hi = min(hi, len(spec))
    return float(np.mean(spec[lo:hi])) if lo < hi else 0.0


# ═══════════════════════════════════════════════
#  Main loop
# ═══════════════════════════════════════════════

def main():
    params = FeedbackParams()
    loop   = FeedbackLoop(WIDTH, HEIGHT, device=DEVICE, params=params)

    cv2.namedWindow("Warp FFT Feedback", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Warp FFT Feedback", WIDTH, HEIGHT)

    frame_count = 0
    t_start     = time.perf_counter()
    fps_display = 0.0

    print("Running — press Q or ESC to quit")
    print("  Z/X  zoom sensitivity   R/T  rotation sensitivity   D/F  decay")

    while True:
        # 1. Audio & FFT
        spec   = get_fft_spectrum()
        bass   = band_mean(spec, BASS_LO, BASS_HI)
        mid    = band_mean(spec, MID_LO,  MID_HI)
        treble = band_mean(spec, TREB_LO, TREB_HI)
        wp.copy(fft_gpu, wp.array(spec, dtype=wp.float32, device=DEVICE))

        # 2. Feedback pass (prev → curr)
        loop.step(bass, mid, treble, time_val=frame_count * 0.05)

        # 3. Draw spectrum bars into curr
        wp.launch(
            spectrum_kernel,
            dim=WIDTH * HEIGHT,
            inputs=[loop.curr, fft_gpu, WIDTH, HEIGHT, FFT_BINS, float(BAR_ALPHA)],
            device=DEVICE,
        )

        # 4. Display
        frame_bgr = loop.to_bgr()

        elapsed = time.perf_counter() - t_start
        if elapsed > 0:
            fps_display = frame_count / elapsed
        cv2.putText(frame_bgr,
                    f"FPS:{fps_display:.1f}  bass:{bass:.2f} mid:{mid:.2f} treb:{treble:.2f}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame_bgr,
                    f"zoom_sens:{params.zoom_sensitivity:.3f}  "
                    f"rot_sens:{params.rot_sensitivity:.4f}  "
                    f"decay:{params.decay:.3f}",
                    (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.imshow("Warp FFT Feedback", frame_bgr)

        # 5. Advance ping-pong
        loop.advance()
        frame_count += 1

        # 6. Key handling
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("z"):
            params.zoom_sensitivity = max(0.0, params.zoom_sensitivity - 0.005)
        elif key == ord("x"):
            params.zoom_sensitivity = min(0.3, params.zoom_sensitivity + 0.005)
        elif key == ord("r"):
            params.rot_sensitivity = max(0.0, params.rot_sensitivity - 0.001)
        elif key == ord("t"):
            params.rot_sensitivity = min(0.1, params.rot_sensitivity + 0.001)
        elif key == ord("d"):
            params.decay = max(0.80, params.decay - 0.005)
        elif key == ord("f"):
            params.decay = min(0.999, params.decay + 0.005)

    cv2.destroyAllWindows()
    if _AUDIO_AVAILABLE:
        _stream.stop()
        _stream.close()

    print(f"Done — {frame_count} frames in {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
