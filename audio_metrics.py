"""
audio_metrics.py — Real-time audio → 0-1 visual curve values
Uses VB-Cable (or any loopback device) as input.

Metrics produced each frame:
  sub_bass    20–80 Hz      rumble / sub
  bass        80–250 Hz     kick body, bass guitar
  low_mid     250–500 Hz    warmth, mud
  mid         500–2000 Hz   vocals, snare body
  high_mid    2000–8000 Hz  presence, attack
  treble      8000–20000 Hz air, cymbals
  energy      overall RMS loudness
  brightness  spectral centroid (low=dark, high=bright)
  flux        spectral flux — how much spectrum changed
  kick        kick drum trigger (bass transient onset)
  onset       general onset strength (any hit/transient)

All values are 0.0–1.0 with envelope following:
  - fast attack (~5ms)  — rises immediately on loud signal
  - slower release       — decays smoothly, good for visuals

Run this file directly to see live printed values.
Import AudioAnalyzer to use in your own project.
"""

import numpy as np
import sounddevice as sd
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable

# ── Config ───────────────────────────────────────────────────────────────────

SAMPLE_RATE   = 44100
BUFFER_SIZE   = 1024        # samples per frame (~23ms at 44100)
CHANNELS      = 2

# Envelope follower time constants (seconds)
ATTACK_TIME   = 0.005       # 5ms — almost instant rise
RELEASE_TIME  = 0.15        # 150ms — smooth decay

# Kick detection
KICK_THRESHOLD       = 0.35  # minimum bass energy to consider a kick
KICK_ONSET_RATIO     = 2.5   # must be this many × the recent average
KICK_COOLDOWN_SEC    = 0.08  # min time between kicks (avoid double-triggers)
KICK_PULSE_DECAY     = 0.12  # seconds to decay kick pulse back to 0

# Onset detection (general transient)
ONSET_FLUX_THRESHOLD = 0.2   # spectral flux level to trigger onset
ONSET_PULSE_DECAY    = 0.08

# Frequency band boundaries (Hz)
BANDS = {
    "sub_bass":  (20,   80),
    "bass":      (80,   250),
    "low_mid":   (250,  500),
    "mid":       (500,  2000),
    "high_mid":  (2000, 8000),
    "treble":    (8000, 20000),
}

# Normalization reference RMS (tune to your typical signal level)
# Lower = more sensitive. Roughly: quiet=0.01, medium=0.05, loud=0.1
NORM_REF = 0.04


# ── Data ─────────────────────────────────────────────────────────────────────

@dataclass
class AudioMetrics:
    # Frequency bands
    sub_bass:   float = 0.0
    bass:       float = 0.0
    low_mid:    float = 0.0
    mid:        float = 0.0
    high_mid:   float = 0.0
    treble:     float = 0.0
    # Derived
    energy:     float = 0.0   # overall RMS
    brightness: float = 0.0   # spectral centroid
    flux:       float = 0.0   # spectral flux
    # Transient triggers
    kick:       float = 0.0   # kick drum pulse
    onset:      float = 0.0   # general onset pulse

    def as_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        bar = lambda v: "█" * int(v * 20) + "░" * (20 - int(v * 20))
        lines = [
            f"  sub_bass  [{bar(self.sub_bass)}] {self.sub_bass:.3f}",
            f"  bass      [{bar(self.bass)}] {self.bass:.3f}",
            f"  low_mid   [{bar(self.low_mid)}] {self.low_mid:.3f}",
            f"  mid       [{bar(self.mid)}] {self.mid:.3f}",
            f"  high_mid  [{bar(self.high_mid)}] {self.high_mid:.3f}",
            f"  treble    [{bar(self.treble)}] {self.treble:.3f}",
            f"  energy    [{bar(self.energy)}] {self.energy:.3f}",
            f"  brightness[{bar(self.brightness)}] {self.brightness:.3f}",
            f"  flux      [{bar(self.flux)}] {self.flux:.3f}",
            f"  kick      [{bar(self.kick)}] {self.kick:.3f}",
            f"  onset     [{bar(self.onset)}] {self.onset:.3f}",
        ]
        return "\n".join(lines)


# ── Envelope follower ─────────────────────────────────────────────────────────

class Envelope:
    """Exponential attack/release smoother. Tracks a 0-1 signal."""
    def __init__(self, attack: float, release: float, sample_rate: int = 60):
        # Convert time constants to per-frame coefficients
        self.a_att = 1.0 - np.exp(-1.0 / (attack  * sample_rate))
        self.a_rel = 1.0 - np.exp(-1.0 / (release * sample_rate))
        self.value = 0.0

    def process(self, target: float) -> float:
        coef = self.a_att if target > self.value else self.a_rel
        self.value += coef * (target - self.value)
        return self.value


# ── Main analyser ─────────────────────────────────────────────────────────────

class AudioAnalyzer:
    """
    Captures audio from `device` and calls `on_frame(metrics)` each buffer.

    Usage:
        def on_frame(m: AudioMetrics):
            print(m.kick, m.brightness)

        a = AudioAnalyzer(device="CABLE Output", on_frame=on_frame)
        a.start()
        time.sleep(30)
        a.stop()
    """

    def __init__(
        self,
        device: Optional[str | int] = None,
        on_frame: Optional[Callable[[AudioMetrics], None]] = None,
        sample_rate: int = SAMPLE_RATE,
        buffer_size: int = BUFFER_SIZE,
    ):
        self.device      = device
        self.on_frame    = on_frame
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.metrics     = AudioMetrics()
        self._stream     = None
        self._running    = False

        # Frame rate estimate for envelope coefficients
        fps = sample_rate / buffer_size

        # Per-band envelope followers
        self._env = {k: Envelope(ATTACK_TIME, RELEASE_TIME, fps) for k in BANDS}
        self._env["energy"]     = Envelope(ATTACK_TIME, RELEASE_TIME, fps)
        self._env["brightness"] = Envelope(0.02, 0.3, fps)
        self._env["flux"]       = Envelope(ATTACK_TIME, RELEASE_TIME * 0.5, fps)

        # State for kick detection
        self._bass_history    = np.zeros(30)   # rolling window
        self._kick_cooldown   = 0.0
        self._kick_pulse      = 0.0
        self._last_frame_time = time.perf_counter()

        # State for onset detection
        self._onset_pulse     = 0.0
        self._prev_spectrum   = None

        # Frequency bin helpers (computed once)
        freqs = np.fft.rfftfreq(buffer_size, 1.0 / sample_rate)
        self._freqs = freqs
        self._band_masks = {
            name: (freqs >= lo) & (freqs < hi)
            for name, (lo, hi) in BANDS.items()
        }

    # ── Audio callback (called by sounddevice on audio thread) ───────────────

    def _callback(self, indata, frames, time_info, status):
        # Mix to mono
        mono = indata.mean(axis=1) if indata.ndim > 1 else indata[:, 0]

        # FFT magnitude spectrum (positive frequencies only)
        window    = np.hanning(len(mono))
        spectrum  = np.abs(np.fft.rfft(mono * window))
        spectrum /= (len(mono) / 2)   # normalise by window length

        now = time.perf_counter()
        dt  = now - self._last_frame_time
        self._last_frame_time = now

        m = self.metrics

        # ── Frequency bands ──────────────────────────────────────────────────
        for name, mask in self._band_masks.items():
            raw = float(np.sqrt(np.mean(spectrum[mask] ** 2))) if mask.any() else 0.0
            clamped = min(raw / NORM_REF, 1.0)
            setattr(m, name, self._env[name].process(clamped))

        # ── Overall energy (RMS of waveform) ─────────────────────────────────
        rms = float(np.sqrt(np.mean(mono ** 2)))
        m.energy = self._env["energy"].process(min(rms / NORM_REF, 1.0))

        # ── Brightness (spectral centroid, normalised to 0-1 over 0–10kHz) ──
        total = spectrum.sum()
        if total > 1e-10:
            centroid = float(np.dot(self._freqs, spectrum) / total)
        else:
            centroid = 0.0
        brightness_norm = min(centroid / 10000.0, 1.0)
        m.brightness = self._env["brightness"].process(brightness_norm)

        # ── Spectral flux (positive difference from last frame) ──────────────
        if self._prev_spectrum is not None and len(spectrum) == len(self._prev_spectrum):
            diff = spectrum - self._prev_spectrum
            flux_raw = float(np.sqrt(np.mean(np.maximum(diff, 0) ** 2)))
            m.flux = self._env["flux"].process(min(flux_raw / NORM_REF, 1.0))
        self._prev_spectrum = spectrum.copy()

        # ── Kick drum detection ───────────────────────────────────────────────
        bass_energy = m.bass  # already enveloped 0-1

        # Shift rolling history and add current value
        self._bass_history = np.roll(self._bass_history, 1)
        self._bass_history[0] = bass_energy
        bass_avg = self._bass_history[1:].mean()  # average of recent frames (excl. current)

        self._kick_cooldown = max(0.0, self._kick_cooldown - dt)

        if (
            bass_energy > KICK_THRESHOLD
            and bass_avg > 0.01
            and bass_energy > bass_avg * KICK_ONSET_RATIO
            and self._kick_cooldown == 0.0
        ):
            self._kick_pulse   = 1.0
            self._kick_cooldown = KICK_COOLDOWN_SEC

        # Decay the kick pulse
        self._kick_pulse = max(0.0, self._kick_pulse - dt / KICK_PULSE_DECAY)
        m.kick = self._kick_pulse

        # ── General onset (spectral flux threshold pulse) ────────────────────
        if m.flux > ONSET_FLUX_THRESHOLD:
            self._onset_pulse = 1.0
        self._onset_pulse = max(0.0, self._onset_pulse - dt / ONSET_PULSE_DECAY)
        m.onset = self._onset_pulse

        # ── Deliver to caller ─────────────────────────────────────────────────
        if self.on_frame:
            self.on_frame(m)

    # ── Control ──────────────────────────────────────────────────────────────

    def start(self):
        self._stream = sd.InputStream(
            device=self.device,
            channels=CHANNELS,
            samplerate=self.sample_rate,
            blocksize=self.buffer_size,
            dtype="float32",
            callback=self._callback,
        )
        self._stream.start()
        self._running = True

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
        self._running = False


# ── Device listing helper ─────────────────────────────────────────────────────

def list_devices():
    """Print all input devices — find your VB-Cable device name here."""
    print("\nAvailable INPUT devices:")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            print(f"  [{i:2d}] {dev['name']}")
    print()


# ── CLI demo ──────────────────────────────────────────────────────────────────

def main():
    list_devices()

    # Set this to your VB-Cable device name or index.
    # Common names: "CABLE Output (VB-Audio Virtual Cable)"
    # You can also pass an integer index from the list above.
    DEVICE = "CABLE Output (VB-Audio Virtual Cable)"

    print(f"Listening on: {DEVICE!r}")
    print("Press Ctrl+C to stop.\n")

    analyzer = AudioAnalyzer(device=DEVICE)
    analyzer.start()

    try:
        while True:
            m = analyzer.metrics
            # Clear terminal and print all metrics
            print("\033[H\033[J", end="")   # ANSI clear screen
            print(f"  {'─'*48}")
            print(f"  AUDIO METRICS  (Ctrl+C to quit)")
            print(f"  {'─'*48}")
            print(m)
            print(f"  {'─'*48}")
            time.sleep(1 / 30)  # ~30fps display refresh
    except KeyboardInterrupt:
        pass
    finally:
        analyzer.stop()
        print("\nStopped.")


if __name__ == "__main__":
    main()
