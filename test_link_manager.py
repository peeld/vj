"""
test_link_manager.py — Unit tests for link_manager.py (Phases 2 & 3).

Run with:
    python -m pytest test_link_manager.py -v
or:
    python test_link_manager.py
"""
import time
import unittest

from link_manager import (
    _flat_to_ns,
    ADSREnvelope,
    EnvelopeDef,
    EVAL_MATH_NS,
    LFO,
    LFODef,
    LinkManager,
    SignalLink,
    SmoothHelper,
    SourceRegistry,
)


class MockPM:
    def __init__(self):
        self.values = {}

    def set(self, key, value):
        self.values[key] = value


# ─────────────────────────────────────────────────────────────────────────────
#  _flat_to_ns
# ─────────────────────────────────────────────────────────────────────────────

class TestFlatToNs(unittest.TestCase):

    def test_empty(self):
        self.assertEqual(_flat_to_ns({}), {})

    def test_no_dots_passed_through(self):
        result = _flat_to_ns({"dt": 0.016, "x": 1.0})
        self.assertAlmostEqual(result["dt"], 0.016)
        self.assertAlmostEqual(result["x"], 1.0)

    def test_dotted_keys_become_namespace_attributes(self):
        result = _flat_to_ns({"audio.bass": 0.5, "audio.mid": 0.3})
        ns = result["audio"]
        self.assertAlmostEqual(ns.bass, 0.5)
        self.assertAlmostEqual(ns.mid,  0.3)

    def test_multiple_prefixes_are_separate_namespaces(self):
        result = _flat_to_ns({"audio.bass": 0.5, "midi.cc7": 0.8, "clock.t": 42.0})
        self.assertAlmostEqual(result["audio"].bass, 0.5)
        self.assertAlmostEqual(result["midi"].cc7,   0.8)
        self.assertAlmostEqual(result["clock"].t,    42.0)

    def test_mixed_dotted_and_plain(self):
        result = _flat_to_ns({"audio.bass": 0.5, "dt": 0.016})
        self.assertAlmostEqual(result["audio"].bass, 0.5)
        self.assertAlmostEqual(result["dt"], 0.016)

    def test_attr_access_in_eval(self):
        ns = _flat_to_ns({"audio.bass": 0.75})
        value = eval("audio.bass", {"__builtins__": {}}, ns)
        self.assertAlmostEqual(value, 0.75)

    def test_underscore_attr_names(self):
        # midi.note36_vel, audio.bass_smooth etc. are valid attribute names
        result = _flat_to_ns({"midi.note36_vel": 0.9, "audio.bass_smooth": 0.4})
        self.assertAlmostEqual(result["midi"].note36_vel,    0.9)
        self.assertAlmostEqual(result["audio"].bass_smooth,  0.4)


# ─────────────────────────────────────────────────────────────────────────────
#  SourceRegistry
# ─────────────────────────────────────────────────────────────────────────────

class TestSourceRegistry(unittest.TestCase):

    def setUp(self):
        self.reg = SourceRegistry()

    def test_update_and_snapshot(self):
        self.reg.update("audio.bass", 0.6)
        snap = self.reg.snapshot()
        self.assertAlmostEqual(snap["audio.bass"], 0.6)

    def test_snapshot_is_independent_copy(self):
        self.reg.update("audio.bass", 0.6)
        snap = self.reg.snapshot()
        snap["audio.bass"] = 99.0
        snap2 = self.reg.snapshot()
        self.assertAlmostEqual(snap2["audio.bass"], 0.6)

    def test_smooth_first_call_initialised_to_value(self):
        # No jump: smooth value on first update must equal the input value.
        self.reg.update("audio.bass", 0.8)
        snap = self.reg.snapshot()
        self.assertAlmostEqual(snap["audio.bass_smooth"], 0.8)

    def test_smooth_moves_toward_target(self):
        self.reg.update("audio.bass", 0.0)
        time.sleep(0.05)   # give the low-pass filter real elapsed time to act
        self.reg.update("audio.bass", 1.0)
        snap = self.reg.snapshot()
        # Must be between previous (0) and target (1), exclusive.
        self.assertGreater(snap["audio.bass_smooth"], 0.0)
        self.assertLess(snap["audio.bass_smooth"],    1.0)

    def test_peak_first_call_equals_value(self):
        self.reg.update("audio.bass", 0.7)
        snap = self.reg.snapshot()
        self.assertAlmostEqual(snap["audio.bass_peak"], 0.7)

    def test_peak_tracks_maximum(self):
        self.reg.update("audio.bass", 0.3)
        self.reg.update("audio.bass", 0.9)
        snap = self.reg.snapshot()
        self.assertGreaterEqual(snap["audio.bass_peak"], 0.9)

    def test_peak_decays_when_signal_drops(self):
        self.reg.update("audio.bass", 1.0)
        for _ in range(30):
            self.reg.update("audio.bass", 0.0)
        snap = self.reg.snapshot()
        self.assertLess(snap["audio.bass_peak"], 1.0)
        self.assertGreaterEqual(snap["audio.bass_peak"], 0.0)

    def test_derived_keys_present(self):
        self.reg.update("midi.cc7", 0.5)
        snap = self.reg.snapshot()
        self.assertIn("midi.cc7",        snap)
        self.assertIn("midi.cc7_smooth", snap)
        self.assertIn("midi.cc7_peak",   snap)

    def test_source_keys_sorted(self):
        self.reg.update("midi.cc7",   0.5)
        self.reg.update("audio.bass", 0.3)
        keys = self.reg.source_keys()
        self.assertEqual(keys, sorted(keys))


# ─────────────────────────────────────────────────────────────────────────────
#  SmoothHelper
# ─────────────────────────────────────────────────────────────────────────────

class TestSmoothHelper(unittest.TestCase):

    def test_first_call_returns_x(self):
        # prev = x on first call → out = x regardless of tau/dt.
        sm = SmoothHelper()
        sm.dt = 0.016
        sm.reset()
        out = sm(0.7, 0.1)
        self.assertAlmostEqual(out, 0.7)

    def test_converges_toward_target_over_frames(self):
        sm = SmoothHelper()
        sm.dt = 0.1   # 100ms per frame, tau=0.1 → converges in ~5 frames
        val = 0.0
        for _ in range(50):
            sm.reset()
            val = sm(1.0, 0.1)
        self.assertGreater(val, 0.99)

    def test_two_calls_per_frame_get_independent_slots(self):
        sm = SmoothHelper()
        sm.dt = 10.0  # large dt → alpha ≈ 1 (near-instant tracking)

        # Frame 1: initialise slots 0 and 1 at 0.2 and 0.8.
        sm.reset()
        sm(0.2, 0.001)
        sm(0.8, 0.001)

        # Frame 2: drive slot 0 toward 1.0, slot 1 toward 0.0.
        sm.reset()
        out0 = sm(1.0, 0.001)
        out1 = sm(0.0, 0.001)

        self.assertGreater(out0, 0.99)  # slot 0 tracked from 0.2 → 1.0
        self.assertLess(out1,    0.01)  # slot 1 tracked from 0.8 → 0.0

    def test_slot_indices_stable_across_frames(self):
        sm = SmoothHelper()
        sm.dt = 0.016
        # Slot 1 sees 0.9 in frame 1; must still see its own state in frame 2.
        sm.reset(); sm(0.5, 0.1); sm(0.9, 0.1)
        sm.reset(); sm(0.5, 0.1); out = sm(0.9, 0.1)
        # Slot 1 was initialised to 0.9 and is fed 0.9 again → stays near 0.9.
        self.assertAlmostEqual(out, 0.9, places=3)

    def test_dt_zero_output_stays_at_previous(self):
        sm = SmoothHelper()
        sm.dt = 0.016
        sm.reset()
        sm(0.5, 0.1)   # initialise slot 0 to 0.5

        sm.dt = 0.0    # alpha = 1 - exp(0) = 0 → no movement
        sm.reset()
        out = sm(1.0, 0.1)
        self.assertAlmostEqual(out, 0.5)


# ─────────────────────────────────────────────────────────────────────────────
#  ADSREnvelope
# ─────────────────────────────────────────────────────────────────────────────

class TestADSREnvelope(unittest.TestCase):

    def test_idle_returns_zero(self):
        env = ADSREnvelope(attack=0.5, decay=0.5, sustain=0.5, release=0.5)
        self.assertAlmostEqual(env.tick(0.1), 0.0)
        self.assertAlmostEqual(env.tick(0.1), 0.0)

    def test_full_cycle(self):
        # attack=1.0, decay=1.0, sustain=0.5, release=1.0, peak=1.0, dt=0.5
        env = ADSREnvelope(attack=1.0, decay=1.0, sustain=0.5, release=1.0)
        env.trigger()

        # Attack: 0 → 1.0 over 1.0 s (two ticks of 0.5 s)
        self.assertAlmostEqual(env.tick(0.5), 0.5)   # halfway up
        self.assertAlmostEqual(env.tick(0.5), 1.0)   # peak; transitions to DECAY

        # Decay: 1.0 → 0.5 over 1.0 s
        self.assertAlmostEqual(env.tick(0.5), 0.75)  # 1.0 + (0.5-1.0)*0.5
        self.assertAlmostEqual(env.tick(0.5), 0.5)   # sustain level; transitions to SUSTAIN

        # Sustain: holds at 0.5
        self.assertAlmostEqual(env.tick(0.5), 0.5)
        self.assertAlmostEqual(env.tick(0.5), 0.5)

        env.gate_off()

        # Release: 0.5 → 0 over 1.0 s
        self.assertAlmostEqual(env.tick(0.5), 0.25)  # 0.5*(1 - 0.5)
        self.assertAlmostEqual(env.tick(0.5), 0.0)   # zero; transitions to IDLE

        # Back to IDLE
        self.assertAlmostEqual(env.tick(0.5), 0.0)
        self.assertEqual(env._state, ADSREnvelope._IDLE)

    def test_zero_attack_jumps_to_peak_immediately(self):
        env = ADSREnvelope(attack=0.0, decay=1.0, sustain=0.5, release=0.5)
        env.trigger()
        out = env.tick(0.1)
        self.assertAlmostEqual(out, 1.0)
        self.assertEqual(env._state, ADSREnvelope._DECAY)

    def test_zero_decay_jumps_to_sustain_immediately(self):
        env = ADSREnvelope(attack=0.0, decay=0.0, sustain=0.6, release=0.5)
        env.trigger()
        env.tick(0.1)          # ATTACK → DECAY (instant)
        out = env.tick(0.1)    # DECAY  → SUSTAIN (instant)
        self.assertAlmostEqual(out, 0.6)
        self.assertEqual(env._state, ADSREnvelope._SUSTAIN)

    def test_zero_release_transitions_to_idle_immediately(self):
        env = ADSREnvelope(attack=0.0, decay=0.0, sustain=0.5, release=0.0)
        env.trigger()
        env.tick(0.1)    # → DECAY
        env.tick(0.1)    # → SUSTAIN
        env.gate_off()
        out = env.tick(0.1)  # → IDLE
        self.assertAlmostEqual(out, 0.0)
        self.assertEqual(env._state, ADSREnvelope._IDLE)

    def test_retrigger_in_attack_restarts_from_zero(self):
        env = ADSREnvelope(attack=1.0, decay=0.5, sustain=0.5, release=0.5)
        env.trigger()
        env.tick(0.4)     # partway through attack (value ≈ 0.4)
        env.trigger()     # retrigger
        out = env.tick(0.1)
        self.assertAlmostEqual(out, 0.1)   # 0.1/1.0 of peak

    def test_retrigger_in_sustain_restarts(self):
        env = ADSREnvelope(attack=0.0, decay=0.0, sustain=0.5, release=0.5)
        env.trigger()
        env.tick(0.1); env.tick(0.1)   # → SUSTAIN
        self.assertEqual(env._state, ADSREnvelope._SUSTAIN)
        env.trigger()
        self.assertEqual(env._state, ADSREnvelope._ATTACK)

    def test_gate_off_in_idle_is_noop(self):
        env = ADSREnvelope(attack=0.5, decay=0.5, sustain=0.5, release=0.5)
        env.gate_off()
        self.assertEqual(env._state, ADSREnvelope._IDLE)
        self.assertAlmostEqual(env.tick(0.1), 0.0)

    def test_gate_off_in_attack_is_noop(self):
        env = ADSREnvelope(attack=1.0, decay=0.5, sustain=0.5, release=0.5)
        env.trigger()
        env.tick(0.3)
        self.assertEqual(env._state, ADSREnvelope._ATTACK)
        env.gate_off()
        self.assertEqual(env._state, ADSREnvelope._ATTACK)  # unchanged

    def test_gate_off_in_decay_is_noop(self):
        env = ADSREnvelope(attack=0.0, decay=1.0, sustain=0.5, release=0.5)
        env.trigger()
        env.tick(0.1)    # → DECAY
        self.assertEqual(env._state, ADSREnvelope._DECAY)
        env.gate_off()
        self.assertEqual(env._state, ADSREnvelope._DECAY)

    def test_custom_peak(self):
        env = ADSREnvelope(attack=1.0, decay=1.0, sustain=0.5, release=0.5, peak=0.5)
        env.trigger()
        out = env.tick(1.0)   # full attack → should reach peak=0.5
        self.assertAlmostEqual(out, 0.5)

    def test_zero_sustain_decay_reaches_zero(self):
        env = ADSREnvelope(attack=0.0, decay=1.0, sustain=0.0, release=0.5, peak=1.0)
        env.trigger()
        env.tick(0.1)                  # ATTACK → DECAY
        out = env.tick(0.5)            # halfway through decay: 1.0 + (0.0-1.0)*0.5
        self.assertAlmostEqual(out, 0.5)

    def test_output_clamped_to_unit_range(self):
        # sustain=2.0 is nonsense input; output must still be clamped to [0, 1].
        env = ADSREnvelope(attack=0.0, decay=0.0, sustain=2.0, release=0.5)
        env.trigger()
        env.tick(0.1); env.tick(0.1)   # → SUSTAIN at sustain=2.0
        out = env.tick(0.1)
        self.assertLessEqual(out,    1.0)
        self.assertGreaterEqual(out, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
#  LFO
# ─────────────────────────────────────────────────────────────────────────────

class TestLFO(unittest.TestCase):

    def _at_phase(self, shape, phase):
        """Return LFO output at a fixed phase (rate=0 so phase doesn't advance)."""
        lfo = LFO(shape=shape, rate_hz=0.0, phase=phase)
        return lfo.tick(1.0)

    def test_sine_quarter_phases(self):
        self.assertAlmostEqual(self._at_phase("sine", 0.00), 0.5, places=5)
        self.assertAlmostEqual(self._at_phase("sine", 0.25), 1.0, places=5)
        self.assertAlmostEqual(self._at_phase("sine", 0.50), 0.5, places=5)
        self.assertAlmostEqual(self._at_phase("sine", 0.75), 0.0, places=5)

    def test_saw_is_linear(self):
        self.assertAlmostEqual(self._at_phase("saw", 0.00),  0.00, places=5)
        self.assertAlmostEqual(self._at_phase("saw", 0.50),  0.50, places=5)
        self.assertAlmostEqual(self._at_phase("saw", 0.99),  0.99, places=5)

    def test_square_halves(self):
        self.assertAlmostEqual(self._at_phase("square", 0.00),  1.0)
        self.assertAlmostEqual(self._at_phase("square", 0.25),  1.0)
        self.assertAlmostEqual(self._at_phase("square", 0.50),  0.0)
        self.assertAlmostEqual(self._at_phase("square", 0.75),  0.0)

    def test_tri_shape(self):
        self.assertAlmostEqual(self._at_phase("tri", 0.00),  0.0, places=5)
        self.assertAlmostEqual(self._at_phase("tri", 0.25),  0.5, places=5)
        self.assertAlmostEqual(self._at_phase("tri", 0.50),  1.0, places=5)
        self.assertAlmostEqual(self._at_phase("tri", 0.75),  0.5, places=5)

    def test_phase_accumulates(self):
        lfo = LFO(shape="saw", rate_hz=1.0, phase=0.0)
        self.assertAlmostEqual(lfo.tick(0.25), 0.25, places=5)
        self.assertAlmostEqual(lfo.tick(0.25), 0.50, places=5)
        self.assertAlmostEqual(lfo.tick(0.25), 0.75, places=5)

    def test_phase_wraps(self):
        lfo = LFO(shape="saw", rate_hz=1.0, phase=0.8)
        lfo.tick(0.3)   # phase → (0.8 + 0.3) % 1 = 0.1
        out = lfo.tick(0.0)   # rate=1, dt=0 → phase stays at 0.1; saw = 0.1
        self.assertAlmostEqual(out, 0.1, places=5)

    def test_initial_phase_offset(self):
        lfo = LFO(shape="saw", rate_hz=0.0, phase=0.75)
        self.assertAlmostEqual(lfo.tick(1.0), 0.75, places=5)

    def test_unknown_shape_returns_zero(self):
        lfo = LFO(shape="blorp", rate_hz=1.0, phase=0.0)
        self.assertAlmostEqual(lfo.tick(0.1), 0.0)

    def test_all_shapes_stay_in_unit_range(self):
        for shape in ("sine", "saw", "square", "tri"):
            lfo = LFO(shape=shape, rate_hz=3.7, phase=0.13)
            for _ in range(120):
                out = lfo.tick(0.013)
                self.assertGreaterEqual(out, 0.0, f"{shape} went below 0")
                self.assertLessEqual(out,    1.0, f"{shape} went above 1")


# ─────────────────────────────────────────────────────────────────────────────
#  Dataclass round-trips
# ─────────────────────────────────────────────────────────────────────────────

class TestDataclassRoundtrips(unittest.TestCase):

    def test_signal_link(self):
        link = SignalLink(
            sink_key="feedback.decay",
            expression="lerp(0.98, 0.999, audio.bass)",
            enabled=True,
        )
        link2 = SignalLink.from_dict(link.to_dict())
        self.assertEqual(link2.sink_key,   link.sink_key)
        self.assertEqual(link2.expression, link.expression)
        self.assertTrue(link2.enabled)

    def test_signal_link_disabled(self):
        link = SignalLink(sink_key="x", expression="0.5", enabled=False)
        self.assertFalse(SignalLink.from_dict(link.to_dict()).enabled)

    def test_envelope_def(self):
        defn = EnvelopeDef(
            name="kick", trigger="audio.onset",
            attack=0.01, decay=0.1, sustain=0.0, release=0.2, peak=1.0,
            gate_off="midi.note36.off",
        )
        d2 = EnvelopeDef.from_dict(defn.to_dict())
        self.assertEqual(d2.name,    defn.name)
        self.assertEqual(d2.trigger, defn.trigger)
        self.assertEqual(d2.gate_off, defn.gate_off)
        self.assertAlmostEqual(d2.attack,  defn.attack)
        self.assertAlmostEqual(d2.decay,   defn.decay)
        self.assertAlmostEqual(d2.sustain, defn.sustain)
        self.assertAlmostEqual(d2.release, defn.release)
        self.assertAlmostEqual(d2.peak,    defn.peak)

    def test_envelope_def_no_gate_off(self):
        defn = EnvelopeDef(name="hit", trigger="audio.onset",
                           attack=0.0, decay=0.1, sustain=0.0, release=0.1)
        self.assertIsNone(EnvelopeDef.from_dict(defn.to_dict()).gate_off)

    def test_lfo_def(self):
        defn = LFODef(name="slow", shape="tri", rate_hz=0.25, phase=0.5)
        d2 = LFODef.from_dict(defn.to_dict())
        self.assertEqual(d2.name,  defn.name)
        self.assertEqual(d2.shape, defn.shape)
        self.assertAlmostEqual(d2.rate_hz, defn.rate_hz)
        self.assertAlmostEqual(d2.phase,   defn.phase)


# ─────────────────────────────────────────────────────────────────────────────
#  evaluate_links
# ─────────────────────────────────────────────────────────────────────────────

class TestEvaluateLinks(unittest.TestCase):

    def setUp(self):
        self.lm = LinkManager()
        self.pm = MockPM()

    def _src(self, key, value):
        self.lm.source_registry.update(key, value)

    def test_constant_expression(self):
        self.lm.add_link(SignalLink(sink_key="out", expression="0.5"))
        self.lm.evaluate_links(self.pm, 0.016)
        self.assertAlmostEqual(self.pm.values["out"], 0.5)

    def test_source_reference(self):
        self._src("audio.bass", 0.8)
        self.lm.add_link(SignalLink(sink_key="out", expression="audio.bass"))
        self.lm.evaluate_links(self.pm, 0.016)
        self.assertAlmostEqual(self.pm.values["out"], 0.8)

    def test_lerp_expression(self):
        self._src("audio.bass", 1.0)
        self.lm.add_link(SignalLink(sink_key="out", expression="lerp(0.0, 2.0, audio.bass)"))
        self.lm.evaluate_links(self.pm, 0.016)
        self.assertAlmostEqual(self.pm.values["out"], 2.0)

    def test_bad_expression_is_silent(self):
        self.lm.add_link(SignalLink(sink_key="out", expression="1 / 0"))
        self.lm.evaluate_links(self.pm, 0.016)   # must not raise
        self.assertNotIn("out", self.pm.values)

    def test_undefined_name_is_silent(self):
        self.lm.add_link(SignalLink(sink_key="out", expression="no_such_source"))
        self.lm.evaluate_links(self.pm, 0.016)
        self.assertNotIn("out", self.pm.values)

    def test_disabled_link_skipped(self):
        self.lm.add_link(SignalLink(sink_key="out", expression="1.0", enabled=False))
        self.lm.evaluate_links(self.pm, 0.016)
        self.assertNotIn("out", self.pm.values)

    def test_dt_available_in_expression(self):
        self.lm.add_link(SignalLink(sink_key="out", expression="dt"))
        self.lm.evaluate_links(self.pm, 0.033)
        self.assertAlmostEqual(self.pm.values["out"], 0.033)

    def test_const_namespace(self):
        self.lm._const_ns.gravity = 9.8
        self.lm.add_link(SignalLink(sink_key="out", expression="const.gravity"))
        self.lm.evaluate_links(self.pm, 0.016)
        self.assertAlmostEqual(self.pm.values["out"], 9.8)

    def test_smooth_initialises_to_x_on_first_call(self):
        self._src("audio.bass", 0.5)
        self.lm.add_link(SignalLink(sink_key="out", expression="smooth(audio.bass, 0.1)"))
        self.lm.evaluate_links(self.pm, 0.016)
        self.assertAlmostEqual(self.pm.values["out"], 0.5)

    def test_two_smooth_calls_independent(self):
        self._src("audio.bass", 1.0)
        self._src("audio.mid",  0.0)
        expr = "smooth(audio.bass, 0.01) - smooth(audio.mid, 0.01)"
        self.lm.add_link(SignalLink(sink_key="out", expression=expr))
        self.lm.evaluate_links(self.pm, 0.016)
        # Both initialise to their own x: 1.0 - 0.0 = 1.0
        self.assertAlmostEqual(self.pm.values["out"], 1.0)

    def test_multiple_links_independent(self):
        self._src("audio.bass", 0.3)
        self._src("audio.mid",  0.7)
        self.lm.add_link(SignalLink(sink_key="sink1", expression="audio.bass"))
        self.lm.add_link(SignalLink(sink_key="sink2", expression="audio.mid"))
        self.lm.evaluate_links(self.pm, 0.016)
        self.assertAlmostEqual(self.pm.values["sink1"], 0.3)
        self.assertAlmostEqual(self.pm.values["sink2"], 0.7)

    def test_remove_link(self):
        self.lm.add_link(SignalLink(sink_key="out", expression="1.0"))
        self.lm.remove_link("out")
        self.lm.evaluate_links(self.pm, 0.016)
        self.assertNotIn("out", self.pm.values)

    def test_disable_then_enable_link(self):
        self.lm.add_link(SignalLink(sink_key="out", expression="1.0"))
        self.lm.disable_link("out")
        self.lm.evaluate_links(self.pm, 0.016)
        self.assertNotIn("out", self.pm.values)

        self.lm.enable_link("out")
        self.lm.evaluate_links(self.pm, 0.016)
        self.assertAlmostEqual(self.pm.values["out"], 1.0)

    def test_smooth_state_persists_across_frames(self):
        # With small tau and large dt, smooth() converges toward target.
        self._src("audio.bass", 0.0)
        self.lm.add_link(SignalLink(sink_key="out", expression="smooth(audio.bass, 0.01)"))
        self.lm.evaluate_links(self.pm, 0.016)   # initialise slot at 0.0

        self._src("audio.bass", 1.0)
        self.lm.evaluate_links(self.pm, 1.0)     # large dt → should move substantially
        self.assertGreater(self.pm.values["out"], 0.5)


# ─────────────────────────────────────────────────────────────────────────────
#  LinkManager envelope / LFO integration
# ─────────────────────────────────────────────────────────────────────────────

class TestLinkManagerEnvelopeLFO(unittest.TestCase):

    def setUp(self):
        self.lm = LinkManager()

    def _make_instant_sustain_env(self, name, sustain):
        """Helper: create an envelope that reaches sustain in two zero-duration ticks."""
        defn = EnvelopeDef(name=name, trigger="x",
                           attack=0.0, decay=0.0, sustain=sustain, release=0.5)
        self.lm.add_envelope(defn)
        env = self.lm._envelopes[name]
        env.trigger()
        env.tick(0.1)   # ATTACK → DECAY
        env.tick(0.1)   # DECAY  → SUSTAIN
        return env

    def test_tick_envelopes_writes_registry(self):
        self._make_instant_sustain_env("kick", 0.5)
        self.lm.tick_envelopes(0.1)
        snap = self.lm.source_registry.snapshot()
        self.assertAlmostEqual(snap["env.kick"], 0.5)

    def test_tick_lfos_writes_registry(self):
        defn = LFODef(name="slow", shape="saw", rate_hz=0.0, phase=0.75)
        self.lm.add_lfo(defn)
        self.lm.tick_lfos(0.1)
        snap = self.lm.source_registry.snapshot()
        self.assertAlmostEqual(snap["lfo.slow"], 0.75)

    def test_env_source_usable_in_expression(self):
        pm = MockPM()
        self._make_instant_sustain_env("hit", 0.8)
        self.lm.tick_envelopes(0.1)   # writes env.hit = 0.8
        self.lm.add_link(SignalLink(sink_key="out", expression="env.hit"))
        self.lm.evaluate_links(pm, 0.016)
        self.assertAlmostEqual(pm.values["out"], 0.8)

    def test_lfo_source_usable_in_expression(self):
        pm = MockPM()
        defn = LFODef(name="wave", shape="saw", rate_hz=0.0, phase=0.6)
        self.lm.add_lfo(defn)
        self.lm.tick_lfos(0.1)        # writes lfo.wave = 0.6
        self.lm.add_link(SignalLink(sink_key="out", expression="lfo.wave"))
        self.lm.evaluate_links(pm, 0.016)
        self.assertAlmostEqual(pm.values["out"], 0.6)

    def test_remove_envelope(self):
        defn = EnvelopeDef(name="e1", trigger="x",
                           attack=0.0, decay=0.0, sustain=0.5, release=0.5)
        self.lm.add_envelope(defn)
        self.lm.remove_envelope("e1")
        self.assertNotIn("e1", self.lm._envelopes)
        self.assertEqual(self.lm._envelope_defs, [])

    def test_remove_lfo(self):
        defn = LFODef(name="l1", shape="sine", rate_hz=1.0, phase=0.0)
        self.lm.add_lfo(defn)
        self.lm.remove_lfo("l1")
        self.assertNotIn("l1", self.lm._lfos)
        self.assertEqual(self.lm._lfo_defs, [])

    def test_multiple_envelopes_independent(self):
        self._make_instant_sustain_env("a", 0.3)
        self._make_instant_sustain_env("b", 0.9)
        self.lm.tick_envelopes(0.1)
        snap = self.lm.source_registry.snapshot()
        self.assertAlmostEqual(snap["env.a"], 0.3)
        self.assertAlmostEqual(snap["env.b"], 0.9)


if __name__ == "__main__":
    unittest.main()
