"""
elements/kaleidoscope.py

Emits n-fold rotationally-symmetric shard groups that radiate away from the
X axis in the YZ plane and drift along it.  Viewed from the X axis they
compose into a kaleidoscope pattern.
"""

import colorsys
from dataclasses import dataclass

import numpy as np
import moderngl

from .base import DrawingElement, FrameContext, register_element_type, Prop

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOUND         = 1.0
N_SLOTS       = 20
N_SYM         = 8
LIFETIME      = 4.5
FADE_IN       = 0.30
FADE_OUT      = 1.60
EMIT_INTERVAL = 0.28
MAX_RADIUS    = 0.82
FILL_FACTOR   = 0.48   # fraction of sector filled by each shard (rest is gap)
HUE_SPREAD    = 0.30   # total hue walk across shards in one group


@dataclass
class ShardGroup:
    cx:         float
    birth:      float
    phase:      float
    speed:      float
    amplitude:  float
    spin:       float
    base_angle: float
    shard_rgb:  np.ndarray   # (n_sym, 3) precomputed per-shard colors


class KaleidoscopeDrawing(DrawingElement, section="kaleidoscope"):
    """
    Kaleidoscope shard emitter.  n-fold symmetric wedge shards radiate
    outward from the X axis in the YZ plane; looking down X reveals a
    kaleidoscope.
    """
    kind = "kaleidoscope"

    n_slots       = Prop("Slot Count",      int,    20,   4,   64,  1,
                         description="Max simultaneous shard groups (requires regen)")
    n_sym         = Prop("Fold Symmetry",   int,     8,   2,   32,  1,
                         description="Rotational fold count per group (requires regen)")
    lifetime      = Prop("Lifetime",        float,  4.5,  0.5, 16.0, 0.1,
                         description="Seconds each shard group lives")
    spin_speed    = Prop("Spin Speed",      float,  0.25, -4.0, 4.0, 0.05,
                         description="Global rotation speed around X axis (rad/s)")
    amplitude     = Prop("Drift Amplitude", float,  1.0,  0.0, 10.0, 0.05,
                         description="Global multiplier on sinusoidal X-axis drift")
    emit_interval = Prop("Emit Interval",   float,  0.28, 0.05, 5.0, 0.05,
                         description="Seconds between group spawns")

    def __init__(self, ctx: moderngl.Context, device=None, **kwargs):
        super().__init__()
        self._ctx  = ctx
        self._rng  = np.random.default_rng()
        self._pool: list[ShardGroup | None] = []
        self._spawn_timer = 0.0
        self._palette: list = []

        self.n_slots       = N_SLOTS
        self.n_sym         = N_SYM
        self.lifetime      = LIFETIME
        self.spin_speed    = 0.25
        self.amplitude     = 1.0
        self.emit_interval = EMIT_INTERVAL

        self._buf_n_sym   = 0
        self._buf_n_slots = 0
        self._geo_ready   = False

        self._prog = self._ctx.program(
            vertex_shader="""
                #version 330
                uniform mat4 mvp;
                in vec3 in_position;
                in vec4 in_color;
                out vec4 v_color;
                void main() {
                    gl_Position = mvp * vec4(in_position, 1.0);
                    v_color = in_color;
                }
            """,
            fragment_shader="""
                #version 330
                in vec4 v_color;
                out vec4 f_color;
                void main() { f_color = v_color; }
            """,
        )

        self.regen()

    # ------------------------------------------------------------------
    # DrawingElement interface
    # ------------------------------------------------------------------

    def set_palette(self, palette: list) -> None:
        if palette:
            self._palette = list(palette)

    def regen(self) -> None:
        self._alloc_buffers(self.n_sym, self.n_slots)

    def step(self, ctx: FrameContext) -> None:
        dt = ctx.frame_time
        t  = ctx.current_time

        for i, sg in enumerate(self._pool):
            if sg is not None and (t - sg.birth) >= self.lifetime:
                self._pool[i] = None

        if self.active:
            self._spawn_timer += dt
            while self._spawn_timer >= self.emit_interval:
                self._spawn_timer -= self.emit_interval
                self._spawn(t)
        else:
            self._spawn_timer = 0.0

    def draw(self, mvp: np.ndarray, ctx: FrameContext) -> None:
        self._update_geo(ctx.current_time)
        self._prog["mvp"].write(mvp.tobytes())
        self._ctx.blend_func = moderngl.ONE, moderngl.ONE
        self._vao.render(moderngl.TRIANGLES)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _alloc_buffers(self, n_sym: int, n_slots: int) -> None:
        n_verts = n_slots * n_sym * 4   # 4 verts per shard (one quad)
        n_idx   = n_slots * n_sym * 6   # 6 indices per shard (2 triangles)

        self._buf_n_sym   = n_sym
        self._buf_n_slots = n_slots
        self._pos_cpu = np.zeros((n_verts, 3), dtype=np.float32)
        self._col_cpu = np.zeros((n_verts, 4), dtype=np.float32)

        # Static index buffer — never changes after alloc
        idx = np.empty(n_idx, dtype=np.uint32)
        for s in range(n_slots):
            for si in range(n_sym):
                bv = (s * n_sym + si) * 4
                bi = (s * n_sym + si) * 6
                # quad: v0=inner-left, v1=inner-right, v2=outer-right, v3=outer-left
                idx[bi+0] = bv;     idx[bi+1] = bv+1; idx[bi+2] = bv+2
                idx[bi+3] = bv;     idx[bi+4] = bv+2; idx[bi+5] = bv+3

        if self._geo_ready:
            self._vao.release()
            self._pos_buf.release()
            self._col_buf.release()
            self._idx_buf.release()

        self._pos_buf = self._ctx.buffer(reserve=n_verts * 3 * 4)
        self._col_buf = self._ctx.buffer(reserve=n_verts * 4 * 4)
        self._idx_buf = self._ctx.buffer(idx.tobytes())
        self._vao = self._ctx.vertex_array(
            self._prog,
            [(self._pos_buf, "3f", "in_position"), (self._col_buf, "4f", "in_color")],
            self._idx_buf,
        )
        self._geo_ready = True
        self._pool = [None] * n_slots

    def _spawn(self, t: float) -> None:
        slot = next((i for i, sg in enumerate(self._pool) if sg is None), None)
        if slot is None:
            oldest, slot = -1.0, 0
            for i, sg in enumerate(self._pool):
                if sg is not None:
                    age = t - sg.birth
                    if age > oldest:
                        oldest, slot = age, i

        n_sym = self._buf_n_sym

        if self._palette:
            base_r, base_g, base_b = self._palette[
                int(self._rng.integers(0, len(self._palette)))
            ]
            base_h, sat, _ = colorsys.rgb_to_hsv(base_r, base_g, base_b)
            sat = max(0.6, sat)
        else:
            base_h = float(self._rng.uniform(0.0, 1.0))
            sat    = float(self._rng.uniform(0.65, 0.95))

        # Precompute per-shard RGB with a hue walk across the group
        shard_rgb = np.array([
            list(colorsys.hsv_to_rgb(
                (base_h + i * HUE_SPREAD / max(n_sym, 1)) % 1.0, sat, 1.0
            ))
            for i in range(n_sym)
        ], dtype=np.float32)

        self._pool[slot] = ShardGroup(
            cx         = float(self._rng.uniform(-BOUND * 0.85, BOUND * 0.85)),
            birth      = t,
            phase      = float(self._rng.uniform(0.0, 2.0 * np.pi)),
            speed      = float(self._rng.uniform(0.06, 0.24)),
            amplitude  = float(self._rng.uniform(0.03, 0.18)),
            spin       = float(self._rng.uniform(-1.2, 1.2)),
            base_angle = float(self._rng.uniform(0.0, 2.0 * np.pi)),
            shard_rgb  = shard_rgb,
        )

    @staticmethod
    def _alpha(age: float, lifetime: float) -> float:
        if age >= lifetime:
            return 0.0
        if age < FADE_IN:
            return age / FADE_IN
        if age > lifetime - FADE_OUT:
            return (lifetime - age) / FADE_OUT
        return 1.0

    def _update_geo(self, t: float) -> None:
        n_sym = self._buf_n_sym
        if n_sym == 0:
            return

        FAR    = np.array([1e6, 1e6, 1e6], dtype=np.float32)
        half_a = (np.pi / n_sym) * FILL_FACTOR
        k_arr  = np.arange(n_sym, dtype=np.float32)
        step   = 2.0 * np.pi / n_sym

        for slot, sg in enumerate(self._pool):
            v0 = slot * n_sym * 4

            if sg is None:
                self._pos_cpu[v0 : v0 + n_sym * 4] = FAR
                continue

            age   = t - sg.birth
            alpha = self._alpha(age, self.lifetime)

            # Radial envelope: expands then contracts (sine bell over lifetime)
            t_n     = min(age / self.lifetime, 1.0)
            r_outer = MAX_RADIUS * np.sin(t_n * np.pi)

            # X-axis drift (same pattern as circleaxis)
            aa  = sg.amplitude * self.amplitude
            cx  = sg.cx + aa * np.sin(aa * t * sg.speed + sg.phase)

            # Counter-rotating adjacent shards: even indices spin +, odd spin −
            # This is what makes the kaleidoscope "open and close" rather than
            # rotating as a rigid body.
            spin_rate = sg.spin + self.spin_speed
            sign      = np.where(k_arr % 2 == 0, 1.0, -1.0).astype(np.float32)
            centers   = (sg.base_angle + k_arr * step + sign * spin_rate * age).astype(np.float32)

            # Per-shard radial breathing in antiphase with the rotation sign —
            # even shards expand while odd shards contract, and vice versa.
            r_scale   = 1.0 + 0.22 * sign * np.sin(age * 3.7 + sg.phase + k_arr * 0.5)
            r_outer_k = np.clip(r_outer * r_scale, 0.0, MAX_RADIUS * 1.25).astype(np.float32)
            r_inner_k = (r_outer_k * 0.08).astype(np.float32)

            a_l = centers - half_a
            a_r = centers + half_a

            # 4 corners per shard: inner-left, inner-right, outer-right, outer-left
            cx_col = np.full(n_sym, cx, dtype=np.float32)
            verts = np.empty((n_sym, 4, 3), dtype=np.float32)
            verts[:, 0] = np.stack([cx_col, r_inner_k * np.cos(a_l), r_inner_k * np.sin(a_l)], axis=1)
            verts[:, 1] = np.stack([cx_col, r_inner_k * np.cos(a_r), r_inner_k * np.sin(a_r)], axis=1)
            verts[:, 2] = np.stack([cx_col, r_outer_k * np.cos(a_r), r_outer_k * np.sin(a_r)], axis=1)
            verts[:, 3] = np.stack([cx_col, r_outer_k * np.cos(a_l), r_outer_k * np.sin(a_l)], axis=1)
            self._pos_cpu[v0 : v0 + n_sym * 4] = verts.reshape(-1, 3)

            # Shimmering brightness per shard, out of phase between even/odd
            shimmer    = (0.65 + 0.35 * np.sin(age * 2.1 + k_arr * 1.3 + sg.phase)).astype(np.float32)
            rgb        = (sg.shard_rgb * shimmer[:, None]).astype(np.float32)  # (n_sym, 3)
            inner      = np.concatenate([rgb * 0.35, np.full((n_sym, 1), alpha * 0.9, dtype=np.float32)], axis=1)
            outer      = np.concatenate([rgb,        np.full((n_sym, 1), alpha * 0.55, dtype=np.float32)], axis=1)

            cols = np.empty((n_sym, 4, 4), dtype=np.float32)
            cols[:, 0] = inner
            cols[:, 1] = inner
            cols[:, 2] = outer
            cols[:, 3] = outer
            self._col_cpu[v0 : v0 + n_sym * 4] = cols.reshape(-1, 4)

        self._pos_buf.write(self._pos_cpu.tobytes())
        self._col_buf.write(self._col_cpu.tobytes())


def _make(ctx, device=None, **kwargs):
    return KaleidoscopeDrawing(ctx, device=device, **kwargs)


register_element_type("kaleidoscope", _make)
