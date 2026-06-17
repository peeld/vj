"""
elements/falling_discs.py

FallingDiscsElement -- coloured flat filled circular discs that slowly fall
from the top to the bottom of the scene, picking colours from the palette.
"""

import colorsys
import random
import numpy as np
import warp as wp
import moderngl

from elements.base import DrawingElement, FrameContext, register_element_type

# ── Pool constants ──────────────────────────────────────────────────────────
_DEFAULT_N      = 200   # disc pool size; changing requires regen()
_N_SEG          = 36    # circular segments per disc
_VERTS_PER_DISC = _N_SEG + 1   # centre + rim
_IDX_PER_DISC   = _N_SEG * 3   # N_SEG fan triangles × 3 verts each

_SPAWN_Y = 1.5     # Y at which new discs appear
_FALL_RANGE = 3.0  # total Y distance fallen over lifetime (SPAWN_Y to -SPAWN_Y)


# ── Warp kernels ────────────────────────────────────────────────────────────

@wp.kernel
def _spawn_discs(
    pos:        wp.array(dtype=wp.vec3),
    age:        wp.array(dtype=wp.float32),
    lifetime:   wp.array(dtype=wp.float32),
    radius:     wp.array(dtype=wp.float32),
    disc_color: wp.array(dtype=wp.vec3),
    rot_speed:  wp.array(dtype=wp.float32),
    rot_phase:  wp.array(dtype=wp.float32),
    palette:    wp.array(dtype=wp.vec3),
    n_palette:  int,
    min_radius: float,
    max_radius: float,
    spd_min:    float,
    spd_max:    float,
    rot_spd_max: float,
    frame:      int,
):
    i = wp.tid()
    if age[i] >= lifetime[i]:
        r    = wp.rand_init(frame * 6271 + i, i)
        x    = (wp.randf(r) - 0.5) * 1.8
        z    = (wp.randf(r) - 0.5) * 1.8
        pos[i] = wp.vec3(x, _SPAWN_Y, z)
        age[i] = 0.0

        rad = min_radius + wp.randf(r) * (max_radius - min_radius)
        radius[i] = rad

        speed = spd_min + wp.randf(r) * (spd_max - spd_min)
        if speed < 1e-4:
            speed = 1e-4
        lifetime[i] = _FALL_RANGE / speed

        rot_speed[i] = (wp.randf(r) - 0.5) * 2.0 * rot_spd_max
        rot_phase[i] = wp.randf(r) * 6.283185307179586

        pi = int(wp.randf(r) * float(n_palette))
        if pi < 0:
            pi = 0
        if pi >= n_palette:
            pi = n_palette - 1
        disc_color[i] = palette[pi]


@wp.kernel
def _advance_discs(
    pos:      wp.array(dtype=wp.vec3),
    age:      wp.array(dtype=wp.float32),
    lifetime: wp.array(dtype=wp.float32),
    dt:       float,
):
    i = wp.tid()
    if age[i] < lifetime[i]:
        age[i] = age[i] + dt
        lt = lifetime[i]
        p  = pos[i]
        if lt > 0.0:
            dy = _FALL_RANGE / lt * dt
            pos[i] = wp.vec3(p[0], p[1] - dy, p[2])


@wp.kernel
def _build_disc_verts(
    pos:        wp.array(dtype=wp.vec3),
    age:        wp.array(dtype=wp.float32),
    lifetime:   wp.array(dtype=wp.float32),
    radius:     wp.array(dtype=wp.float32),
    disc_color: wp.array(dtype=wp.vec3),
    rot_speed:  wp.array(dtype=wp.float32),
    rot_phase:  wp.array(dtype=wp.float32),
    out_pos:    wp.array(dtype=wp.vec3),
    out_color:  wp.array(dtype=wp.vec4),
    n_seg:      int,
):
    i    = wp.tid()
    base = i * (n_seg + 1)

    p   = pos[i]
    rad = radius[i]
    a   = age[i]
    lt  = lifetime[i]
    c   = disc_color[i]
    rot = rot_phase[i] + a * rot_speed[i]

    alpha = 0.0
    if lt > 0.0:
        t = a / lt
        if t >= 0.0 and t < 1.0:
            if t < 0.08:
                alpha = t / 0.08
            else:
                if t > 0.85:
                    alpha = (1.0 - t) / 0.15
                else:
                    alpha = 1.0

    far = wp.vec3(1.0e6, 1.0e6, 1.0e6)

    if alpha < 0.001:
        out_pos[base]   = far
        out_color[base] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        for j in range(n_seg):
            out_pos[base + 1 + j]   = far
            out_color[base + 1 + j] = wp.vec4(0.0, 0.0, 0.0, 0.0)
    else:
        out_pos[base]   = p
        out_color[base] = wp.vec4(c[0], c[1], c[2], alpha)
        for j in range(n_seg):
            angle = rot + float(j) / float(n_seg) * 6.283185307179586
            vx = p[0] + rad * wp.sin(angle)
            vy = p[1] + rad * wp.cos(angle)
            out_pos[base + 1 + j]   = wp.vec3(vx, vy, p[2])
            out_color[base + 1 + j] = wp.vec4(c[0] * 0.82, c[1] * 0.82, c[2] * 0.82, alpha * 0.88)


# ── Element ─────────────────────────────────────────────────────────────────

class FallingDiscsElement(DrawingElement):
    """Coloured flat filled circular discs that slowly fall top-to-bottom.

    Tunable parameters (all take effect on the next spawn cycle unless noted):
        min_size        float  minimum disc radius in world units
        max_size        float  maximum disc radius in world units
        fall_speed_min  float  minimum fall speed (world units / second)
        fall_speed_max  float  maximum fall speed (world units / second)
        count           int    pool size -- requires regen() after changing
    """
    kind = "falling_discs"

    def __init__(self, ctx: moderngl.Context, device: str | None = None, **kwargs):
        super().__init__()
        self._ctx      = ctx
        self.device    = device
        self._frame    = 0
        self._has_cuda = wp.get_cuda_device_count() > 0

        # Tunable parameters
        self.count              = _DEFAULT_N   # requires regen() to take effect
        self.min_size           = 0.001
        self.max_size           = 0.01
        self.fall_speed_min     = 0.15
        self.fall_speed_max     = 0.50
        self.rotation_speed_max = 0.5   # radians/sec; each disc gets a random value in [-max, +max]

        self._palette:     list[tuple[float, float, float]] = self._default_palette()
        self._palette_arr: wp.array | None = None

        self._build(ctx)

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _default_palette() -> list[tuple[float, float, float]]:
        base = random.random()
        n = 7
        return [colorsys.hsv_to_rgb((base + i / n) % 1.0, 0.75, 1.0) for i in range(n)]

    def _make_palette_arr(self) -> wp.array:
        pal  = self._palette or self._default_palette()
        data = np.array([(r, g, b) for r, g, b in pal], dtype=np.float32)
        return wp.array(data, dtype=wp.vec3, device=self.device)

    def _build_index_buffer(self, n: int, n_seg: int) -> np.ndarray:
        vpd   = n_seg + 1
        disc  = np.arange(n,     dtype=np.uint32)
        seg   = np.arange(n_seg, dtype=np.uint32)
        D, J  = np.meshgrid(disc, seg, indexing='ij')
        base  = (D * vpd).astype(np.uint32)
        idx   = np.stack([base, base + 1 + J, base + 1 + (J + 1) % n_seg], axis=-1)
        return idx.reshape(-1).astype(np.uint32)

    def _scatter_initial(self) -> None:
        """Scatter discs at random Y positions so the scene starts populated."""
        n   = self.count
        rng = np.random.default_rng()
        pal = self._palette or self._default_palette()

        speeds = rng.uniform(self.fall_speed_min, self.fall_speed_max, n).astype(np.float32)
        lts    = (_FALL_RANGE / np.maximum(speeds, 1e-4)).astype(np.float32)
        frac   = rng.uniform(0.0, 1.0, n).astype(np.float32)
        ages   = (frac * lts).astype(np.float32)

        x    = rng.uniform(-0.9, 0.9, n).astype(np.float32)
        z    = rng.uniform(-0.9, 0.9, n).astype(np.float32)
        y    = (_SPAWN_Y - _FALL_RANGE * frac).astype(np.float32)
        pos  = np.stack([x, y, z], axis=1).astype(np.float32)

        radii  = rng.uniform(self.min_size, self.max_size, n).astype(np.float32)
        pi     = rng.integers(0, len(pal), n)
        colors = np.array([pal[k] for k in pi], dtype=np.float32)

        rot_speeds = rng.uniform(-self.rotation_speed_max, self.rotation_speed_max, n).astype(np.float32)
        rot_phases = rng.uniform(0.0, 6.283185307179586, n).astype(np.float32)

        self._pos        = wp.array(pos,        dtype=wp.vec3, device=self.device)
        self._age        = wp.array(ages,                      device=self.device)
        self._lifetime   = wp.array(lts,                       device=self.device)
        self._radius     = wp.array(radii,                     device=self.device)
        self._disc_color = wp.array(colors,     dtype=wp.vec3, device=self.device)
        self._rot_speed  = wp.array(rot_speeds,                device=self.device)
        self._rot_phase  = wp.array(rot_phases,                device=self.device)

    def _build(self, ctx: moderngl.Context) -> None:
        n           = self.count
        n_seg       = _N_SEG
        total_verts = n * _VERTS_PER_DISC
        total_idx   = n * _IDX_PER_DISC

        # Per-disc intermediate vertex arrays (used by kernel, then copied to GL)
        self._out_pos   = wp.zeros(total_verts, dtype=wp.vec3, device=self.device)
        self._out_color = wp.zeros(total_verts, dtype=wp.vec4, device=self.device)

        self._scatter_initial()

        # GL buffers
        self.pos_buf   = ctx.buffer(reserve=total_verts * 3 * 4)
        self.color_buf = ctx.buffer(reserve=total_verts * 4 * 4)
        self.index_buf = ctx.buffer(self._build_index_buffer(n, n_seg).tobytes())

        self.prog = ctx.program(
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
                void main() {
                    f_color = v_color;
                }
            """,
        )

        self.vao = ctx.vertex_array(
            self.prog,
            [
                (self.pos_buf,   "3f", "in_position"),
                (self.color_buf, "4f", "in_color"),
            ],
            self.index_buf,
        )

        if self._has_cuda:
            self._pos_reg   = wp.RegisteredGLBuffer(self.pos_buf.glo,   wp.get_preferred_device())
            self._color_reg = wp.RegisteredGLBuffer(self.color_buf.glo, wp.get_preferred_device())

        self._palette_arr = self._make_palette_arr()

    # ── DrawingElement interface ─────────────────────────────────────────

    def step(self, ctx: FrameContext) -> None:
        self._frame += 1
        n       = self.count
        n_seg   = _N_SEG
        total_v = n * _VERTS_PER_DISC
        n_pal   = len(self._palette or self._default_palette())

        wp.launch(
            _spawn_discs,
            dim=n,
            inputs=[
                self._pos, self._age, self._lifetime, self._radius, self._disc_color,
                self._rot_speed, self._rot_phase,
                self._palette_arr, n_pal,
                self.min_size, self.max_size,
                self.fall_speed_min, self.fall_speed_max,
                self.rotation_speed_max,
                self._frame,
            ],
            device=self.device,
        )

        wp.launch(
            _advance_discs,
            dim=n,
            inputs=[self._pos, self._age, self._lifetime, ctx.frame_time],
            device=self.device,
        )

        wp.launch(
            _build_disc_verts,
            dim=n,
            inputs=[
                self._pos, self._age, self._lifetime, self._radius, self._disc_color,
                self._rot_speed, self._rot_phase,
                self._out_pos, self._out_color, n_seg,
            ],
            device=self.device,
        )

        if self._has_cuda:
            mapped_pos = self._pos_reg.map(dtype=wp.vec3, shape=(total_v,))
            wp.copy(mapped_pos, self._out_pos)
            self._pos_reg.unmap()
            mapped_col = self._color_reg.map(dtype=wp.vec4, shape=(total_v,))
            wp.copy(mapped_col, self._out_color)
            self._color_reg.unmap()
        else:
            self.pos_buf.write(self._out_pos.numpy().tobytes())
            self.color_buf.write(self._out_color.numpy().tobytes())

    def draw(self, mvp, ctx: FrameContext) -> None:
        self.prog["mvp"].write(mvp.tobytes())
        self.vao.render(moderngl.TRIANGLES)

    def regen(self) -> None:
        self._palette_arr = self._make_palette_arr()
        self._scatter_initial()
        self._frame = 0

    def set_palette(self, palette: list) -> None:
        if palette:
            self._palette = list(palette)
            self._palette_arr = self._make_palette_arr()


def _make(ctx, device=None, **kwargs):
    return FallingDiscsElement(ctx, device=device, **kwargs)


register_element_type("falling_discs", _make)
