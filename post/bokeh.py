"""
post/bokeh.py — BokehEffect

Depth-of-field post-effect: pixels further from the focus depth receive a
disc blur proportional to their Circle-of-Confusion (CoC) radius, producing
the characteristic bokeh look where distant objects become soft discs.

Pipeline (per frame)
--------------------
1. Read colour (RGBA uint8) and depth (float32 NDC) from the scene FBO.
2. Upload both to Warp float32 arrays on the GPU.
3. Warp kernel: per-pixel CoC → disc-gather of colour neighbours.
4. Copy result back to a ModernGL texture and return it.

Parameters (adjustable at runtime via key bindings)
----------------------------------------------------
focus_depth : float  [0..1]
    Normalised linear depth that is perfectly in focus.
    0 = near plane, 1 = far plane.  Default 0.15.

focus_range : float  [0..0.5]
    Half-width of the sharp band around focus_depth.
    |linear_depth − focus_depth| ≤ focus_range → no blur.
    Default 0.05.

max_radius : int
    Maximum disc-blur radius in pixels for fully out-of-focus pixels.
    Default 12.

near / far : float
    Camera clip planes (must match the scene's projection matrix).
    Defaults match viewport.py  (near=0.1, far=50.0).

Key bindings (when this effect is the active post-effect)
----------------------------------------------------------
    ↑ / ↓      focus_depth  ± 0.02
    ← / →      max_radius   ∓ / ± 1 px
    [ / ]      focus_range  ∓ / ± 0.01
"""

from __future__ import annotations

import numpy as np
import moderngl
import warp as wp

from post.base import PostEffect, _QUAD_VERT, _QUAD_FRAG, _QUAD_VERTS, DEVICE


# ─── Warp kernel ──────────────────────────────────────────────────────────────

@wp.kernel
def _bokeh_kernel(
    color:       wp.array(dtype=wp.float32),  # RGBA flat  [h*w*4]  in [0..1]
    depth:       wp.array(dtype=wp.float32),  # NDC depth  [h*w]    in [0..1]
    out:         wp.array(dtype=wp.float32),  # RGBA flat  [h*w*4]  output
    img_w:       int,
    img_h:       int,
    focus_depth: float,   # normalised linear depth in focus
    focus_range: float,   # half-width of sharp band
    max_radius:  int,     # maximum disc radius in pixels
    near:        float,   # camera near clip
    far:         float,   # camera far  clip
):
    tid = wp.tid()
    if tid >= img_w * img_h:
        return

    px = tid % img_w
    py = tid // img_w

    # ── Linearise NDC depth ────────────────────────────────────────────
    # OpenGL maps:  linear_z = (2*n*f) / (f+n - ndc*(f-n))
    ndc_d  = depth[tid]
    lin_d  = (2.0 * near * far) / (far + near - ndc_d * (far - near))
    norm_d = wp.clamp(lin_d / far, 0.0, 1.0)   # in [0..1]

    # ── Circle of Confusion radius (pixels) ───────────────────────────
    dist   = wp.abs(norm_d - focus_depth) - focus_range
    coc    = wp.clamp(dist / (1.0 - focus_range + 1e-6), 0.0, 1.0)
    radius = int(float(max_radius) * coc)       # 0 = sharp, max_radius = fully blurred

    # ── Disc gather ────────────────────────────────────────────────────
    # Loop over square neighbourhood clamped to actual radius; use the
    # tightest bounds so in-focus pixels (radius==0) cost only one sample.
    r_lo = -radius
    r_hi =  radius + 1

    acc_r  = float(0.0)
    acc_g  = float(0.0)
    acc_b  = float(0.0)
    acc_a  = float(0.0)
    count  = int(0)

    for dy in range(r_lo, r_hi):
        for dx in range(r_lo, r_hi):
            if dx * dx + dy * dy <= radius * radius:
                sx = px + dx
                sy = py + dy
                if sx >= 0 and sx < img_w and sy >= 0 and sy < img_h:
                    s      = (sy * img_w + sx) * 4
                    acc_r  = acc_r + color[s]
                    acc_g  = acc_g + color[s + 1]
                    acc_b  = acc_b + color[s + 2]
                    acc_a  = acc_a + color[s + 3]
                    count  = count + 1

    base = tid * 4
    if count > 0:
        inv          = 1.0 / float(count)
        out[base]     = acc_r * inv
        out[base + 1] = acc_g * inv
        out[base + 2] = acc_b * inv
        out[base + 3] = acc_a * inv
    else:
        # Passthrough guard (can only trigger at image corners with radius=0)
        out[base]     = color[base]
        out[base + 1] = color[base + 1]
        out[base + 2] = color[base + 2]
        out[base + 3] = color[base + 3]


# ─── PostEffect ───────────────────────────────────────────────────────────────

class BokehEffect(PostEffect):
    """
    Depth-of-field bokeh post-effect.

    Usage::

        from post import BokehEffect

        class MyScene(Scene):
            post_effect = BokehEffect(focus_depth=0.2, max_radius=14)
    """

    name = "bokeh"

    def __init__(
        self,
        focus_depth: float = 0.15,
        focus_range: float = 0.05,
        max_radius:  int   = 12,
        near:        float = 0.1,
        far:         float = 50.0,
    ) -> None:
        self.focus_depth = focus_depth
        self.focus_range = focus_range
        self.max_radius  = max_radius
        self.near        = near
        self.far         = far

        self._ctx:       moderngl.Context     | None = None
        self._fbo:       moderngl.Framebuffer | None = None
        self._out_tex:   moderngl.Texture     | None = None
        self._quad_prog: moderngl.Program     | None = None
        self._quad_vao:  moderngl.VertexArray | None = None

        self._w: int = 1
        self._h: int = 1

        self._color_buf: wp.array | None = None   # float32 [h*w*4]
        self._depth_buf: wp.array | None = None   # float32 [h*w]
        self._out_buf:   wp.array | None = None   # float32 [h*w*4]

    # ── PostEffect interface ──────────────────────────────────────────────────

    def setup(self, ctx: moderngl.Context, w: int, h: int) -> None:
        self._ctx = ctx
        self._w, self._h = w, h
        self._build_fbo(w, h)
        self._build_quad(ctx)
        self._alloc_warp(w, h)
        self._print_params()

    @property
    def fbo(self) -> moderngl.Framebuffer:
        return self._fbo

    def bind_scene_fbo(self) -> None:
        self._fbo.use()
        self._ctx.enable(moderngl.DEPTH_TEST)
        self._ctx.clear(0.04, 0.04, 0.06, 1.0)

    def process(
        self,
        scene_fbo: moderngl.Framebuffer,
        t:  float,
        dt: float,
    ) -> moderngl.Texture:
        # TODO: PBO zero-copy refactor (same pattern as GlitchEffect/FeedbackPostEffect).
        # Needs two pack PBOs: one for colour (GL_RGBA / GL_UNSIGNED_BYTE) and one for
        # depth (GL_DEPTH_COMPONENT / GL_FLOAT), both registered READ_ONLY.  Output via
        # an unpack PBO registered WRITE_DISCARD.  Not done: effect is unused/broken.
        w, h = self._w, self._h

        # ── Read colour buffer (RGBA uint8 → float32) ──────────────────
        raw_color = scene_fbo.color_attachments[0].read()
        color_np  = np.frombuffer(raw_color, dtype=np.uint8).reshape(h, w, 4)
        color_f32 = (color_np.astype(np.float32) * (1.0 / 255.0)).ravel()
        wp.copy(self._color_buf, wp.array(color_f32, dtype=wp.float32, device=DEVICE))

        # ── Read depth buffer (float32 NDC, one value per pixel) ───────
        raw_depth = scene_fbo.depth_attachment.read()
        depth_f32 = np.frombuffer(raw_depth, dtype=np.float32).ravel()
        wp.copy(self._depth_buf, wp.array(depth_f32, dtype=wp.float32, device=DEVICE))

        # ── Run bokeh kernel ───────────────────────────────────────────
        wp.launch(
            _bokeh_kernel,
            dim=w * h,
            inputs=[
                self._color_buf,
                self._depth_buf,
                self._out_buf,
                w, h,
                float(self.focus_depth),
                float(self.focus_range),
                int(self.max_radius),
                float(self.near),
                float(self.far),
            ],
            device=DEVICE,
        )

        # ── Write result to output texture ─────────────────────────────
        result    = self._out_buf.numpy().reshape(h, w, 4)
        result_u8 = (np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8)
        self._out_tex.write(result_u8.tobytes())

        return self._out_tex

    def blit_to_screen(self, screen: moderngl.Framebuffer) -> None:
        screen.use()
        self._ctx.disable(moderngl.DEPTH_TEST)
        self._ctx.clear(0.0, 0.0, 0.0, 1.0)
        self._out_tex.use(0)
        self._quad_prog["tex"].value = 0
        self._quad_vao.render(moderngl.TRIANGLE_STRIP)

    def resize(self, w: int, h: int) -> None:
        if self._ctx is None:
            return
        self._w, self._h = w, h
        self._build_fbo(w, h)
        self._alloc_warp(w, h)

    def on_key(self, key, action, keys) -> None:
        if action != keys.ACTION_PRESS:
            return

        if key == keys.UP:
            self.focus_depth = round(min(0.98, self.focus_depth + 0.02), 3)
            self._print_params()
        elif key == keys.DOWN:
            self.focus_depth = round(max(0.02, self.focus_depth - 0.02), 3)
            self._print_params()
        elif key == keys.RIGHT:
            self.max_radius = min(32, self.max_radius + 1)
            self._print_params()
        elif key == keys.LEFT:
            self.max_radius = max(1, self.max_radius - 1)
            self._print_params()
        elif key == keys.PAGE_UP:
            self.focus_range = round(max(0.0, self.focus_range - 0.01), 3)
            self._print_params()
        elif key == keys.PAGE_DOWN:
            self.focus_range = round(min(0.5, self.focus_range + 0.01), 3)
            self._print_params()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_fbo(self, w: int, h: int) -> None:
        self._fbo     = self._ctx.framebuffer(
            color_attachments=[self._ctx.texture((w, h), 4)],
            depth_attachment=self._ctx.depth_texture((w, h)),
        )
        self._out_tex = self._ctx.texture((w, h), 4)

    def _build_quad(self, ctx: moderngl.Context) -> None:
        vbo = ctx.buffer(_QUAD_VERTS.tobytes())
        self._quad_prog = ctx.program(
            vertex_shader=_QUAD_VERT,
            fragment_shader=_QUAD_FRAG,
        )
        self._quad_vao = ctx.vertex_array(
            self._quad_prog, [(vbo, "2f", "in_pos")]
        )

    def _alloc_warp(self, w: int, h: int) -> None:
        self._color_buf = wp.zeros(w * h * 4, dtype=wp.float32, device=DEVICE)
        self._depth_buf = wp.zeros(w * h,     dtype=wp.float32, device=DEVICE)
        self._out_buf   = wp.zeros(w * h * 4, dtype=wp.float32, device=DEVICE)

    def _print_params(self) -> None:
        print(
            f"[bokeh]  focus_depth={self.focus_depth:.3f}  "
            f"focus_range={self.focus_range:.3f}  "
            f"max_radius={self.max_radius}px  "
            f"(↑↓ depth | ←→ radius | [] range)"
        )
