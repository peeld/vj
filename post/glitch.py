"""
post/glitch.py — GlitchEffect

Takes 30 rectangular horizontal slices of the scene, offsets each left or
right by a random amount, and leaves the original pixels underneath.  The
slices regenerate on a configurable interval for a living-corruption look.

Kernel design (pull model)
--------------------------
For every output pixel (px, py) we scan all 30 rects.  If (px, py) falls
inside the *shifted* destination of rect i, we read from the *unshifted*
source position (px - offset, py) instead of the original location.
Pixels not claimed by any rect pass through unchanged.  Because the source
data is never cleared, the original image shows through everywhere the
rect used to live — the classic data-corruption ghost.
"""

from __future__ import annotations

import numpy as np
import moderngl
import warp as wp

from post.base import (
    PostEffect, DEVICE, NULL_OFFSET,
    _QUAD_VERT, _QUAD_FRAG, _QUAD_VERTS,
    glBindBuffer, glBindFramebuffer, glBindTexture,
    glReadPixels, glTexSubImage2D,
    GL_PIXEL_PACK_BUFFER, GL_PIXEL_UNPACK_BUFFER,
    GL_RGBA, GL_UNSIGNED_BYTE, GL_TEXTURE_2D,
    GL_READ_FRAMEBUFFER,
)


N_RECTS = 30


# ─── Warp kernels ─────────────────────────────────────────────────────────────

@wp.kernel
def _decode_rgba_u8(
    src: wp.array(dtype=wp.uint8),
    dst: wp.array(dtype=wp.float32),
):
    """Pack-PBO uint8 RGBA → float32 RGBA (no Y-flip; preserves OpenGL bottom-up order)."""
    tid  = wp.tid()
    base = tid * 4
    dst[base]     = float(src[base])     / 255.0
    dst[base + 1] = float(src[base + 1]) / 255.0
    dst[base + 2] = float(src[base + 2]) / 255.0
    dst[base + 3] = float(src[base + 3]) / 255.0


@wp.kernel
def _pack_rgba_u8(
    src: wp.array(dtype=wp.float32),
    dst: wp.array(dtype=wp.uint8),
):
    """float32 RGBA → uint8 RGBA; writes directly into the mapped unpack PBO."""
    tid  = wp.tid()
    base = tid * 4
    dst[base]     = wp.uint8(int(wp.clamp(src[base],     0.0, 1.0) * 255.0))
    dst[base + 1] = wp.uint8(int(wp.clamp(src[base + 1], 0.0, 1.0) * 255.0))
    dst[base + 2] = wp.uint8(int(wp.clamp(src[base + 2], 0.0, 1.0) * 255.0))
    dst[base + 3] = wp.uint8(255)


@wp.kernel
def _glitch_kernel(
    src:     wp.array(dtype=wp.float32),
    dst:     wp.array(dtype=wp.float32),
    rect_x:  wp.array(dtype=wp.int32),
    rect_y:  wp.array(dtype=wp.int32),
    rect_w:  wp.array(dtype=wp.int32),
    rect_h:  wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    n_rects: int,
    img_w:   int,
    img_h:   int,
):
    tid = wp.tid()
    if tid >= img_w * img_h:
        return

    px = tid % img_w
    py = tid // img_w

    # Default: copy original pixel
    base = tid * 4
    r = src[base]
    g = src[base + 1]
    b = src[base + 2]
    a = src[base + 3]

    # Scan rects — last one that claims this pixel wins
    for i in range(n_rects):
        rx  = rect_x[i]
        ry  = rect_y[i]
        rw  = rect_w[i]
        rh  = rect_h[i]
        off = offsets[i]

        # Destination x-span after horizontal shift, clamped to image bounds
        dst_x0 = wp.clamp(rx + off,      0, img_w)
        dst_x1 = wp.clamp(rx + rw + off, 0, img_w)

        if px >= dst_x0 and px < dst_x1 and py >= ry and py < ry + rh:
            # Pull from the unshifted source column
            src_x = px - off
            if src_x >= 0 and src_x < img_w:
                s = (py * img_w + src_x) * 4
                r = src[s]
                g = src[s + 1]
                b = src[s + 2]
                a = src[s + 3]

    dst[base]     = r
    dst[base + 1] = g
    dst[base + 2] = b
    dst[base + 3] = a


# ─── PostEffect ───────────────────────────────────────────────────────────────

class GlitchEffect(PostEffect):
    """
    Glitch post-effect: 30 horizontal slices shifted left/right each frame.

    Parameters
    ----------
    regen_interval : float
        Seconds between slice regenerations (default 0.1 s).
    max_offset : int | None
        Maximum pixel offset.  Defaults to w // 3.
    max_slice_h : int | None
        Maximum slice height in pixels.  Defaults to h // 10.
    """

    name = "glitch"

    def __init__(
        self,
        regen_interval: float = 0.1,
        max_offset:    int | None = None,
        max_slice_h:   int | None = None,
    ) -> None:
        self._ctx:         moderngl.Context            | None = None
        self._fbo:         moderngl.Framebuffer        | None = None
        self._display_tex: moderngl.Texture            | None = None
        self._quad_prog:   moderngl.Program            | None = None
        self._quad_vao:    moderngl.VertexArray        | None = None

        self._w = 1
        self._h = 1

        self._pack_pbo:   moderngl.Buffer          | None = None
        self._reg_pack:   wp.RegisteredGLBuffer    | None = None
        self._unpack_pbo: moderngl.Buffer          | None = None
        self._reg_unpack: wp.RegisteredGLBuffer    | None = None

        self._src_buf: wp.array | None = None
        self._dst_buf: wp.array | None = None

        self._rect_x:  wp.array | None = None
        self._rect_y:  wp.array | None = None
        self._rect_w:  wp.array | None = None
        self._rect_h:  wp.array | None = None
        self._offsets: wp.array | None = None

        self._regen_interval = regen_interval
        self._max_offset     = max_offset
        self._max_slice_h    = max_slice_h
        self._last_regen     = -999.0
        self._rng            = np.random.default_rng()

    # ── PostEffect interface ──────────────────────────────────────────────────

    def setup(self, ctx: moderngl.Context, w: int, h: int) -> None:
        self._ctx  = ctx
        self._w, self._h = w, h
        self._build_fbo(w, h)
        self._build_quad(ctx)
        self._alloc_warp(w, h)
        self._regen_rects()

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
        if t - self._last_regen >= self._regen_interval:
            self._regen_rects()
            self._last_regen = t

        w, h = self._w, self._h
        n    = w * h

        # FBO → pack PBO (GPU-side DMA, no PCIe transfer)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, self._pack_pbo.glo)
        glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, NULL_OFFSET)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        glBindFramebuffer(GL_READ_FRAMEBUFFER, 0)

        # Decode: pack PBO uint8 → float32; map() syncs GL→CUDA
        raw_gpu = self._reg_pack.map(dtype=wp.uint8, shape=(n * 4,))
        wp.launch(_decode_rgba_u8, dim=n, inputs=[raw_gpu, self._src_buf], device=DEVICE)
        wp.synchronize_device(DEVICE)
        self._reg_pack.unmap()

        # Shift slices (float32 → float32)
        wp.launch(
            _glitch_kernel,
            dim=n,
            inputs=[
                self._src_buf, self._dst_buf,
                self._rect_x, self._rect_y,
                self._rect_w, self._rect_h,
                self._offsets,
                N_RECTS, w, h,
            ],
            device=DEVICE,
        )

        # Pack: float32 → unpack PBO uint8; then GPU-side DMA to display texture
        out_gpu = self._reg_unpack.map(dtype=wp.uint8, shape=(n * 4,))
        wp.launch(_pack_rgba_u8, dim=n, inputs=[self._dst_buf, out_gpu], device=DEVICE)
        wp.synchronize_device(DEVICE)
        self._reg_unpack.unmap()

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._unpack_pbo.glo)
        glBindTexture(GL_TEXTURE_2D, self._display_tex.glo)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, NULL_OFFSET)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

        return self._display_tex

    def blit_to_screen(self, screen: moderngl.Framebuffer) -> None:
        screen.use()
        self._ctx.disable(moderngl.DEPTH_TEST)
        self._ctx.clear(0.0, 0.0, 0.0, 1.0)
        self._display_tex.use(0)
        self._quad_prog["tex"].value = 0
        self._quad_vao.render(moderngl.TRIANGLE_STRIP)

    def resize(self, w: int, h: int) -> None:
        if self._ctx is None:
            return
        self._w, self._h = w, h
        self._build_fbo(w, h)
        self._alloc_warp(w, h)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_fbo(self, w: int, h: int) -> None:
        # Unregister old CUDA registrations before their GL buffers are released.
        self._reg_pack   = None
        self._reg_unpack = None

        self._fbo = self._ctx.framebuffer(
            color_attachments=[self._ctx.texture((w, h), 4)],
            depth_attachment=self._ctx.depth_texture((w, h)),
        )

        n = w * h * 4
        self._pack_pbo   = self._ctx.buffer(reserve=n)
        self._reg_pack   = wp.RegisteredGLBuffer(
            self._pack_pbo.glo, device=DEVICE,
            flags=wp.RegisteredGLBuffer.READ_ONLY)

        self._unpack_pbo = self._ctx.buffer(reserve=n)
        self._reg_unpack = wp.RegisteredGLBuffer(
            self._unpack_pbo.glo, device=DEVICE,
            flags=wp.RegisteredGLBuffer.WRITE_DISCARD)

        self._display_tex = self._ctx.texture((w, h), 4)

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
        n = w * h * 4
        self._src_buf = wp.zeros(n, dtype=wp.float32, device=DEVICE)
        self._dst_buf = wp.zeros(n, dtype=wp.float32, device=DEVICE)

    def _regen_rects(self) -> None:
        w, h = self._w, self._h
        max_off = self._max_offset  or max(10, w // 3)
        max_sh  = self._max_slice_h or max(2,  h // 10)

        rx_np  = np.zeros(N_RECTS, dtype=np.int32)
        ry_np  = np.zeros(N_RECTS, dtype=np.int32)
        rw_np  = np.zeros(N_RECTS, dtype=np.int32)
        rh_np  = np.zeros(N_RECTS, dtype=np.int32)
        off_np = np.zeros(N_RECTS, dtype=np.int32)

        for i in range(N_RECTS):
            rw = int(self._rng.integers(20, max(21, w // 2)))
            rh = int(self._rng.integers(1,  max(2, max_sh)))
            rx = int(self._rng.integers(0,  max(1, w - rw)))
            ry = int(self._rng.integers(0,  max(1, h - rh)))
            direction = int(self._rng.choice([-1, 1]))
            off = int(direction * self._rng.integers(8, max(9, max_off)))
            rx_np[i], ry_np[i], rw_np[i], rh_np[i], off_np[i] = rx, ry, rw, rh, off

        self._rect_x  = wp.array(rx_np,  dtype=wp.int32, device=DEVICE)
        self._rect_y  = wp.array(ry_np,  dtype=wp.int32, device=DEVICE)
        self._rect_w  = wp.array(rw_np,  dtype=wp.int32, device=DEVICE)
        self._rect_h  = wp.array(rh_np,  dtype=wp.int32, device=DEVICE)
        self._offsets = wp.array(off_np, dtype=wp.int32, device=DEVICE)
