"""
elements/video.py — VideoGLSurface (Step 3) and VideoElement (Step 4).

Data flow:
    numpy RGBA uint8 (CPU) → wp.copy → wp.RegisteredGLBuffer (PBO, GPU)
        → glTexSubImage2D → moderngl Texture
"""

from __future__ import annotations

import ctypes

import numpy as np
import moderngl
import warp as wp
from OpenGL.GL import (
    glBindBuffer, glBindTexture, glTexSubImage2D,
    GL_TEXTURE_2D, GL_PIXEL_UNPACK_BUFFER,
    GL_RGBA, GL_UNSIGNED_BYTE,
)

from .base import DrawingElement, FrameContext, register_element_type, Prop


# ── GLSL programs ─────────────────────────────────────────────────────────────

_VERT_BG = """
#version 330
in vec2 in_position;
out vec2 uv;
void main() {
    uv = in_position * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;
    gl_Position = vec4(in_position, 0.9999, 1.0);
}
"""

_VERT_3D = """
#version 330
in vec2 in_position;
uniform mat4 mvp;
out vec2 uv;
void main() {
    uv = in_position * 0.5 + 0.5;
    uv.y = 1.0 - uv.y;
    gl_Position = mvp * vec4(in_position, 0.0, 1.0);
}
"""

_FRAG = """
#version 330
uniform sampler2D video_tex;
uniform float opacity;
in vec2 uv;
out vec4 out_color;
void main() {
    out_color = texture(video_tex, uv) * vec4(1.0, 1.0, 1.0, opacity);
}
"""

# Full-screen quad: two triangles covering NDC [-1, 1]
_QUAD_VERTS = np.array([
    -1, -1,  1, -1,  -1,  1,
     1, -1,  1,  1,  -1,  1,
], dtype='f4')


class VideoGLSurface:
    """CPU numpy RGBA → GL texture via Warp PBO (zero kernel, one wp.copy).

    Owns the PBO and GL texture for a single video frame size.
    Call upload_from_rgba() each frame from the GL thread.
    """

    def __init__(self, ctx: moderngl.Context, width: int, height: int, device: str = "cuda"):
        self._ctx = ctx
        self.width = width
        self.height = height
        self._device = device
        self._n = width * height * 4

        self.texture = ctx.texture((width, height), 4)
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)

        self._pbo = ctx.buffer(reserve=self._n)
        # WRITE_DISCARD: we always overwrite the full PBO each frame,
        # so CUDA never needs to read the old contents before writing.
        self._registered = wp.RegisteredGLBuffer(
            self._pbo.glo, device=device,
            flags=wp.RegisteredGLBuffer.WRITE_DISCARD)

    def upload_from_rgba(self, rgba: np.ndarray) -> None:
        """Push a (H, W, 4) uint8 numpy frame to the GL texture. Call from GL thread."""
        cpu_wp = wp.array(rgba.flatten(), dtype=wp.uint8, device="cpu")
        pbo_wp = self._registered.map(dtype=wp.uint8, shape=(self._n,))
        wp.copy(pbo_wp, cpu_wp)
        wp.synchronize_device(self._device)
        self._registered.unmap()

        # ctypes.c_void_p(0) is essential — passing None causes PyOpenGL to allocate
        # a CPU buffer and silently bypass the PBO, defeating the GPU-side copy.
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbo.glo)
        glBindTexture(GL_TEXTURE_2D, self.texture.glo)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, self.width, self.height,
                        GL_RGBA, GL_UNSIGNED_BYTE, ctypes.c_void_p(0))
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    def release(self):
        self._registered = None     # unregister before releasing PBO
        self._pbo.release()
        self.texture.release()


# ── VideoElement ──────────────────────────────────────────────────────────────

class VideoElement(DrawingElement, section="video"):
    """Renders a video frame as either a fullscreen background quad or a
    world-space 3D quad."""

    kind = "video"

    active       = Prop("Active", bool, True, widget_hint="check",
                        description="Play / pause video playback")
    opacity      = Prop("Opacity", float, 1.0, 0.0, 1.0, 0.05)
    mode         = Prop("Mode", str, "background",
                        choices=["background", "3d"], widget_hint="combo")
    pos_x        = Prop("Pos X", float, 0.0, -10.0, 10.0, 0.1)
    pos_y        = Prop("Pos Y", float, 0.0, -10.0, 10.0, 0.1)
    pos_z        = Prop("Pos Z", float, 0.0, -10.0, 10.0, 0.1)
    scale        = Prop("Scale", float, 1.0,  0.01, 20.0, 0.1)
    play_pos     = Prop("Play Pos", float, 0.0, 0.0, 1.0, 0.001,
                        widget_hint="readonly",
                        description="Normalised playback position (0–1)")
    cam_distance = Prop("Cam Distance", float, 0.0, 0.0, 200.0, 0.01,
                        widget_hint="readonly",
                        description="Distance from camera to video plane (3D mode only)")

    def __init__(self, ctx: moderngl.Context, device: str = "cuda", **kwargs):
        super().__init__()
        self._ctx = ctx
        self._device = device
        self._player = None
        self._surface: VideoGLSurface | None = None

        self._prog_bg = ctx.program(vertex_shader=_VERT_BG, fragment_shader=_FRAG)
        self._prog_3d = ctx.program(vertex_shader=_VERT_3D, fragment_shader=_FRAG)

        verts = ctx.buffer(_QUAD_VERTS.tobytes())
        self._vao_bg = ctx.vertex_array(self._prog_bg, [(verts, '2f', 'in_position')])
        self._vao_3d = ctx.vertex_array(self._prog_3d, [(verts, '2f', 'in_position')])

        self.opacity      = 1.0
        self.mode         = "background"
        self.pos_x        = 0.0
        self.pos_y        = 0.0
        self.pos_z        = 0.0
        self.scale        = 1.0
        self.play_pos     = 0.0
        self.cam_distance = 0.0

    def set_player(self, player) -> None:
        """Wire in a VideoPlayer. Called from Qt thread — only stores the reference;
        VideoGLSurface is created lazily in step() on the GL thread."""
        if (self._surface is not None and
                (self._surface.width != player.width or
                 self._surface.height != player.height)):
            self._surface.release()
            self._surface = None
        self._player = player

    def step(self, ctx: FrameContext) -> None:
        if self._player is None:
            return

        # active → play / pause
        if self._player.playing != self.active:
            if self.active:
                self._player.play()
            else:
                self._player.pause()

        # play_pos (0–1)
        dur = self._player.duration
        self.play_pos = (self._player.position / dur) if dur > 0 else 0.0

        # cam_distance — meaningful only in 3D mode
        if self.mode == "3d" and ctx.cam_eye is not None:
            cam = np.asarray(ctx.cam_eye, dtype='f4')
            pos = np.array([self.pos_x, self.pos_y, self.pos_z], dtype='f4')
            self.cam_distance = float(np.linalg.norm(cam - pos))
        else:
            self.cam_distance = 0.0

        if self._surface is None:
            self._surface = VideoGLSurface(
                self._ctx, self._player.width, self._player.height, self._device)
        rgba = self._player.get_current_frame()
        if rgba is not None:
            self._surface.upload_from_rgba(rgba)

    def draw(self, mvp, ctx: FrameContext) -> None:
        if self._surface is None:
            return
        self._surface.texture.use(location=0)
        if self.mode == "background":
            self._prog_bg['video_tex'] = 0
            self._prog_bg['opacity'] = self.opacity
            self._ctx.depth_func = '<='
            self._vao_bg.render(moderngl.TRIANGLES)
            self._ctx.depth_func = '<'
        else:
            model = np.eye(4, dtype='f4')
            model[0, 0] = self.scale
            model[1, 1] = self.scale
            model[3, 0] = self.pos_x
            model[3, 1] = self.pos_y
            model[3, 2] = self.pos_z
            model_mvp = (mvp @ model).astype('f4')
            self._prog_3d['mvp'].write(model_mvp.tobytes())
            self._prog_3d['video_tex'] = 0
            self._prog_3d['opacity'] = self.opacity
            self._vao_3d.render(moderngl.TRIANGLES)

    def regen(self) -> None:
        pass


register_element_type(
    "video",
    lambda ctx, device="cuda", **kw: VideoElement(ctx, device, **kw),
)
