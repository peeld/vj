"""
post/base.py — PostEffect abstract base + shared GL helpers.

All concrete post-effects import DEVICE, the quad shaders, and PostEffect
from here so the boilerplate lives in exactly one place.
"""

from __future__ import annotations

import ctypes
from abc import ABC, abstractmethod

import numpy as np
import moderngl
import warp as wp
from OpenGL.GL import (
    glBindBuffer, glBindFramebuffer, glBindTexture,
    glReadPixels, glTexSubImage2D,
    GL_PIXEL_PACK_BUFFER, GL_PIXEL_UNPACK_BUFFER,
    GL_RGBA, GL_UNSIGNED_BYTE, GL_TEXTURE_2D,
    GL_READ_FRAMEBUFFER,
)


# ── Warp device ────────────────────────────────────────────────────────────────
wp.init()
DEVICE: str = "cuda" if wp.get_cuda_device_count() > 0 else "cpu"

# Must be passed as the `data` argument of glReadPixels / glTexSubImage2D when
# a PBO is bound.  Passing None causes PyOpenGL to allocate a CPU buffer and
# silently bypass the PBO, defeating zero-copy entirely.
NULL_OFFSET = ctypes.c_void_p(0)


# ── Fullscreen-quad shaders (shared by all effects that blit to screen) ────────

_QUAD_VERT = """
#version 330
in vec2 in_pos;
out vec2 uv;
void main() {
    uv = vec2(in_pos.x * 0.5 + 0.5,
              0.5 - in_pos.y * 0.5);
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_QUAD_FRAG = """
#version 330
uniform sampler2D tex;
in  vec2 uv;
out vec4 f_color;
void main() { f_color = texture(tex, uv); }
"""

_QUAD_VERTS = np.array(
    [-1.0, -1.0,  1.0, -1.0,  -1.0,  1.0,  1.0,  1.0], dtype=np.float32
)


# ══════════════════════════════════════════════════════════════════════════════
#  Abstract base
# ══════════════════════════════════════════════════════════════════════════════

class PostEffect(ABC):
    """
    Interface for a post-processing unit.

    gui_merged (and any host) calls these in order each frame:

        effect.bind_scene_fbo()          # 1. activate & clear the effect's FBO
        draw_scene(mvp, t)               # 2. render 3-D scene into it
        effect.process(effect.fbo, t, 0) # 3. apply the effect
        effect.blit_to_screen(ctx.screen)# 4. put the result on screen

    One-time calls:
        effect.setup(ctx, w, h)          # at startup
        effect.resize(w, h)              # on window resize
        effect.on_key(key, action, keys) # forwarded key events (optional)

    Writing your own
    ----------------
    Subclass PostEffect and implement the five abstract members::

        class MyEffect(PostEffect):
            name = "my_effect"

            def setup(self, ctx, w, h): ...
            def process(self, scene_fbo, t, dt) -> moderngl.Texture: ...
            def bind_scene_fbo(self): ...
            def blit_to_screen(self, screen): ...

            @property
            def fbo(self) -> moderngl.Framebuffer: ...
    """

    # Human-readable label shown when switching effects.
    name: str = "effect"

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @abstractmethod
    def setup(self, ctx: moderngl.Context, w: int, h: int) -> None:
        """Allocate GPU resources sized to the viewport."""

    @abstractmethod
    def process(
        self,
        scene_fbo: moderngl.Framebuffer,
        t:  float,
        dt: float,
    ) -> moderngl.Texture:
        """
        Process the rendered scene and return a texture to blit to screen.
        ``scene_fbo`` is the FBO returned by ``self.fbo`` after the scene
        has been drawn into it.
        """

    def resize(self, w: int, h: int) -> None:
        """Recreate any size-dependent resources when the window is resized."""

    def on_key(self, key, action, keys) -> None:
        """Optional: handle keyboard events forwarded from the host."""

    # ── Frame interface (must match for all effects) ───────────────────────────

    @property
    @abstractmethod
    def fbo(self) -> moderngl.Framebuffer:
        """The off-screen FBO the scene should be rendered into."""

    @abstractmethod
    def bind_scene_fbo(self) -> None:
        """Activate ``self.fbo`` and clear it, ready for scene drawing."""

    @abstractmethod
    def blit_to_screen(self, screen: moderngl.Framebuffer) -> None:
        """Blit the last processed result to *screen*."""
