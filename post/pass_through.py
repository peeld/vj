"""
post/pass_through.py — PassThroughEffect

Renders the scene into an off-screen FBO and blits it straight to the screen
with no processing.  Useful as a switcher baseline and for side-by-side
comparison against feedback effects.
"""

from __future__ import annotations

import moderngl

from post.base import PostEffect, _QUAD_VERT, _QUAD_FRAG, _QUAD_VERTS


class PassThroughEffect(PostEffect):
    """
    No-op post-effect: scene → FBO → screen, no feedback, no trails.

    Included in the effect switcher so you can flip back to a clean render
    mid-session without disabling the post pipeline entirely.
    """

    name = "pass_through"

    def __init__(self) -> None:
        self._ctx:       moderngl.Context     | None = None
        self._fbo:       moderngl.Framebuffer | None = None
        self._quad_prog: moderngl.Program     | None = None
        self._quad_vao:  moderngl.VertexArray | None = None

    # ── PostEffect interface ──────────────────────────────────────────────────

    def setup(self, ctx: moderngl.Context, w: int, h: int) -> None:
        self._ctx = ctx
        self._build_fbo(w, h)
        self._build_quad(ctx)

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
        # Nothing to do — the scene is already in self._fbo.
        return scene_fbo.color_attachments[0]

    def blit_to_screen(self, screen: moderngl.Framebuffer) -> None:
        screen.use()
        self._ctx.disable(moderngl.DEPTH_TEST)
        self._ctx.clear(0.0, 0.0, 0.0, 1.0)
        self._fbo.color_attachments[0].use(0)
        self._quad_prog["tex"].value = 0
        self._quad_vao.render(moderngl.TRIANGLE_STRIP)

    def resize(self, w: int, h: int) -> None:
        if self._ctx is not None:
            self._build_fbo(w, h)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_fbo(self, w: int, h: int) -> None:
        self._fbo = self._ctx.framebuffer(
            color_attachments=[self._ctx.texture((w, h), 4)],
            depth_attachment=self._ctx.depth_texture((w, h)),
        )

    def _build_quad(self, ctx: moderngl.Context) -> None:
        vbo = ctx.buffer(_QUAD_VERTS.tobytes())
        self._quad_prog = ctx.program(vertex_shader=_QUAD_VERT, fragment_shader=_QUAD_FRAG)
        self._quad_vao  = ctx.vertex_array(self._quad_prog, [(vbo, "2f", "in_pos")])
