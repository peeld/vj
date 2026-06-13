"""
viewport.py
Viewport3D — the one base class you never need to touch.

Provides:
  - moderngl-window lifecycle (GL context, window, event loop)
  - Orbit camera (yaw / pitch / distance → MVP matrix)
  - Mouse drag to orbit, scroll to zoom
  - Off-screen FBO + fullscreen-quad blit for post-effects
  - P key to toggle post-effect on/off at runtime
  - ESC to quit
  - Auto-rotation (cam_yaw += dt) — disable via Scene.auto_rotate = False

Usage
-----
    from viewport import run
    run(MyScene)

    # Optional overrides:
    run(MyScene, title="My App", window_size=(1920, 1080))
"""

from __future__ import annotations

import numpy as np
import moderngl
import moderngl_window as mglw
import warp as wp

from scene import Scene


# ── Warp init (once per process) ──────────────────────────────────────────────
wp.init()
_DEVICE = "cuda" if wp.get_cuda_device_count() > 0 else "cpu"
print(f"[viewport] warp device: {_DEVICE}")


# ── Fullscreen-quad shaders ────────────────────────────────────────────────────

_QUAD_VERT = """
#version 330
in  vec2 in_pos;
out vec2 uv;
void main() {
    // Flip Y so (0,0) = top-left (image space) to match the feedback buffer.
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


# ══════════════════════════════════════════════════════════════════════════════
#  Viewport3D
# ══════════════════════════════════════════════════════════════════════════════

class Viewport3D(mglw.WindowConfig):
    """
    Do not subclass this directly — use ``run(YourScene)`` instead.
    ``run()`` injects ``_scene_class`` before handing off to moderngl-window.
    """

    # Set by run() ─────────────────────────────────────────────────────────────
    _scene_class: type[Scene] = None

    # moderngl-window config ───────────────────────────────────────────────────
    title        = "Warp Viewport"
    window_size  = (1280, 720)
    gl_version   = (3, 3)
    resizable    = True
    aspect_ratio = None

    # ── Init ──────────────────────────────────────────────────────────────────

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # ── GL state ──────────────────────────────────────────────────────
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # ── Instantiate scene ─────────────────────────────────────────────
        self._scene: Scene = self._scene_class()
        self._scene.setup(self.ctx)

        # ── Camera (seed from scene class attrs if overridden) ────────────
        self.cam_yaw   = float(self._scene_class.cam_yaw)
        self.cam_pitch = float(self._scene_class.cam_pitch)
        self.cam_dist  = float(self._scene_class.cam_dist)
        self._drag       = False
        self._last_mouse = (0, 0)

        self.time = 0.0

        # ── Post-effect ───────────────────────────────────────────────────
        self._post         = self._scene.post_effect   # may be None
        self._post_enabled = self._post is not None

        w, h = self.window_size

        if self._post is not None:
            self._post.setup(self.ctx, w, h)
            self._fbo = self._make_fbo(w, h)
            self._setup_quad()
            print(f"[viewport] post-effect: {type(self._post).__name__}  (P to toggle)")
        else:
            self._fbo      = None
            self._quad_vao = None
            self._quad_prog = None

    # ── FBO / quad helpers ────────────────────────────────────────────────────

    def _make_fbo(self, w: int, h: int) -> moderngl.Framebuffer:
        return self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((w, h), 4)],
            depth_attachment=self.ctx.depth_texture((w, h)),
        )

    def _setup_quad(self) -> None:
        quad = np.array([-1.0, -1.0,  1.0, -1.0,  -1.0,  1.0,  1.0,  1.0],
                        dtype=np.float32)
        vbo             = self.ctx.buffer(quad.tobytes())
        self._quad_prog = self.ctx.program(vertex_shader=_QUAD_VERT,
                                           fragment_shader=_QUAD_FRAG)
        self._quad_vao  = self.ctx.vertex_array(
            self._quad_prog, [(vbo, "2f", "in_pos")]
        )

    # ── Per-frame ─────────────────────────────────────────────────────────────

    def on_render(self, current_time: float, frame_time: float):
        self.time += frame_time
        mvp = self._build_mvp()

        self._scene.step(self.time, frame_time)

        if self._post is not None and self._post_enabled:
            self._render_with_post(mvp, current_time, frame_time)
        else:
            self._render_direct(mvp)

        if self._scene.auto_rotate:
            self.cam_yaw += frame_time

    def _render_direct(self, mvp: np.ndarray) -> None:
        """Scene rendered straight to the screen framebuffer."""
        self.ctx.screen.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.04, 0.04, 0.06, 1.0)
        self._scene.draw(mvp)

    def _render_with_post(
        self, mvp: np.ndarray, t: float, dt: float
    ) -> None:
        """Scene → off-screen FBO → post-effect → blit to screen."""
        w, h = self.window_size

        # 1. Render scene to FBO
        self._fbo.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.04, 0.04, 0.06, 1.0)
        self._scene.draw(mvp)

        # 2. Post-effect processes the FBO, returns a texture
        result_tex = self._post.process(self._fbo, t, dt)

        # 3. Blit result texture to screen via fullscreen quad
        self.ctx.screen.use()
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        result_tex.use(0)
        self._quad_prog["tex"].value = 0
        self._quad_vao.render(moderngl.TRIANGLE_STRIP)

    # ── Camera ────────────────────────────────────────────────────────────────

    def _build_mvp(self) -> np.ndarray:
        yaw   = np.radians(self.cam_yaw)
        pitch = np.radians(np.clip(self.cam_pitch, -89.0, 89.0))

        # Spherical → Cartesian eye position
        eye = np.array([
            self.cam_dist * np.cos(pitch) * np.sin(yaw),
            self.cam_dist * np.sin(pitch),
            self.cam_dist * np.cos(pitch) * np.cos(yaw),
        ], dtype=np.float64)

        # Look-at view matrix
        fwd      = -eye / (np.linalg.norm(eye) + 1e-12)
        world_up = np.array([0.0, 1.0, 0.0])
        right    = np.cross(fwd, world_up)
        right   /= np.linalg.norm(right) + 1e-12
        up       = np.cross(right, fwd)

        view = np.array([
            [ right[0],  right[1],  right[2], -np.dot(right, eye)],
            [    up[0],     up[1],     up[2], -np.dot(up,    eye)],
            [  -fwd[0],   -fwd[1],   -fwd[2],  np.dot(fwd,   eye)],
            [        0,         0,         0,                    1],
        ], dtype=np.float32)

        # Perspective projection
        w, h  = self.window_size
        aspect = w / max(h, 1)
        fov    = np.radians(45.0)
        near, far = 0.1, 50.0
        f = 1.0 / np.tan(fov / 2)

        proj = np.array([
            [f / aspect,  0,                            0,                          0],
            [          0, f,                            0,                          0],
            [          0, 0,   -(far + near) / (far - near), -2*far*near/(far - near)],
            [          0, 0,                           -1,                          0],
        ], dtype=np.float32)

        # Column-major for GLSL — transpose the row-major result
        mvp = (proj @ view).T
        return np.ascontiguousarray(mvp, dtype=np.float32)

    # ── Input handling ────────────────────────────────────────────────────────

    def on_key_event(self, key, action, modifiers):
        keys = self.wnd.keys

        if action != keys.ACTION_PRESS:
            # Forward release events to scene/post too in case they care
            self._scene.on_key(key, action, keys)
            if self._post is not None:
                self._post.on_key(key, action, keys)
            return

        if key == keys.ESCAPE:
            self.wnd.close()
            return

        if key == keys.P and self._post is not None:
            self._post_enabled = not self._post_enabled
            state = "ON" if self._post_enabled else "OFF"
            print(f"[viewport] post-effect {state}")
            return

        # Forward to scene first, then post-effect
        self._scene.on_key(key, action, keys)
        if self._post is not None:
            self._post.on_key(key, action, keys)

    def on_mouse_drag_event(self, x: int, y: int, dx: int, dy: int):
        self.cam_yaw   += dx * 0.3
        self.cam_pitch  = float(np.clip(self.cam_pitch + dy * 0.3, -89.0, 89.0))

    def on_mouse_scroll_event(self, x_offset: float, y_offset: float):
        self.cam_dist = float(np.clip(self.cam_dist - y_offset * 0.2, 0.3, 30.0))

    # ── Resize ────────────────────────────────────────────────────────────────

    def on_resize(self, width: int, height: int):
        if self._post is not None:
            self._post.resize(width, height)
            self._fbo = self._make_fbo(width, height)


# ══════════════════════════════════════════════════════════════════════════════
#  Public entry point
# ══════════════════════════════════════════════════════════════════════════════

def run(
    scene_class:  type[Scene],
    title:        str | None          = None,
    window_size:  tuple[int, int]     = (1280, 720),
) -> None:
    """
    Launch a Viewport3D window running ``scene_class``.

    Parameters
    ----------
    scene_class  : Scene subclass to instantiate.
    title        : Window title (defaults to ``scene_class.title``).
    window_size  : Initial window dimensions in pixels.

    Example
    -------
        from viewport import run
        run(MyScene, title="My Experiment")
    """
    cls = type(
        f"_{scene_class.__name__}App",
        (Viewport3D,),
        {
            "_scene_class": scene_class,
            "title":        title or getattr(scene_class, "title", scene_class.__name__),
            "window_size":  window_size,
        },
    )
    mglw.run_window_config(cls)
