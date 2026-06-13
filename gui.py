"""
gui.py
WarpCubeGUI -- user interaction, canvas management, and optional GPU feedback post-effect.

Responsibilities:
  - Window / GL context lifecycle (via moderngl-window)
  - Camera state (yaw, pitch, distance) and MVP construction
  - Mouse / keyboard input
  - Per-frame orchestration: step operations, upload data, issue draws
  - Optional GPU feedback-loop post-processing (zoom / rotate / ripple / trails)
  - Optional elliptical orbit camera (always points at scene centre)

Everything else lives in:
  data.py          -- Warp arrays and configuration (PointCloudData, BallData)
  operation.py     -- Warp kernel dispatch (PointCloudOperation, BallOperation)
  drawable.py      -- Shaders + GPU buffers + draw calls (PointsDrawable, LinesDrawable, ShapeDrawable)
  warp_feedback.py -- FeedbackLoop and FeedbackParams
  nn_graph.py      -- NNGraph: animated KNN graph (Warp kernels + GL drawables)
  camera.py        -- OrbitCamera: spherical camera with optional auto-orbit

Controls:
  Mouse drag   -- orbit camera  (also temporarily overrides orbit-cam)
  Scroll       -- zoom          (also temporarily overrides orbit-cam)
  O            -- toggle elliptical orbit camera on / off
  R            -- randomise point cloud
  P            -- toggle post-effect on / off
  ESC          -- quit

Feedback tweaks (only active while post-effect is on):
  Z / X        -- scene blend ↓ / ↑  (how much fresh 3D bleeds in each frame)
  D / F        -- decay ↓ / ↑        (trail length)
  Q / W        -- rotation speed ↓ / ↑
  A / S        -- zoom ↓ / ↑
  H / J        -- hue shift speed ↓ / ↑  (colour cycling on trails)
  C / V        -- chromatic aberration ↓ / ↑
  B / N        -- saturation boost ↓ / ↑
  K / L        -- smear strength ↓ / ↑
  M            -- cycle smear pattern (swirl → swirl_cw → rightward → …)
"""

from dataclasses import dataclass

import moderngl
import moderngl_window as mglw
import warp as wp

from drawlib.data import BallData, PointCloudData
from drawlib.operation import BallOperation, PointCloudOperation
from drawlib.drawable import DynamicLinesDrawable, LinesDrawable, PointsDrawable, ShapeDrawable
from drawlib.post_effect import FeedbackPostEffect
from drawlib.warp_feedback import FeedbackParams
from drawlib.nn_graph import NNGraph
from drawlib.helpers import build_wireframe
from drawlib.camera import OrbitCamera

# ── Warp device ────────────────────────────────────────────────────────────────
wp.init()
DEVICE = "cuda" if wp.get_cuda_device_count() > 0 else "cpu"
print(f"[warp] device: {DEVICE}")


# ── Scene-specific feedback preset ────────────────────────────────────────────

@dataclass
class WarpSceneParams(FeedbackParams):
    base_zoom        : float = 1.008
    zoom_sensitivity : float = 0.0
    base_rot         : float = 0.003
    rot_sensitivity  : float = 0.0
    decay            : float = 0.985
    ripple_strength  : float = 8.0
    ripple_freq      : float = 10.0
    hue_shift        : float = 0.018
    chroma_offset    : float = 0.012
    sat_boost        : float = 1.15
    smear_strength   : float = 0.0


# ── Main GUI class ─────────────────────────────────────────────────────────────

class WarpCubeGUI(mglw.WindowConfig):
    title        = "Warp -- Colored 3D Point Cloud"
    window_size  = (1280, 720)
    gl_version   = (3, 3)
    resizable    = True
    aspect_ratio = None

    # How strongly the fresh 3-D frame bleeds into the feedback buffer.
    # 0 = pure echo chamber, 1 = no trails.
    SCENE_ALPHA: float = 0.3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # -- Data -----------------------------------------------------------------
        self.cloud_data = PointCloudData()
        self.ball_data  = BallData(cube_half=self.cloud_data.cube_half)

        # -- Operations -----------------------------------------------------------
        self.cloud_op = PointCloudOperation(self.cloud_data)
        self.ball_op  = BallOperation(self.ball_data)

        # -- Drawables ------------------------------------------------------------
        self.cloud_draw = PointsDrawable(self.ctx)
        self.cloud_draw.setup(
            self.cloud_data.positions_numpy(),
            self.cloud_data.colors_numpy(),
        )

        self.wire_draw = LinesDrawable(self.ctx)
        self.wire_draw.setup(*build_wireframe(self.cloud_data.cube_half))

        self.ball_draw = ShapeDrawable(self.ctx)
        self.ball_draw.setup(
            self.ball_data.positions_numpy(),
            BallData.COLORS,
        )

        # -- NN graph -------------------------------------------------------------
        self.nn_graph = NNGraph(self.ctx, device=DEVICE)

        # -- GL state -------------------------------------------------------------
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # -- Camera ---------------------------------------------------------------
        self.camera = OrbitCamera()
        self._drag  = False

        self.time = 0.0

        # -- Post-effect (feedback loop) ------------------------------------------
        self.post_effect = True   # default ON; toggle with P

        w, h = self.window_size
        self._post_effect = FeedbackPostEffect(
            params        = WarpSceneParams(),
            scene_alpha   = self.SCENE_ALPHA,
            smear_pattern = "swirl",
        )
        self._post_effect.setup(self.ctx, w, h)

        print("[gui] post-effect ON  (P to toggle)")
        print("[gui] orbit camera ON  (O to toggle; mouse/scroll overrides for 2s)")

    # -- Per-frame ----------------------------------------------------------------

    def on_render(self, current_time: float, frame_time: float):
        self.time += frame_time

        # Simulate
        self.ball_op.step(frame_time)
        self.cloud_op.step(self.time, frame_time, self.ball_data)

        # Step NN graph (animate points, rebuild KNN + edge buffer)
        self.nn_graph.step(self.time)

        # Upload updated GPU data via CUDA-GL interop
        self.cloud_draw.write_warp(self.cloud_data.wp_pos, self.cloud_data.wp_col)
        self.ball_draw.write_warp(self.ball_data.wp_pos)
        self.nn_graph.upload()

        self.camera.tick(frame_time)
        mvp = self.camera.mvp(self.window_size)

        if self.post_effect:
            self._render_with_post(current_time, mvp)
        else:
            self._render_direct(mvp)

    def _draw_scene(self, mvp):
        self.cloud_draw.draw(mvp)
        self.wire_draw.draw(mvp)
        self.nn_graph.draw(mvp)
        # self.ball_draw.draw(mvp, point_size=80.0)

    def _render_direct(self, mvp):
        """Render the 3-D scene straight to the screen framebuffer."""
        self.ctx.screen.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.04, 0.04, 0.06, 1.0)
        self._draw_scene(mvp)

    def _render_with_post(self, current_time: float, mvp):
        """Render scene → post-effect FBO, process, blit result to screen."""
        self._post_effect.bind_scene_fbo()
        self._draw_scene(mvp)
        self._post_effect.process(self._post_effect.fbo, current_time, dt=0.0)
        self._post_effect.blit_to_screen(self.ctx.screen)


    # -- Input --------------------------------------------------------------------

    def on_mouse_press_event(self, x, y, button):
        if button == 1:
            self._drag = True

    def on_mouse_release_event(self, x, y, button):
        if button == 1:
            self._drag = False

    def on_mouse_drag_event(self, x, y, dx, dy):
        if self._drag:
            self.camera.on_drag(dx, dy)

    def on_mouse_scroll_event(self, x_offset, y_offset):
        self.camera.on_scroll(y_offset)

    def on_key_event(self, key, action, modifiers):
        if action != self.wnd.keys.ACTION_PRESS:
            return

        keys = self.wnd.keys

        if key == keys.R:
            self.cloud_data.randomize()
            self.cloud_draw.update(
                self.cloud_data.positions_numpy(),
                self.cloud_data.colors_numpy(),
            )
            import random
            nn_seed = random.randint(0, 2**31 - 1)
            self.nn_graph.randomize(nn_seed)
            print(f"Points randomized  (nn_graph seed={nn_seed})")

        elif key == keys.ESCAPE:
            self.wnd.close()

        elif key == keys.P:
            self.post_effect = not self.post_effect
            state = "ON" if self.post_effect else "OFF"
            print(f"post-effect: {state}")

        elif key == keys.O:
            self.camera.orbit_enabled = not self.camera.orbit_enabled
            if self.camera.orbit_enabled:
                self.camera._user_idle = 0.0   # resume orbit immediately
                print("orbit camera: ON  (mouse/scroll to take over; auto-resumes after 2s)")
            else:
                print("orbit camera: OFF")

        # ── Feedback tweaks: delegate to post-effect ──────────────────────────
        elif self.post_effect:
            self._post_effect.on_key(key, action, keys)


    def on_resize(self, width: int, height: int):
        self._post_effect.resize(width, height)


if __name__ == "__main__":
    mglw.run_window_config(WarpCubeGUI)
