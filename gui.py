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

import numpy as np
import moderngl
import moderngl_window as mglw
import warp as wp

from drawlib.data import BallData, PointCloudData
from drawlib.operation import BallOperation, PointCloudOperation
from drawlib.drawable import DynamicLinesDrawable, LinesDrawable, PointsDrawable, ShapeDrawable
from drawlib.post_effect import FeedbackPostEffect
from drawlib.warp_feedback import FeedbackParams

# ── Warp device ────────────────────────────────────────────────────────────────
wp.init()
DEVICE = "cuda" if wp.get_cuda_device_count() > 0 else "cpu"
print(f"[warp] device: {DEVICE}")



# ── NN-graph constants ─────────────────────────────────────────────────────────

_NN_N          = 500
_NN_K          = 5
_NN_N_EDGES    = _NN_N * _NN_K
_NN_N_VERTS    = _NN_N_EDGES * 2   # two vertices per segment


# ── NN-graph kernels ───────────────────────────────────────────────────────────

@wp.kernel
def _nn_init_points(
    positions: wp.array(dtype=wp.vec3),
    base_pos:  wp.array(dtype=wp.vec3),
    colors:    wp.array(dtype=wp.vec4),
    seed:      int,
):
    """Randomise positions, record as animation base, colour by XYZ."""
    i  = wp.tid()
    r0 = wp.rand_init(seed, i * 3 + 0)
    r1 = wp.rand_init(seed, i * 3 + 1)
    r2 = wp.rand_init(seed, i * 3 + 2)
    x  = wp.randf(r0) * 2.0 - 1.0
    y  = wp.randf(r1) * 2.0 - 1.0
    z  = wp.randf(r2) * 2.0 - 1.0
    p  = wp.vec3(x, y, z)
    positions[i] = p
    base_pos[i]  = p
    r = wp.clamp(x * 0.5 + 0.7, 0.2, 1.0)
    g = wp.clamp(y * 0.5 + 0.6, 0.2, 1.0)
    b = wp.clamp(z * 0.5 + 0.9, 0.3, 1.0)
    colors[i] = wp.vec4(r, g, b, 1.0)


@wp.kernel
def _nn_animate_points(
    positions: wp.array(dtype=wp.vec3),
    base_pos:  wp.array(dtype=wp.vec3),
    t:         float,
    amplitude: float,
):
    """Sinusoidal drift of each point around its base position."""
    i  = wp.tid()
    bp = base_pos[i]
    fi = float(i)
    dx = wp.sin(t * 0.6  + fi * 0.137) * amplitude
    dy = wp.cos(t * 0.45 + fi * 0.251) * amplitude
    dz = wp.sin(t * 0.8  + fi * 0.389) * amplitude
    positions[i] = wp.vec3(bp[0] + dx, bp[1] + dy, bp[2] + dz)


@wp.kernel
def _nn_find_knn5(
    positions:  wp.array(dtype=wp.vec3),
    nn_indices: wp.array(dtype=int),
    nn_dists:   wp.array(dtype=float),
    n:          int,
):
    """Brute-force K=5 nearest-neighbour search (O(N^2), trivial at N=500)."""
    i  = wp.tid()
    pi = positions[i]

    bi0 = int(-1); bd0 = float(1.0e18)
    bi1 = int(-1); bd1 = float(1.0e18)
    bi2 = int(-1); bd2 = float(1.0e18)
    bi3 = int(-1); bd3 = float(1.0e18)
    bi4 = int(-1); bd4 = float(1.0e18)

    for j in range(n):
        if j == i:
            continue
        pj = positions[j]
        dx = pi[0] - pj[0]
        dy = pi[1] - pj[1]
        dz = pi[2] - pj[2]
        d2 = dx * dx + dy * dy + dz * dz

        if d2 < bd0:
            bi4 = bi3; bd4 = bd3
            bi3 = bi2; bd3 = bd2
            bi2 = bi1; bd2 = bd1
            bi1 = bi0; bd1 = bd0
            bi0 = j;   bd0 = d2
        elif d2 < bd1:
            bi4 = bi3; bd4 = bd3
            bi3 = bi2; bd3 = bd2
            bi2 = bi1; bd2 = bd1
            bi1 = j;   bd1 = d2
        elif d2 < bd2:
            bi4 = bi3; bd4 = bd3
            bi3 = bi2; bd3 = bd2
            bi2 = j;   bd2 = d2
        elif d2 < bd3:
            bi4 = bi3; bd4 = bd3
            bi3 = j;   bd3 = d2
        elif d2 < bd4:
            bi4 = j;   bd4 = d2

    base = i * 5
    nn_indices[base + 0] = bi0;  nn_dists[base + 0] = wp.sqrt(bd0)
    nn_indices[base + 1] = bi1;  nn_dists[base + 1] = wp.sqrt(bd1)
    nn_indices[base + 2] = bi2;  nn_dists[base + 2] = wp.sqrt(bd2)
    nn_indices[base + 3] = bi3;  nn_dists[base + 3] = wp.sqrt(bd3)
    nn_indices[base + 4] = bi4;  nn_dists[base + 4] = wp.sqrt(bd4)


@wp.kernel
def _nn_build_edges(
    positions:  wp.array(dtype=wp.vec3),
    colors:     wp.array(dtype=wp.vec4),
    nn_indices: wp.array(dtype=int),
    nn_dists:   wp.array(dtype=float),
    edge_pos:   wp.array(dtype=wp.vec3),
    edge_col:   wp.array(dtype=wp.vec4),
    fade_dist:  float,
):
    """Write flat edge VBO from precomputed KNN; alpha fades with segment length."""
    i         = wp.tid()
    pi        = positions[i]
    ci        = colors[i]
    nn_base   = i * 5
    edge_base = i * 5 * 2

    for k in range(5):
        j    = nn_indices[nn_base + k]
        dist = nn_dists[nn_base + k]
        v    = edge_base + k * 2

        if j < 0:
            edge_pos[v]     = pi
            edge_pos[v + 1] = pi
            edge_col[v]     = wp.vec4(0.0, 0.0, 0.0, 0.0)
            edge_col[v + 1] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        else:
            pj    = positions[j]
            cj    = colors[j]
            t     = wp.clamp(1.0 - dist / fade_dist, 0.0, 1.0)
            alpha = t * t * 0.95
            edge_pos[v]     = pi
            edge_pos[v + 1] = pj
            edge_col[v]     = wp.vec4(ci[0], ci[1], ci[2], alpha)
            edge_col[v + 1] = wp.vec4(cj[0], cj[1], cj[2], alpha)


# ── Fullscreen-quad shaders ────────────────────────────────────────────────────

_QUAD_VERT = """
#version 330
in vec2 in_pos;
out vec2 uv;
void main() {
    // uv (0,0) = bottom-left in GL convention; flip Y so (0,0) = top-left
    // to match the feedback buffer's image-space layout.
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


# ── Main GUI class ─────────────────────────────────────────────────────────────

class WarpCubeGUI(mglw.WindowConfig):
    title        = "Warp -- Colored 3D Point Cloud"
    window_size  = (1280, 720)
    gl_version   = (3, 3)
    resizable    = True
    aspect_ratio = None

    # How strongly the fresh 3-D frame bleeds into the feedback buffer.
    # 0 = pure echo chamber, 1 = no trails.
    SCENE_ALPHA: float = 0.13

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
        self.wire_draw.setup(*self._build_wireframe(self.cloud_data.cube_half))

        self.ball_draw = ShapeDrawable(self.ctx)
        self.ball_draw.setup(
            self.ball_data.positions_numpy(),
            BallData.COLORS,
        )

        # -- NN-graph drawables ---------------------------------------------------
        self._nn_wp_pos      = wp.zeros(_NN_N, dtype=wp.vec3)
        self._nn_wp_base_pos = wp.zeros(_NN_N, dtype=wp.vec3)
        self._nn_wp_col      = wp.zeros(_NN_N, dtype=wp.vec4)
        self._nn_wp_nn_idx   = wp.zeros(_NN_N * _NN_K, dtype=int)
        self._nn_wp_nn_dist  = wp.zeros(_NN_N * _NN_K, dtype=float)
        self._nn_wp_edge_pos = wp.zeros(_NN_N_VERTS, dtype=wp.vec3)
        self._nn_wp_edge_col = wp.zeros(_NN_N_VERTS, dtype=wp.vec4)

        wp.launch(
            _nn_init_points,
            dim=_NN_N,
            inputs=[self._nn_wp_pos, self._nn_wp_base_pos, self._nn_wp_col, 42],
            device=DEVICE,
        )

        self.nn_pts_draw  = PointsDrawable(self.ctx)
        self.nn_pts_draw.setup(
            self._nn_wp_pos.numpy(),
            self._nn_wp_col.numpy(),
        )

        self.nn_edge_draw = DynamicLinesDrawable(self.ctx)
        self.nn_edge_draw.setup(_NN_N_EDGES)

        # -- GL state -------------------------------------------------------------
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # -- Camera ---------------------------------------------------------------
        self.cam_yaw   =  35.0
        self.cam_pitch = -25.0
        self.cam_dist  =   1
        self._drag       = False
        self._last_mouse = (0, 0)

        # -- Orbit camera ---------------------------------------------------------
        # Elliptical orbit that always looks at the scene centre (0,0,0).
        # The path is a 3-D Lissajous ellipse:
        #   x = orbit_a * cos(omega * t)
        #   y = orbit_b * sin(omega * phi * t)   (phi = golden ratio → slow drift)
        #   z = orbit_a * sin(omega * t)
        # User mouse/scroll input overrides the orbit for ORBIT_RESUME_DELAY
        # seconds, after which automatic flight resumes.
        self.orbit_cam        = True    # O to toggle
        self._orbit_t         = 0.0    # accumulated orbit time
        self._orbit_a         = 1.5    # XZ semi-axis (horizontal radius)
        self._orbit_b         = 0.6    # Y semi-axis  (vertical amplitude)
        self._orbit_speed     = 0.22   # rad/s (horizontal)
        self._orbit_phi       = (1 + 5 ** 0.5) / 2 * 0.5   # ≈ 0.809 (vertical freq multiplier)
        self._user_idle       = 0.0    # counts down after user input; orbit resumes at 0
        self._ORBIT_RESUME_DELAY = 2.0  # seconds before orbit reclaims camera

        self.time = 0.0

        # -- Post-effect (feedback loop) ------------------------------------------
        self.post_effect = True   # default ON; toggle with P

        w, h = self.window_size

        self._post_effect = FeedbackPostEffect(
            params=FeedbackParams(
                base_zoom        = 1.008,
                zoom_sensitivity = 0.0,
                base_rot         = 0.003,
                rot_sensitivity  = 0.0,
                decay            = 0.985,
                ripple_strength  = 8.0,
                ripple_freq      = 10.0,
                hue_shift        = 0.018,
                chroma_offset    = 0.012,
                sat_boost        = 1.15,
                smear_strength   = 0.0,
            ),
            scene_alpha   = self.SCENE_ALPHA,
            smear_pattern = "swirl",
        )
        self._post_effect.setup(self.ctx, w, h)

        # Off-screen FBO (scene renders here; post-effect reads it)
        self._fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((w, h), 4)],
            depth_attachment=self.ctx.depth_texture((w, h)),
        )

        # Fullscreen quad for blitting the post-effect result
        quad = np.array([-1.0, -1.0,  1.0, -1.0,  -1.0,  1.0,  1.0,  1.0],
                        dtype=np.float32)
        self._quad_vbo  = self.ctx.buffer(quad.tobytes())
        self._quad_prog = self.ctx.program(vertex_shader=_QUAD_VERT,
                                           fragment_shader=_QUAD_FRAG)
        self._quad_vao  = self.ctx.vertex_array(
            self._quad_prog, [(self._quad_vbo, "2f", "in_pos")]
        )

        print("[gui] post-effect ON  (P to toggle)")
        print("[gui] orbit camera ON  (O to toggle; mouse/scroll overrides for 2s)")

    # -- Per-frame ----------------------------------------------------------------

    def on_render(self, current_time: float, frame_time: float):
        self.time += frame_time
        w, h = self.window_size

        # Simulate
        self.ball_op.step(frame_time)
        self.cloud_op.step(self.time, frame_time, self.ball_data)

        # Step NN graph (animate points, rebuild KNN + edge buffer)
        wp.launch(_nn_animate_points, dim=_NN_N,
                  inputs=[self._nn_wp_pos, self._nn_wp_base_pos, self.time, 0.08],
                  device=DEVICE)
        wp.launch(_nn_find_knn5, dim=_NN_N,
                  inputs=[self._nn_wp_pos, self._nn_wp_nn_idx, self._nn_wp_nn_dist, _NN_N],
                  device=DEVICE)
        wp.launch(_nn_build_edges, dim=_NN_N,
                  inputs=[self._nn_wp_pos, self._nn_wp_col,
                          self._nn_wp_nn_idx, self._nn_wp_nn_dist,
                          self._nn_wp_edge_pos, self._nn_wp_edge_col, 0.5],
                  device=DEVICE)

        # Upload updated GPU data via CUDA-GL interop
        self.cloud_draw.write_warp(self.cloud_data.wp_pos, self.cloud_data.wp_col)
        self.ball_draw.write_warp(self.ball_data.wp_pos)
        self.nn_pts_draw.write_warp(self._nn_wp_pos, self._nn_wp_col)
        self.nn_edge_draw.write_warp(self._nn_wp_edge_pos, self._nn_wp_edge_col)

        mvp = self._build_mvp()

        if self.post_effect:
            self._render_with_post(current_time, mvp, w, h)
        else:
            self._render_direct(mvp)

        self._tick_orbit_cam(frame_time)

    def _tick_orbit_cam(self, dt: float) -> None:
        """Advance the elliptical orbit camera, unless user input is active."""
        if not self.orbit_cam:
            return
        if self._user_idle > 0.0:
            self._user_idle -= dt
            return
        # Advance orbit time and drive yaw/pitch/dist from ellipse position
        self._orbit_t += dt
        ox = self._orbit_a * np.cos(self._orbit_t * self._orbit_speed)
        oy = self._orbit_b * np.sin(self._orbit_t * self._orbit_speed * self._orbit_phi)
        oz = self._orbit_a * np.sin(self._orbit_t * self._orbit_speed)
        dist = float(np.sqrt(ox * ox + oy * oy + oz * oz))
        self.cam_dist  = dist
        self.cam_yaw   = float(np.degrees(np.arctan2(ox, oz)))
        self.cam_pitch = float(np.degrees(np.arcsin(np.clip(oy / (dist + 1e-9), -1.0, 1.0))))

    def _render_direct(self, mvp: np.ndarray):
        """Render the 3-D scene straight to the screen framebuffer."""
        self.ctx.screen.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.04, 0.04, 0.06, 1.0)
        self.cloud_draw.draw(mvp)
        self.wire_draw.draw(mvp)
        self.nn_edge_draw.draw(mvp)
        self.nn_pts_draw.draw(mvp)
        # self.ball_draw.draw(mvp, point_size=80.0)

    def _render_with_post(self, current_time: float, mvp: np.ndarray, w: int, h: int):
        """Render scene → FBO, hand off to FeedbackPostEffect, blit result."""

        # 1. Render 3-D scene → off-screen FBO
        self._fbo.use()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.04, 0.04, 0.06, 1.0)
        self.cloud_draw.draw(mvp)
        self.wire_draw.draw(mvp)
        self.nn_edge_draw.draw(mvp)
        self.nn_pts_draw.draw(mvp)
        # self.ball_draw.draw(mvp, point_size=80.0)

        # 2. Post-effect: FBO → feedback loop → returns display texture
        display_tex = self._post_effect.process(self._fbo, current_time, dt=0.0)

        # 3. Blit to screen
        self.ctx.screen.use()
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)
        display_tex.use(0)
        self._quad_prog["tex"].value = 0
        self._quad_vao.render(moderngl.TRIANGLE_STRIP)

    # -- Helpers ------------------------------------------------------------------

    @staticmethod
    def _build_wireframe(half: float):
        """Return (vertices, colors, indices) numpy arrays for a unit wireframe cube."""
        h = half
        vertices = np.array([
            [-h,-h,-h], [ h,-h,-h], [ h, h,-h], [-h, h,-h],
            [-h,-h, h], [ h,-h, h], [ h, h, h], [-h, h, h],
        ], dtype=np.float32)
        indices = np.array([
            0,1, 1,2, 2,3, 3,0,
            4,5, 5,6, 6,7, 7,4,
            0,4, 1,5, 2,6, 3,7,
        ], dtype=np.uint32)
        colors = np.full((8, 4), [1.0, 1.0, 1.0, 0.45], dtype=np.float32)
        return vertices, colors, indices

    def _build_mvp(self) -> np.ndarray:
        yaw   = np.radians(self.cam_yaw)
        pitch = np.radians(self.cam_pitch)
        d     = self.cam_dist

        eye = np.array([
            d * np.cos(pitch) * np.sin(yaw),
            d * np.sin(pitch),
            d * np.cos(pitch) * np.cos(yaw),
        ], dtype=np.float32)

        center = np.zeros(3, dtype=np.float32)
        up     = np.array([0, 1, 0], dtype=np.float32)

        f = center - eye; f /= np.linalg.norm(f)
        r = np.cross(f, up); r /= np.linalg.norm(r)
        u = np.cross(r, f)
        view = np.eye(4, dtype=np.float32)
        view[0,:3] = r;  view[0,3] = -r.dot(eye)
        view[1,:3] = u;  view[1,3] = -u.dot(eye)
        view[2,:3] = -f; view[2,3] =  f.dot(eye)

        fov  = np.radians(55)
        w, h = self.window_size
        asp  = w / max(h, 1)
        near, far = 0.1, 100.0
        t = np.tan(fov / 2)
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0,0] =  1/(asp*t)
        proj[1,1] =  1/t
        proj[2,2] = -(far+near)/(far-near)
        proj[2,3] = -2*far*near/(far-near)
        proj[3,2] = -1

        return (proj @ view).T

    # -- Input --------------------------------------------------------------------

    def on_mouse_press_event(self, x, y, button):
        if button == 1:
            self._drag = True
            self._last_mouse = (x, y)

    def on_mouse_release_event(self, x, y, button):
        if button == 1:
            self._drag = False

    def on_mouse_drag_event(self, x, y, dx, dy):
        if self._drag:
            if self.orbit_cam:
                self._user_idle = self._ORBIT_RESUME_DELAY
            self.cam_yaw   += dx * 0.4
            self.cam_pitch  = np.clip(self.cam_pitch - dy * 0.4, -89, 89)

    def on_mouse_scroll_event(self, x_offset, y_offset):
        if self.orbit_cam:
            self._user_idle = self._ORBIT_RESUME_DELAY
        self.cam_dist = np.clip(self.cam_dist - y_offset * 0.2, 1.0, 12.0)

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
            wp.launch(
                _nn_init_points, dim=_NN_N,
                inputs=[self._nn_wp_pos, self._nn_wp_base_pos, self._nn_wp_col, nn_seed],
                device=DEVICE,
            )
            print(f"Points randomized  (nn_graph seed={nn_seed})")

        elif key == keys.ESCAPE:
            self.wnd.close()

        elif key == keys.P:
            self.post_effect = not self.post_effect
            state = "ON" if self.post_effect else "OFF"
            print(f"post-effect: {state}")

        elif key == keys.O:
            self.orbit_cam = not self.orbit_cam
            if self.orbit_cam:
                self._user_idle = 0.0   # resume orbit immediately
                print("orbit camera: ON  (mouse/scroll to take over; auto-resumes after 2s)")
            else:
                print("orbit camera: OFF")


        # ── Feedback tweaks: delegate to post-effect ──────────────────────────
        elif self.post_effect:
            self._post_effect.on_key(key, action, keys)


    def on_resize(self, width: int, height: int):
        """Recreate size-dependent GPU resources when the window is resized."""
        self._fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.texture((width, height), 4)],
            depth_attachment=self.ctx.depth_texture((width, height)),
        )
        self._post_effect.resize(width, height)


if __name__ == "__main__":
    mglw.run_window_config(WarpCubeGUI)
