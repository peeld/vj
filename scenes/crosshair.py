"""
scenes/crosshair.py
1000 random points distributed through a volume.  A custom shader receives
the 3-D world positions directly, projects them on the GPU, and constructs
full-screen crosshair lines entirely inside the vertex shader.

Instanced rendering — no CPU projection, no NDC VBO:
  - 1 instance  = 1 point  (vec3 world position + vec4 colour)
  - 4 vertices  = 2 LINES primitives
    gl_VertexID 0,1 → horizontal line  (-1, ndc_y) → ( 1, ndc_y)
    gl_VertexID 2,3 → vertical   line  (ndc_x, -1) → (ndc_x,  1)

The vertex shader applies the MVP, divides to NDC, then picks the correct
endpoint.  Points behind the camera are pushed off-screen and clipped.
Additive blending makes overlapping lines stack into bright hot-spots.

Run with:
    python run_crosshair.py
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import moderngl

from drawlib.scene import Scene


# ── Constants ─────────────────────────────────────────────────────────────────

N_POINTS  = 100
CUBE_HALF = 2.0


# ── Shader ────────────────────────────────────────────────────────────────────

_CROSS_VERT = """
#version 330

uniform mat4 mvp;

in vec3 in_position;  // per-instance: 3-D world position
in vec4 in_color;     // per-instance: RGBA

out vec4 v_color;

void main() {
    v_color = in_color;

    // Project point into clip space
    vec4 clip = mvp * vec4(in_position, 1.0);

    // Points behind the camera get degenerate coords — GL clips them away
    if (clip.w <= 0.0) {
        gl_Position = vec4(2.0, 2.0, 0.0, 1.0);
        return;
    }

    // Perspective divide -> NDC
    vec2 ndc = clip.xy / clip.w;

    // Construct one of 4 line endpoints
    if      (gl_VertexID == 0) gl_Position = vec4(-1.0, ndc.y, 0.0, 1.0);
    else if (gl_VertexID == 1) gl_Position = vec4( 1.0, ndc.y, 0.0, 1.0);
    else if (gl_VertexID == 2) gl_Position = vec4(ndc.x, -1.0, 0.0, 1.0);
    else                       gl_Position = vec4(ndc.x,  1.0, 0.0, 1.0);
}
"""

_CROSS_FRAG = """
#version 330

in  vec4 v_color;
out vec4 f_color;

void main() {
    f_color = v_color;
}
"""


# ── CrosshairDrawable ─────────────────────────────────────────────────────────

class CrosshairDrawable:
    """
    Draws a pair of full-screen crosshair lines for every point in a cloud.

    The vertex shader does all the projection work; this class just owns the
    VBOs (positions + colours), the VAO, and issues the instanced draw call.
    """

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._n  = 0

    def setup(self, positions: np.ndarray, colors: np.ndarray) -> None:
        """
        positions : (N, 3) float32  – world-space 3-D positions
        colors    : (N, 4) float32  – per-point RGBA
        """
        self._n = len(positions)

        self.prog = self.ctx.program(
            vertex_shader=_CROSS_VERT,
            fragment_shader=_CROSS_FRAG,
        )

        self._vbo_pos = self.ctx.buffer(positions.astype(np.float32).tobytes())
        self._vbo_col = self.ctx.buffer(colors.astype(np.float32).tobytes())

        # /i  sets attribute divisor = 1 → one value consumed per instance
        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (self._vbo_pos, "3f /i", "in_position"),
                (self._vbo_col, "4f /i", "in_color"),
            ],
        )

    def draw(self, mvp: np.ndarray) -> None:
        self.prog["mvp"].write(mvp.tobytes())

        # Additive blending: faint lines stack into bright overlap zones
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        self.vao.render(moderngl.LINES, vertices=4, instances=self._n)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA


# ── Scene ─────────────────────────────────────────────────────────────────────

class CrosshairScene(Scene):
    """
    1000 random points in a cube volume.
    Each point projects to screen space and draws a full-screen H+V crosshair.
    Camera auto-rotates so the line grid continuously reshuffles.
    """

    title       = "Warp — Crosshair Volume"
    auto_rotate = True
    cam_dist    = 5.0
    cam_pitch   = -20.0

    def setup(self, ctx: moderngl.Context) -> None:
        rng = np.random.default_rng(7)

        positions = rng.uniform(
            -CUBE_HALF, CUBE_HALF, (N_POINTS, 3)
        ).astype(np.float32)

        # Hue along X axis, low alpha so lines are faint individually
        t  = (positions[:, 0] + CUBE_HALF) / (2.0 * CUBE_HALF)
        h6 = t * 6.0
        r  = np.clip(np.abs(h6 - 3.0) - 1.0, 0.0, 1.0)
        g  = np.clip(2.0 - np.abs(h6 - 2.0), 0.0, 1.0)
        b  = np.clip(2.0 - np.abs(h6 - 4.0), 0.0, 1.0)
        alpha  = np.full(N_POINTS, 0.6, dtype=np.float32)
        colors = np.stack([r, g, b, alpha], axis=1).astype(np.float32)

        self._cross = CrosshairDrawable(ctx)
        self._cross.setup(positions, colors)

    def step(self, t: float, dt: float) -> None:
        pass

    def draw(self, mvp: np.ndarray) -> None:
        self._cross.draw(mvp)
