"""
drawable.py
Drawable base class and concrete drawables for points, lines, and shapes.

Each Drawable owns its shader program and GPU buffers (VBO/VAO), but holds
NO simulation data.

Data flow -- two paths:
  write_warp(wp_array, ...)   GPU -> GPU via CUDA-GL interop (preferred, zero CPU)
  update(np_array, ...)       CPU -> GPU fallback, or for static/one-shot uploads
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import moderngl
import warp as wp


class Drawable(ABC):
    def __init__(self, ctx: moderngl.Context):
        self.ctx  = ctx
        self.prog: moderngl.Program     = None
        self.vao:  moderngl.VertexArray = None

    @abstractmethod
    def setup(self, *args, **kwargs): ...

    def update(self, *args, **kwargs): ...

    @abstractmethod
    def draw(self, mvp: np.ndarray, **kwargs): ...


def _register(gl_buffer: moderngl.Buffer) -> wp.RegisteredGLBuffer:
    return wp.RegisteredGLBuffer(gl_buffer.glo, wp.get_preferred_device())


# ---------------------------------------------------------------------------
# Points
# ---------------------------------------------------------------------------

_POINTS_VERT = """
    #version 330
    uniform mat4 mvp;
    in vec3 in_position;
    in vec4 in_color;
    out vec4 v_color;
    void main() {
        gl_Position = mvp * vec4(in_position, 1.0);
        v_color = in_color;
        gl_PointSize = 2.0;
    }
"""

_POINTS_FRAG = """
    #version 330
    in vec4 v_color;
    out vec4 f_color;
    void main() {
        vec2 c = gl_PointCoord * 2.0 - 1.0;
        if (dot(c, c) > 1.0) discard;
        if (v_color[3] < 0.01) discard;
        f_color = v_color;
    }
"""


class PointsDrawable(Drawable):
    def setup(self, positions: np.ndarray, colors: np.ndarray):
        self._n = len(positions)
        self.prog = self.ctx.program(vertex_shader=_POINTS_VERT, fragment_shader=_POINTS_FRAG)
        self._vbo_pos = self.ctx.buffer(positions.tobytes())
        self._vbo_col = self.ctx.buffer(colors.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [
            (self._vbo_pos, "3f", "in_position"),
            (self._vbo_col, "4f", "in_color"),
        ])
        self._reg_pos = _register(self._vbo_pos)
        self._reg_col = _register(self._vbo_col)

    def write_warp(self, wp_pos: wp.array, wp_col: wp.array):
        gl_pos = self._reg_pos.map(dtype=wp.vec3, shape=(self._n,))
        wp.copy(gl_pos, wp_pos)
        self._reg_pos.unmap()
        gl_col = self._reg_col.map(dtype=wp.vec4, shape=(self._n,))
        wp.copy(gl_col, wp_col)
        self._reg_col.unmap()

    def update(self, positions: np.ndarray, colors: np.ndarray):
        self._vbo_pos.write(positions.tobytes())
        self._vbo_col.write(colors.tobytes())

    def draw(self, mvp: np.ndarray, **kwargs):
        self.prog["mvp"].write(mvp.tobytes())
        self.vao.render(moderngl.POINTS)


# ---------------------------------------------------------------------------
# Lines (static indexed)
# ---------------------------------------------------------------------------

_LINES_VERT = """
    #version 330
    uniform mat4 mvp;
    in vec3 in_position;
    in vec4 in_color;
    out vec4 v_color;
    void main() {
        gl_Position = mvp * vec4(in_position, 1.0);
        v_color = in_color;
    }
"""

_LINES_FRAG = """
    #version 330
    in vec4 v_color;
    out vec4 f_color;
    void main() { f_color = v_color; }
"""


class LinesDrawable(Drawable):
    def setup(self, vertices: np.ndarray, colors: np.ndarray, indices: np.ndarray):
        self.prog = self.ctx.program(vertex_shader=_LINES_VERT, fragment_shader=_LINES_FRAG)
        self._vbo_pos = self.ctx.buffer(vertices.tobytes())
        self._vbo_col = self.ctx.buffer(colors.tobytes())
        self._ibo     = self.ctx.buffer(indices.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [
            (self._vbo_pos, "3f", "in_position"),
            (self._vbo_col, "4f", "in_color"),
        ], self._ibo)

    def update(self, vertices: np.ndarray = None, colors: np.ndarray = None):
        if vertices is not None: self._vbo_pos.write(vertices.tobytes())
        if colors  is not None: self._vbo_col.write(colors.tobytes())

    def draw(self, mvp: np.ndarray, **kwargs):
        self.prog["mvp"].write(mvp.tobytes())
        self.vao.render(moderngl.LINES)


# ---------------------------------------------------------------------------
# Dynamic Lines & Triangles  (shared shaders, CUDA-GL interop)
# ---------------------------------------------------------------------------

_DYN_VERT = """
    #version 330
    uniform mat4 mvp;
    in vec3 in_position;
    in vec4 in_color;
    out vec4 v_color;
    void main() {
        gl_Position = mvp * vec4(in_position, 1.0);
        v_color = in_color;
    }
"""

_DYN_FRAG = """
    #version 330
    in vec4 v_color;
    out vec4 f_color;
    void main() { f_color = v_color; }
"""

# Keep the old names as aliases so existing code keeps working
_DYN_LINES_VERT = _DYN_VERT
_DYN_LINES_FRAG = _DYN_FRAG


def _make_dynamic_drawable(ctx, n_verts):
    """Shared helper: allocate VBOs + interop registers for n_verts vertices."""
    prog = ctx.program(vertex_shader=_DYN_VERT, fragment_shader=_DYN_FRAG)
    vbo_pos = ctx.buffer(np.zeros((n_verts, 3), dtype=np.float32).tobytes())
    vbo_col = ctx.buffer(np.zeros((n_verts, 4), dtype=np.float32).tobytes())
    vao = ctx.vertex_array(prog, [
        (vbo_pos, "3f", "in_position"),
        (vbo_col, "4f", "in_color"),
    ])
    reg_pos = _register(vbo_pos)
    reg_col = _register(vbo_col)
    return prog, vbo_pos, vbo_col, vao, reg_pos, reg_col


class DynamicLinesDrawable(Drawable):
    """N line segments, non-indexed. 2 verts per segment. CUDA-GL interop."""

    def setup(self, n_segments: int):
        self._n_verts = n_segments * 2
        (self.prog, self._vbo_pos, self._vbo_col,
         self.vao, self._reg_pos, self._reg_col) = _make_dynamic_drawable(self.ctx, self._n_verts)

    def write_warp(self, wp_pos: wp.array, wp_col: wp.array):
        gl_pos = self._reg_pos.map(dtype=wp.vec3, shape=(self._n_verts,))
        wp.copy(gl_pos, wp_pos)
        self._reg_pos.unmap()
        gl_col = self._reg_col.map(dtype=wp.vec4, shape=(self._n_verts,))
        wp.copy(gl_col, wp_col)
        self._reg_col.unmap()

    def update(self, positions: np.ndarray = None, colors: np.ndarray = None):
        if positions is not None: self._vbo_pos.write(positions.tobytes())
        if colors    is not None: self._vbo_col.write(colors.tobytes())

    def draw(self, mvp: np.ndarray, **kwargs):
        self.prog["mvp"].write(mvp.tobytes())
        self.vao.render(moderngl.LINES)


class DynamicTrianglesDrawable(Drawable):
    """N billboard quads stored as 6 non-indexed verts each. CUDA-GL interop.

    Vertex layout per quad (base = i*6):
        0 head_left   1 head_right   2 tail_right   <- triangle A
        3 head_left   4 tail_right   5 tail_left    <- triangle B
    """

    def setup(self, n_quads: int):
        self._n_verts = n_quads * 6
        (self.prog, self._vbo_pos, self._vbo_col,
         self.vao, self._reg_pos, self._reg_col) = _make_dynamic_drawable(self.ctx, self._n_verts)

    def write_warp(self, wp_pos: wp.array, wp_col: wp.array):
        gl_pos = self._reg_pos.map(dtype=wp.vec3, shape=(self._n_verts,))
        wp.copy(gl_pos, wp_pos)
        self._reg_pos.unmap()
        gl_col = self._reg_col.map(dtype=wp.vec4, shape=(self._n_verts,))
        wp.copy(gl_col, wp_col)
        self._reg_col.unmap()

    def update(self, positions: np.ndarray = None, colors: np.ndarray = None):
        if positions is not None: self._vbo_pos.write(positions.tobytes())
        if colors    is not None: self._vbo_col.write(colors.tobytes())

    def draw(self, mvp: np.ndarray, **kwargs):
        self.prog["mvp"].write(mvp.tobytes())
        self.vao.render(moderngl.TRIANGLES)


# ---------------------------------------------------------------------------
# Shapes  (sphere-shaded point sprites -- bouncing balls)
# ---------------------------------------------------------------------------

_SHAPES_VERT = """
    #version 330
    uniform mat4 mvp;
    uniform float point_size;
    in vec3 in_position;
    in vec4 in_color;
    out vec4 v_color;
    void main() {
        vec4 clip    = mvp * vec4(in_position, 1.0);
        gl_Position  = clip;
        gl_PointSize = point_size / clip.w;
        v_color = in_color;
    }
"""

_SHAPES_FRAG = """
    #version 330
    in vec4 v_color;
    out vec4 f_color;
    void main() {
        vec2 c   = gl_PointCoord * 2.0 - 1.0;
        float r2 = dot(c, c);
        if (r2 > 1.0) discard;
        vec3 normal   = vec3(c.x, -c.y, sqrt(1.0 - r2));
        vec3 light    = normalize(vec3(1.0, 2.0, 3.0));
        float diffuse = max(dot(normal, light), 0.0);
        float ambient = 0.25;
        vec3 rgb      = v_color.rgb * (ambient + diffuse * 0.75);
        float spec    = pow(max(reflect(-light, normal).z, 0.0), 32.0) * 0.6;
        f_color = vec4(rgb + spec, 1.0);
    }
"""


class ShapeDrawable(Drawable):
    """Sphere-shaded point sprites (balls). Colors static; positions dynamic."""

    def setup(self, positions: np.ndarray, colors: np.ndarray):
        self._n = len(positions)
        self.prog = self.ctx.program(vertex_shader=_SHAPES_VERT, fragment_shader=_SHAPES_FRAG)
        self._vbo_pos = self.ctx.buffer(positions.tobytes())
        self._vbo_col = self.ctx.buffer(colors.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [
            (self._vbo_pos, "3f", "in_position"),
            (self._vbo_col, "4f", "in_color"),
        ])
        self._reg_pos = _register(self._vbo_pos)

    def write_warp(self, wp_pos: wp.array):
        gl_pos = self._reg_pos.map(dtype=wp.vec3, shape=(self._n,))
        wp.copy(gl_pos, wp_pos)
        self._reg_pos.unmap()

    def update(self, positions: np.ndarray = None, colors: np.ndarray = None):
        if positions is not None: self._vbo_pos.write(positions.tobytes())
        if colors    is not None: self._vbo_col.write(colors.tobytes())

    def draw(self, mvp: np.ndarray, point_size: float = 80.0, **kwargs):
        self.prog["mvp"].write(mvp.tobytes())
        self.prog["point_size"].value = point_size
        self.vao.render(moderngl.POINTS)


# ---------------------------------------------------------------------------
# Ribbons  (static indexed triangle mesh)
# ---------------------------------------------------------------------------

class RibbonDrawable(Drawable):
    """Static filled quad-strip ribbons as indexed triangles."""

    def setup(self, vertices: np.ndarray, colors: np.ndarray, indices: np.ndarray):
        self.prog = self.ctx.program(vertex_shader=_LINES_VERT, fragment_shader=_LINES_FRAG)
        self._vbo_pos = self.ctx.buffer(vertices.tobytes())
        self._vbo_col = self.ctx.buffer(colors.tobytes())
        self._ibo     = self.ctx.buffer(indices.tobytes())
        self.vao = self.ctx.vertex_array(self.prog, [
            (self._vbo_pos, "3f", "in_position"),
            (self._vbo_col, "4f", "in_color"),
        ], self._ibo)

    def update(self, vertices: np.ndarray = None, colors: np.ndarray = None):
        if vertices is not None: self._vbo_pos.write(vertices.tobytes())
        if colors   is not None: self._vbo_col.write(colors.tobytes())

    def release(self):
        """Release all GPU resources held by this drawable."""
        for attr in ("vao", "_ibo", "_vbo_col", "_vbo_pos", "prog"):
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.release()
                setattr(self, attr, None)

    def draw(self, mvp: np.ndarray, **kwargs):
        self.prog["mvp"].write(mvp.tobytes())
        self.vao.render(moderngl.TRIANGLES)
