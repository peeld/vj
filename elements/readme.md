# Building scene elements

This document plus `base.py` are everything you need to build a new scene
element. No other project file is required or referenced.

## GPU-first rule

Per-frame computation (anything in `step()`) must run on the GPU as Warp
kernels. CPU/NumPy code is acceptable only for:

- one-time setup in `__init__`
- rebuilding buffers in `regen()` (rare, not a per-frame call)
- small sequential control logic that doesn't touch per-element data
- the fallback path when no CUDA device is available (see "Compute")

If you find yourself writing a NumPy loop or array op inside `step()`,
move it into a Warp kernel instead. CPU per-frame work is the thing this
document asks you to avoid.

## The contract

An element is any class implementing `DrawingElement` from `base.py`:

```python
class DrawingElement(ABC):
    kind: str = "element"

    def __init__(self) -> None:
        self.name: str = self.kind
        self.visible: bool = True

    @abstractmethod
    def step(self, ctx: FrameContext) -> None: ...

    @abstractmethod
    def draw(self, mvp, ctx: FrameContext) -> None: ...

    def regen(self) -> None: ...

    def set_palette(self, palette: list) -> None: ...
```

- `step(ctx)` is called every frame regardless of visibility. Advance your
  simulation here. If you want to cheapen work while hidden, check
  `self.visible` yourself — the host doesn't skip the call for you.
- `draw(mvp, ctx)` is called only when `visible` is `True`. Issue your GPU
  draw calls here.
- `regen()` is an optional hook to reseed or rebuild the element from
  scratch. Default is a no-op.
- `set_palette(palette)` is an optional hook to receive a new color
  palette. Default is a no-op.
- `kind` is a class-level string identifying the element type. `name`
  defaults to `kind` and is used as a unique label, so only one live
  instance per `kind` is expected at a time.

`FrameContext` is the single argument passed to `step`/`draw` each frame:

| field | type | meaning |
|---|---|---|
| `time` | float | seconds since the element/scene was created |
| `current_time` | float | wall-clock seconds, monotonic |
| `frame_time` | float | seconds elapsed since the previous frame (delta time) |
| `cam_eye` | vec3-like | camera position in world space |
| `cam_fwd` | vec3-like | camera forward unit vector |
| `cam_right` | vec3-like | camera right unit vector |
| `cam_up` | vec3-like | camera up unit vector |

Use `frame_time` to scale any per-frame increment so motion speed is
independent of frame rate. Use the `cam_*` vectors for camera-facing
billboards or distance-based effects (see "Camera" below).

## Construction and registration

Construct with this signature so a generic host can build any registered
kind without special-casing it:

```python
def __init__(self, ctx: moderngl.Context, device: str | None = None, **kwargs):
    ...
```

- `ctx` is the active `moderngl.Context`, used to create programs, buffers,
  and vertex arrays.
- `device` is the Warp device string (`"cuda"`, `"cpu"`, or `None` to let
  Warp pick). Store it and pass it to every `wp.array`/`wp.launch` call you
  make so the element honors whatever device it was constructed with.
- `**kwargs` lets a host pass element-specific construction parameters
  without the registry needing to know the element's signature.

Register a factory at import time so a host can discover and construct
your type by name:

```python
from elements.base import register_element_type

def _make(ctx, device=None, **kwargs):
    return MyElement(ctx, device=device, **kwargs)

register_element_type("my_element", _make)
```

A module that defines an element should call `register_element_type` once
at module scope. Importing the module is what makes the type available —
the host typically imports every element module up front purely for this
side effect.

## Rendering

Rendering uses raw `moderngl` calls — there is no wrapper class involved.

Build a shader program and a vertex array once (in `__init__` or `regen()`,
never per-frame):

```python
self.prog = ctx.program(
    vertex_shader="""
        #version 330
        uniform mat4 mvp;
        in vec3 in_position;
        in vec4 in_color;
        out vec4 v_color;
        void main() {
            gl_Position = mvp * vec4(in_position, 1.0);
            v_color = in_color;
        }
    """,
    fragment_shader="""
        #version 330
        in vec4 v_color;
        out vec4 f_color;
        void main() {
            f_color = v_color;
        }
    """,
)

self.pos_buf   = ctx.buffer(reserve=N * 3 * 4)   # 3 floats * 4 bytes
self.color_buf = ctx.buffer(reserve=N * 4 * 4)   # 4 floats * 4 bytes

self.vao = ctx.vertex_array(
    self.prog,
    [
        (self.pos_buf,   "3f", "in_position"),
        (self.color_buf, "4f", "in_color"),
    ],
)
```

Each frame, update buffer contents (see "Compute" for the GPU-resident way
to do this) and issue the draw call:

```python
def draw(self, mvp, ctx: FrameContext) -> None:
    self.prog["mvp"].write(mvp.tobytes())
    self.vao.render(moderngl.POINTS)  # or LINES, LINE_STRIP, TRIANGLES, ...
```

Notes:

- Vertex attribute names (`in_position`, `in_color`, ...) must match the
  `in` declarations in the vertex shader exactly.
- `mvp` is a 4x4 model-view-projection matrix; write it as raw bytes to a
  `mat4` uniform every frame since the camera moves.
- Blend state (`ctx.blend_func`), depth test, and point-size mode are
  global GL state owned by the host and set up once outside any element.
  If your `draw()` temporarily changes blend mode (e.g. additive glow),
  restore the previous mode before returning — other elements draw
  immediately after yours in the same frame.
- Point size is normally set per-vertex from the vertex shader (writing
  `gl_PointSize`) rather than as global state, when individual points need
  to vary in size.

## Compute

Per-frame simulation should be a Warp kernel, launched from `step()`, that
writes directly into arrays you then either upload to your `moderngl`
buffers or — preferably — map directly via CUDA-GL interop.

A minimal kernel and launch:

```python
import warp as wp

@wp.kernel
def _advance(pos: wp.array(dtype=wp.vec3), age: wp.array(dtype=wp.float32), dt: float):
    i = wp.tid()
    age[i] = age[i] + dt
    pos[i] = pos[i] + wp.vec3(0.0, dt, 0.0)

wp.launch(_advance, dim=N, inputs=[self.pos, self.age, ctx.frame_time], device=self.device)
```

- `wp.tid()` gives the thread's index into the launch dimension; this is
  how each thread knows which array element it owns.
- Inputs and outputs are both passed via `inputs=[...]`; Warp arrays are
  mutated in place.
- Avoid Python conditional expressions (`x if cond else y`) inside kernel
  bodies — use full `if`/`else` blocks instead.
- Call `wp.synchronize()` only when the CPU needs to read back kernel
  results (e.g. before a `.numpy()` call). Don't synchronize on a path
  that stays GPU-resident — it stalls the pipeline for no reason.

For randomness inside a kernel, seed per-frame (not once at construction)
so randomness doesn't repeat or stay static:

```python
@wp.kernel
def _spawn(pos: wp.array(dtype=wp.vec3), frame: int):
    i = wp.tid()
    r = wp.rand_init(frame, i)
    pos[i] = wp.vec3(wp.randf(r) - 0.5, wp.randf(r) - 0.5, wp.randf(r) - 0.5)
```

Increment a frame counter in `step()` and pass it as the seed each launch.

### Getting data to the screen: GPU-GL interop

Prefer mapping your `moderngl` buffer directly as a Warp array and writing
into it from a kernel, avoiding any CPU round-trip:

```python
self._reg = wp.RegisteredGLBuffer(self.pos_buf.glo, wp.get_preferred_device())
```

Each frame:

```python
mapped = self._reg.map(dtype=wp.vec3, shape=(N,))
wp.launch(_advance, dim=N, inputs=[mapped, self.age, ctx.frame_time], device=self.device)
self._reg.unmap()
```

Important: `wp.RegisteredGLBuffer` always registers against
`wp.get_preferred_device()`, not against `self.device`. GL interop requires
an actual CUDA device; if the element's chosen compute device is `"cpu"`,
or no CUDA device exists at all, this path is unavailable.

When no CUDA device is present, fall back to a CPU round-trip instead —
this is the one per-frame CPU path this document doesn't discourage,
because there is no GPU-resident alternative:

```python
if wp.get_cuda_device_count() > 0:
    mapped = self._reg.map(dtype=wp.vec3, shape=(N,))
    wp.launch(_advance, dim=N, inputs=[mapped, self.age, ctx.frame_time], device=self.device)
    self._reg.unmap()
else:
    wp.launch(_advance, dim=N, inputs=[self.pos, self.age, ctx.frame_time], device=self.device)
    self.pos_buf.write(self.pos.numpy().tobytes())
```

Check `wp.get_cuda_device_count()` once at construction and branch on it,
rather than re-checking every frame.

## Fixed-size pools

For elements with a variable number of "live" things (particles, sparks,
ribbons that spawn and die), allocate buffers once at the maximum count
and never resize them per spawn/despawn:

- Represent "dead" with alpha `0.0` and/or by parking the position far
  outside the camera frustum (e.g. `(1e6, 1e6, 1e6)`).
- Track an `age` (or `alive`) scalar per slot in its own `wp.array`.
- A spawn kernel looks for slots whose age exceeds a lifetime and resets
  them; an advance kernel ages every slot and updates position/color.
- Only tear down and recreate buffers when a parameter that changes the
  buffer's *size* changes (e.g. the user raises the max count) — this
  happens in `regen()`, not in `step()`.

This keeps every frame's GPU work at a constant, predictable cost with no
allocation churn.

## World space and camera

The scene is centered near the origin with extent roughly `[-1, 1]` on
each axis. Build new elements to roughly this scale so they compose with
others already in the scene.

`mvp` (passed into `draw()`) already encodes camera position and
projection — you do not construct the camera yourself. Use the `cam_eye`/
`cam_fwd`/`cam_right`/`cam_up` fields on `FrameContext` when an effect
needs to react to camera orientation, e.g. billboarding a quad to face the
camera. Warp has no built-in `normalize()`, so unit vectors are computed
by hand with an explicit zero-length guard:

```python
@wp.kernel
def _billboard_right(cam_right: wp.vec3, cam_up: wp.vec3, half_size: float,
                      out_offset: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    v = cam_right
    length = wp.length(v)
    if length > 0.0:
        v = v / length
    else:
        v = wp.vec3(1.0, 0.0, 0.0)
    out_offset[i] = v * half_size
```

## Color

Colors are RGBA floats in `[0, 1]`; alpha `0.0` means fully invisible.

`set_palette(palette)` receives a list of `(r, g, b)` tuples. Store it and
apply it lazily the next time colors are generated (e.g. on the next
spawn or `regen()`) rather than rewriting every existing vertex
immediately:

```python
def set_palette(self, palette: list) -> None:
    if palette:
        self._palette = list(palette)
```

If no palette has been pushed yet, generate a reasonable default
internally rather than requiring one. A self-contained way to do this
with only the standard library:

```python
import colorsys, random

def _default_palette(n: int = 5) -> list[tuple[float, float, float]]:
    base_hue = random.random()
    out = []
    for i in range(n):
        hue = (base_hue + i / n) % 1.0
        out.append(colorsys.hsv_to_rgb(hue, 0.6, 1.0))
    return out
```

When coloring many vertices at once, pick one palette entry per spawned
instance (e.g. by index modulo palette length, or randomly) rather than
interpolating the whole palette across every vertex — this keeps spawn
logic simple and matches how `set_palette` is meant to be consumed.

## Tunable parameters

Expose tunable values as plain public instance attributes, not via
getters/setters:

```python
self.speed = 1.0
self.spawn_rate = 20.0
```

This lets an external system read and write parameters with plain
`getattr`/`setattr` without needing per-element bindings. Document, in a
comment near each attribute, whether changing it takes effect immediately
(most simulation parameters) or requires calling `regen()` (anything that
changes a buffer's size or shader).

## Worked example

A fixed-pool of `N` spark particles, entirely GPU-resident per frame.

```python
import numpy as np
import warp as wp
import moderngl

from elements.base import DrawingElement, FrameContext, register_element_type

N = 4096


@wp.kernel
def _spawn(pos: wp.array(dtype=wp.vec3), age: wp.array(dtype=wp.float32),
           lifetime: float, frame: int):
    i = wp.tid()
    if age[i] >= lifetime:
        r = wp.rand_init(frame, i)
        pos[i] = wp.vec3(wp.randf(r) - 0.5, 0.0, wp.randf(r) - 0.5)
        age[i] = 0.0


@wp.kernel
def _advance(pos: wp.array(dtype=wp.vec3), age: wp.array(dtype=wp.float32),
             color: wp.array(dtype=wp.vec4), dt: float, lifetime: float):
    i = wp.tid()
    age[i] = age[i] + dt
    pos[i] = pos[i] + wp.vec3(0.0, dt, 0.0)
    t = age[i] / lifetime
    alpha = 1.0 - t
    if alpha < 0.0:
        alpha = 0.0
    color[i] = wp.vec4(1.0, 0.8, 0.3, alpha)


class Sparks(DrawingElement):
    kind = "sparks"

    def __init__(self, ctx: moderngl.Context, device: str | None = None, **kwargs):
        super().__init__()
        self.ctx = ctx
        self.device = device
        self.speed = 1.0
        self.lifetime = 2.0
        self._frame = 0
        self._has_cuda = wp.get_cuda_device_count() > 0

        self.pos = wp.zeros(N, dtype=wp.vec3, device=device)
        self.age = wp.array(np.full(N, 1.0e6, dtype=np.float32), device=device)
        self.color = wp.zeros(N, dtype=wp.vec4, device=device)

        self.prog = ctx.program(
            vertex_shader="""
                #version 330
                uniform mat4 mvp;
                in vec3 in_position;
                in vec4 in_color;
                out vec4 v_color;
                void main() {
                    gl_Position = mvp * vec4(in_position, 1.0);
                    gl_PointSize = 4.0;
                    v_color = in_color;
                }
            """,
            fragment_shader="""
                #version 330
                in vec4 v_color;
                out vec4 f_color;
                void main() { f_color = v_color; }
            """,
        )
        self.pos_buf = ctx.buffer(reserve=N * 3 * 4)
        self.color_buf = ctx.buffer(reserve=N * 4 * 4)
        self.vao = ctx.vertex_array(
            self.prog,
            [(self.pos_buf, "3f", "in_position"), (self.color_buf, "4f", "in_color")],
        )

        if self._has_cuda:
            self._pos_reg = wp.RegisteredGLBuffer(self.pos_buf.glo, wp.get_preferred_device())
            self._color_reg = wp.RegisteredGLBuffer(self.color_buf.glo, wp.get_preferred_device())

    def step(self, ctx: FrameContext) -> None:
        self._frame += 1
        dt = ctx.frame_time * self.speed

        wp.launch(_spawn, dim=N, inputs=[self.pos, self.age, self.lifetime, self._frame],
                  device=self.device)
        wp.launch(_advance, dim=N, inputs=[self.pos, self.age, self.color, dt, self.lifetime],
                  device=self.device)

        if self._has_cuda:
            mapped_pos = self._pos_reg.map(dtype=wp.vec3, shape=(N,))
            wp.copy(mapped_pos, self.pos)
            self._pos_reg.unmap()
            mapped_color = self._color_reg.map(dtype=wp.vec4, shape=(N,))
            wp.copy(mapped_color, self.color)
            self._color_reg.unmap()
        else:
            self.pos_buf.write(self.pos.numpy().tobytes())
            self.color_buf.write(self.color.numpy().tobytes())

    def draw(self, mvp, ctx: FrameContext) -> None:
        self.prog["mvp"].write(mvp.tobytes())
        self.vao.render(moderngl.POINTS)

    def regen(self) -> None:
        self.age = wp.array(np.full(N, 1.0e6, dtype=np.float32), device=self.device)
        self.pos = wp.zeros(N, dtype=wp.vec3, device=self.device)
        self._frame = 0

    def set_palette(self, palette: list) -> None:
        self._palette = list(palette) if palette else None


def _make(ctx, device=None, **kwargs):
    return Sparks(ctx, device=device, **kwargs)


register_element_type("sparks", _make)
```

This covers every required piece: pool allocation, per-frame Warp kernels
for spawn and advance, GPU-GL interop with a CPU fallback, palette and
parameter hooks, and registration.

## Checklist

- [ ] Subclass `DrawingElement`, set `kind`, implement `step` and `draw`.
- [ ] Constructor takes `(ctx, device=None, **kwargs)`.
- [ ] Per-frame work in `step()` is Warp kernels, not NumPy.
- [ ] Buffers are allocated once at max size; no per-frame resizing.
- [ ] `draw()` restores any GL state it temporarily changed.
- [ ] GPU-GL interop falls back to a CPU round-trip when no CUDA device
      is present.
- [ ] `regen()` rebuilds state without needing a new instance.
- [ ] `set_palette()` stores the palette and applies it lazily.
- [ ] Tunable values are plain public attributes.
- [ ] Module calls `register_element_type(kind, factory)` at import time.
