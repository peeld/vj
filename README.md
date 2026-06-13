# Warp GPU Visualiser — `gui.py`

An interactive 3-D particle visualiser built on [NVIDIA Warp](https://github.com/NVIDIA/warp), ModernGL, and moderngl-window. All simulation runs on the GPU via Warp kernels; rendered frames are optionally fed through a GPU feedback/post-processing loop before display.

---

## Architecture

The codebase is split into five modules with a strict separation of concerns:

```
gui.py            Window lifecycle, camera, input, per-frame orchestration
data.py           Warp arrays and simulation configuration
operation.py      Warp kernel dispatch (simulation step)
drawable.py       Shaders, GPU buffers, and draw calls
warp_feedback.py  Ping-pong feedback buffer and post-effect kernel
kernels.py        Pure @wp.kernel definitions (no Python logic)
```

### Data flow per frame (`on_render`)

```
BallOperation.step()         ← physics on GPU
PointCloudOperation.step()   ← particle simulation on GPU
     │
     ▼
Drawable.write_warp()        ← GPU→GPU via CUDA-GL interop (zero CPU round-trip)
     │
     ▼
_render_direct()             ← straight to screen framebuffer
   — or —
_render_with_post()          ← scene → off-screen FBO → inject_scene_kernel
                                → FeedbackLoop.step() → blit to screen
```

---

## Module reference

### `data.py` — simulation state

| Class | Owns |
|---|---|
| `PointCloudData` | `wp_pos` (vec3), `wp_vel` (vec3), `wp_col` (vec4) for 250 k points |
| `BallData` | `wp_pos` (vec3), `wp_vel` (vec3) for 5 bouncing balls; fixed colours |

Both classes expose `.positions_numpy()` / `.colors_numpy()` helpers for the initial GPU upload (used once in `__init__`). After that, kernels write directly into the Warp arrays in-place.

### `kernels.py` — GPU kernels

Pure `@wp.kernel` functions; they take only `wp.array` arguments and plain scalars. No Python logic, no class state. Kernels are compiled once at import time.

| Kernel | Does |
|---|---|
| `step_points` | Advances positions by velocity × dt |
| `respawn_escaped` | Wraps/respawns points that leave the cube |
| `influence_points` | Repels points from each ball within a radius |
| `bounce_balls` | Elastic ball–wall bouncing |
| `color_by_interaction` | Blends point colour toward the nearest ball's colour |

### `operation.py` — simulation step

`PointCloudOperation` and `BallOperation` wrap `wp.launch()` calls into named methods and own simulation-level constants (radius, damping, etc.). They hold no GPU arrays themselves — they receive data objects and write back into them.

### `drawable.py` — rendering

Each `Drawable` subclass owns its shader program and one or more VBOs/VAOs but holds no simulation data.

| Class | Renders |
|---|---|
| `PointsDrawable` | Round point-sprite particle cloud (positions + RGBA) |
| `LinesDrawable` | Indexed line segments (wireframe cube) |
| `ShapeDrawable` | Sphere-shaded point sprites (bouncing balls) |

**Two upload paths:**
- `write_warp(wp_array)` — GPU→GPU via `wp.RegisteredGLBuffer`. Use this every frame for dynamic geometry.
- `update(np_array)` — CPU→GPU fallback. Use for static or one-shot uploads.

### `warp_feedback.py` — post-effect

`FeedbackLoop` maintains a ping-pong RGBA float32 buffer pair on the GPU. Each call to `step()` runs the feedback kernel (zoom, rotation, ripple, hue-shift, chromatic aberration, saturation boost) from `prev` into `curr`. The result is read back to an OpenGL texture and blitted to the screen via a fullscreen quad.

`FeedbackParams` is a `dataclass` — every effect parameter lives here and can be tweaked at runtime (the keyboard bindings in `gui.py` do exactly this).

### `gui.py` — orchestration

`WarpCubeGUI` subclasses `mglw.WindowConfig`. Responsibilities:

- **`__init__`** — constructs one `*Data`, one `*Operation`, and one `*Drawable` per scene object; sets up the off-screen FBO, fullscreen quad, and `FeedbackLoop`.
- **`on_render`** — the single per-frame entry point. Runs simulation, uploads to GL, builds MVP, dispatches to `_render_direct` or `_render_with_post`.
- **`_build_mvp`** — pure math: yaw/pitch/distance → view × projection matrix.
- **`_build_wireframe`** — generates cube edge vertices and index buffer.
- **`inject_scene_kernel`** — module-level `@wp.kernel` that alpha-blends the freshly rendered FBO pixels into the feedback `curr` buffer.

---

## Controls

| Input | Action |
|---|---|
| Mouse drag | Orbit camera |
| Scroll | Zoom |
| `R` | Randomise point cloud |
| `P` | Toggle post-effect on/off |
| `ESC` | Quit |

**Feedback tweaks (post-effect only):**

| Keys | Parameter |
|---|---|
| `Z` / `X` | Scene blend ↓ / ↑ (how much fresh 3-D bleeds in) |
| `D` / `F` | Decay ↓ / ↑ (trail length) |
| `Q` / `W` | Rotation speed ↓ / ↑ |
| `A` / `S` | Zoom ↓ / ↑ |
| `H` / `J` | Hue-shift speed ↓ / ↑ |
| `C` / `V` | Chromatic aberration ↓ / ↑ |
| `B` / `N` | Saturation boost ↓ / ↑ |

---

## Adding a new scene element

Follow these four steps; they map directly to the four modules.

### 1. Add data — `data.py`

Create a new class (or extend an existing one) that allocates Warp arrays in `__init__`. Use `wp.array(..., dtype=wp.vec3)` etc. Provide `.positions_numpy()` / `.colors_numpy()` helpers for the initial upload.

```python
class RingData:
    NUM_POINTS = 4_000

    def __init__(self, radius: float = 1.2):
        theta = np.linspace(0, 2 * np.pi, self.NUM_POINTS, dtype=np.float32)
        pos   = np.stack([np.cos(theta) * radius,
                          np.zeros_like(theta),
                          np.sin(theta) * radius], axis=1)
        self.wp_pos = wp.array(pos, dtype=wp.vec3)
        self.wp_col = wp.array(
            np.ones((self.NUM_POINTS, 4), dtype=np.float32), dtype=wp.vec4
        )

    def positions_numpy(self): return self.wp_pos.numpy()
    def colors_numpy(self):    return self.wp_col.numpy()
```

### 2. Add kernels — `kernels.py`

Write pure `@wp.kernel` functions. They must take only `wp.array` arguments and plain scalars/structs — no Python objects. Keep all logic here; keep `operation.py` thin.

```python
@wp.kernel
def rotate_ring(
    positions: wp.array(dtype=wp.vec3),
    speed:     float,
    dt:        float,
):
    i  = wp.tid()
    p  = positions[i]
    x  = p[0]; z = p[2]
    c  = wp.cos(speed * dt); s = wp.sin(speed * dt)
    positions[i] = wp.vec3(x * c - z * s, p[1], x * s + z * c)
```

### 3. Add an operation — `operation.py`

Create a class that holds constants and calls `wp.launch()`. It should accept the relevant `*Data` instance in `__init__` and expose a `step(dt)` method.

```python
class RingOperation:
    SPEED = 1.5

    def __init__(self, data: RingData):
        self.data = data

    def step(self, dt: float):
        wp.launch(rotate_ring,
                  dim=self.data.NUM_POINTS,
                  inputs=[self.data.wp_pos, self.SPEED, dt])
```

### 4. Add a drawable — `drawable.py`

Subclass `Drawable`. In `setup()`, compile the shader, create VBOs, and register buffers for interop. In `write_warp()`, map → copy → unmap. In `draw()`, set uniforms and call `self.vao.render(...)`.

You can reuse `PointsDrawable` or `LinesDrawable` if the geometry type already exists — just call `setup()` with your data.

### 5. Wire it up in `gui.py`

Inside `__init__`, construct data → operation → drawable in order. In `on_render`, call `op.step()`, then `draw.write_warp()`, then `draw.draw(mvp)` alongside the existing calls. Add a key binding in `on_key_event` if needed.

```python
# __init__
self.ring_data = RingData()
self.ring_op   = RingOperation(self.ring_data)
self.ring_draw = PointsDrawable(self.ctx)          # reuse existing drawable
self.ring_draw.setup(self.ring_data.positions_numpy(),
                     self.ring_data.colors_numpy())

# on_render  (inside the simulate section)
self.ring_op.step(frame_time)
self.ring_draw.write_warp(self.ring_data.wp_pos, self.ring_data.wp_col)

# _render_direct / _render_with_post  (inside the draw section)
self.ring_draw.draw(mvp)
```

---

## Guidelines

**Keep modules single-responsibility.** Kernels go in `kernels.py` — never inline `@wp.kernel` definitions inside operation or data classes. Data classes hold arrays and configuration; they do not call `wp.launch()`. Drawables hold shaders and buffers; they do not touch Warp arrays directly.

**Prefer `write_warp` over `update`.** The CUDA-GL interop path avoids any CPU readback. Only fall back to `update(np_array)` for geometry that changes rarely or is generated on the CPU (e.g. a one-shot wireframe).

**One `wp.launch` call per kernel per frame is the norm.** If you need multiple kernels per step, add them as separate methods on the operation class and call them in sequence from `on_render`. Don't batch unrelated work into a single kernel.

**`FeedbackParams` fields can be mutated at runtime.** The keyboard bindings already do this. If you add new post-effect parameters, add them to `FeedbackParams` first, then add key bindings that increment/decrement them — keep the pattern consistent.

**Resize awareness.** The off-screen FBO and `FeedbackLoop` are sized to `window_size` at init. If you add a custom texture or FBO, make sure to recreate it in `on_resize`.
