"""
Marching Cubes with NVIDIA Warp
================================
Warp features demonstrated:
  @wp.struct      - bundle SDF parameters as a GPU-friendly value type
  @wp.func        - reusable device functions (inlined into kernels)
  @wp.kernel      - GPU kernels dispatched with wp.launch()
  wp.array        - GPU array allocation (wp.zeros, wp.empty)
  wp.atomic_add   - race-safe counter across threads
  wp.MarchingCubes- built-in iso-surface extractor
  wp.launch       - kernel dispatch
  wp.synchronize  - host/device sync
  .numpy()        - pull results back to CPU

Scene: sphere + torus blended with smooth-union CSG.
Outputs: marching_cubes.obj  (mesh)
         marching_cubes.png  (matplotlib render)
"""

import numpy as np
import warp as wp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ── Init ──────────────────────────────────────────────────────────────────────
wp.init()
try:
    wp.get_device("cuda")
    DEVICE = "cuda"
except Exception:
    DEVICE = "cpu"
print(f"Using device: {DEVICE}")


# =============================================================================
# 1.  @wp.struct  -  GPU-friendly parameter bundle
# =============================================================================
@wp.struct
class SDFParams:
    sphere_r: float   # sphere radius
    torus_R:  float   # torus major radius
    torus_r:  float   # torus minor radius
    blend:    float   # smooth-union blend width


# =============================================================================
# 2.  @wp.func  -  device functions (inlined into kernels at compile time)
# =============================================================================
@wp.func
def sd_sphere(p: wp.vec3, r: float) -> float:
    return wp.length(p) - r


@wp.func
def sd_torus(p: wp.vec3, big_r: float, small_r: float) -> float:
    # project onto xz-plane, measure distance from the major circle
    q = wp.vec2(wp.sqrt(p[0]*p[0] + p[2]*p[2]) - big_r, p[1])
    return wp.length(q) - small_r


@wp.func
def smooth_union(a: float, b: float, k: float) -> float:
    """Smooth-min CSG blend (Inigo Quilez)."""
    h = wp.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return wp.lerp(b, a, h) - k * h * (1.0 - h)


@wp.func
def scene_sdf(p: wp.vec3, prm: SDFParams) -> float:
    """Combined SDF for the whole scene."""
    d1 = sd_sphere(p, prm.sphere_r)
    d2 = sd_torus(p, prm.torus_R, prm.torus_r)
    return smooth_union(d1, d2, prm.blend)


@wp.func
def sdf_gradient(p: wp.vec3, eps: float, prm: SDFParams) -> wp.vec3:
    """Central-difference surface normal (gradient of SDF)."""
    ex = wp.vec3(eps, 0.0, 0.0)
    ey = wp.vec3(0.0, eps, 0.0)
    ez = wp.vec3(0.0, 0.0, eps)
    dx = scene_sdf(p + ex, prm) - scene_sdf(p - ex, prm)
    dy = scene_sdf(p + ey, prm) - scene_sdf(p - ey, prm)
    dz = scene_sdf(p + ez, prm) - scene_sdf(p - ez, prm)
    return wp.normalize(wp.vec3(dx, dy, dz))


# =============================================================================
# 3.  @wp.kernel  -  fill a 3-D SDF volume
#     wp.launch with dim=(nx,ny,nz) gives each thread its own (ix,iy,iz)
# =============================================================================
@wp.kernel
def fill_volume(
    field:    wp.array3d(dtype=float),   # shape (nx, ny, nz)
    nx: int,  ny: int,  nz: int,
    lo: wp.vec3,  hi: wp.vec3,
    prm: SDFParams,
    n_inside: wp.array(dtype=int),       # wp.atomic_add target
):
    ix, iy, iz = wp.tid()               # 3-D thread ID from dim=(nx,ny,nz)

    # grid coords -> world position
    tx = float(ix) / float(nx - 1)
    ty = float(iy) / float(ny - 1)
    tz = float(iz) / float(nz - 1)
    p = wp.vec3(
        lo[0] + tx * (hi[0] - lo[0]),
        lo[1] + ty * (hi[1] - lo[1]),
        lo[2] + tz * (hi[2] - lo[2]),
    )

    d = scene_sdf(p, prm)
    field[ix, iy, iz] = d

    # wp.atomic_add: race-safe accumulation across all threads
    if d < 0.0:
        wp.atomic_add(n_inside, 0, 1)


# =============================================================================
# 4.  @wp.kernel  -  grid-space verts -> world-space + per-vertex normals
# =============================================================================
@wp.kernel
def post_process(
    verts_in:  wp.array(dtype=wp.vec3),   # grid-space verts from MC
    nx: int,   ny: int,   nz: int,
    lo: wp.vec3,  hi: wp.vec3,
    prm: SDFParams,
    verts_out: wp.array(dtype=wp.vec3),   # world-space verts
    normals:   wp.array(dtype=wp.vec3),   # surface normals
):
    vid = wp.tid()
    g = verts_in[vid]

    # grid -> world
    world = wp.vec3(
        lo[0] + (g[0] / float(nx - 1)) * (hi[0] - lo[0]),
        lo[1] + (g[1] / float(ny - 1)) * (hi[1] - lo[1]),
        lo[2] + (g[2] / float(nz - 1)) * (hi[2] - lo[2]),
    )
    verts_out[vid] = world
    normals[vid]   = sdf_gradient(world, 0.001, prm)


# =============================================================================
# 5.  Scene / grid setup
# =============================================================================
params = SDFParams()
params.sphere_r = 0.45
params.torus_R  = 0.60
params.torus_r  = 0.18
params.blend    = 0.12

N     = 64                      # grid resolution per axis
lo    = wp.vec3(-1.0, -1.0, -1.0)
hi    = wp.vec3( 1.0,  1.0,  1.0)
total = N * N * N

# wp.zeros - GPU array allocation; 3-D shape for the volume
field    = wp.zeros((N, N, N), dtype=float, device=DEVICE)
n_inside = wp.zeros(1,        dtype=int,   device=DEVICE)


# =============================================================================
# 6.  wp.launch  -  evaluate SDF on the entire grid
# =============================================================================
wp.launch(
    kernel=fill_volume,
    dim=(N, N, N),                      # 3-D dispatch
    inputs=[field, N, N, N, lo, hi, params, n_inside],
    device=DEVICE,
)
wp.synchronize()   # host/device sync before reading back

count_inside = n_inside.numpy()[0]
print(f"Voxels inside surface: {count_inside:,} / {total:,}  "
      f"({100 * count_inside / total:.1f}%)")


# =============================================================================
# 7.  wp.MarchingCubes  -  extract iso-surface at threshold = 0
# =============================================================================
mc = wp.MarchingCubes(
    nx=N, ny=N, nz=N,
    max_verts=total * 3,
    max_tris=total * 2,
    device=DEVICE,
)
mc.surface(field=field, threshold=0.0)
wp.synchronize()

n_verts = mc.verts.shape[0]
n_tris  = mc.indices.shape[0] // 3
print(f"Extracted: {n_verts:,} vertices, {n_tris:,} triangles")


# =============================================================================
# 8.  Post-process: world-space transform + normals (single kernel pass)
# =============================================================================
world_verts = wp.zeros(n_verts, dtype=wp.vec3, device=DEVICE)
normals     = wp.zeros(n_verts, dtype=wp.vec3, device=DEVICE)

wp.launch(
    kernel=post_process,
    dim=n_verts,
    inputs=[mc.verts, N, N, N, lo, hi, params, world_verts, normals],
    device=DEVICE,
)
wp.synchronize()


# =============================================================================
# 9.  Pull to CPU  (.numpy())
# =============================================================================
v = world_verts.numpy()              # (n_verts, 3)
n = normals.numpy()                  # (n_verts, 3)
f = mc.indices.numpy().reshape(-1, 3)  # (n_tris, 3)


# =============================================================================
# 10.  Save OBJ
# =============================================================================
obj_path = "marching_cubes.obj"
with open(obj_path, "w") as fh:
    fh.write("# Warp marching cubes export\n")
    for vi in v:
        fh.write(f"v  {vi[0]:.6f} {vi[1]:.6f} {vi[2]:.6f}\n")
    for ni in n:
        fh.write(f"vn {ni[0]:.6f} {ni[1]:.6f} {ni[2]:.6f}\n")
    for tri in f:
        a, b, c = int(tri[0]) + 1, int(tri[1]) + 1, int(tri[2]) + 1
        fh.write(f"f {a}//{a} {b}//{b} {c}//{c}\n")
print(f"Saved: {obj_path}")


# =============================================================================
# 11.  Matplotlib render (sample a subset - mpl is CPU-bound)
# =============================================================================
MAX_TRIS = 3000
step = max(1, n_tris // MAX_TRIS)
tris = [v[tri] for tri in f[::step]]

z_mid  = np.array([t[:, 2].mean() for t in tris])
z_lo, z_hi = z_mid.min(), z_mid.max()
z_norm = (z_mid - z_lo) / (z_hi - z_lo + 1e-8)

fig = plt.figure(figsize=(9, 7), facecolor="#1a1a2e")
ax  = fig.add_subplot(111, projection="3d", facecolor="#1a1a2e")

col = Poly3DCollection(tris, linewidth=0, alpha=0.85)
col.set_facecolor(plt.cm.plasma(z_norm))
ax.add_collection3d(col)

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
    pane.fill = False
    pane.set_edgecolor("#444")
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.label.set_color("white")
    axis.set_tick_params(colors="white")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(
    f"Warp Marching Cubes  |  {N}^3 grid  |  {n_verts:,} verts  |  {n_tris:,} tris",
    color="white", pad=12,
)
ax.view_init(elev=22, azim=40)

png_path = "marching_cubes.png"
plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {png_path}")
plt.close()
