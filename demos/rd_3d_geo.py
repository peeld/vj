#!/usr/bin/env python3
"""
rd_3d_geo.py  —  3-D Gray-Scott reaction-diffusion + real-time computational geometry

Pipeline
--------
  Warp (CUDA/CPU)     3-D voxel Gray-Scott PDE with fractal 3-D fbm warp
       ↓ V field
  scikit-image        marching_cubes  →  triangle mesh (verts, faces, normals)
       ↓
  scipy               trilinear sample of gradient magnitude → vertex colours
       ↓
  Open3D              real-time rotating mesh display + live topology stats

Computational geometry output each mesh update
----------------------------------------------
  • Vertex / face counts (mesh resolution tracks reaction front complexity)
  • Euler characteristic  χ = V - E + F   (E ≈ 3F/2 for closed manifold)
  • Estimated genus  g ≈ (2 − χ) / 2
    – g=0  sphere-like blobs
    – g=1  torus/ring structures
    – g>1  sponge / gyroid-like topology
  These numbers flip live as spots merge, split, and form rings.

Usage
-----
  python rd_3d_geo.py                        # 64³, auto device
  python rd_3d_geo.py --device cuda          # force CUDA
  python rd_3d_geo.py --size 96             # bigger grid (needs CUDA for smooth fps)
  python rd_3d_geo.py --image photo.png     # seed from image midplane
  python rd_3d_geo.py --preset 1 --level 0.3

Controls (Open3D window)
------------------------
  Q / ESC    quit
  P          next preset  (spots → coral → mitosis → worms)
  F          toggle fractal warp on/off
  ]  [       isovalue ±0.02
  +  -       sim steps per display frame ±1
  R          reset field
  S          save mesh snapshot (.ply)
"""

import sys, os, time, argparse
import numpy as np
import warp as wp
from skimage.measure import marching_cubes
from scipy.ndimage import map_coordinates

try:
    import open3d as o3d
    HAS_O3D = True
except ImportError:
    HAS_O3D = False

wp.init()

# ── Gray-Scott presets: (name, Du, Dv, F, k) ─────────────────────────────────
PRESETS = [
    ("spots",   0.16, 0.08, 0.035, 0.065),
    ("coral",   0.16, 0.08, 0.055, 0.062),
    ("mitosis", 0.28, 0.12, 0.028, 0.053),
    ("worms",   0.16, 0.08, 0.078, 0.061),
]

# ── Warp kernels ──────────────────────────────────────────────────────────────

@wp.func
def sw3(a: wp.array3d(dtype=wp.float32),
        d: int, i: int, j: int,
        D: int, H: int, W: int) -> wp.float32:
    """Toroidal (wrap) sample of a 3-D array."""
    return a[(d % D + D) % D, (i % H + H) % H, (j % W + W) % W]


@wp.func
def fbm3(fx: float, fy: float, fz: float, t: float) -> wp.vec3f:
    """
    2-octave 3-D fractional Brownian Motion domain warp.

    Each spatial component of the displacement is a product of sinusoids
    in the three axes; octave 2 is phase-shifted by octave 1's output,
    giving recursive self-similar structure in all three dimensions.
    """
    s  = wp.float32(3.0)
    s2 = wp.float32(6.3)
    tt = t * wp.float32(0.05)

    # Octave 1 — cross-axis sinusoidal basis
    qx = wp.sin(fx * s + tt) * \
         wp.cos(fy * s * wp.float32(1.1) + fz * wp.float32(0.7) + tt * wp.float32(0.8))
    qy = wp.cos(fy * s + tt * wp.float32(1.2)) * \
         wp.sin(fz * s * wp.float32(0.9) + fx * wp.float32(0.5) - tt * wp.float32(0.6))
    qz = wp.sin(fz * s * wp.float32(1.1) - tt * wp.float32(0.9)) * \
         wp.cos(fx * s + fy * wp.float32(0.8) + tt)

    # Octave 2 — warped by octave 1
    half = wp.float32(0.5)
    dx = wp.sin((fx + qx * half) * s2 + tt * wp.float32(1.7)) * \
         wp.cos((fy + qy * half) * s2)
    dy = wp.cos((fy + qy * half) * s2 - tt * wp.float32(1.3)) * \
         wp.sin((fz + qz * half) * s2)
    dz = wp.sin((fz + qz * half) * s2 + tt * wp.float32(2.1)) * \
         wp.cos((fx + qx * half) * s2)

    return wp.vec3f(dx, dy, dz)


@wp.kernel
def gs3_step(
    u_in:  wp.array3d(dtype=wp.float32),
    v_in:  wp.array3d(dtype=wp.float32),
    u_out: wp.array3d(dtype=wp.float32),
    v_out: wp.array3d(dtype=wp.float32),
    D: int, H: int, W: int,
    Du: float, Dv: float, F: float, k: float,
    dt: float, warp_px: float, t: float, frac: int,
):
    """
    3-D Gray-Scott timestep with fractal-warped Laplacian centre.

    The 6 cardinal neighbours (±x, ±y, ±z) come from the standard position,
    but the stencil centre is displaced by the 3-D fbm field.  This creates
    anisotropic diffusion whose anisotropy is self-similar at multiple scales.

    dU/dt = Du·∇²U  −  U·V²  +  F·(1−U)
    dV/dt = Dv·∇²V  +  U·V²  −  (F+k)·V
    """
    d, i, j = wp.tid()

    # Normalised position in [-1, 1]³
    fx = wp.float32(j) / wp.float32(W) * wp.float32(2.0) - wp.float32(1.0)
    fy = wp.float32(i) / wp.float32(H) * wp.float32(2.0) - wp.float32(1.0)
    fz = wp.float32(d) / wp.float32(D) * wp.float32(2.0) - wp.float32(1.0)

    # Fractal displacement
    dd = wp.float32(0.0)
    di = wp.float32(0.0)
    dj = wp.float32(0.0)
    if frac == 1:
        disp = fbm3(fx, fy, fz, t)
        dd = disp[2]
        di = disp[1]
        dj = disp[0]

    wd = d + int(dd * warp_px)
    wi = i + int(di * warp_px)
    wj = j + int(dj * warp_px)

    # Warped stencil centre
    wcu = sw3(u_in, wd, wi, wj, D, H, W)
    wcv = sw3(v_in, wd, wi, wj, D, H, W)

    # 6-neighbour Laplacian (3-D), displaced centre
    lu = (sw3(u_in, d-1, i,   j,   D, H, W) + sw3(u_in, d+1, i,   j,   D, H, W) +
          sw3(u_in, d,   i-1, j,   D, H, W) + sw3(u_in, d,   i+1, j,   D, H, W) +
          sw3(u_in, d,   i,   j-1, D, H, W) + sw3(u_in, d,   i,   j+1, D, H, W) -
          wp.float32(6.0) * wcu)

    lv = (sw3(v_in, d-1, i,   j,   D, H, W) + sw3(v_in, d+1, i,   j,   D, H, W) +
          sw3(v_in, d,   i-1, j,   D, H, W) + sw3(v_in, d,   i+1, j,   D, H, W) +
          sw3(v_in, d,   i,   j-1, D, H, W) + sw3(v_in, d,   i,   j+1, D, H, W) -
          wp.float32(6.0) * wcv)

    u_v = u_in[d, i, j]
    v_v = v_in[d, i, j]
    uvv = u_v * v_v * v_v

    u_out[d, i, j] = wp.clamp(
        u_v + dt * (Du * lu - uvv + F * (wp.float32(1.0) - u_v)),
        wp.float32(0.0), wp.float32(1.0))
    v_out[d, i, j] = wp.clamp(
        v_v + dt * (Dv * lv + uvv - (F + k) * v_v),
        wp.float32(0.0), wp.float32(1.0))


# ── Seed ──────────────────────────────────────────────────────────────────────

def make_seed_3d(D, H, W, device, img_path=None):
    """
    Initialise U, V concentration fields.

    If an image is given, its luminance seeds V in the central Z slab
    (fades to zero toward the top and bottom of the grid), so the emerging
    3-D topology is directly shaped by the image content.
    Otherwise, random spherical blobs scatter V throughout the volume.
    """
    rng = np.random.default_rng()
    U = np.ones( (D, H, W), np.float32)
    V = np.zeros((D, H, W), np.float32)

    if img_path and os.path.isfile(img_path):
        from PIL import Image
        img  = Image.open(img_path).convert("L").resize((W, H), Image.LANCZOS)
        luma = np.array(img, dtype=np.float32) / 255.0
        # Seed middle half of Z axis, weight fades out to 0 at ±D/4 from centre
        for dz in range(D):
            w = max(0.0, 1.0 - abs(dz - D / 2.0) / (D / 4.0))
            noise = rng.uniform(-0.03, 0.03, (H, W)).astype(np.float32)
            V[dz] = np.clip(0.45 * luma * w + noise, 0.0, 0.5)
            U[dz] = np.clip(1.0 - 0.3 * luma * w, 0.5, 1.0)
    else:
        n = max(20, D * H * W // (10 ** 3))
        for _ in range(n):
            cd = rng.integers(D // 8, 7 * D // 8)
            ch = rng.integers(H // 8, 7 * H // 8)
            cw = rng.integers(W // 8, 7 * W // 8)
            r  = rng.integers(2, max(3, min(D, H, W) // 8))
            dz, dy, dx = np.ogrid[:D, :H, :W]
            mask = (dz - cd) ** 2 + (dy - ch) ** 2 + (dx - cw) ** 2 < r ** 2
            U[mask] = 0.50
            V[mask] = 0.25
        noise = rng.uniform(-0.015, 0.015, (D, H, W)).astype(np.float32)
        U = np.clip(U + noise, 0.0, 1.0)
        V = np.clip(V + noise, 0.0, 1.0)

    return (wp.array(U, dtype=wp.float32, device=device),
            wp.array(V, dtype=wp.float32, device=device))


# ── Geometry extraction ────────────────────────────────────────────────────────

def extract_geometry(v_np: np.ndarray, level: float):
    """
    Run marching cubes on the V concentration field, then:
      • compute vertex normals (already returned by scikit-image)
      • trilinearly sample gradient magnitude at each vertex → vertex colour
      • estimate Euler characteristic χ = V − E + F (E ≈ 3F/2 for manifold)
      • derive approximate genus  g ≈ (2 − χ) / 2

    Returns a dict, or None if the isovalue has no crossings yet.
    """
    try:
        verts, faces, normals, _ = marching_cubes(
            v_np, level=level, allow_degenerate=False)
    except (ValueError, RuntimeError):
        return None

    if len(verts) == 0 or len(faces) == 0:
        return None

    # ── Gradient magnitude ────────────────────────────────────────────────────
    # np.gradient returns (dV/dz, dV/dy, dV/dx) for a (D, H, W) array
    gz, gy, gx = np.gradient(v_np)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)

    # Trilinear interpolation of grad_mag at marching-cubes vertex positions
    # verts are in grid coords [0, N-1]; map_coordinates handles subpixel
    vertex_grad = map_coordinates(grad_mag, verts.T, order=1, mode='nearest')
    gmax = vertex_grad.max()
    if gmax > 1e-8:
        vertex_grad /= gmax
    vertex_grad = np.clip(vertex_grad, 0.0, 1.0)

    # ── Vertex colour: teal (low gradient) → magenta-white (high gradient) ───
    # Low-gradient regions = chemical plateau (inside a blob or open ocean)
    # High-gradient regions = reaction front, sharp V boundary
    colors = np.stack([
        vertex_grad * 0.95,                       # R
        (1.0 - vertex_grad) * 0.75,               # G
        0.55 + vertex_grad * 0.45,                # B
    ], axis=1)

    # ── Euler characteristic (closed-manifold estimate) ───────────────────────
    n_v = len(verts)
    n_f = len(faces)
    n_e = (3 * n_f) // 2          # E = 3F/2 for closed triangulated manifold
    euler = n_v - n_e + n_f       # χ = V − E + F
    genus = max(0, (2 - euler) // 2)

    return dict(verts=verts, faces=faces, normals=normals,
                colors=colors, n_v=n_v, n_f=n_f, euler=euler, genus=genus)


# ── Visualiser ────────────────────────────────────────────────────────────────

def run_open3d(args, D, H, W, device):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("RD 3-D  |  computational geometry", width=960, height=720)

    opt = vis.get_render_option()
    opt.light_on          = True
    opt.mesh_show_back_face = True

    # We'll swap between two mesh objects so Open3D doesn't stutter on update
    mesh = o3d.geometry.TriangleMesh()
    mesh_added = False

    # Add a bounding box wireframe so the grid extent is always visible
    bbox = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        o3d.geometry.AxisAlignedBoundingBox([0, 0, 0], [W, H, D]))
    bbox.paint_uniform_color([0.3, 0.3, 0.3])
    vis.add_geometry(bbox)

    # Add coordinate frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=W * 0.15)
    vis.add_geometry(frame)

    # ── Mutable simulation state (modified from key callbacks) ────────────────
    st = dict(preset=args.preset, level=args.level, frac=1,
              spf=args.spf, running=True, reset=False, save=False)

    def cb_quit(v):    st['running'] = False;               return False
    def cb_reset(v):   st['reset']   = True;                return False
    def cb_preset(v):  st['preset']  = (st['preset']+1) % len(PRESETS); return False
    def cb_frac(v):    st['frac']    = 1 - st['frac'];      return False
    def cb_lup(v):     st['level']   = min(st['level']+0.02, 0.85); return False
    def cb_ldn(v):     st['level']   = max(st['level']-0.02, 0.05); return False
    def cb_spfup(v):   st['spf']     = min(st['spf']+1, 30);        return False
    def cb_spfdn(v):   st['spf']     = max(st['spf']-1, 1);         return False
    def cb_save(v):    st['save']    = True;                         return False

    vis.register_key_callback(ord('Q'),  cb_quit)
    vis.register_key_callback(256,       cb_quit)    # ESC
    vis.register_key_callback(ord('R'),  cb_reset)
    vis.register_key_callback(ord('P'),  cb_preset)
    vis.register_key_callback(ord('F'),  cb_frac)
    vis.register_key_callback(ord(']'),  cb_lup)
    vis.register_key_callback(ord('['),  cb_ldn)
    vis.register_key_callback(ord('='),  cb_spfup)
    vis.register_key_callback(ord('-'),  cb_spfdn)
    vis.register_key_callback(ord('S'),  cb_save)

    # ── Buffers ───────────────────────────────────────────────────────────────
    u_a, v_a = make_seed_3d(D, H, W, device, args.image)
    u_b = wp.zeros((D, H, W), dtype=wp.float32, device=device)
    v_b = wp.zeros((D, H, W), dtype=wp.float32, device=device)

    step = 0
    geo_step = 0        # counts toward next geometry update
    fps_times = []

    print("\n  Running — topology stats will appear below:\n")
    print(f"  {'step':>6}  {'verts':>7}  {'faces':>7}  {'χ':>5}  "
          f"{'genus':>5}  {'level':>6}  preset")
    print("  " + "─" * 58)

    while st['running']:
        t0 = time.perf_counter()

        if st['reset']:
            u_a, v_a = make_seed_3d(D, H, W, device, args.image)
            step = 0;  geo_step = 0;  st['reset'] = False

        pname, Du, Dv, F, k = PRESETS[st['preset']]

        # ── Simulation steps ──────────────────────────────────────────────────
        for _ in range(st['spf']):
            wp.launch(gs3_step, dim=(D, H, W),
                      inputs=[u_a, v_a, u_b, v_b,
                               D, H, W, Du, Dv, F, k,
                               args.dt, args.warp, float(step), st['frac']],
                      device=device)
            u_a, u_b = u_b, u_a
            v_a, v_b = v_b, v_a
            step    += 1
            geo_step += 1

        # ── Geometry update ───────────────────────────────────────────────────
        if geo_step >= args.geo_every:
            geo_step = 0
            v_np = v_a.numpy()
            geo  = extract_geometry(v_np, st['level'])

            if geo is not None:
                mesh.vertices       = o3d.utility.Vector3dVector(geo['verts'])
                mesh.triangles      = o3d.utility.Vector3iVector(geo['faces'])
                mesh.vertex_normals = o3d.utility.Vector3dVector(geo['normals'])
                mesh.vertex_colors  = o3d.utility.Vector3dVector(geo['colors'])

                if not mesh_added:
                    vis.add_geometry(mesh)
                    ctr = vis.get_view_control()
                    ctr.set_lookat([W / 2, H / 2, D / 2])
                    ctr.set_front([1, 0.6, 0.8])
                    ctr.set_up([0, 1, 0])
                    ctr.set_zoom(0.55)
                    mesh_added = True
                else:
                    vis.update_geometry(mesh)

                # Save snapshot
                if st['save']:
                    fname = f"mesh_{step:07d}.ply"
                    o3d.io.write_triangle_mesh(fname, mesh)
                    print(f"  Saved {fname}")
                    st['save'] = False

                # Topology readout
                print(f"  {step:6d}  {geo['n_v']:7d}  {geo['n_f']:7d}  "
                      f"{geo['euler']:5d}  {geo['genus']:5d}  "
                      f"{st['level']:6.3f}  {pname}  "
                      f"frac={'on' if st['frac'] else 'off'}")

        if not vis.poll_events():
            break
        vis.update_renderer()

        fps_times.append(time.perf_counter() - t0)
        if len(fps_times) >= 15:
            fps = 1.0 / (sum(fps_times) / len(fps_times))
            fps_times.clear()
            sps = fps * st['spf']
            vis.get_render_option()          # keep window responsive
            # update title via window name (Open3D doesn't expose set_window_title)
            print(f"  [{fps:5.1f} fps  {sps:5.0f} steps/s  "
                  f"spf={st['spf']}  warp={args.warp:.0f}px]", end="\r")

    vis.destroy_window()


def run_matplotlib_fallback(args, D, H, W, device):
    """
    Minimal fallback using matplotlib 3-D surface plot.
    Updates every geo_every sim steps.  Much slower than Open3D.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_title("RD 3-D  (install open3d for real-time rendering)")
    plt.ion()

    u_a, v_a = make_seed_3d(D, H, W, device, args.image)
    u_b = wp.zeros((D, H, W), dtype=wp.float32, device=device)
    v_b = wp.zeros((D, H, W), dtype=wp.float32, device=device)

    preset_idx = args.preset
    pname, Du, Dv, F, k = PRESETS[preset_idx]
    step = 0
    geo_step = 0

    while plt.fignum_exists(fig.number):
        for _ in range(args.spf):
            wp.launch(gs3_step, dim=(D, H, W),
                      inputs=[u_a, v_a, u_b, v_b,
                               D, H, W, Du, Dv, F, k,
                               args.dt, args.warp, float(step), 1],
                      device=device)
            u_a, u_b = u_b, u_a
            v_a, v_b = v_b, v_a
            step += 1;  geo_step += 1

        if geo_step >= args.geo_every:
            geo_step = 0
            geo = extract_geometry(v_a.numpy(), args.level)
            if geo is not None:
                ax.cla()
                ax.set_xlim(0, W);  ax.set_ylim(0, H);  ax.set_zlim(0, D)
                ax.set_xlabel("X");  ax.set_ylabel("Y");  ax.set_zlabel("Z")
                verts, faces = geo['verts'], geo['faces']
                tris = verts[faces]
                cmap = plt.cm.cool
                fc   = cmap(geo['colors'][:, 2].mean()
                            if len(geo['colors']) else 0.5)
                poly = Poly3DCollection(tris[::max(1, len(tris)//2000)],
                                        alpha=0.5, facecolor=fc, edgecolor='none')
                ax.add_collection3d(poly)
                ax.set_title(f"step={step}  V={geo['n_v']}  F={geo['n_f']}"
                             f"  χ={geo['euler']}  genus≈{geo['genus']}")
                plt.pause(0.001)

    plt.ioff();  plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image",     default=None,    help="Seed image (luminance → V midplane)")
    ap.add_argument("--size",      type=int, default=64,   help="Grid size N³ (default 64)")
    ap.add_argument("--depth",     type=int, default=None, help="Override Z depth")
    ap.add_argument("--preset",    type=int, default=0,    help="Preset index 0–3")
    ap.add_argument("--level",     type=float, default=0.25, help="Isosurface level (default 0.25)")
    ap.add_argument("--dt",        type=float, default=1.0,  help="Timestep size (default 1.0)")
    ap.add_argument("--spf",       type=int,   default=5,    help="Sim steps per display frame (default 5)")
    ap.add_argument("--geo-every", type=int,   default=10,   help="Geometry update every N sim steps (default 10)")
    ap.add_argument("--warp",      type=float, default=8.0,  help="Fractal warp pixel strength (default 8)")
    ap.add_argument("--device",    default="auto",           help="Warp device: auto | cpu | cuda")
    args = ap.parse_args()

    # ── Device selection ──────────────────────────────────────────────────────
    if args.device == "auto":
        device = "cuda" if wp.is_cuda_available() else "cpu"
    else:
        device = args.device

    N = args.size
    D = args.depth or N
    H = W = N

    print("━" * 62)
    print("  RD 3-D  |  Computational Geometry Demo  (Warp)")
    print("━" * 62)
    print(f"  Grid      : {W}×{H}×{D}  ({W*H*D/1e6:.2f}M voxels)")
    print(f"  Device    : {device}")
    print(f"  Preset    : {PRESETS[args.preset][0]}")
    print(f"  Isovalue  : {args.level}")
    print(f"  spf       : {args.spf}  (steps per display frame)")
    print(f"  Geo every : {args.geo_every} sim steps")
    print(f"  Renderer  : {'Open3D' if HAS_O3D else 'matplotlib (install open3d for better perf)'}")
    print("━" * 62)

    if HAS_O3D:
        run_open3d(args, D, H, W, device)
    else:
        print("\n  open3d not found — using matplotlib fallback.")
        print("  Install with:  pip install open3d\n")
        run_matplotlib_fallback(args, D, H, W, device)


if __name__ == "__main__":
    main()
