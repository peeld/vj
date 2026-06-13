#!/usr/bin/env python3
"""
rd_3d_organic.py  —  3-D Gray-Scott, organic morphing, custom OpenGL shader

Gray-Scott F and k parameters breathe slowly (±8 % at a ~2000-step period),
keeping the chemistry alive and the shape continuously morphing without
destabilising the pattern.  Marching cubes extracts the evolving isosurface
and uploads it live to GPU.

Shader highlights
-----------------
  3-point Blinn-Phong   key (warm) / fill (cool) / back (rim)
  Animated key light    slowly orbits so highlights walk across the form
  Fresnel rim glow      cyan halo at silhouette edges — pops the volume
  Subsurface scatter    warm translucency on back-lit thin areas
  Gamma correction      sRGB output for correct perceived brightness

Usage
-----
  python rd_3d_organic.py                            # 64³ grid, auto device
  python rd_3d_organic.py --device cuda          # force CUDA
  python rd_3d_organic.py --size 80              # larger grid (needs CUDA)
  python rd_3d_organic.py --image photo.png      # seed from image slice
  python rd_3d_organic.py --preset 1             # mitosis breathing

Controls
--------
  Left drag     orbit camera
  Scroll        zoom
  A             toggle auto-rotate  (default: on)
  P             next preset
  ] / [         isovalue  ±0.02
  R             reset field
  Q / ESC       quit
"""

import sys, os, time, math, argparse
import numpy as np
import warp as wp
import pygame
import moderngl
from skimage.measure import marching_cubes
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter


wp.init()

# ── Breathing presets ─────────────────────────────────────────────────────────
# (name, Du, Dv, F_base, k_base, breath_period_steps)
# F oscillates ±8 %, k oscillates ±5 %, at slightly different phases.
# Keeps the chemistry alive without destabilising the pattern.
PRESETS = [
    ("coral",   0.16, 0.08, 0.055, 0.062, 2000),
    ("mitosis", 0.28, 0.12, 0.028, 0.053, 3000),
    ("blobs",   0.16, 0.08, 0.025, 0.055, 1800),
    ("worms",   0.16, 0.08, 0.078, 0.061, 2500),
]

# ── Warp kernels ──────────────────────────────────────────────────────────────

@wp.func
def sw3(a: wp.array3d(dtype=wp.float32),
        d: int, i: int, j: int,
        D: int, H: int, W: int) -> wp.float32:
    return a[(d % D + D) % D, (i % H + H) % H, (j % W + W) % W]


@wp.func
def fbm3(fx: float, fy: float, fz: float, t: float) -> wp.vec3f:
    """2-octave 3-D fbm displacement — creates fractal-warped Laplacian centre."""
    s  = wp.float32(3.0)
    s2 = wp.float32(6.3)
    tt = t * wp.float32(0.04)
    qx = wp.sin(fx * s + tt) * wp.cos(fy * s * wp.float32(1.1) + fz * wp.float32(0.7) + tt * wp.float32(0.8))
    qy = wp.cos(fy * s + tt * wp.float32(1.2)) * wp.sin(fz * s * wp.float32(0.9) + fx * wp.float32(0.5) - tt * wp.float32(0.6))
    qz = wp.sin(fz * s * wp.float32(1.1) - tt * wp.float32(0.9)) * wp.cos(fx * s + fy * wp.float32(0.8) + tt)
    h  = wp.float32(0.5)
    dx = wp.sin((fx + qx * h) * s2 + tt * wp.float32(1.7)) * wp.cos((fy + qy * h) * s2)
    dy = wp.cos((fy + qy * h) * s2 - tt * wp.float32(1.3)) * wp.sin((fz + qz * h) * s2)
    dz = wp.sin((fz + qz * h) * s2 + tt * wp.float32(2.1)) * wp.cos((fx + qx * h) * s2)
    return wp.vec3f(dx, dy, dz)


@wp.kernel
def gs3_step(
    u_in:  wp.array3d(dtype=wp.float32), v_in:  wp.array3d(dtype=wp.float32),
    u_out: wp.array3d(dtype=wp.float32), v_out: wp.array3d(dtype=wp.float32),
    D: int, H: int, W: int,
    Du: float, Dv: float, F: float, k: float,
    dt: float, warp_px: float, t: float,
):
    d, i, j = wp.tid()
    fx = wp.float32(j) / wp.float32(W) * wp.float32(2.0) - wp.float32(1.0)
    fy = wp.float32(i) / wp.float32(H) * wp.float32(2.0) - wp.float32(1.0)
    fz = wp.float32(d) / wp.float32(D) * wp.float32(2.0) - wp.float32(1.0)

    disp = fbm3(fx, fy, fz, t)
    wd = d + int(disp[2] * warp_px)
    wi = i + int(disp[1] * warp_px)
    wj = j + int(disp[0] * warp_px)

    wcu = sw3(u_in, wd, wi, wj, D, H, W)
    wcv = sw3(v_in, wd, wi, wj, D, H, W)

    lu = (sw3(u_in,d-1,i,j,D,H,W) + sw3(u_in,d+1,i,j,D,H,W) +
          sw3(u_in,d,i-1,j,D,H,W) + sw3(u_in,d,i+1,j,D,H,W) +
          sw3(u_in,d,i,j-1,D,H,W) + sw3(u_in,d,i,j+1,D,H,W) - wp.float32(6.0) * wcu)
    lv = (sw3(v_in,d-1,i,j,D,H,W) + sw3(v_in,d+1,i,j,D,H,W) +
          sw3(v_in,d,i-1,j,D,H,W) + sw3(v_in,d,i+1,j,D,H,W) +
          sw3(v_in,d,i,j-1,D,H,W) + sw3(v_in,d,i,j+1,D,H,W) - wp.float32(6.0) * wcv)

    u_v = u_in[d,i,j];  v_v = v_in[d,i,j]
    uvv = u_v * v_v * v_v
    u_out[d,i,j] = wp.clamp(u_v + dt*(Du*lu - uvv + F*(wp.float32(1.0)-u_v)), wp.float32(0.0), wp.float32(1.0))
    v_out[d,i,j] = wp.clamp(v_v + dt*(Dv*lv + uvv - (F+k)*v_v),               wp.float32(0.0), wp.float32(1.0))




# ── AO kernels ────────────────────────────────────────────────────────────────

@wp.func
def sample_v3(
    v:   wp.array3d(dtype=wp.float32),
    pos: wp.vec3f,
    D: int, H: int, W: int,
) -> wp.float32:
    """Nearest-neighbour voxel sample with clamped boundary."""
    iz = wp.clamp(int(pos[0] + wp.float32(0.5)), 0, D - 1)
    iy = wp.clamp(int(pos[1] + wp.float32(0.5)), 0, H - 1)
    ix = wp.clamp(int(pos[2] + wp.float32(0.5)), 0, W - 1)
    return v[iz, iy, ix]


@wp.kernel
def ao_kernel(
    verts:     wp.array(dtype=wp.vec3f),
    normals:   wp.array(dtype=wp.vec3f),
    v_field:   wp.array3d(dtype=wp.float32),
    sdirs:     wp.array(dtype=wp.vec3f),   # hemisphere sample dirs (local z-up frame)
    n_samp:    int,
    n_steps:   int,
    level:     float,
    ao_radius: float,
    ao_out:    wp.array(dtype=wp.float32),
    D: int, H: int, W: int,
):
    """
    Hemisphere-sampled AO via voxel ray marching.
    Mirrors the GLSL AO() function — casts n_samp rays per vertex, marches
    n_steps steps to ao_radius voxels, weights by cos(theta), returns sqrt(1-occ).
    """
    vid = wp.tid()
    pos = verts[vid]
    n   = wp.normalize(normals[vid])

    # Build orthonormal basis aligned to surface normal (local z → n)
    ref = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(1.0))
    if n[2] * n[2] > wp.float32(0.81):
        ref = wp.vec3f(wp.float32(1.0), wp.float32(0.0), wp.float32(0.0))
    t = wp.normalize(wp.cross(n, ref))
    b = wp.cross(n, t)

    step_size = wp.float32(ao_radius) / wp.float32(n_steps)
    occ       = wp.float32(0.0)
    weight    = wp.float32(0.0)

    for i in range(n_samp):
        sd     = sdirs[i]
        d      = sd[0] * t + sd[1] * b + sd[2] * n
        cos_th = sd[2]          # local-z = cos(angle with normal)

        hit_dist  = wp.float32(ao_radius)  # declared mutable via wp.float32()
        hit_found = int(0)                  # int() = mutable in Warp dynamic loop
        for s in range(n_steps):
            dist = step_size * wp.float32(s + 1)
            pt   = pos + d * dist
            val  = sample_v3(v_field, pt, D, H, W)
            if val > wp.float32(level) and hit_found == int(0):
                hit_dist  = dist
                hit_found = int(1)

        occ    += cos_th * (wp.float32(1.0) - hit_dist / wp.float32(ao_radius))
        weight += cos_th

    if weight > wp.float32(0.0):
        raw = wp.clamp(occ / weight, wp.float32(0.0), wp.float32(1.0))
        ao_out[vid] = wp.sqrt(wp.float32(1.0) - raw)   # sqrt() like the GLSL shader
    else:
        ao_out[vid] = wp.float32(1.0)

# ── Seed ──────────────────────────────────────────────────────────────────────

def make_seed(D, H, W, device, img_path=None):
    rng = np.random.default_rng()
    U   = np.ones( (D, H, W), np.float32)
    V   = np.zeros((D, H, W), np.float32)
    if img_path and os.path.isfile(img_path):
        from PIL import Image
        img  = Image.open(img_path).convert("L").resize((W, H), Image.LANCZOS)
        luma = np.array(img, dtype=np.float32) / 255.0
        for dz in range(D):
            w = max(0.0, 1.0 - abs(dz - D / 2.0) / (D / 4.0))
            n = rng.uniform(-0.03, 0.03, (H, W)).astype(np.float32)
            V[dz] = np.clip(0.45 * luma * w + n, 0.0, 0.5)
            U[dz] = np.clip(1.0  - 0.3 * luma * w, 0.5, 1.0)
    else:
        n_blobs = max(20, D * H * W // 800)
        for _ in range(n_blobs):
            cd = rng.integers(D//8, 7*D//8)
            ch = rng.integers(H//8, 7*H//8)
            cw = rng.integers(W//8, 7*W//8)
            r  = rng.integers(2, max(3, min(D,H,W)//7))
            dz, dy, dx = np.ogrid[:D, :H, :W]
            m = (dz-cd)**2 + (dy-ch)**2 + (dx-cw)**2 < r**2
            U[m] = 0.50;  V[m] = 0.25
        n = rng.uniform(-0.015, 0.015, (D,H,W)).astype(np.float32)
        U = np.clip(U+n, 0, 1);  V = np.clip(V+n, 0, 1)
    return (wp.array(U, dtype=wp.float32, device=device),
            wp.array(V, dtype=wp.float32, device=device))



def compute_ao(geo, v_field_wp, level, device, ao_samp_wp, ao_radius=4.5, n_steps=10):
    """Upload mesh verts/normals to GPU, run ao_kernel, return (N,) float32 AO values."""
    verts_np   = geo['verts']    # (N,3) float32, voxel-space coords
    normals_np = geo['normals']  # (N,3) float32
    n_verts    = len(verts_np)
    if n_verts == 0:
        return np.ones(0, np.float32)

    verts_wp   = wp.array(verts_np,   dtype=wp.vec3f, device=device)
    normals_wp = wp.array(normals_np, dtype=wp.vec3f, device=device)
    ao_wp      = wp.zeros(n_verts, dtype=wp.float32, device=device)
    D, H, W    = v_field_wp.shape
    n_samp     = ao_samp_wp.shape[0]

    wp.launch(ao_kernel, dim=n_verts,
              inputs=[verts_wp, normals_wp, v_field_wp, ao_samp_wp,
                      n_samp, n_steps, float(level), float(ao_radius),
                      ao_wp, D, H, W],
              device=device)
    wp.synchronize()
    return ao_wp.numpy()


# ── Geometry ──────────────────────────────────────────────────────────────────

def extract_geo(v_np, level):
    """Marching cubes + gradient-based vertex colour (amber → teal)."""

    v_np = gaussian_filter(v_np, sigma=1.0)
    try:
        verts, faces, normals, _ = marching_cubes(v_np, level=level,
                                                   allow_degenerate=False)
    except (ValueError, RuntimeError):
        return None
    if len(verts) == 0 or len(faces) == 0:
        return None

    gz, gy, gx = np.gradient(v_np)
    grad = np.sqrt(gx**2 + gy**2 + gz**2)
    g_v  = np.clip(map_coordinates(grad, verts.T, order=1, mode='nearest'), 0, None)
    gmax = g_v.max()
    if gmax > 1e-8:
        g_v /= gmax
    g_v = np.sqrt(g_v)   # perceptual gamma

    # Amber (low gradient / plateau) → teal (high gradient / reaction front)
    colors = np.stack([
        0.88 - g_v * 0.60,   # R
        0.52 - g_v * 0.18,   # G
        0.18 + g_v * 0.62,   # B
    ], axis=1).astype(np.float32)

    return dict(verts=verts.astype(np.float32),
                normals=normals.astype(np.float32),
                colors=colors,
                faces=faces.astype(np.uint32))


# ── Matrix utilities (row-major; pass .T to OpenGL) ──────────────────────────

def perspective(fov_deg, aspect, near, far):
    f  = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    nf = 1.0 / (near - far)
    return np.array([
        [f/aspect, 0,  0,                   0],
        [0,        f,  0,                   0],
        [0,        0,  (far+near)*nf,  2*far*near*nf],
        [0,        0,  -1,                  0],
    ], dtype=np.float32)


def lookat(eye, center, up):
    f = center - eye;   f /= np.linalg.norm(f)
    r = np.cross(f, up); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    M = np.eye(4, dtype=np.float32)
    M[0,:3] = r;  M[0,3] = -float(np.dot(r, eye))
    M[1,:3] = u;  M[1,3] = -float(np.dot(u, eye))
    M[2,:3] = -f; M[2,3] =  float(np.dot(f, eye))
    return M


# ── GLSL shaders ──────────────────────────────────────────────────────────────

VERT = """
#version 330

in vec3  in_vert;
in vec3  in_norm;
in vec3  in_color;
in float in_ao;

uniform mat4 mvp;
uniform mat4 model;     // for world-space position
uniform mat3 norm_mat;  // transpose(inverse(model)) upper-left 3×3

out vec3  v_pos;
out vec3  v_norm;
out vec3  v_color;
out float v_ao;

void main() {
    vec4 world = model * vec4(in_vert, 1.0);
    v_pos       = world.xyz;
    v_norm      = normalize(norm_mat * in_norm);
    v_color     = in_color;
    v_ao        = in_ao;
    gl_Position = mvp * vec4(in_vert, 1.0);
}
"""

FRAG = """
#version 330

in vec3  v_pos;
in vec3  v_norm;
in vec3  v_color;
in float v_ao;

uniform vec3  cam_pos;
uniform float time;         // slowly orbits the key light
// uniform float view_lo;  // discard pixels with NdotV below this  (0=silhouette, 1=face-on)
// uniform float view_hi;  // discard pixels with NdotV above this  (default 1.0)

out vec4 f_color;

void main() {
    vec3 N = normalize(v_norm);
    vec3 V = normalize(cam_pos - v_pos);

    // Per-pixel angle to camera: NdotV = 1 → face-on, 0 → edge-on.
    float NdotV = max(dot(N, V), 0.0);

    // Smooth angular band: full opacity inside [view_lo, view_hi],
    // fades over a soft edge of 0.08 on each side.
    // float band_w = 0.08;
    // float alpha  = smoothstep(view_lo - band_w, view_lo + band_w, NdotV)
    //              * smoothstep(view_hi + band_w, view_hi - band_w, NdotV);
    // if (alpha < 0.001) discard;  // skip fully transparent frags early
    
    float alpha = 1.0;

    // ── Key light: warm, slowly orbits the mesh ───────────────────────────
    float lt = time * 0.18;   // one full orbit ≈ 35 s
    vec3 L1  = normalize(vec3(cos(lt) * 1.2, 1.4, sin(lt) * 1.2));
    float d1 = max(dot(N, L1), 0.0);

    // ── Fill light: cool, fixed lower-left ───────────────────────────────
    vec3 L2  = normalize(vec3(-1.1, -0.6, 0.4));
    float d2 = max(dot(N, L2), 0.0) * 0.28;

    // ── Back light: gives depth separation at silhouette ─────────────────
    vec3 L3  = normalize(vec3(-0.4, 0.7, -1.3));
    float d3 = max(dot(N, L3), 0.0) * 0.22;

    // ── Blinn-Phong specular (key light only) ────────────────────────────
    vec3  H    = normalize(L1 + V);
    float spec = pow(max(dot(N, H), 0.0), 90.0) * 0.55;

    // ── Subsurface scatter approx: warm glow where back-light shines ─────
    //    through the mesh toward the viewer
    float sss = pow(max(dot(-L1, V), 0.0), 6.0) * 0.30;

    // ── Hemisphere-sampled AO (computed per-vertex in Warp, see ao_kernel) ─
    float ao = v_ao;  // 0=fully occluded, 1=fully open

    // ── Compose ──────────────────────────────────────────────────────────
    vec3 amb  = v_color * 0.12 * ao;
    vec3 diff = v_color * (d1 + d2 + d3) * (0.35 + 0.65 * ao);

    // Key-light colour: very slightly warm white
    vec3 key_col = vec3(1.00, 0.97, 0.92);
    vec3 spec_c  = key_col * spec;

    // Scatter: warm amber bleed
    vec3 sss_c = vec3(1.00, 0.65, 0.25) * sss;

    vec3 color = amb + diff + spec_c + sss_c;
    
    // Subtle vignette in screen-space (cheaper than SSAO)
    // (handled outside shader for portability — left as placeholder)

    // Filmic tone-map (Reinhard) + gamma
    color = color / (color + vec3(1.0));
    color = pow(color, vec3(1.0 / 2.2));
        
    f_color = vec4(clamp(color, 0.0, 1.0), alpha);
}
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image",   default=None)
    ap.add_argument("--size",    type=int,   default=64,   help="Grid N³ (default 64)")
    ap.add_argument("--preset",  type=int,   default=0,    help="0-3 (default 0 coral)")
    ap.add_argument("--level",   type=float, default=0.22, help="Isovalue (default 0.22)")
    ap.add_argument("--dt",      type=float, default=0.01)
    ap.add_argument("--spf",     type=int,   default=1,    help="Sim steps per frame (default 6)")
    ap.add_argument("--warp",    type=float, default=7.0,  help="Fractal warp px (default 7)")
    ap.add_argument("--device",  default="auto")
    ap.add_argument("--width",   type=int,   default=960)
    ap.add_argument("--height",  type=int,   default=720)
    args = ap.parse_args()

    # Device
    device = ("cuda" if wp.is_cuda_available() else "cpu") \
             if args.device == "auto" else args.device

    N = args.size;  D = H = W = N

    print(f"  Grid {N}³  |  device={device}  |  preset={PRESETS[args.preset][0]}")

    # ── Pygame + ModernGL ─────────────────────────────────────────────────────
    pygame.init()
    screen = pygame.display.set_mode(
        (args.width, args.height),
        pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
    pygame.display.set_caption("RD 3-D Organic")
    ctx = moderngl.create_context()
    ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE | moderngl.BLEND)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
    ctx.clear_color = (0.04, 0.04, 0.06, 1.0)   # near-black background

    prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)

    # Pre-allocate GPU buffers (sized for worst-case ~400 k verts at 64³)
    MAX_V = 600_000;  MAX_F = 1_200_000
    vbo_p = ctx.buffer(reserve=MAX_V * 12, dynamic=True)   # 3×f32
    vbo_n = ctx.buffer(reserve=MAX_V * 12, dynamic=True)
    vbo_c  = ctx.buffer(reserve=MAX_V * 12, dynamic=True)
    vbo_ao = ctx.buffer(reserve=MAX_V *  4, dynamic=True)   # 1×f32
    ibo    = ctx.buffer(reserve=MAX_F * 12, dynamic=True)   # 3×u32
    vao   = ctx.vertex_array(prog,
                [(vbo_p,  '3f', 'in_vert'),
                 (vbo_n,  '3f', 'in_norm'),
                 (vbo_c,  '3f', 'in_color'),
                 (vbo_ao, '1f', 'in_ao')],
                ibo)
    n_indices = 0   # updated when new mesh arrives

    # ── Camera state ──────────────────────────────────────────────────────────
    center      = np.array([W/2, H/2, D/2], dtype=np.float32)
    theta       = 0.3          # horizontal orbit angle (rad)
    phi         = 0.25         # vertical orbit angle (rad)
    radius      = N * 2.2      # zoom distance
    auto_rotate = True
    dragging    = False
    drag_start  = (0, 0)
    theta_start = theta
    phi_start   = phi

    def camera_eye():
        return center + radius * np.array([
            math.sin(theta) * math.cos(phi),
            math.sin(phi),
            math.cos(theta) * math.cos(phi),
        ], dtype=np.float32)

    # ── Simulation buffers ────────────────────────────────────────────────────
    pidx      = args.preset
    level     = args.level
    u_a, v_a  = make_seed(D, H, W, device, args.image)
    u_b       = wp.zeros((D,H,W), dtype=wp.float32, device=device)
    v_b       = wp.zeros((D,H,W), dtype=wp.float32, device=device)
    step      = 0
    t_start   = time.perf_counter()

    # Pre-compute hemisphere sample directions for AO kernel (fixed, z-up frame)
    _ao_rng     = np.random.default_rng(1337)
    _ao_thetas  = _ao_rng.uniform(0, 2*np.pi, 32).astype(np.float32)
    _ao_cphi    = _ao_rng.uniform(0, 1, 32).astype(np.float32)
    _ao_sphi    = np.sqrt(1 - _ao_cphi**2).astype(np.float32)
    _ao_sdirs   = np.stack([np.cos(_ao_thetas)*_ao_sphi,
                            np.sin(_ao_thetas)*_ao_sphi,
                            _ao_cphi], axis=1).astype(np.float32)
    ao_samp_wp  = wp.array(_ao_sdirs, dtype=wp.vec3f, device=device)

    # Geo extraction timing
    last_geo_t = 0.0
    GEO_INTERVAL = 0.12   # seconds between marching-cubes calls (~8 fps geo)

    wireframe = False
    view_lo   = 0.0   # NdotV lower bound (0=silhouette, raise to hide grazing faces)
    view_hi   = 1.0   # NdotV upper bound (1=face-on, lower to hide dead-on faces)
    clock = pygame.time.Clock()
    print("  Running — left-drag orbit, scroll zoom, A=autorotate, P=preset, ]/[=isovalue, W=wireframe\n")
    print("  Angle window: ,/. → low bound   -/= → high bound  (default 0.0–1.0 = all faces)\n")

    while True:
        # ── Events ───────────────────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit();  sys.exit()
            elif ev.type == pygame.KEYDOWN:
                k = ev.key
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    pygame.quit();  sys.exit()
                elif k == pygame.K_r:
                    u_a, v_a = make_seed(D, H, W, device, args.image)
                    step = 0
                elif k == pygame.K_p:
                    pidx = (pidx + 1) % len(PRESETS)
                    print(f"  Preset → {PRESETS[pidx][0]}")
                elif k == pygame.K_a:
                    auto_rotate = not auto_rotate
                elif k == pygame.K_RIGHTBRACKET:
                    level = min(level + 0.02, 0.85);  print(f"  level → {level:.3f}")
                elif k == pygame.K_LEFTBRACKET:
                    level = max(level - 0.02, 0.04);  print(f"  level → {level:.3f}")
                elif k == pygame.K_COMMA:
                    view_lo = max(0.0, round(view_lo - 0.05, 2))
                    print(f"  view_lo → {view_lo:.2f}  ({math.degrees(math.acos(view_lo)):.0f}°)")
                elif k == pygame.K_PERIOD:
                    view_lo = min(view_hi - 0.05, round(view_lo + 0.05, 2))
                    print(f"  view_lo → {view_lo:.2f}  ({math.degrees(math.acos(view_lo)):.0f}°)")
                elif k == pygame.K_MINUS:
                    view_hi = max(view_lo + 0.05, round(view_hi - 0.05, 2))
                    print(f"  view_hi → {view_hi:.2f}  ({math.degrees(math.acos(view_hi)):.0f}°)")
                elif k == pygame.K_EQUALS:
                    view_hi = min(1.0, round(view_hi + 0.05, 2))
                    print(f"  view_hi → {view_hi:.2f}  ({math.degrees(math.acos(view_hi)):.0f}°)")
                elif k == pygame.K_w:
                    wireframe = not wireframe
                    ctx.wireframe = wireframe
                    print(f"  wireframe → {'on' if wireframe else 'off'}")
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                dragging = True;  drag_start = ev.pos
                theta_start = theta;  phi_start = phi
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                dragging = False
            elif ev.type == pygame.MOUSEMOTION and dragging:
                dx = (ev.pos[0] - drag_start[0]) / args.width  * math.pi * 2
                dy = (ev.pos[1] - drag_start[1]) / args.height * math.pi
                theta = theta_start - dx
                phi   = max(-1.3, min(1.3, phi_start + dy))
            elif ev.type == pygame.MOUSEWHEEL:
                radius = max(N * 0.8, min(N * 6, radius - ev.y * N * 0.1))
            elif ev.type == pygame.VIDEORESIZE:
                args.width, args.height = ev.w, ev.h

        # ── Auto-rotate ───────────────────────────────────────────────────────
        if auto_rotate and not dragging:
            theta += 0.0004   # ~14°/s at 60fps

        # ── Breathing parameters ──────────────────────────────────────────────
        pname, Du, Dv, F0, k0, period = PRESETS[pidx]
        phase = (step % period) / period * 2 * math.pi
        F_t = F0 * (1.0 + 0.08 * math.sin(phase))
        k_t = k0 * (1.0 + 0.05 * math.cos(phase * 1.3))

        # ── Simulation ────────────────────────────────────────────────────────
        for _ in range(args.spf):
            wp.launch(gs3_step, dim=(D,H,W),
                      inputs=[u_a, v_a, u_b, v_b,
                               D, H, W, Du, Dv, F_t, k_t,
                               args.dt, args.warp, float(step)],
                      device=device)
            u_a, u_b = u_b, u_a
            v_a, v_b = v_b, v_a
            step += 1

        # ── Geometry update (rate-limited) ────────────────────────────────────
        now = time.perf_counter()
        if now - last_geo_t >= GEO_INTERVAL:
            last_geo_t = now
            geo = extract_geo(v_a.numpy(), level)
            if geo is not None:
                ao_np = compute_ao(geo, v_a, level, device, ao_samp_wp)
                n = len(geo['verts'])
                f = len(geo['faces'])
                vbo_p.write(geo['verts'].tobytes())
                vbo_n.write(geo['normals'].tobytes())
                vbo_c.write(geo['colors'].tobytes())
                vbo_ao.write(ao_np.tobytes())
                ibo.write(geo['faces'].tobytes())
                n_indices = f * 3

        # ── Render ────────────────────────────────────────────────────────────
        W_px, H_px = args.width, args.height
        ctx.viewport = (0, 0, W_px, H_px)
        ctx.clear()

        if n_indices > 0:
            eye = camera_eye()

            # Normalise grid to [-1,1] cube for the model matrix
            scale = 2.0 / N
            model = np.diag([scale, scale, scale, 1.0]).astype(np.float32)
            model[:3, 3] = -1.0    # shift so grid centre → origin

            view  = lookat(eye * scale - center * scale + np.array([0,0,0],np.float32),
                           np.zeros(3, np.float32),
                           np.array([0,1,0], np.float32))

            # Build a cleaner view: orbit around normalised-space origin
            eye_n = np.array([
                math.sin(theta) * math.cos(phi),
                math.sin(phi),
                math.cos(theta) * math.cos(phi),
            ], dtype=np.float32) * (radius * scale)
            view  = lookat(eye_n, np.zeros(3,np.float32), np.array([0,1,0],np.float32))
            proj  = perspective(45.0, W_px / H_px, 0.01, 500.0)
            mvp   = proj @ view @ model

            t_now = time.perf_counter() - t_start

            prog['mvp'].write(mvp.T.astype(np.float32).tobytes())
            prog['model'].write(model.T.astype(np.float32).tobytes())
            # Normal matrix = upper-left 3×3 of model (uniform scale → same as model)
            prog['norm_mat'].write(model[:3,:3].T.astype(np.float32).tobytes())
            prog['cam_pos'].value = tuple(eye_n.tolist())
            prog['time'].value    = float(t_now)
            # prog['view_lo'].value = float(view_lo)
            # prog['view_hi'].value = float(view_hi)

            vao.render(moderngl.TRIANGLES, vertices=n_indices)

        pygame.display.flip()

        fps = clock.tick(120)
        sps = args.spf * (1000.0 / max(fps, 1))
        pygame.display.set_caption(
            f"RD 3-D Organic  |  {pname}  |  "
            f"{clock.get_fps():.0f} fps  {sps:.0f} steps/s  |  "
            f"level={level:.2f}  lo={view_lo:.2f}  hi={view_hi:.2f}  "
            f"step={step}  {'[wireframe]' if wireframe else ''}"
        )


if __name__ == "__main__":
        main()
