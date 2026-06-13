#!/usr/bin/env python3
"""
reaction_diffusion_fractal.py
==============================
Gray-Scott reaction-diffusion with fractal domain-warping feedback,
implemented in NVIDIA Warp for parallel CPU/GPU execution.

HOW IT WORKS
------------
1. The input image seeds initial chemical concentrations:
     U (substrate)  = 1 - 0.3 * luminance   (bright → less substrate)
     V (activator)  = 0.5 * luminance        (bright → more activator)
   so the emerging spot/stripe topology is shaped by the original image.

2. Each timestep runs the Gray-Scott PDE:
     dU/dt = Du·∇²U - U·V² + F·(1 - U)
     dV/dt = Dv·∇²V + U·V² - (F + k)·V

3. Fractal feedback: before computing the Laplacian, a fractal transform
   warps the sampling coordinates. Two modes are available:
     fbm   — 3-octave fractional Brownian Motion; each octave phase-shifts
              using the previous octave's output, creating self-similar warp.
     julia — z → z² + c iteration; the bounded orbit is mapped to a
              displacement field, so diffusion samples from positions
              sculpted by complex dynamics.
   The fractal Laplacian is blended with the standard one and the blend
   ramps from 0 → 0.6 over the first half of the run, so early-stage
   pattern seeding is clean and fractal feedback intensifies as structures mature.

USAGE
-----
  # Synthetic seed (random blobs), fbm fractal, spots preset, 400 steps:
  python reaction_diffusion_fractal.py

  # Seed from image, Julia fractal, coral preset, GIF output:
  python reaction_diffusion_fractal.py --image photo.png --fractal julia --preset coral --gif

  # Larger grid, fire colormap, save every 10 steps:
  python reaction_diffusion_fractal.py --size 512 --colormap fire --save-every 10

  # Run on CUDA if available:
  python reaction_diffusion_fractal.py --device cuda

GRAY-SCOTT PRESETS
------------------
  spots    F=0.035 k=0.065  — isolated circular spots
  stripes  F=0.060 k=0.062  — long parallel stripes
  coral    F=0.055 k=0.062  — branching coral/maze
  mitosis  F=0.028 k=0.053  — self-replicating spots
  blobs    F=0.025 k=0.055  — large slow-moving blobs
  worms    F=0.078 k=0.061  — tangled worm-like filaments
"""

import warp as wp
import numpy as np
from PIL import Image
import os
import argparse
import time as _time

wp.init()

# ─────────────────────────────────────────────────────────────────────────────
# Parameter presets
# ─────────────────────────────────────────────────────────────────────────────

PRESETS = {
    "spots":   dict(Du=0.16, Dv=0.08, F=0.035, k=0.065),
    "stripes": dict(Du=0.16, Dv=0.08, F=0.060, k=0.062),
    "coral":   dict(Du=0.16, Dv=0.08, F=0.055, k=0.062),
    "mitosis": dict(Du=0.28, Dv=0.12, F=0.028, k=0.053),
    "blobs":   dict(Du=0.16, Dv=0.08, F=0.025, k=0.055),
    "worms":   dict(Du=0.16, Dv=0.08, F=0.078, k=0.061),
}

# ─────────────────────────────────────────────────────────────────────────────
# Warp helper functions (compiled for CPU or CUDA)
# ─────────────────────────────────────────────────────────────────────────────

@wp.func
def swrap(a: wp.array2d(dtype=wp.float32), i: int, j: int, h: int, w: int) -> wp.float32:
    """Sample a 2-D array with toroidal (wrap-around) boundary conditions."""
    return a[(i % h + h) % h, (j % w + w) % w]


@wp.func
def fbm_warp(fx: float, fy: float, t: float) -> wp.vec2f:
    """
    3-octave fractional Brownian Motion domain warp.

    Each octave feeds its output as a phase perturbation into the next,
    producing recursive self-similarity: zoom into any region and you see
    a structure statistically similar to the whole.

    Returns a displacement vector in [-1, 1]².
    """
    s1 = wp.float32(2.5)
    s2 = wp.float32(5.1)
    s3 = wp.float32(10.3)

    # Octave 1 — base sinusoidal wave
    qx = wp.sin(fx * s1 + t) * wp.cos(fy * s1 * wp.float32(1.1) + t * wp.float32(0.8))
    qy = wp.cos(fx * s1 * wp.float32(0.9) - t * wp.float32(0.6)) * wp.sin(fy * s1 + t)

    # Octave 2 — phase-shifted by octave 1
    rx = wp.sin(fx * s2 + qx * wp.float32(1.7) + t * wp.float32(1.3)) * \
         wp.cos(fy * s2 + qy * wp.float32(1.7))
    ry = wp.cos(fx * s2 + qx * wp.float32(1.7) - t * wp.float32(0.9)) * \
         wp.sin(fy * s2 + qy * wp.float32(1.7))

    # Octave 3 — phase-shifted by octave 2 (finest detail)
    dx = wp.sin((fx + rx * wp.float32(0.4)) * s3 + t * wp.float32(2.0)) * \
         wp.cos((fy + ry * wp.float32(0.4)) * s3)
    dy = wp.cos((fx + rx * wp.float32(0.4)) * s3 - t * wp.float32(1.7)) * \
         wp.sin((fy + ry * wp.float32(0.4)) * s3)

    return wp.vec2f(dx, dy)


@wp.func
def julia_warp(fx: float, fy: float) -> wp.vec2f:
    """
    Julia set orbit displacement field.

    Maps each pixel position (normalised to ≈[-1.6, 1.6]) through repeated
    complex iteration  z → z² + c  (c = -0.7269 + 0.1889i, giving a rich
    connected Julia set).  The bounded orbit endpoint is mapped via sin to a
    smooth displacement in [-1, 1]², so diffusion samples from positions
    whose structure reflects the fractal basin geometry.

    A soft clamp prevents divergence while preserving the orbit's shape.
    """
    cx = wp.float32(-0.7269)
    cy = wp.float32(0.1889)
    zx = fx * wp.float32(1.6)
    zy = fy * wp.float32(1.6)

    for _i in range(20):
        zx2 = zx * zx - zy * zy + cx
        zy2 = wp.float32(2.0) * zx * zy + cy
        zx = zx2
        zy = zy2
        # Soft clamp: redirect escaping orbits back toward origin
        msq = zx * zx + zy * zy
        if msq > wp.float32(16.0):
            inv = wp.float32(4.0) / msq
            zx = zx * inv
            zy = zy * inv

    return wp.vec2f(wp.sin(zx * wp.float32(0.5)), wp.sin(zy * wp.float32(0.5)))


# ─────────────────────────────────────────────────────────────────────────────
# Main simulation kernel (one PDE timestep)
# ─────────────────────────────────────────────────────────────────────────────

@wp.kernel
def gs_step(
    u_in:  wp.array2d(dtype=wp.float32),
    v_in:  wp.array2d(dtype=wp.float32),
    u_out: wp.array2d(dtype=wp.float32),
    v_out: wp.array2d(dtype=wp.float32),
    H: int,
    W: int,
    Du: float,
    Dv: float,
    F:  float,
    k:  float,
    frac_strength: float,
    frac_type: int,   # 0 = fbm | 1 = julia | 2 = none
    t: float,
    blend: float,     # weight of fractal Laplacian vs standard (0..1)
):
    """
    One Gray-Scott reaction-diffusion step with fractal Laplacian blending.

    For each pixel (i, j):
      1. Compute a fractal displacement d at the normalised position.
      2. Compute a standard 5-point Laplacian centred at (i, j).
      3. Compute a fractal-warped Laplacian centred at (i+di, j+dj).
      4. Blend the two Laplacians; `blend` ramps up over the simulation.
      5. Apply Gray-Scott update equations.
    """
    i, j = wp.tid()

    # Normalise pixel to [-1, 1] for the fractal functions
    fx = wp.float32(j) / wp.float32(W) * wp.float32(2.0) - wp.float32(1.0)
    fy = wp.float32(i) / wp.float32(H) * wp.float32(2.0) - wp.float32(1.0)

    # Compute fractal displacement vector
    d = wp.vec2f(0.0, 0.0)
    if frac_type == 0:
        d = fbm_warp(fx, fy, t * wp.float32(0.08))
    elif frac_type == 1:
        d = julia_warp(fx, fy)
    # frac_type == 2 → d stays zero (no warp)

    # Convert displacement to integer pixel offsets
    wi = i + int(d[1] * float(H) * frac_strength * wp.float32(0.04))
    wj = j + int(d[0] * float(W) * frac_strength * wp.float32(0.04))

    # ── Standard 5-point Laplacian ──
    lu = (swrap(u_in, i-1, j,   H, W) + swrap(u_in, i+1, j,   H, W) +
          swrap(u_in, i,   j-1, H, W) + swrap(u_in, i,   j+1, H, W) -
          wp.float32(4.0) * u_in[i, j])

    lv = (swrap(v_in, i-1, j,   H, W) + swrap(v_in, i+1, j,   H, W) +
          swrap(v_in, i,   j-1, H, W) + swrap(v_in, i,   j+1, H, W) -
          wp.float32(4.0) * v_in[i, j])

    # ── Fractal-warped Laplacian (centre shifted to displaced position) ──
    wcu = swrap(u_in, wi, wj, H, W)
    wcv = swrap(v_in, wi, wj, H, W)

    lu_f = (swrap(u_in, wi-1, wj,   H, W) + swrap(u_in, wi+1, wj,   H, W) +
            swrap(u_in, wi,   wj-1, H, W) + swrap(u_in, wi,   wj+1, H, W) -
            wp.float32(4.0) * wcu)

    lv_f = (swrap(v_in, wi-1, wj,   H, W) + swrap(v_in, wi+1, wj,   H, W) +
            swrap(v_in, wi,   wj-1, H, W) + swrap(v_in, wi,   wj+1, H, W) -
            wp.float32(4.0) * wcv)

    # ── Blend the two Laplacians ──
    b0 = wp.float32(1.0) - blend
    flu = lu * b0 + lu_f * blend
    flv = lv * b0 + lv_f * blend

    # ── Gray-Scott reaction-diffusion equations ──
    #   dU/dt = Du·∇²U  -  U·V²  +  F·(1 - U)
    #   dV/dt = Dv·∇²V  +  U·V²  -  (F + k)·V
    u_v = u_in[i, j]
    v_v = v_in[i, j]
    uvv = u_v * v_v * v_v           # autocatalytic term U·V²

    u_out[i, j] = wp.clamp(u_v + Du * flu - uvv + F * (wp.float32(1.0) - u_v),
                            wp.float32(0.0), wp.float32(1.0))
    v_out[i, j] = wp.clamp(v_v + Dv * flv + uvv - (F + k) * v_v,
                            wp.float32(0.0), wp.float32(1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Colourisation (runs on CPU via NumPy — not performance-critical)
# ─────────────────────────────────────────────────────────────────────────────

def colorize(v_np: np.ndarray, mode: str) -> np.ndarray:
    """Map V concentration [0,1] to an RGB image array [0,1]³."""
    t = np.clip(v_np, 0.0, 1.0)

    if mode == "plasma":
        # Dark purple → magenta → yellow-white
        r = np.clip(t * 3.5 - 0.5, 0, 1)
        g = np.clip(t * 3.0 - 1.0, 0, 1) * 0.55
        b = np.clip(1.0 - t * 2.5,  0, 1)
    elif mode == "fire":
        r = np.clip(t * 3.0,        0, 1)
        g = np.clip(t * 3.0 - 1.0, 0, 1)
        b = np.clip(t * 3.0 - 2.0, 0, 1)
    elif mode == "ice":
        r = np.clip(t * 2.0 - 0.5, 0, 1) * 0.25
        g = np.clip(t * 2.5,        0, 1) * 0.85
        b = np.clip(t * 2.0,        0, 1)
    elif mode == "bioluminescent":
        r = np.clip(t * 2.0 - 1.0, 0, 1) * 0.4
        g = np.clip(t * 3.0 - 0.5, 0, 1)
        b = np.clip(t * 2.0,        0, 1) * 0.9
    else:  # gray
        return np.stack([t, t, t], axis=-1)

    return np.stack([r, g, b], axis=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Seeding utilities
# ─────────────────────────────────────────────────────────────────────────────

def seed_from_image(path: str, size: int):
    """
    Seed (U, V) from an image file.

    Bright pixels → high V (more activator), low U (less substrate).
    This means bright regions of the image grow patterns faster,
    and the topological structure of the emerging chemistry reflects
    the original image content.
    """
    img  = Image.open(path).convert("L").resize((size, size), Image.LANCZOS)
    luma = np.array(img, dtype=np.float32) / 255.0

    rng   = np.random.default_rng(42)
    noise = rng.uniform(-0.02, 0.02, luma.shape).astype(np.float32)

    U = np.clip(1.0 - 0.3 * luma + noise, 0.0, 1.0)
    V = np.clip(0.5 * luma + noise,        0.0, 1.0)
    return U, V


def seed_synthetic(size: int):
    """
    Create a synthetic seed: U=1 everywhere, small random blobs of V
    distributed across the grid to kick-start the reaction.
    """
    rng = np.random.default_rng(42)
    U   = np.ones((size, size),  dtype=np.float32)
    V   = np.zeros((size, size), dtype=np.float32)

    n_seeds = max(12, size // 16)
    for _ in range(n_seeds):
        cy = rng.integers(size // 8, 7 * size // 8)
        cx = rng.integers(size // 8, 7 * size // 8)
        r  = rng.integers(max(2, size // 30), max(4, size // 14))
        y, x = np.ogrid[:size, :size]
        mask    = (y - cy) ** 2 + (x - cx) ** 2 < r ** 2
        U[mask] = 0.50
        V[mask] = 0.25

    noise = rng.uniform(-0.01, 0.01, (size, size)).astype(np.float32)
    return np.clip(U + noise, 0, 1), np.clip(V + noise, 0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--image",        default=None,          help="Input image to seed concentrations")
    ap.add_argument("--size",         type=int,   default=256,  help="Grid size N → N×N (default: 256)")
    ap.add_argument("--steps",        type=int,   default=400,  help="Total simulation steps (default: 400)")
    ap.add_argument("--preset",       default="spots",       choices=list(PRESETS),
                    help="Gray-Scott parameter preset (default: spots)")
    ap.add_argument("--fractal",      default="fbm",         choices=["fbm", "julia", "none"],
                    help="Fractal warp mode (default: fbm)")
    ap.add_argument("--frac-strength",type=float, default=1.0, help="Fractal displacement strength (default: 1.0)")
    ap.add_argument("--colormap",     default="plasma",
                    choices=["plasma", "fire", "ice", "bioluminescent", "gray"],
                    help="Output colormap (default: plasma)")
    ap.add_argument("--save-every",   type=int,   default=20,  help="Save a frame every N steps (default: 20)")
    ap.add_argument("--gif",          action="store_true",    help="Create animated GIF from saved frames")
    ap.add_argument("--gif-fps",      type=int,   default=12,  help="GIF playback FPS (default: 12)")
    ap.add_argument("--outdir",       default="rd_fractal_out", help="Output directory for frames")
    ap.add_argument("--device",       default="cpu",          help="Warp device: cpu or cuda (default: cpu)")
    args = ap.parse_args()

    device = args.device
    size   = args.size
    p      = PRESETS[args.preset]
    ftype  = {"fbm": 0, "julia": 1, "none": 2}[args.fractal]

    os.makedirs(args.outdir, exist_ok=True)

    # Header
    print("━" * 60)
    print("  Reaction-Diffusion Fractal Feedback  (Warp)")
    print("━" * 60)
    print(f"  Grid     : {size}×{size}")
    print(f"  Preset   : {args.preset}  (F={p['F']} k={p['k']})")
    print(f"  Fractal  : {args.fractal}  strength={args.frac_strength}")
    print(f"  Steps    : {args.steps}  device={device}")
    print(f"  Colormap : {args.colormap}   outdir={args.outdir}")
    print("━" * 60)

    # ── Initialise concentration fields ──
    if args.image and os.path.isfile(args.image):
        print(f"  Seeding from image : {args.image}")
        U_np, V_np = seed_from_image(args.image, size)
    else:
        if args.image:
            print(f"  [warning] Image not found: {args.image!r} — using synthetic seed")
        else:
            print("  Seeding : synthetic random blobs")
        U_np, V_np = seed_synthetic(size)

    # Upload to Warp (2-D arrays, shape (size, size))
    u_a = wp.array(U_np, dtype=wp.float32, device=device)
    v_a = wp.array(V_np, dtype=wp.float32, device=device)
    u_b = wp.zeros((size, size), dtype=wp.float32, device=device)
    v_b = wp.zeros((size, size), dtype=wp.float32, device=device)

    frames: list[Image.Image] = []
    t0 = _time.perf_counter()

    print("\n  Running...\n")

    for step in range(args.steps + 1):

        # ── Save frame ──
        if step % args.save_every == 0:
            v_np  = v_a.numpy()
            rgb   = (colorize(v_np, args.colormap) * 255).astype(np.uint8)
            frame = Image.fromarray(rgb, mode="RGB")
            fname = os.path.join(args.outdir, f"frame_{step:05d}.png")
            frame.save(fname)
            frames.append(frame.copy())

            elapsed = _time.perf_counter() - t0
            pct     = 100.0 * step / args.steps if args.steps else 100
            bar_len = 30
            filled  = int(bar_len * step / max(args.steps, 1))
            bar     = "█" * filled + "░" * (bar_len - filled)
            print(f"  [{bar}] {pct:5.1f}%  step {step:4d}  ({elapsed:.1f}s)", flush=True)

        if step == args.steps:
            break

        # ── Blend ramps from 0 → 0.6 over the first half of the run ──
        blend = min(1.0, (step / max(args.steps, 1)) * 2.0) * 0.6

        # ── One Gray-Scott + fractal step ──
        wp.launch(
            gs_step,
            dim=(size, size),
            inputs=[
                u_a, v_a, u_b, v_b,
                size, size,
                float(p["Du"]), float(p["Dv"]),
                float(p["F"]),  float(p["k"]),
                float(args.frac_strength),
                ftype,
                float(step),
                float(blend),
            ],
            device=device,
        )

        # Ping-pong: swap input/output buffers
        u_a, u_b = u_b, u_a
        v_a, v_b = v_b, v_a

    elapsed = _time.perf_counter() - t0
    sps     = args.steps / elapsed if elapsed > 0 else 0
    print(f"\n  Finished in {elapsed:.1f}s  ({sps:.0f} steps/s)")
    print(f"  {len(frames)} frames → {os.path.abspath(args.outdir)}/")

    # ── Optional GIF ──
    if args.gif and frames:
        gif_path = os.path.join(args.outdir, "animation.gif")
        duration_ms = max(1, 1000 // args.gif_fps)
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
        print(f"  GIF saved : {os.path.abspath(gif_path)}")

    final_path = os.path.join(args.outdir, f"frame_{args.steps:05d}.png")
    print(f"  Final frame : {os.path.abspath(final_path)}")
    print()


if __name__ == "__main__":
    main()
