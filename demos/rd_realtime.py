#!/usr/bin/env python3
"""
rd_realtime.py — Gray-Scott + fractal warp, real-time pygame viewer

Usage
-----
  python rd_realtime.py                          # 512×512 synthetic seed
  python rd_realtime.py --image photo.png        # seed from image
  python rd_realtime.py --size 768 --dt 1.5      # larger, faster/rougher
  python rd_realtime.py --device cuda            # GPU if available

Controls
--------
  SPACE / R    reset field
  P            next preset  (spots → coral → stripes → mitosis → worms → blobs)
  F            cycle fractal  (fbm → julia → none)
  C            cycle colormap (bio → fire → plasma → ice)
  + / -        steps per frame ±1   (more = faster evolution, less smooth)
  ] / [        warp pixel strength ±2
  S            save screenshot
  Q / ESC      quit
"""

import sys, os, time, argparse
import numpy as np
import warp as wp

wp.init()

# ── presets: name, Du, Dv, F, k ──────────────────────────────────────────────
PRESETS = [
    ("spots",   0.16, 0.08, 0.035, 0.065),
    ("coral",   0.16, 0.08, 0.055, 0.062),
    ("stripes", 0.16, 0.08, 0.060, 0.062),
    ("mitosis", 0.28, 0.12, 0.028, 0.053),
    ("worms",   0.16, 0.08, 0.078, 0.061),
    ("blobs",   0.16, 0.08, 0.025, 0.055),
]
CMAPS = ["bio", "fire", "plasma", "ice"]

# ── Warp kernels ──────────────────────────────────────────────────────────────

@wp.func
def sw(a: wp.array2d(dtype=wp.float32), i: int, j: int, h: int, w: int) -> wp.float32:
    """Toroidal (wrap) sample."""
    return a[(i % h + h) % h, (j % w + w) % w]


@wp.kernel
def gs_step(
    u_in:  wp.array2d(dtype=wp.float32),
    v_in:  wp.array2d(dtype=wp.float32),
    u_out: wp.array2d(dtype=wp.float32),
    v_out: wp.array2d(dtype=wp.float32),
    H: int, W: int,
    Du: float, Dv: float, F: float, k: float,
    dt: float, warp_px: float, t: float, frac: int,
):
    """
    Gray-Scott timestep with fractal-warped Laplacian centre.

    The 4 neighbours come from the standard position (i,j), but the
    centre of the stencil is sampled at the fractal-displaced position (wi,wj).
    This single displaced lookup is cheap, creates noise and self-similar
    texture, and avoids the cost of a full second Laplacian.
    """
    i, j = wp.tid()
    fx = wp.float32(j) / wp.float32(W) * wp.float32(2.0) - wp.float32(1.0)
    fy = wp.float32(i) / wp.float32(H) * wp.float32(2.0) - wp.float32(1.0)

    # Fractal displacement
    di = wp.float32(0.0)
    dj = wp.float32(0.0)

    if frac == 0:                              # single-octave fbm
        s  = wp.float32(4.0)
        tt = t * wp.float32(0.06)
        di = wp.sin(fx * s + tt) * wp.cos(fy * s * wp.float32(1.1) + tt * wp.float32(0.8))
        dj = wp.cos(fx * s * wp.float32(0.9) - tt * wp.float32(0.7)) * wp.sin(fy * s + tt)

    elif frac == 1:                            # julia set orbit
        cx = wp.float32(-0.7269)
        cy = wp.float32(0.1889)
        zx = fx * wp.float32(1.6)
        zy = fy * wp.float32(1.6)
        for _n in range(10):
            zx2 = zx * zx - zy * zy + cx
            zy2 = wp.float32(2.0) * zx * zy + cy
            zx = zx2
            zy = zy2
            msq = zx * zx + zy * zy
            if msq > wp.float32(16.0):
                inv = wp.float32(4.0) / msq
                zx = zx * inv
                zy = zy * inv
        di = wp.sin(zx * wp.float32(0.5))
        dj = wp.sin(zy * wp.float32(0.5))
    # frac == 2 → no warp (di=dj=0)

    wi = i + int(di * warp_px)
    wj = j + int(dj * warp_px)

    # Laplacian: 4 standard neighbours, displaced centre
    wcu = sw(u_in, wi, wj, H, W)
    wcv = sw(v_in, wi, wj, H, W)
    lu = (sw(u_in, i-1, j, H, W) + sw(u_in, i+1, j, H, W) +
          sw(u_in, i, j-1, H, W) + sw(u_in, i, j+1, H, W) - wp.float32(4.0) * wcu)
    lv = (sw(v_in, i-1, j, H, W) + sw(v_in, i+1, j, H, W) +
          sw(v_in, i, j-1, H, W) + sw(v_in, i, j+1, H, W) - wp.float32(4.0) * wcv)

    u_v = u_in[i, j]
    v_v = v_in[i, j]
    uvv = u_v * v_v * v_v

    u_out[i, j] = wp.clamp(u_v + dt * (Du * lu - uvv + F * (wp.float32(1.0) - u_v)),
                            wp.float32(0.0), wp.float32(1.0))
    v_out[i, j] = wp.clamp(v_v + dt * (Dv * lv + uvv - (F + k) * v_v),
                            wp.float32(0.0), wp.float32(1.0))


@wp.kernel
def to_rgb(
    v:   wp.array2d(dtype=wp.float32),
    rgb: wp.array(dtype=wp.uint8),
    H: int, W: int, cmap: int,
):
    """Map V concentration to RGB in parallel (no Python loop)."""
    i, j = wp.tid()
    t   = wp.clamp(v[i, j], wp.float32(0.0), wp.float32(1.0))
    idx = (i * W + j) * 3

    r = wp.float32(0.0)
    g = wp.float32(0.0)
    b = wp.float32(0.0)

    if cmap == 0:   # bioluminescent
        r = wp.clamp(t * wp.float32(2.0) - wp.float32(1.0), wp.float32(0.0), wp.float32(1.0)) * wp.float32(0.4)
        g = wp.clamp(t * wp.float32(3.0) - wp.float32(0.5), wp.float32(0.0), wp.float32(1.0))
        b = wp.clamp(t * wp.float32(2.0),                   wp.float32(0.0), wp.float32(1.0)) * wp.float32(0.9)
    elif cmap == 1: # fire
        r = wp.clamp(t * wp.float32(3.0),                   wp.float32(0.0), wp.float32(1.0))
        g = wp.clamp(t * wp.float32(3.0) - wp.float32(1.0), wp.float32(0.0), wp.float32(1.0))
        b = wp.clamp(t * wp.float32(3.0) - wp.float32(2.0), wp.float32(0.0), wp.float32(1.0))
    elif cmap == 2: # plasma
        r = wp.clamp(t * wp.float32(3.5) - wp.float32(0.5), wp.float32(0.0), wp.float32(1.0))
        g = wp.clamp(t * wp.float32(3.0) - wp.float32(1.0), wp.float32(0.0), wp.float32(1.0)) * wp.float32(0.55)
        b = wp.clamp(wp.float32(1.0) - t * wp.float32(2.5), wp.float32(0.0), wp.float32(1.0))
    elif cmap == 3: # ice
        r = wp.clamp(t * wp.float32(2.0) - wp.float32(0.5), wp.float32(0.0), wp.float32(1.0)) * wp.float32(0.25)
        g = wp.clamp(t * wp.float32(2.5),                   wp.float32(0.0), wp.float32(1.0)) * wp.float32(0.85)
        b = wp.clamp(t * wp.float32(2.0),                   wp.float32(0.0), wp.float32(1.0))

    rgb[idx    ] = wp.uint8(r * wp.float32(255.0))
    rgb[idx + 1] = wp.uint8(g * wp.float32(255.0))
    rgb[idx + 2] = wp.uint8(b * wp.float32(255.0))


# ── Seed ──────────────────────────────────────────────────────────────────────

def make_seed(H, W, img_path, device):
    rng = np.random.default_rng()
    if img_path and os.path.isfile(img_path):
        from PIL import Image
        img  = Image.open(img_path).convert("L").resize((W, H), Image.LANCZOS)
        luma = np.array(img, dtype=np.float32) / 255.0
        noise = rng.uniform(-0.05, 0.05, luma.shape).astype(np.float32)
        U = np.clip(1.0 - 0.4 * luma + noise, 0, 1)
        V = np.clip(0.5  * luma + noise,       0, 1)
    else:
        U = np.ones( (H, W), np.float32)
        V = np.zeros((H, W), np.float32)
        n = max(20, (H * W) // 256)
        for _ in range(n):
            cy = rng.integers(H // 8, 7 * H // 8)
            cx = rng.integers(W // 8, 7 * W // 8)
            r  = rng.integers(2, max(3, min(H, W) // 15))
            y, x = np.ogrid[:H, :W]
            m = (y - cy) ** 2 + (x - cx) ** 2 < r ** 2
            U[m] = 0.5
            V[m] = 0.25
        noise = rng.uniform(-0.02, 0.02, (H, W)).astype(np.float32)
        U = np.clip(U + noise, 0, 1)
        V = np.clip(V + noise, 0, 1)

    return (wp.array(U, dtype=wp.float32, device=device),
            wp.array(V, dtype=wp.float32, device=device))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image",  default=None,    help="Seed image path")
    ap.add_argument("--size",   type=int, default=512,  help="Grid size (default 512)")
    ap.add_argument("--width",  type=int, default=None, help="Override width")
    ap.add_argument("--height", type=int, default=None, help="Override height")
    ap.add_argument("--dt",     type=float, default=1.0,  help="Timestep size; >1 = faster/rougher (default 1.0)")
    ap.add_argument("--steps",  type=int,   default=2,    help="Steps per frame (default 2)")
    ap.add_argument("--warp",   type=float, default=10.0, help="Warp pixel strength (default 10)")
    ap.add_argument("--preset", type=int,   default=0,    help="Preset index 0-5")
    ap.add_argument("--frac",   type=int,   default=0,    help="Fractal 0=fbm 1=julia 2=none")
    ap.add_argument("--cmap",   type=int,   default=0,    help="Colormap 0=bio 1=fire 2=plasma 3=ice")
    ap.add_argument("--device", default="cpu",            help="Warp device: cpu or cuda")
    args = ap.parse_args()

    W      = args.width  or args.size
    H      = args.height or args.size
    device = args.device

    import pygame
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    clock  = pygame.time.Clock()

    # Mutable state
    pidx    = args.preset % len(PRESETS)
    frac    = args.frac
    cmap    = args.cmap
    spf     = args.steps
    warp_px = args.warp
    dt      = args.dt
    step    = 0

    # Buffers
    u_a, v_a = make_seed(H, W, args.image, device)
    u_b = wp.zeros((H, W), dtype=wp.float32, device=device)
    v_b = wp.zeros((H, W), dtype=wp.float32, device=device)
    rgb = wp.zeros(H * W * 3, dtype=wp.uint8, device=device)

    print(__doc__)
    print(f"Grid {W}×{H}  |  device={device}  |  dt={dt}")
    print(f"Preset: {PRESETS[pidx][0]}  |  Fractal: {['fbm','julia','none'][frac]}\n")

    fps_hist = []
    running  = True

    while running:
        t0 = time.perf_counter()

        # ── Events ────────────────────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                kk = ev.key
                if kk in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif kk in (pygame.K_SPACE, pygame.K_r):
                    u_a, v_a = make_seed(H, W, args.image, device)
                    step = 0;  print("Reset")
                elif kk == pygame.K_p:
                    pidx = (pidx + 1) % len(PRESETS)
                    print(f"Preset → {PRESETS[pidx][0]}")
                elif kk == pygame.K_f:
                    frac = (frac + 1) % 3
                    print(f"Fractal → {['fbm','julia','none'][frac]}")
                elif kk == pygame.K_c:
                    cmap = (cmap + 1) % len(CMAPS)
                    print(f"Colormap → {CMAPS[cmap]}")
                elif kk in (pygame.K_EQUALS, pygame.K_PLUS):
                    spf = min(spf + 1, 20);  print(f"spf → {spf}")
                elif kk == pygame.K_MINUS:
                    spf = max(spf - 1, 1);   print(f"spf → {spf}")
                elif kk == pygame.K_RIGHTBRACKET:
                    warp_px = min(warp_px + 2, 60);  print(f"warp → {warp_px:.0f}px")
                elif kk == pygame.K_LEFTBRACKET:
                    warp_px = max(warp_px - 2,  0);  print(f"warp → {warp_px:.0f}px")
                elif kk == pygame.K_s:
                    from PIL import Image
                    fname = f"rd_{step:07d}.png"
                    Image.fromarray(rgb.numpy().reshape(H, W, 3)).save(fname)
                    print(f"Screenshot → {fname}")

        _, Du, Dv, F, k = PRESETS[pidx]

        # ── Simulation ────────────────────────────────────────────────────────
        for _ in range(spf):
            wp.launch(gs_step,
                      dim=(H, W),
                      inputs=[u_a, v_a, u_b, v_b,
                               H, W, Du, Dv, F, k,
                               dt, warp_px, float(step), frac],
                      device=device)
            u_a, u_b = u_b, u_a
            v_a, v_b = v_b, v_a
            step += 1

        # ── Render ────────────────────────────────────────────────────────────
        wp.launch(to_rgb, dim=(H, W),
                  inputs=[v_a, rgb, H, W, cmap],
                  device=device)

        # Zero-copy to pygame: numpy view → frombuffer (buffer protocol, no alloc)
        rgb_np = rgb.numpy().reshape(H, W, 3)
        surf   = pygame.image.frombuffer(rgb_np, (W, H), 'RGB')
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # ── FPS title ─────────────────────────────────────────────────────────
        fps_hist.append(1.0 / max(time.perf_counter() - t0, 1e-6))
        if len(fps_hist) >= 20:
            fps = sum(fps_hist) / len(fps_hist);  fps_hist.clear()
            pygame.display.set_caption(
                f"RD Fractal  |  {PRESETS[pidx][0]}  {['fbm','julia','none'][frac]}"
                f"  {CMAPS[cmap]}  |  {fps:.0f} fps  {fps*spf:.0f} steps/s"
                f"  |  spf={spf}  warp={warp_px:.0f}px"
            )

    pygame.quit()


if __name__ == "__main__":
    main()
