#!/usr/bin/env python3
"""
nebula_raymarch.py — Real-time volumetric nebula via GPU ray marching
======================================================================
All compute runs on the GPU via NVIDIA Warp kernels.

Each pixel launches a perspective ray through a 3D density volume
built from two levels of domain-warped fractal Brownian motion (fBm).
Colour and opacity accumulate via a physically-based emission-absorption
model.  Stars are added with a per-pixel integer hash — exactly 1 pixel
each, no blobs.

Usage
-----
  python nebula_raymarch.py                   # 512×512, CUDA
  python nebula_raymarch.py --size 256        # smaller / faster
  python nebula_raymarch.py --steps 96        # more detail, slower
  python nebula_raymarch.py --device cpu      # CPU fallback
  python nebula_raymarch.py --speed 0.5       # slow animation

Controls
--------
  S        save PNG screenshot
  Q / ESC  quit
"""

import argparse
import time

import numpy as np
import warp as wp

wp.init()

# ── Compile-time constant: ray-march steps ────────────────────────────────────
# Increase for more detail; decrease for more speed.
MARCH_STEPS = 64


# ── GPU helper functions ──────────────────────────────────────────────────────

@wp.func
def ridged_fbm(p: wp.vec3f) -> wp.float32:
    """
    Ridged multifractal noise — each octave contributes (1 - |noise|).

    Regular fBm sums smooth signed noise → soft blobs.
    Ridged fBm flips the distribution: the flat parts become dark voids
    and the peaks become sharp bright ridges.  That's what filamentary
    nebulae actually look like.

    The slight frequency offsets (2.01, 4.03, 8.07, 16.11) break
    axis-aligned grid artefacts.
    """
    v  = wp.float32(0.5000) * (wp.float32(1.0) - wp.abs(wp.noise(wp.rand_init(17),  p)))
    v += wp.float32(0.2500) * (wp.float32(1.0) - wp.abs(wp.noise(wp.rand_init(31),  p * wp.float32(2.01))))
    v += wp.float32(0.1250) * (wp.float32(1.0) - wp.abs(wp.noise(wp.rand_init(53),  p * wp.float32(4.03))))
    v += wp.float32(0.0625) * (wp.float32(1.0) - wp.abs(wp.noise(wp.rand_init(97),  p * wp.float32(8.07))))
    v += wp.float32(0.0313) * (wp.float32(1.0) - wp.abs(wp.noise(wp.rand_init(113), p * wp.float32(16.11))))
    return v


@wp.func
def density_at(p: wp.vec3f, anim_t: wp.float32) -> wp.float32:
    """
    Two-level domain-warped ridged density field.

    Warp level 1: sample ridged_fbm at pa to get q (3 channels)
    Warp level 2: sample ridged_fbm at (pa + 4*q) to get r (2 channels)
    Final density: ridged_fbm at (pa + 4*r)

    Stronger warp (4.0 vs old 2.2) + ridged noise creates
    the twisted, interleaved filament structure seen in real nebulae.
    Power cube at the end crushes the mid-range to zero so filaments
    read as sharp bright threads against true black.
    """
    pa = p + wp.vec3f(
        anim_t * wp.float32(0.040),
        anim_t * wp.float32(0.030),
        anim_t * wp.float32(0.025),
    )

    # Sample at 2.5× world scale → filaments are ~2.5× finer
    ps = pa * wp.float32(2.5)

    # Warp level 1 — two channels, moderate strength so it twists without
    # displacing sample points by more than ~1 noise period
    qx = ridged_fbm(ps)
    qy = ridged_fbm(ps + wp.vec3f(wp.float32(1.7), wp.float32(9.2), wp.float32(3.8)))
    q  = wp.vec3f(qx, qy, wp.float32(0.0))

    # Final density at domain-warped position
    d = ridged_fbm(ps + wp.float32(1.5) * q)

    # Hard threshold: values below 0.38 → absolute zero (void between filaments)
    # Steep ramp above that → filaments snap into existence rather than fading in
    d = wp.clamp((d - wp.float32(0.88)) * wp.float32(7.0), wp.float32(0.0), wp.float32(1.0))
    d = d * d   # additional power to keep peak density high, crush the ramp tail

    # Elliptical falloff — slightly wider than tall (like Crab Nebula)
    ex = p[0] / wp.float32(1.4)
    ey = p[1] / wp.float32(1.1)
    ez = p[2] / wp.float32(1.0)
    r_ell   = wp.sqrt(ex * ex + ey * ey + ez * ez)
    falloff = wp.clamp(wp.float32(1.0) - r_ell, wp.float32(0.0), wp.float32(1.0))
    d       = d * falloff * falloff * falloff   # cubic falloff: hard edge at the boundary

    return d


@wp.func
def nebula_color(d: wp.float32, pos: wp.vec3f) -> wp.vec3f:
    """
    Radius-based colour zones, matching real nebula emission lines:
      outer shell  (r > 0.7) → orange-red   (SII  6716/6731 Å filaments)
      mid region   (r ~ 0.4) → teal/green   (OIII 5007 Å)
      inner core   (r < 0.2) → blue-white   (hot synchrotron / continuum)

    A low-frequency noise field adds local hue patches on top of the
    radius zones so adjacent filaments differ in shade.
    """
    # Normalised radial distance from nebula centre
    r_len  = wp.length(pos)
    r_norm = wp.clamp(r_len / wp.float32(1.3), wp.float32(0.0), wp.float32(1.0))

    # ── Zone weights ─────────────────────────────────────────────────────────
    # outer_w: peaks at r_norm = 1, fades inward
    outer_w = wp.clamp((r_norm - wp.float32(0.45)) / wp.float32(0.35), wp.float32(0.0), wp.float32(1.0))
    # inner_w: peaks at r_norm = 0, fades outward
    inner_w = wp.clamp(wp.float32(1.0) - r_norm / wp.float32(0.30), wp.float32(0.0), wp.float32(1.0))
    # mid fills the rest
    mid_w   = wp.clamp(wp.float32(1.0) - outer_w - inner_w, wp.float32(0.0), wp.float32(1.0))

    # ── Zone base colours ─────────────────────────────────────────────────────
    # SII outer: deep orange-red
    or_r = wp.float32(1.00); or_g = wp.float32(0.30); or_b = wp.float32(0.02)
    # OIII mid: teal-green
    tl_r = wp.float32(0.05); tl_g = wp.float32(0.85); tl_b = wp.float32(0.65)
    # Inner core: pale blue-white
    bw_r = wp.float32(0.55); bw_g = wp.float32(0.75); bw_b = wp.float32(1.00)

    base_r = or_r * outer_w + tl_r * mid_w + bw_r * inner_w
    base_g = or_g * outer_w + tl_g * mid_w + bw_g * inner_w
    base_b = or_b * outer_w + tl_b * mid_w + bw_b * inner_w

    # ── Local hue noise — adds variation within each zone ────────────────────
    hue = wp.noise(wp.rand_init(7), pos * wp.float32(0.7)) * wp.float32(0.25)
    rc  = wp.clamp(base_r + hue * wp.float32( 0.5), wp.float32(0.0), wp.float32(1.0))
    gc  = wp.clamp(base_g + hue * wp.float32( 0.2), wp.float32(0.0), wp.float32(1.0))
    bc  = wp.clamp(base_b + hue * wp.float32(-0.4), wp.float32(0.0), wp.float32(1.0))

    return wp.vec3f(rc, gc, bc)


# ── Main render kernel ────────────────────────────────────────────────────────

@wp.kernel
def render_nebula(
    pixels:  wp.array(dtype=wp.vec3f),
    W:       int,
    H:       int,
    t_near:  wp.float32,
    t_far:   wp.float32,
    anim_t:  wp.float32,
):
    """
    One GPU thread per pixel.

    Ray setup
    ---------
    Camera sits at (0, 0, -2.5) and looks toward +z through a virtual
    screen at z = -1.  The perspective focal length is 1.5 (moderate FOV).

    March loop
    ----------
    We step from t_near to t_far in MARCH_STEPS equal steps, accumulating
    colour via:
        acc   += transmittance * sigma_e * dt * colour
        T     *= exp(-sigma_a * dt)
    where sigma_a and sigma_e are both proportional to the local density.

    Early exit when transmittance drops below 0.005 (fully opaque).
    """
    pi, pj = wp.tid()

    aspect = wp.float32(W) / wp.float32(H)

    # Pixel centre in [-1, 1] (v is flipped so +v = up)
    u = (wp.float32(pj) + wp.float32(0.5)) / wp.float32(W) * wp.float32(2.0) - wp.float32(1.0)
    v = wp.float32(1.0) - (wp.float32(pi) + wp.float32(0.5)) / wp.float32(H) * wp.float32(2.0)

    ro = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(-1.5))
    rd = wp.normalize(wp.vec3f(u * aspect, v, wp.float32(1.5)))

    dt            = (t_far - t_near) / wp.float32(MARCH_STEPS)
    acc           = wp.vec3f(wp.float32(0.0), wp.float32(0.0), wp.float32(0.0))
    transmittance = wp.float32(1.0)

    for s in range(MARCH_STEPS):
        tc  = t_near + (wp.float32(s) + wp.float32(0.5)) * dt
        pos = ro + rd * tc

        d = density_at(pos, anim_t)

        if d > wp.float32(0.005):
            emit_col = nebula_color(d, pos)

            # High absorption: filaments block the ray quickly so you see their
            # surface rather than integrating through the whole depth.
            # Combined with the hard threshold in density_at, this gives
            # hard-edged filaments against true-black voids.
            sigma_a   = d * wp.float32(12.0)
            step_T    = wp.exp(-sigma_a * dt)
            emit      = d * wp.float32(30.0) * dt

            acc           += transmittance * emit * emit_col
            transmittance *= step_T

        if transmittance < wp.float32(0.005):
            break

    # ── Composite ─────────────────────────────────────────────────────────────
    final = acc

    # Reinhard tone-map (component-wise) → gamma-2 correction (≈ sRGB)
    # Warp has no vec3f/vec3f overload, so divide each channel separately.
    final = wp.vec3f(
        wp.sqrt(final[0] / (final[0] + wp.float32(1.0))),
        wp.sqrt(final[1] / (final[1] + wp.float32(1.0))),
        wp.sqrt(final[2] / (final[2] + wp.float32(1.0))),
    )

    pixels[pi * W + pj] = final


# ── Convert float pixels to uint8 RGB ────────────────────────────────────────

@wp.kernel
def pixels_to_rgb8(
    pixels: wp.array(dtype=wp.vec3f),
    rgb:    wp.array(dtype=wp.uint8),
    n:      int,
):
    idx = wp.tid()
    if idx < n:
        c           = pixels[idx]
        scale       = wp.float32(255.0)
        rgb[idx * 3    ] = wp.uint8(wp.clamp(c[0], wp.float32(0.0), wp.float32(1.0)) * scale)
        rgb[idx * 3 + 1] = wp.uint8(wp.clamp(c[1], wp.float32(0.0), wp.float32(1.0)) * scale)
        rgb[idx * 3 + 2] = wp.uint8(wp.clamp(c[2], wp.float32(0.0), wp.float32(1.0)) * scale)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--size",   type=int,   default=512,
                    help="Render resolution (square, default 512)")
    ap.add_argument("--steps",  type=int,   default=MARCH_STEPS,
                    help=f"Ray-march steps (default {MARCH_STEPS}); more = slower but denser")
    ap.add_argument("--device", default="cuda",
                    help="Warp device: cuda or cpu")
    ap.add_argument("--speed",  type=float, default=1.0,
                    help="Animation speed multiplier (default 1.0)")
    args = ap.parse_args()

    W = H  = args.size
    device  = args.device

    import pygame
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Nebula — Ray March")

    pixels = wp.zeros(W * H,     dtype=wp.vec3f, device=device)
    rgb    = wp.zeros(W * H * 3, dtype=wp.uint8, device=device)

    # Trigger kernel compilation on first launch
    print("Compiling kernels …", end=" ", flush=True)
    wp.launch(render_nebula,
              dim=(H, W),
              inputs=[pixels, W, H,
                      wp.float32(0.5), wp.float32(5.5),
                      wp.float32(0.0)],
              device=device)
    wp.synchronize()
    print("done.")
    print(__doc__)
    print(f"Grid {W}×{H}  |  device={device}  |  march steps={MARCH_STEPS}\n")

    t0       = time.time()
    fps_hist = []
    running  = True

    while running:
        frame_t0 = time.perf_counter()

        # ── Events ────────────────────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif ev.key == pygame.K_s:
                    from PIL import Image
                    fname = f"nebula_{int(time.time())}.png"
                    Image.fromarray(rgb.numpy().reshape(H, W, 3)).save(fname)
                    print(f"Screenshot → {fname}")

        # ── Render ────────────────────────────────────────────────────────────
        anim_t = (time.time() - t0) * args.speed

        wp.launch(render_nebula,
                  dim=(H, W),
                  inputs=[pixels, W, H,
                          wp.float32(0.5), wp.float32(5.5),
                          wp.float32(anim_t)],
                  device=device)

        wp.launch(pixels_to_rgb8,
                  dim=(W * H,),
                  inputs=[pixels, rgb, W * H],
                  device=device)

        # Zero-copy path: Warp → numpy view → pygame surface
        rgb_np = rgb.numpy().reshape(H, W, 3)
        surf   = pygame.image.frombuffer(rgb_np, (W, H), "RGB")
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        # ── FPS counter ───────────────────────────────────────────────────────
        fps_hist.append(1.0 / max(time.perf_counter() - frame_t0, 1e-9))
        if len(fps_hist) >= 20:
            fps = sum(fps_hist) / len(fps_hist)
            fps_hist.clear()
            pygame.display.set_caption(
                f"Nebula Ray March  |  {fps:.1f} fps  |  {device}  |  {MARCH_STEPS} steps"
            )

    pygame.quit()


if __name__ == "__main__":
    main()
