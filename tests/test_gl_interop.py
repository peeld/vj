"""
test_gl_interop.py -- Zero-copy GL<->Warp transfer path explorer

Questions this answers:
  A. PIXEL_PACK_BUFFER: FBO -> Warp without CPU?
  B. PIXEL_UNPACK_BUFFER: Warp -> GL texture without CPU?
  D. Full zero-copy round-trip: FBO->PBO->kernel->PBO->texture?
  E. Double-buffered PBO: does pipelining glReadPixels reduce latency?
  C. GLTextureResource: direct CUDA texture mapping? (run last; may corrupt GL state on fail)

Current pipeline (two PCIe stalls per frame):
  GL render -> fbo.read()[GPU->CPU] -> wp.copy[CPU->GPU] -> kernel
            -> .numpy()[GPU->CPU] -> texture.write()[CPU->GPU] -> blit

Target pipeline (GPU-only):
  GL render -> PBO(glReadPixels) -> wp.map -> kernel -> wp.map
            -> PBO(glTexSubImage2D) -> blit

Run:
  python -X utf8 test_gl_interop.py [width height]
"""

import sys
import time
import ctypes
import numpy as np

print("Importing warp...")
import warp as wp
wp.init()

print("Importing moderngl...")
import moderngl

print("Importing PyOpenGL...")
from OpenGL.GL import (
    glBindBuffer, glReadPixels, glTexSubImage2D,
    glFinish, glBindTexture, glGetError, glBindFramebuffer,
    GL_PIXEL_PACK_BUFFER, GL_PIXEL_UNPACK_BUFFER,
    GL_RGBA, GL_UNSIGNED_BYTE, GL_TEXTURE_2D,
    GL_NO_ERROR, GL_FRAMEBUFFER,
)

NULL_OFFSET = ctypes.c_void_p(0)   # PBO offset 0 as explicit null pointer

W = int(sys.argv[1]) if len(sys.argv) > 2 else 1920
H = int(sys.argv[2]) if len(sys.argv) > 2 else 1080
TIMING_FRAMES = 200
DEVICE = "cuda:0"
SEP = "-" * 64

def header(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def ok(label, detail=""):
    print(f"  [PASS] {label}" + (f"  ({detail})" if detail else ""))

def fail(label, detail=""):
    print(f"  [FAIL] {label}" + (f"  ({detail[:120]})" if detail else ""))

def clear_gl_errors():
    while glGetError() != GL_NO_ERROR:
        pass

# ── GL context + shared resources ─────────────────────────────────────────────

header("Setup")
print(f"  Resolution: {W}x{H}   CUDA: {DEVICE}")
ctx = moderngl.create_standalone_context()
print(f"  OpenGL: {ctx.info['GL_VERSION']}   Vendor: {ctx.info['GL_VENDOR']}")

scene_tex = ctx.texture((W, H), 4)
scene_fbo = ctx.framebuffer(color_attachments=[scene_tex])
scene_fbo.use()
ctx.clear(1.0, 0.0, 0.5, 1.0)
glFinish()

display_tex = ctx.texture((W, H), 4)
display_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

EXPECTED = np.frombuffer(scene_fbo.color_attachments[0].read(), dtype=np.uint8)[:4].copy()
print(f"  Scene pixel[0]: {EXPECTED.tolist()}")

def pixels_match(arr_u8, label):
    p = np.asarray(arr_u8, dtype=np.uint8)[:4]
    ok_val = np.allclose(p.astype(int), EXPECTED.astype(int), atol=1)
    (ok if ok_val else fail)(f"Pixel correct ({label})",
                              "" if ok_val else f"got {p.tolist()} expected {EXPECTED.tolist()}")
    return ok_val

# ─────────────────────────────────────────────────────────────────────────────
# BASELINE
# ─────────────────────────────────────────────────────────────────────────────

header("BASELINE: fbo.read() -> numpy -> texture.write()")
glFinish()
t0 = time.perf_counter()
for _ in range(TIMING_FRAMES):
    raw = scene_fbo.color_attachments[0].read()
    gpu_in = wp.array(np.frombuffer(raw, dtype=np.uint8), dtype=wp.uint8, device=DEVICE)
    out_np = gpu_in.numpy()
    display_tex.write(out_np.tobytes())
    glFinish()
baseline_ms = (time.perf_counter() - t0) / TIMING_FRAMES * 1000
print(f"  Baseline: {baseline_ms:.3f} ms/frame  ({TIMING_FRAMES} frames, {W}x{H})")
clear_gl_errors()

# ─────────────────────────────────────────────────────────────────────────────
# TEST A: PIXEL_PACK_BUFFER -- FBO -> PBO -> Warp (no CPU readback)
# ─────────────────────────────────────────────────────────────────────────────

header("TEST A: FBO -> PIXEL_PACK_BUFFER -> wp.RegisteredGLBuffer")
print("  glReadPixels into a bound PBO writes directly into GPU VRAM")
print("  Key: pass ctypes.c_void_p(0) not None (None makes PyOpenGL alloc a CPU buffer)")
print()

pack_pbo_ms = None
try:
    pack_pbo = ctx.buffer(reserve=W * H * 4)
    reg_pack = wp.RegisteredGLBuffer(pack_pbo.glo, device=DEVICE,
                                     flags=wp.RegisteredGLBuffer.READ_ONLY)
    ok("wp.RegisteredGLBuffer created")

    scene_fbo.use()
    glBindBuffer(GL_PIXEL_PACK_BUFFER, pack_pbo.glo)
    glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, NULL_OFFSET)
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
    glFinish()
    if glGetError() != GL_NO_ERROR:
        raise RuntimeError("GL error after glReadPixels -> PBO")
    ok("glReadPixels -> PBO (GPU-side, no CPU)")

    warp_arr = reg_pack.map(dtype=wp.uint8, shape=(W * H * 4,))
    pixels_match(warp_arr.numpy()[:4], "map().numpy()")
    reg_pack.unmap()
    ok("map / unmap lifecycle")

    glFinish()
    t0 = time.perf_counter()
    for _ in range(TIMING_FRAMES):
        scene_fbo.use()
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pack_pbo.glo)
        glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, NULL_OFFSET)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        warp_arr = reg_pack.map(dtype=wp.uint8, shape=(W * H * 4,))
        reg_pack.unmap()
        glFinish()
    pack_pbo_ms = (time.perf_counter() - t0) / TIMING_FRAMES * 1000
    print(f"\n  PBO pack:  {pack_pbo_ms:.3f} ms/frame  (baseline: {baseline_ms:.3f})")

except Exception as e:
    fail("TEST A", str(e))
    clear_gl_errors()

# ─────────────────────────────────────────────────────────────────────────────
# TEST B: PIXEL_UNPACK_BUFFER -- Warp -> PBO -> GL texture (no CPU upload)
# ─────────────────────────────────────────────────────────────────────────────

header("TEST B: Warp -> PIXEL_UNPACK_BUFFER -> GL texture")
print("  glTexSubImage2D with a bound PBO uploads from GPU VRAM, no PCIe write")
print()

unpack_pbo_ms = None
try:
    unpack_pbo = ctx.buffer(reserve=W * H * 4)
    reg_unpack = wp.RegisteredGLBuffer(unpack_pbo.glo, device=DEVICE,
                                       flags=wp.RegisteredGLBuffer.WRITE_DISCARD)
    ok("wp.RegisteredGLBuffer created")

    src_cpu = np.zeros(W * H * 4, dtype=np.uint8)
    src_cpu[1::4] = 255; src_cpu[2::4] = 64; src_cpu[3::4] = 255
    EXPECTED_B = np.array([0, 255, 64, 255], dtype=np.uint8)
    src_gpu = wp.array(src_cpu, dtype=wp.uint8, device=DEVICE)

    warp_dst = reg_unpack.map(dtype=wp.uint8, shape=(W * H * 4,))
    wp.copy(warp_dst, src_gpu)
    wp.synchronize_device(DEVICE)
    reg_unpack.unmap()
    ok("wp.copy GPU array -> registered PBO")

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, unpack_pbo.glo)
    glBindTexture(GL_TEXTURE_2D, display_tex.glo)
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, NULL_OFFSET)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
    glFinish()
    if glGetError() != GL_NO_ERROR:
        raise RuntimeError("GL error after glTexSubImage2D from PBO")
    ok("glTexSubImage2D from PBO (GPU-side upload)")

    pix_b = np.frombuffer(display_tex.read(), dtype=np.uint8)[:4]
    ok_b = np.allclose(pix_b.astype(int), EXPECTED_B.astype(int), atol=1)
    (ok if ok_b else fail)(f"Pixel correct after PBO upload  {pix_b.tolist()}")

    glFinish()
    t0 = time.perf_counter()
    for _ in range(TIMING_FRAMES):
        warp_dst = reg_unpack.map(dtype=wp.uint8, shape=(W * H * 4,))
        wp.copy(warp_dst, src_gpu)
        wp.synchronize_device(DEVICE)
        reg_unpack.unmap()
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, unpack_pbo.glo)
        glBindTexture(GL_TEXTURE_2D, display_tex.glo)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, NULL_OFFSET)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        glFinish()
    unpack_pbo_ms = (time.perf_counter() - t0) / TIMING_FRAMES * 1000
    print(f"\n  PBO unpack: {unpack_pbo_ms:.3f} ms/frame")

except Exception as e:
    fail("TEST B", str(e))
    clear_gl_errors()

# ─────────────────────────────────────────────────────────────────────────────
# TEST D: Full zero-copy round-trip (pack PBO -> kernel -> unpack PBO)
# ─────────────────────────────────────────────────────────────────────────────

header("TEST D: Full round-trip -- FBO -> PBO -> kernel -> PBO -> texture")
print("  Complete post-processing loop: zero CPU copies, all GPU DMA")
print()

@wp.kernel
def copy_kernel(src: wp.array(dtype=wp.uint8),
                dst: wp.array(dtype=wp.uint8)):
    i = wp.tid()
    dst[i] = src[i]

clear_gl_errors()
fullzero_ms = None
try:
    n = W * H * 4
    fbo_pbo = ctx.buffer(reserve=n)
    out_pbo = ctx.buffer(reserve=n)
    reg_in  = wp.RegisteredGLBuffer(fbo_pbo.glo, device=DEVICE,
                                    flags=wp.RegisteredGLBuffer.READ_ONLY)
    reg_out = wp.RegisteredGLBuffer(out_pbo.glo, device=DEVICE,
                                    flags=wp.RegisteredGLBuffer.WRITE_DISCARD)

    def zero_copy_frame():
        # Step 1: FBO -> pack PBO via glReadPixels (GPU DMA, no CPU)
        # Use raw glBindFramebuffer to avoid moderngl state side-effects
        glBindFramebuffer(GL_FRAMEBUFFER, scene_fbo.glo)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, fbo_pbo.glo)
        glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, NULL_OFFSET)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        # Step 2: map PBOs to Warp, run kernel, unmap
        src_arr = reg_in.map(dtype=wp.uint8, shape=(n,))
        dst_arr = reg_out.map(dtype=wp.uint8, shape=(n,))
        wp.launch(copy_kernel, dim=n, inputs=[src_arr, dst_arr], device=DEVICE)
        wp.synchronize_device(DEVICE)
        reg_in.unmap()
        reg_out.unmap()
        # Step 3: unpack PBO -> display texture (GPU DMA, no CPU)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, out_pbo.glo)
        glBindTexture(GL_TEXTURE_2D, display_tex.glo)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, NULL_OFFSET)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    # Correctness check
    zero_copy_frame()
    glFinish()
    if glGetError() != GL_NO_ERROR:
        fail("Full round-trip GL ops")
    else:
        pixels_match(np.frombuffer(display_tex.read(), dtype=np.uint8), "full zero-copy")
        ok("Full zero-copy round-trip correct")

    # Timing
    glFinish()
    t0 = time.perf_counter()
    for _ in range(TIMING_FRAMES):
        zero_copy_frame()
        glFinish()
    fullzero_ms = (time.perf_counter() - t0) / TIMING_FRAMES * 1000

    print(f"\n  Full zero-copy: {fullzero_ms:.3f} ms/frame")
    print(f"  Baseline:       {baseline_ms:.3f} ms/frame")
    if fullzero_ms:
        saved = baseline_ms - fullzero_ms
        print(f"  Savings:        {saved:+.3f} ms/frame ({saved/baseline_ms*100:.0f}%)")

except Exception as e:
    fail("TEST D", str(e))
    clear_gl_errors()

# ─────────────────────────────────────────────────────────────────────────────
# TEST E: Double-buffered pack PBO (pipeline glReadPixels async)
# ─────────────────────────────────────────────────────────────────────────────

header("TEST E: Double-buffered PIXEL_PACK PBO")
print("  Frame N:   issue glReadPixels -> PBO[write]  (queued to GPU)")
print("  Frame N:   process PBO[read] from last frame (guaranteed ready)")
print("  This hides glReadPixels GPU latency behind the kernel + unpack.")
print()

clear_gl_errors()
doublebuf_ms = None
try:
    n = W * H * 4
    pbo_a = ctx.buffer(reserve=n)
    pbo_b = ctx.buffer(reserve=n)
    reg_a = wp.RegisteredGLBuffer(pbo_a.glo, device=DEVICE, flags=wp.RegisteredGLBuffer.READ_ONLY)
    reg_b = wp.RegisteredGLBuffer(pbo_b.glo, device=DEVICE, flags=wp.RegisteredGLBuffer.READ_ONLY)
    pbos = [pbo_a, pbo_b]
    regs = [reg_a, reg_b]

    # Warm-up: prime both PBOs
    for i in range(2):
        glBindFramebuffer(GL_FRAMEBUFFER, scene_fbo.glo)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos[i].glo)
        glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, NULL_OFFSET)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
    glFinish()

    idx = [0, 1]  # [write, read]

    def double_buf_frame():
        w, r = idx[0], idx[1]
        # Issue readback into write PBO (async: queued but not waited)
        glBindFramebuffer(GL_FRAMEBUFFER, scene_fbo.glo)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbos[w].glo)
        glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, NULL_OFFSET)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        # Process read PBO — guaranteed done from prev frame's glFinish
        arr = regs[r].map(dtype=wp.uint8, shape=(n,))
        regs[r].unmap()
        idx[0], idx[1] = r, w

    glFinish()
    t0 = time.perf_counter()
    for _ in range(TIMING_FRAMES):
        double_buf_frame()
        glFinish()
    doublebuf_ms = (time.perf_counter() - t0) / TIMING_FRAMES * 1000
    print(f"  Double-buffered: {doublebuf_ms:.3f} ms/frame")
    if pack_pbo_ms:
        print(f"  Single-buffered: {pack_pbo_ms:.3f} ms/frame")
    ok("Double-buffered PBO timing complete")

except Exception as e:
    fail("TEST E", str(e))
    clear_gl_errors()

# ─────────────────────────────────────────────────────────────────────────────
# TEST C: wp.GLTextureResource (run last; failed CUDA registration may corrupt
#          PyOpenGL state for subsequent glBindFramebuffer calls)
# ─────────────────────────────────────────────────────────────────────────────

header("TEST C: wp.GLTextureResource -- direct GL texture CUDA mapping")
print("  If this works, it eliminates PBOs entirely (no glReadPixels needed).")
print("  Note: requires CUDA GL interop; may not work with standalone contexts.")
print()

gltex_read_ms = gltex_write_ms = None
clear_gl_errors()
try:
    from pyglet.gl import GL_TEXTURE_2D as PYGLET_TEX_2D
    ok("pyglet import ok")

    glFinish()
    try:
        tex_resource = wp.GLTextureResource(scene_tex.glo, PYGLET_TEX_2D, device=DEVICE)
        ok("wp.GLTextureResource created for FBO color texture")

        try:
            mapped_tex = tex_resource.map()
            ok(f"map() -> {type(mapped_tex).__name__}")

            out_arr = wp.zeros(W * H * 4, dtype=wp.uint8, device=DEVICE)
            try:
                mapped_tex.copy_to(out_arr.reshape((H, W, 4)))
                tex_resource.unmap()
                pixels_match(out_arr.numpy()[:4], "GLTextureResource.copy_to()")

                glFinish()
                t0 = time.perf_counter()
                for _ in range(TIMING_FRAMES):
                    tex_resource.map()
                    mapped_tex.copy_to(out_arr.reshape((H, W, 4)))
                    tex_resource.unmap()
                    glFinish()
                gltex_read_ms = (time.perf_counter() - t0) / TIMING_FRAMES * 1000
                print(f"\n  GLTextureResource read: {gltex_read_ms:.3f} ms/frame")

            except Exception as e:
                fail("copy_to()", str(e))
                try: tex_resource.unmap()
                except Exception: pass

            try:
                out_tex_res = wp.GLTextureResource(display_tex.glo, PYGLET_TEX_2D, device=DEVICE)
                src_wp = wp.full((H, W, 4), 200, dtype=wp.uint8, device=DEVICE)
                EXPECTED_C = np.array([200, 200, 200, 200], dtype=np.uint8)

                mapped_out = out_tex_res.map()
                mapped_out.copy_from(src_wp)
                out_tex_res.unmap()
                ok("GLTextureResource write (copy_from)")

                pix_c = np.frombuffer(display_tex.read(), dtype=np.uint8)[:4]
                ok_c = np.allclose(pix_c.astype(int), EXPECTED_C.astype(int), atol=1)
                (ok if ok_c else fail)(f"Write pixel  {pix_c.tolist()}")

                glFinish()
                t0 = time.perf_counter()
                for _ in range(TIMING_FRAMES):
                    mapped_out = out_tex_res.map()
                    mapped_out.copy_from(src_wp)
                    out_tex_res.unmap()
                    glFinish()
                gltex_write_ms = (time.perf_counter() - t0) / TIMING_FRAMES * 1000
                print(f"  GLTextureResource write: {gltex_write_ms:.3f} ms/frame")

            except Exception as e:
                fail("GLTextureResource write path", str(e))

        except Exception as e:
            fail("map()", str(e))
            try: tex_resource.unmap()
            except Exception: pass

    except Exception as e:
        fail("wp.GLTextureResource creation", str(e))
        print()
        print("  Likely cause: moderngl standalone context lacks CUDA GL interop.")
        print("  The full app context (WGL with NVIDIA driver) may support it.")
        print("  Workaround: use PBO paths (Tests A+B+D) which work on this context.")

except ImportError as e:
    fail("pyglet import", str(e))

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

header("SUMMARY")
print(f"  Resolution: {W}x{H}   Timing: {TIMING_FRAMES} frames each")
print()

rows = [
    ("Baseline: fbo.read() + texture.write()",         baseline_ms),
    ("Test A:   PIXEL_PACK_BUFFER  (FBO->Warp)",        pack_pbo_ms),
    ("Test B:   PIXEL_UNPACK_BUFFER (Warp->texture)",   unpack_pbo_ms),
    ("Test D:   Full zero-copy round-trip",             fullzero_ms),
    ("Test E:   Double-buffered PIXEL_PACK",            doublebuf_ms),
    ("Test C:   GLTextureResource read",                gltex_read_ms),
    ("Test C:   GLTextureResource write",               gltex_write_ms),
]
for label, ms in rows:
    s = f"  {label:<50}"
    print(s + (f"{ms:6.3f} ms/frame" if ms is not None else "   N/A (failed)"))

print()
if baseline_ms and fullzero_ms:
    saved = baseline_ms - fullzero_ms
    pct = saved / baseline_ms * 100
    print(f"  Zero-copy savings vs baseline: {saved:+.3f} ms/frame ({pct:.0f}%)")
    print()
    if pct > 30:
        print("  RECOMMENDATION: PBO zero-copy path is a major win.")
        print("  Integrate it into the post-processing pipeline.")
    elif pct > 10:
        print("  RECOMMENDATION: Moderate win; worth integrating at 60fps.")
    else:
        print("  NOTE: Minimal savings; check if bottleneck is elsewhere.")

if pack_pbo_ms and unpack_pbo_ms:
    print()
    print(f"  Individual costs (pack + unpack with kernel): ~{pack_pbo_ms + unpack_pbo_ms:.2f} ms/frame")
    print(f"  (vs {baseline_ms:.2f} ms for the CPU round-trip)")
print()
