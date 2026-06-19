"""
Quick prerequisite check for video support.
Run with: python test_video_prereqs.py
"""
import sys

results = []

# --- Warp + CUDA ---
try:
    import warp as wp
    wp.init()
    results.append(("warp", True, f"v{wp.__version__}"))
except Exception as e:
    results.append(("warp", False, str(e)))

# --- wp.RegisteredGLBuffer ---
try:
    has_gl_buf = hasattr(wp, "RegisteredGLBuffer")
    results.append(("wp.RegisteredGLBuffer", has_gl_buf,
                    "available" if has_gl_buf else "MISSING"))
except Exception as e:
    results.append(("wp.RegisteredGLBuffer", False, str(e)))

# --- PyAV ---
try:
    import av
    ver = av.__version__
    results.append(("av (PyAV)", True, f"v{ver}"))

    # Check NVDEC codec availability
    codecs = [str(c) for c in av.codecs_available]
    for codec in ("h264_cuvid", "hevc_cuvid", "av1_cuvid"):
        has = codec in codecs
        results.append((f"  codec: {codec}", has,
                        "NVDEC available" if has else "not available (software decode will be used)"))
except ImportError as e:
    results.append(("av (PyAV)", False, f"not installed — pip install av"))
except Exception as e:
    results.append(("av (PyAV)", False, str(e)))

# --- PyOpenGL (needed for PBO glTexSubImage2D call) ---
try:
    import OpenGL.GL as gl
    results.append(("PyOpenGL", True, "ok"))
except ImportError:
    results.append(("PyOpenGL", False, "not installed — pip install PyOpenGL"))

# --- numpy ---
try:
    import numpy as np
    results.append(("numpy", True, f"v{np.__version__}"))
except ImportError as e:
    results.append(("numpy", False, str(e)))

# --- Print summary ---
print("\n=== Video Prerequisite Check ===\n")
ok_critical = True
for name, ok, note in results:
    status = "OK " if ok else "FAIL"
    # codec lines are informational, not failures
    if not ok and not name.startswith("  codec"):
        ok_critical = False
    print(f"  [{status}] {name:<35} {note}")

print()
if ok_critical:
    print("All critical checks passed — ready to implement video support.")
else:
    print("Some checks failed — install missing packages above.")
print()

# --- Warp CUDA device info ---
try:
    print("=== Warp CUDA Devices ===\n")
    for d in wp.get_devices():
        if d.is_cuda:
            print(f"  {d.name}: {d.arch}, {d.total_memory // (1024**3)} GiB")
    print()
except Exception:
    pass
