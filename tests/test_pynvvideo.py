"""
Test PyNvVideoCodec import, decode, and warp GPU interop.

Root cause of original DLL failure (diagnosed 2026-06-18):
  Both PyNvVideoCodec_121.pyd and PyNvVideoCodec_130.pyd link against
  cudart64_12.dll (CUDA 12 runtime).  CUDA 13 only provides cudart64_13.dll.
  nvcuda.dll and nvcuvid.dll are present in System32 (installed by the driver).

Fix:
  pip install nvidia-cuda-runtime-cu12
  This provides cudart64_12.dll inside the venv — no system changes needed.

Confirmed working (2026-06-18, RTX 4070 Laptop, CUDA 13.3 driver):
  - PyNvVideoCodec 2.1.0 imports cleanly after fix
  - Decodes H.264 MP4 (Big_Buck_Bunny_720_10s_30MB.mp4, 1280x720)
  - Delivers NV12 GPU frames: GetPtrToPlane(0/1) → int CUDA device pointers
  - wp.array(ptr=...) wraps GPU NV12 planes zero-copy — warp interop confirmed

Run:
  python tests/test_pynvvideo.py [path/to/video.mp4]
  python tests/test_pynvvideo.py Big_Buck_Bunny_720_10s_30MB.mp4
"""
import ctypes
import glob
import os
import sys
import site

# ---------------------------------------------------------------------------
# Step 0: set up DLL search paths before any CUDA imports
# ---------------------------------------------------------------------------

def _add_dll_dirs():
    dirs_added = []

    # nvidia-cuda-runtime-cu12 wheel (pip install nvidia-cuda-runtime-cu12)
    for sp in site.getsitepackages():
        cand = os.path.join(sp, "nvidia", "cuda_runtime", "bin")
        if os.path.isdir(cand):
            os.add_dll_directory(cand)
            dirs_added.append(cand)

    # System32 (nvcuda.dll, nvcuvid.dll)
    os.add_dll_directory(r"C:\Windows\System32")
    dirs_added.append(r"C:\Windows\System32")

    # PyNvVideoCodec package dir (bundled avcodec/avutil/avformat DLLs)
    try:
        import importlib.util
        spec = importlib.util.find_spec("PyNvVideoCodec")
        if spec:
            pkg = os.path.dirname(spec.origin)
            os.add_dll_directory(pkg)
            dirs_added.append(pkg)
    except Exception:
        pass

    return dirs_added

_add_dll_dirs()

# ---------------------------------------------------------------------------
# Step 1: DLL diagnostic
# ---------------------------------------------------------------------------
print("=== DLL diagnostic ===\n")

DLLS_NEEDED = {
    "cudart64_12.dll": "CUDA 12 runtime — fix: pip install nvidia-cuda-runtime-cu12",
    "nvcuda.dll":      "CUDA driver API — installed by NVIDIA driver",
    "nvcuvid.dll":     "NVDEC video decode — installed by NVIDIA driver",
}

SEARCH_PATHS = [r"C:\Windows\System32", r"C:\Windows\SysWOW64"]
for cuda_glob in [r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin",
                  r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*\bin\x64"]:
    SEARCH_PATHS.extend(glob.glob(cuda_glob))
for sp in site.getsitepackages():
    SEARCH_PATHS.append(os.path.join(sp, "nvidia", "cuda_runtime", "bin"))
SEARCH_PATHS += [p for p in os.environ.get("PATH", "").split(";") if p]

def find_dll(name):
    for base in SEARCH_PATHS:
        path = os.path.join(base, name)
        if os.path.isfile(path):
            return path
    return None

missing = []
for dll, note in DLLS_NEEDED.items():
    path = find_dll(dll)
    if path:
        print(f"  [OK  ] {dll:<25} {path}")
    else:
        print(f"  [MISS] {dll:<25} {note}")
        missing.append(dll)
print()

# ---------------------------------------------------------------------------
# Step 2: import PyNvVideoCodec
# ---------------------------------------------------------------------------
print("=== Import PyNvVideoCodec ===\n")

try:
    import PyNvVideoCodec as nvc
    print(f"  OK — version: {nvc.__version__}")
    HAVE_NVC = True
except (ImportError, OSError) as e:
    print(f"  FAIL ({type(e).__name__}): {e}")
    HAVE_NVC = False
except Exception as e:
    print(f"  FAIL ({type(e).__name__}): {e}")
    HAVE_NVC = False

if not HAVE_NVC:
    print()
    if "cudart64_12.dll" in missing:
        print("Fix: pip install nvidia-cuda-runtime-cu12")
        print("(provides cudart64_12.dll inside the venv, no system changes needed)")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Step 3: driver version
# ---------------------------------------------------------------------------
print()
print("=== PyNvVideoCodec info ===\n")
try:
    from PyNvVideoCodec.VersionCheck import DriverWrapper
    ver = DriverWrapper().GetDriverVersion()
    print(f"  Driver NVENC version: {ver}")
except Exception as e:
    print(f"  DriverWrapper: {e}")

# ---------------------------------------------------------------------------
# Step 4: decode test
# ---------------------------------------------------------------------------
video_path = sys.argv[1] if len(sys.argv) > 1 else None

if video_path is None:
    print()
    print("No video file given — skipping decode test.")
    print("Usage: python tests/test_pynvvideo.py path/to/video.mp4")
    sys.exit(0)

print()
print(f"=== Decode test: {video_path} ===\n")

from PyNvVideoCodec.decoders.SimpleDecoder import SimpleDecoder

try:
    decoder = SimpleDecoder(video_path, gpu_id=0)
except Exception as e:
    print(f"  SimpleDecoder init failed: {e}")
    sys.exit(1)

MAX_FRAMES = 5
frame_count = 0
frame_info = None

for frame in decoder:
    if frame_count >= MAX_FRAMES:
        break

    if frame_count == 0:
        # NV12 layout: shape = (H*1.5, W), dtype uint8
        total_h, W = frame.shape
        H = total_h * 2 // 3
        frame_info = (W, H)
        print(f"  Resolution:  {W}x{H}  (NV12 buffer: {W}x{total_h})")
        print(f"  dtype:       {frame.dtype}")
        print(f"  format:      {frame.format}")
        y_ptr  = frame.GetPtrToPlane(0)
        uv_ptr = frame.GetPtrToPlane(1)
        print(f"  Y  plane ptr: {y_ptr}")
        print(f"  UV plane ptr: {uv_ptr}")

    print(f"  Frame {frame_count}: {type(frame).__name__}")
    frame_count += 1

print(f"\n  Decoded {frame_count} frame(s) OK.")

# ---------------------------------------------------------------------------
# Step 5: warp GPU interop — wrap NV12 plane pointers as wp.array (zero-copy)
# ---------------------------------------------------------------------------
print()
print("=== Warp GPU interop ===\n")

try:
    import warp as wp
    wp.init()
except Exception as e:
    print(f"  warp init failed: {e}")
    sys.exit(0)

# Re-open decoder to get a fresh frame (previous iterator may be exhausted)
decoder2 = SimpleDecoder(video_path, gpu_id=0)
for frame in decoder2:
    total_h, W = frame.shape
    H = total_h * 2 // 3
    y_ptr  = frame.GetPtrToPlane(0)
    uv_ptr = frame.GetPtrToPlane(1)

    try:
        # Y plane: (H × W) uint8 luma
        y_wp  = wp.array(ptr=y_ptr,  dtype=wp.uint8, shape=(H * W,),      device="cuda:0")
        # UV plane: (H/2 × W) uint8 interleaved chroma pairs
        uv_wp = wp.array(ptr=uv_ptr, dtype=wp.uint8, shape=(H // 2 * W,), device="cuda:0")

        print(f"  Y  wp.array: shape={y_wp.shape}  dtype={y_wp.dtype}  device={y_wp.device}")
        print(f"  UV wp.array: shape={uv_wp.shape}  dtype={uv_wp.dtype}  device={uv_wp.device}")

        # Read back a slice to confirm real pixel data (non-zero luma)
        y_sample = y_wp.numpy()[:16]
        print(f"  Y luma sample (should be non-zero): {y_sample}")
        print(f"\n  warp interop: OK -- GPU NV12 planes accessible as wp.array, zero-copy")
        print(f"  Next step: NV12->RGBA warp kernel for fully GPU decode path")
    except Exception as e:
        import traceback
        print(f"  warp interop failed: {e}")
        traceback.print_exc()

    break   # only need one frame
