"""Try wrapping DecodedFrame GPU pointers with warp."""
import os, sys
import site

for sp in site.getsitepackages():
    cand = os.path.join(sp, "nvidia", "cuda_runtime", "bin")
    if os.path.isdir(cand):
        os.add_dll_directory(cand)
os.add_dll_directory(r"C:\Windows\System32")

import warp as wp
wp.init()

import PyNvVideoCodec as nvc
from PyNvVideoCodec.decoders.SimpleDecoder import SimpleDecoder

path = sys.argv[1] if len(sys.argv) > 1 else "Big_Buck_Bunny_720_10s_30MB.mp4"
decoder = SimpleDecoder(path, gpu_id=0)

for i, frame in enumerate(decoder):
    if i >= 1:
        break

    # NV12 layout: Y plane (H x W) + UV plane (H/2 x W interleaved)
    # shape is (H*1.5, W) total, so actual video H = shape[0] * 2 // 3
    total_h, W = frame.shape
    H = total_h * 2 // 3   # actual luma height
    print(f"Video resolution: {W}x{H}  (NV12 buffer: {W}x{total_h})")

    y_ptr  = frame.GetPtrToPlane(0)
    uv_ptr = frame.GetPtrToPlane(1)
    print(f"Y  plane ptr: {y_ptr}")
    print(f"UV plane ptr: {uv_ptr}")

    try:
        # Wrap as warp arrays (zero-copy, GPU memory)
        y_wp  = wp.array(ptr=y_ptr,  dtype=wp.uint8, shape=(H * W,),      device="cuda:0")
        uv_wp = wp.array(ptr=uv_ptr, dtype=wp.uint8, shape=(H // 2 * W,), device="cuda:0")
        print(f"\nwp.array Y  — shape={y_wp.shape}  dtype={y_wp.dtype}  device={y_wp.device}")
        print(f"wp.array UV — shape={uv_wp.shape}  dtype={uv_wp.dtype}  device={uv_wp.device}")

        # Verify: read a small slice back to CPU to confirm it's real data (not zeros)
        y_cpu = y_wp.numpy()[:16]
        print(f"\nFirst 16 Y luma bytes (should be non-zero): {y_cpu}")
        print("warp interop: OK — GPU NV12 accessible as wp.array without copy")
    except Exception as e:
        import traceback
        print(f"warp interop failed: {e}")
        traceback.print_exc()
