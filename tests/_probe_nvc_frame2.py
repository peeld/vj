"""Deep probe of DecodedFrame: GPU pointer, nv12_to_rgb, warp interop."""
import os, sys
import site

for sp in site.getsitepackages():
    cand = os.path.join(sp, "nvidia", "cuda_runtime", "bin")
    if os.path.isdir(cand):
        os.add_dll_directory(cand)
os.add_dll_directory(r"C:\Windows\System32")

import PyNvVideoCodec as nvc
from PyNvVideoCodec.decoders.SimpleDecoder import SimpleDecoder
import warp as wp
wp.init()

path = sys.argv[1] if len(sys.argv) > 1 else "Big_Buck_Bunny_720_10s_30MB.mp4"
decoder = SimpleDecoder(path, gpu_id=0)

for i, frame in enumerate(decoder):
    if i >= 1:
        break

    print(f"=== DecodedFrame (frame 0) ===")
    print(f"  format:   {frame.format}")
    print(f"  shape:    {frame.shape}   (NV12: Y plane + UV plane stacked, total H = video_H * 1.5)")
    print(f"  dtype:    {frame.dtype}")

    # cuda attribute
    c = frame.cuda
    print(f"\n  frame.cuda type: {type(c)}")
    print(f"  frame.cuda dir:  {[a for a in dir(c) if not a.startswith('__')]}")
    for attr in ("ptr", "__cuda_array_interface__", "__array_interface__"):
        val = getattr(c, attr, "<no attr>")
        print(f"  cuda.{attr}: {val}")

    # Check for __cuda_array_interface__ (needed for warp/cupy interop)
    if hasattr(c, "__cuda_array_interface__"):
        cai = c.__cuda_array_interface__
        print(f"\n  __cuda_array_interface__: {cai}")

    # Try GetPtrToPlane (CUDA device pointer to Y and UV planes)
    print()
    for plane_idx in range(2):
        try:
            ptr = frame.GetPtrToPlane(plane_idx)
            print(f"  GetPtrToPlane({plane_idx}): {ptr}  (type={type(ptr).__name__})")
        except Exception as e:
            print(f"  GetPtrToPlane({plane_idx}): failed — {e}")

    # Try nv12_to_rgb
    print()
    try:
        rgb = frame.nv12_to_rgb()
        print(f"  nv12_to_rgb(): type={type(rgb).__name__}  shape={getattr(rgb, 'shape', '?')}  dtype={getattr(rgb, 'dtype', '?')}")
        if hasattr(rgb, "__cuda_array_interface__"):
            print(f"  nv12_to_rgb CAI: {rgb.__cuda_array_interface__}")
    except Exception as e:
        print(f"  nv12_to_rgb(): failed — {e}")

    # Try wrapping the frame directly with warp
    print()
    try:
        # Attempt wp.from_dlpack or wp.array from cuda array interface
        if hasattr(frame.cuda, "__cuda_array_interface__"):
            cai = frame.cuda.__cuda_array_interface__
            wp_arr = wp.array(ptr=cai["data"][0], dtype=wp.uint8,
                             shape=cai["shape"], device="cuda")
            print(f"  wp.array from CAI: shape={wp_arr.shape}  dtype={wp_arr.dtype}")
    except Exception as e:
        print(f"  wp.array from CAI: failed — {e}")
