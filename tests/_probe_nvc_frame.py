"""Probe PyNvVideoCodec DecodedFrame attributes to understand the output format."""
import os, sys
import site

# Add cudart64_12.dll path
for sp in site.getsitepackages():
    cand = os.path.join(sp, "nvidia", "cuda_runtime", "bin")
    if os.path.isdir(cand):
        os.add_dll_directory(cand)

os.add_dll_directory(r"C:\Windows\System32")

import PyNvVideoCodec as nvc
from PyNvVideoCodec.decoders.SimpleDecoder import SimpleDecoder

path = sys.argv[1] if len(sys.argv) > 1 else "Big_Buck_Bunny_720_10s_30MB.mp4"
decoder = SimpleDecoder(path, gpu_id=0)

for i, frame in enumerate(decoder):
    if i >= 1:
        break
    print(f"type:          {type(frame)}")
    print(f"dir:           {[a for a in dir(frame) if not a.startswith('__')]}")
    for attr in ("format", "width", "height", "shape", "dtype", "device",
                 "cuda_ptr", "data_ptr", "planes", "pitch"):
        val = getattr(frame, attr, "<no attr>")
        if callable(val):
            try:
                val = val()
            except Exception as e:
                val = f"<call failed: {e}>"
        print(f"{attr:<14} {val}")

    # Try converting to numpy
    import numpy as np
    for method in ("numpy", "to_numpy", "as_numpy", "to_ndarray"):
        fn = getattr(frame, method, None)
        if fn:
            try:
                arr = fn()
                print(f"\n{method}():  shape={arr.shape}  dtype={arr.dtype}")
            except Exception as e:
                print(f"\n{method}(): failed — {e}")
            break
