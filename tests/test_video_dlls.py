"""
Diagnose the PyNvVideoCodec DLL load failure.
"""
import ctypes
import os
import sys
import glob

print("=== DLL Dependency Diagnostic ===\n")

# Key DLLs PyNvVideoCodec needs
DLLS_TO_CHECK = [
    "nvcuvid.dll",          # NVDEC — lives in driver, not CUDA toolkit
    "cudart64_*.dll",       # CUDA runtime
    "cuda.dll",             # CUDA driver API
    "nvml.dll",             # NVIDIA Management Library
]

# Common locations
SEARCH_PATHS = [
    r"C:\Windows\System32",
    r"C:\Windows\SysWOW64",
    r"C:\Program Files\NVIDIA Corporation\NvToolsExt\bin\x64",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
]
# Also check PATH
SEARCH_PATHS += os.environ.get("PATH", "").split(";")

def find_dll(name):
    if "*" in name:
        for base in SEARCH_PATHS:
            matches = glob.glob(os.path.join(base, name))
            if matches:
                return matches[0]
        return None
    for base in SEARCH_PATHS:
        candidate = os.path.join(base, name)
        if os.path.isfile(candidate):
            return candidate
    return None

def try_load(path):
    try:
        ctypes.WinDLL(path)
        return True, None
    except OSError as e:
        return False, str(e)

for dll in DLLS_TO_CHECK:
    path = find_dll(dll)
    if path:
        ok, err = try_load(path)
        status = "OK  " if ok else "ERR "
        detail = path if ok else f"{path} -> {err}"
    else:
        status = "MISS"
        detail = "(not found in search paths)"
    print(f"  [{status}] {dll:<30} {detail}")

# Try loading the _PyNvVideoCodec .pyd directly for a more specific error
print("\n=== PyNvVideoCodec .pyd location ===\n")
try:
    import importlib.util
    spec = importlib.util.find_spec("PyNvVideoCodec")
    if spec:
        pkg_path = os.path.dirname(spec.origin)
        print(f"  Package path: {pkg_path}")
        pyds = glob.glob(os.path.join(pkg_path, "**", "_PyNvVideoCodec*.pyd"), recursive=True)
        pyds += glob.glob(os.path.join(pkg_path, "_PyNvVideoCodec*.pyd"))
        for pyd in pyds:
            print(f"  .pyd found:   {pyd}")
            ok, err = try_load(pyd)
            if ok:
                print(f"             -> loaded OK (unexpected!)")
            else:
                print(f"             -> {err}")
    else:
        print("  importlib cannot find PyNvVideoCodec spec")
except Exception as e:
    print(f"  Error: {e}")

print("\n=== CUDA / Driver in PATH ===\n")
for p in os.environ.get("PATH", "").split(";"):
    if "cuda" in p.lower() or "nvidia" in p.lower():
        print(f"  {p}")
