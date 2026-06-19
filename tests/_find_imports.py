"""Use pefile (if available) to dump the import table of the failing .pyd files."""
import os, sys

pkg = r"C:\Users\Al\Documents\warp\warp-env\Lib\site-packages\PyNvVideoCodec"

try:
    import pefile
    HAS_PEFILE = True
except ImportError:
    HAS_PEFILE = False
    print("pefile not installed -- trying fallback approach")

pyds = [
    "PyNvVideoCodec_130.cp312-win_amd64.pyd",
    "PyNvVideoCodec_121.cp312-win_amd64.pyd",
    "VersionCheck.cp312-win_amd64.pyd",
]

if HAS_PEFILE:
    for name in pyds:
        path = os.path.join(pkg, name)
        if not os.path.exists(path):
            continue
        pe = pefile.PE(path)
        imports = []
        if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
            for entry in pe.DIRECTORY_ENTRY_IMPORT:
                imports.append(entry.dll.decode())
        print(f"\n{name}  imports:")
        for dll in sorted(imports):
            print(f"  {dll}")
else:
    # Fallback: use dumpbin if available
    import subprocess
    dumpbin = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.43.34808\bin\Hostx64\x64\dumpbin.exe"
    if not os.path.exists(dumpbin):
        # Try to find it
        for root, dirs, files in os.walk(r"C:\Program Files\Microsoft Visual Studio"):
            for f in files:
                if f.lower() == "dumpbin.exe":
                    dumpbin = os.path.join(root, f)
                    break

    if os.path.exists(dumpbin):
        for name in pyds:
            path = os.path.join(pkg, name)
            if not os.path.exists(path):
                continue
            result = subprocess.run([dumpbin, "/dependents", path],
                                   capture_output=True, text=True)
            print(f"\n{name}:")
            for line in result.stdout.splitlines():
                if ".dll" in line.lower():
                    print(f"  {line.strip()}")
    else:
        print("dumpbin.exe not found either.")
        print("\nInstall pefile with: pip install pefile")
        print("Or run from a Visual Studio Developer Command Prompt and use: dumpbin /dependents <pyd>")
