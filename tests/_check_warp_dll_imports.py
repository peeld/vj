"""Check what cudart version warp.dll and warp-clang.dll link against."""
import pefile, os

dlls = [
    r"C:\Users\Al\Documents\warp\warp-env\Lib\site-packages\warp\bin\warp.dll",
    r"C:\Users\Al\Documents\warp\warp-env\Lib\site-packages\warp\bin\warp-clang.dll",
]

for path in dlls:
    pe = pefile.PE(path)
    imports = []
    if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            imports.append(entry.dll.decode())
    print(f"\n{os.path.basename(path)}  imports:")
    for dll in sorted(imports):
        if "cuda" in dll.lower() or "nv" in dll.lower():
            print(f"  ** {dll}")
        else:
            print(f"     {dll}")
