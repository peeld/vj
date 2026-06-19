import ctypes, os, sys

pkg = r"C:\Users\Al\Documents\warp\warp-env\Lib\site-packages\PyNvVideoCodec"

os.add_dll_directory(r"C:\Windows\System32")
os.add_dll_directory(pkg)

for pyd_name in [
    "VersionCheck.cp312-win_amd64.pyd",
    "PyNvVideoCodec_130.cp312-win_amd64.pyd",
    "PyNvVideoCodec_121.cp312-win_amd64.pyd",
]:
    path = os.path.join(pkg, pyd_name)
    if not os.path.exists(path):
        print(f"  SKIP (not found): {pyd_name}")
        continue
    try:
        ctypes.WinDLL(path)
        print(f"  OK  : {pyd_name}")
    except OSError as e:
        print(f"  FAIL: {pyd_name}")
        print(f"        {e}")
