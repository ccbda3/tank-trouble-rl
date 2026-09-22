import ctypes
import glob
import importlib.util
import os
import shutil

LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR = 0x00000100
LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000


def _runtime():
    sys32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    names = ["ucrtbase.dll", "concrt140.dll", "msvcp140.dll", "msvcp140_1.dll",
             "msvcp140_2.dll", "msvcp140_atomic_wait.dll", "msvcp140_codecvt_ids.dll",
             "vcruntime140.dll", "vcruntime140_1.dll"]
    names += [os.path.basename(p) for p in glob.glob(os.path.join(sys32, "api-ms-win-crt-*.dll"))]
    for n in names:
        p = os.path.join(sys32, n)
        if os.path.exists(p):
            try:
                ctypes.CDLL(p)
            except OSError:
                pass


def _torch_lib():
    spec = importlib.util.find_spec("torch")
    if spec is None or not spec.origin:
        return None
    return os.path.join(os.path.dirname(spec.origin), "lib")


def _fill_nvrtc(lib):
    pkg = os.path.dirname(os.path.dirname(lib))
    for pat in ("nvidia/*/bin/nvrtc*.dll", "nvidia/*/lib/nvrtc*.dll"):
        for src in glob.glob(os.path.join(pkg, *pat.split("/"))):
            dst = os.path.join(lib, os.path.basename(src))
            if not os.path.exists(dst):
                try:
                    shutil.copy2(src, dst)
                except OSError:
                    pass


def _preload(lib):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.LoadLibraryExW.restype = ctypes.c_void_p
    kernel32.LoadLibraryExW.argtypes = [ctypes.c_wchar_p, ctypes.c_void_p, ctypes.c_uint32]
    flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS
    libs = sorted(glob.glob(os.path.join(lib, "*.dll")))
    left = []
    for _ in range(3):
        left = [os.path.basename(p) for p in libs
                if not kernel32.LoadLibraryExW(p, None, flags)]
        if not left:
            break
    return left


def apply():
    lib = _torch_lib()
    if lib is None:
        return None
    _runtime()
    _fill_nvrtc(lib)
    try:
        os.add_dll_directory(lib)
    except OSError:
        pass
    return _preload(lib)
