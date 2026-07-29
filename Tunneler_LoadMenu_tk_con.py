# YASARA PLUGIN
# TOPIC:       Tunnels
# TITLE:       Tunneler
# AUTHOR:      M.J.L.J. Fürst
# LICENSE:     GPL (www.gnu.org)
# PLATFORMS:   ExternalPython,MacOS
# DESCRIPTION: This plugin loads the menu for the tunnel inspection
#
 
"""
MainMenu: Analyze
  PullDownMenu: Tunnels
    Submenu: Predict Tunnels
      Request: loadmenu
"""

# Tunneler GUI entry point — tkinter dialog for tunnel detection and inspection.

# This file is loaded directly by YASARA as a plugin. It:
#   1. Checks and optionally installs Python dependencies (numpy, sklearn, etc.)
#   2. Imports the core pipeline from Tunneler_function_con and geometry
#      functions from Tunneler_diameter_functions
#   3. Builds a 3-tab tkinter dialog (Create / Appearance / Inspect) via
#      tunneler_dialog(), which runs as a mainloop

# The giant tunneler_dialog() closure contains ~70 nested functions that share
# state through local variables. This is intentional — it avoids global state —
# but makes the function very long. Section banners below help navigate it.


# ============================================================
#  DEPENDENCY CHECKS & IMPORTS
# ============================================================

import sys
import os

# YASARA runs plugins from the plg/ folder (it sets the working directory there)
# and provides the `yasara` module as plg/yasara.py. Normally sys.path[0] is the
# script's folder (plg/), so the import just works -- but when this file is a
# SYMLINK (e.g. a dev checkout linked into plg/), Python 3.11+ sets sys.path[0]
# to the link *target* instead, and `import yasara` fails. Make the import robust
# either way by putting the plugin folder (cwd) on the path first.
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
from yasara import *
from Tunneler_env_check import ensure_dependencies


# ============================================================
#  TOP-LEVEL UTILITIES
#  (Some of these duplicate functions in Tunneler_function_con.py
#   because they are needed before that module is imported, or
#   because the UI file was developed independently.)
# ============================================================

def rescale_floats_to_range(float_list, min_int, max_int, min_float=None, max_float=None):
    """Linearly scale a list of floats to an integer range [min_int, max_int].

    Used to map distance values to YASARA color codes.
    """
    if min_float == None:
        min_float = min(float_list)
    if max_float == None:
        max_float = max(float_list)
    scaled_list = [int((x - min_float) / (max_float - min_float) * (max_int - min_int) + min_int) for x in float_list]
    return scaled_list


# Distance-gradient palettes. Two kinds:
#   ('hue',  [native YASARA colour numbers])  -- a plain 0-360 wheel sweep (wheel: 0/360=
#       blue, 60=magenta, 120=red, 180=yellow, 240=green, 300=cyan; >360 keeps a sweep
#       monotonic since n == n+360). Exact + fast, strip matches atoms.
#   ('cmap', 'matplotlib-name')  -- a real perceptual colormap, sampled to hex and passed
#       to ColorAtom. Hex carries NO performance penalty (the ColorPar "slowdown" warning
#       is only about changing DEFAULT scheme colours; per-atom ColorAtom always uses
#       YASARA's fast texture engine). YASARA snaps hex to its displayable gamut -- which
#       importantly includes the GRAY CIRCLE, so desaturated/diverging palettes (coolwarm's
#       near-white centre) render, impossible on the pure wheel. It has no dark/lightness
#       axis though, so very dark colours snap imperfectly -- hence _CMAP trims the extremes.
DIST_PALETTES = {
    'Rainbow':   ('hue',  [120, 300]),   # native 0-360 sweep: red -> yellow -> green -> cyan
    'Viridis':   ('cmap', 'viridis'),
    'Plasma':    ('cmap', 'plasma'),
    'Turbo':     ('cmap', 'turbo'),
    'Cool-Warm': ('cmap', 'coolwarm'),
    'Spectral':  ('cmap', 'Spectral'),
}
DIST_BANDS = 128                 # gradient quantisation; also caps ColorAtom calls/tunnel
                                 # and keeps the cache's 'c<band>' segment tag <= 4 chars
_CMAP_LO, _CMAP_HI = 0.05, 0.95  # trim colormap dark extremes (snap poorly / can wrap hue)
_CMAP_CACHE = {}
# Above this many tunnel points, the points/balls distance recolour (the group_and_color
# loop) is slow enough (~seconds) to warrant the progress popup; below it recolours fast
# enough that a popup would just flicker. Spheres/shapes always get the popup.
_RECOLOR_POPUP_MIN = 50000


def _palette_num(stops, t):
    """Interpolate a 'hue' palette (list of native YASARA colour numbers) at fraction t in
    [0,1], piecewise-linear across evenly-spaced stops."""
    if t <= 0:
        return stops[0]
    if t >= 1:
        return stops[-1]
    seg = t * (len(stops) - 1)
    i = int(seg)
    f = seg - i
    return stops[i] * (1 - f) + stops[i + 1] * f


def _get_cmap(name):
    cm = _CMAP_CACHE.get(name)
    if cm is None:
        try:
            import matplotlib
            cm = matplotlib.colormaps[name]      # matplotlib >= 3.5
        except Exception:
            import matplotlib.pyplot as _plt
            cm = _plt.get_cmap(name)             # older fallback
        _CMAP_CACHE[name] = cm
    return cm


def _cmap_rgb255(name, u):
    """Sample a matplotlib colormap at u in [0,1] (trimmed to [_CMAP_LO,_CMAP_HI]) ->
    (r,g,b) ints 0-255."""
    uu = _CMAP_LO + max(0.0, min(1.0, u)) * (_CMAP_HI - _CMAP_LO)
    r, g, b, _a = _get_cmap(name)(uu)
    return int(r * 255), int(g * 255), int(b * 255)


def _palette_color(name, u):
    """Colour spec for palette `name` at fraction u in [0,1], ready for ColorAtom: a native
    YASARA hue int for a 'hue' palette, or an 'rrggbb' hex string for a matplotlib cmap."""
    kind, spec = DIST_PALETTES[name]
    if kind == 'hue':
        return int(round(_palette_num(spec, u)))
    return '%02x%02x%02x' % _cmap_rgb255(spec, u)


def _ycolor(cv):
    """Coerce one colour cell to a value ColorAtom/LoadWOb accept: pass a hex string
    ('rrggbb') through unchanged, coerce anything numeric to a plain int (native wheel/gray
    colour). The distinction is by TYPE (str vs number), never by parsing -- a hex like
    '440154' is all-digits-looking but must NOT be read as the decimal 440154."""
    if isinstance(cv, str) or isinstance(cv, np.str_):
        return str(cv)
    return int(cv)

# Verify the scientific stack is present AND functional before importing it.
# ensure_dependencies() diagnoses the running Python (rejects YASARA's bundled
# 'epy'), functionally probes each dependency (a bare `import matplotlib` misses
# the tkagg/Tcl breakage), and offers a correct install/upgrade. It must run
# BEFORE the heavy imports below, which would otherwise crash on a bad env.
_env_status = ensure_dependencies(interactive=True)
if _env_status != 'ok':
    # 'repaired-restart': packages were fixed but this process still holds the
    # old modules, so the user was asked to relaunch. 'abort': env unusable and
    # already explained. Either way, stop before the heavy imports.
    plugin.end()

import numpy as np
from scipy.spatial import cKDTree
import re
import os
import tempfile
from configparser import ConfigParser
from Tunneler_function_con import *
import threading
import tkinter as tk
import tkinter.ttk as ttk
from Tunneler_diameter_functions import *
from sklearn.cluster import DBSCAN
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def _attach_elapsed_timer(win):
    """Add a live MM:SS elapsed-time label to a progress-bar popup.

    The work runs in a worker thread while the tk event loop keeps ticking on the
    main thread, so a self-rescheduling `after()` updates the label live. Elapsed is
    computed from an absolute start, so the display stays accurate even if ticks are
    delayed; the loop stops itself once the window is destroyed.
    """
    timer_label = ttk.Label(win, text="00:00")
    timer_label.pack(pady=(0, 5))
    timer_start = time.perf_counter()
    def _set_label():
        el = int(time.perf_counter() - timer_start)
        timer_label.config(text=f'{el // 60:02d}:{el % 60:02d}')
    def _tick():
        try:
            if not win.winfo_exists():
                return
            _set_label()
            win.after(250, _tick)
        except tk.TclError:
            return
    # Manual tick for work that runs on the MAIN thread (e.g. the sphere build), where
    # the tk event loop is blocked so after() never fires -- the worker calls this + an
    # update_idletasks() to keep the clock moving.
    win._elapsed_manual = _set_label
    win.after(0, _tick)


# --- Sphere .obj geometry cache ----------------------------------------------
# The dominant cost of a sphere rebuild is writing the (multi-hundred-MB) .obj
# file, not YASARA's parse of it (~4.7s vs ~1.25s for a 118k-point tunnel). The
# geometry is independent of alpha (alpha is only a LoadWOb arg, and there is no
# in-place alpha command), so an alpha-only rebuild can reuse the exact files and
# just re-LoadWOb them -- ~4x faster. We keep the files in a dedicated temp dir.
#
# Crash-safety: the dir is wiped on plugin load (clearing any files a previously
# crashed YASARA left behind), on atexit for clean exits, and by _sph_cache_clear
# during a session. Everything lives under one directory so a stale-file leak is
# impossible to accumulate across runs.
import shutil, atexit
_SPH_OBJ_DIR = os.path.join(tempfile.gettempdir(), 'tunneler_sphere_obj_cache')
shutil.rmtree(_SPH_OBJ_DIR, ignore_errors=True)   # drop leftovers from a prior (maybe crashed) run
os.makedirs(_SPH_OBJ_DIR, exist_ok=True)
atexit.register(lambda: shutil.rmtree(_SPH_OBJ_DIR, ignore_errors=True))


# ============================================================
#  "Follow YASARA" window behaviour (macOS)
#
#  Instead of pinning the dialog permanently on top (which also floats it
#  over unrelated apps like a browser), we keep it -topmost only while the
#  frontmost application is "ours" (YASARA itself, or this dialog's own Tk
#  process) and drop it to a normal level when any other app is in front.
#  Net effect: the dialog rides forward whenever YASARA is foregrounded and
#  sinks behind whatever else you switch to.
#
#  macOS ships `lsappinfo`, which reports the frontmost app's pid + name
#  with no extra dependency and no Automation-permission prompt. If it's
#  unavailable (non-macOS, or the query fails), callers fall back to plain
#  always-on-top. Toggling -topmost only restacks the window level; it does
#  NOT activate our app, so it never steals keyboard focus from YASARA.
# ============================================================
def _macos_frontmost_app():
    """Return (pid, name) of the frontmost macOS app, or (None, None) on failure."""
    if sys.platform != 'darwin':
        return (None, None)
    try:
        import subprocess
        asn = subprocess.run(['lsappinfo', 'front'], capture_output=True,
                             text=True, timeout=1.5).stdout.strip()
        if not asn:
            return (None, None)
        out = subprocess.run(['lsappinfo', 'info', '-only', 'pid', '-only', 'name', asn],
                            capture_output=True, text=True, timeout=1.5).stdout
        pid_m = re.search(r'"pid"\s*=\s*(\d+)', out)
        name_m = re.search(r'"LSDisplayName"\s*=\s*"([^"]*)"', out)
        pid = int(pid_m.group(1)) if pid_m else None
        name = name_m.group(1) if name_m else ''
        return (pid, name)
    except Exception:
        return (None, None)


def _macos_set_accessory(on):
    """Hide (on=True) / show (on=False) this process in the macOS Dock + Cmd-Tab
    switcher by setting the NSApplication activation policy (Accessory=1, Regular=0).

    An 'accessory' app still shows windows and can take focus/keyboard input, it
    just has no Dock icon and no app-switcher entry -> the dialog feels attached
    to YASARA rather than a standalone app. Driven through the Obj-C runtime with
    ctypes so no pyobjc dependency is needed. Returns True on success, else False
    (non-macOS or any failure) so callers can silently keep the default policy.
    """
    if sys.platform != 'darwin':
        return False
    try:
        import ctypes
        objc = ctypes.CDLL('/usr/lib/libobjc.dylib')
        ctypes.CDLL('/System/Library/Frameworks/AppKit.framework/AppKit')  # ensure NSApplication is registered
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        NSApplication = objc.objc_getClass(b'NSApplication')
        # app = [NSApplication sharedApplication]  (returns Tk's existing NSApp)
        objc.objc_msgSend.restype = ctypes.c_void_p
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        app = objc.objc_msgSend(NSApplication, objc.sel_registerName(b'sharedApplication'))
        if not app:
            return False
        # [app setActivationPolicy: policy]  -> BOOL
        objc.objc_msgSend.restype = ctypes.c_bool
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long]
        ok = objc.objc_msgSend(app, objc.sel_registerName(b'setActivationPolicy:'), 1 if on else 0)
        return bool(ok)
    except Exception:
        return False


def _shell_mask(centers, spacing):
    """Boolean mask over `centers` (N x 3): True = surface point, False = fully-interior.

    The tunnel cloud is a cubic grid of spacing `spacing`, so a point is INTERIOR iff
    all 26 grid neighbours are present (self + 26 = 27 points within the corner radius
    spacing*sqrt(3)). Surface points are missing at least one neighbour. Rotation-
    invariant (pairwise distances are preserved under the scene's global-frame rotation).

    Used to render only the visible shell of an opaque tunnel -- the buried interior
    contributes nothing to an opaque view. The source point cloud is never modified;
    this only filters the arrays fed to the mesh builder / hide, so pathfinding,
    cross-section and volume (which read all atoms) stay correct.
    """
    centers = np.asarray(centers, dtype=float)
    if len(centers) == 0:
        return np.ones(0, dtype=bool)
    tree = cKDTree(centers)
    r = spacing * np.sqrt(3) * 1.05          # reaches the 26-neighbourhood corner
    counts = tree.query_ball_point(centers, r, return_length=True)
    return np.asarray(counts) < 27           # 27 = self + 26 neighbours -> interior


_ICOSPHERE_CACHE = {}

def _icosphere(level):
    """Return (vertices Nx3, faces Mx3 0-indexed) of a unit icosphere at the given
    subdivision level, matching YASARA's ShowSphere tessellation levels (0=20 faces,
    1=80, 2=320, 3=1280). Cached per level."""
    level = int(level)
    if level in _ICOSPHERE_CACHE:
        return _ICOSPHERE_CACHE[level]
    phi = (1 + 5 ** 0.5) / 2
    V = [(-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0), (0, -1, phi), (0, 1, phi),
         (0, -1, -phi), (0, 1, -phi), (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1)]
    V = [tuple(np.array(v) / np.linalg.norm(v)) for v in V]
    F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4),
         (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8),
         (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(level):
        mid = {}; nV = list(V); nF = []
        def _mid(a, b):
            k = (a, b) if a < b else (b, a)
            if k in mid:
                return mid[k]
            p = np.array(V[a]) + np.array(V[b]); p = p / np.linalg.norm(p)
            nV.append(tuple(p)); mid[k] = len(nV) - 1
            return mid[k]
        for a, b, c in F:
            ab, bc, ca = _mid(a, b), _mid(b, c), _mid(c, a)
            nF += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        V, F = nV, nF
    res = (np.array(V, float), np.array(F, int))
    _ICOSPHERE_CACHE[level] = res
    return res

def _write_obj_rows(f, arr, rowfmt, chunk=1000000):
    """Write rows of `arr` to binary file `f` as text via C-level (rowfmt*k) % tuple,
    chunked to bound peak memory. ~2x faster than a per-row generator join for the
    millions of lines a large sphere mesh needs."""
    n = len(arr); i = 0
    while i < n:
        blk = arr[i:i + chunk]
        f.write(((rowfmt * len(blk)) % tuple(blk.ravel().tolist())).encode())
        i += chunk


# --- Parallel .obj writer -----------------------------------------------------
# Formatting millions of floats to ASCII is CPU-bound and GIL-held, so for big tunnels
# the write is fanned out across worker subprocesses (each a run of Tunneler_meshwrite.py
# -- a standalone script, NOT a multiprocessing.Pool: 'spawn' on macOS would re-execute
# this whole plugin in every worker). Measured ~3-4x on a 118k-point tunnel. It always
# falls back to the serial writer on any failure or below the size threshold.
_PARALLEL_WRITE_MIN_VERTS = 800000                       # skip fan-out below this (spawn overhead)
_WRITE_NPROC = min(12, max(2, (os.cpu_count() or 4) - 2))

def _find_meshworker():
    """Locate Tunneler_meshwrite.py. It sits beside this file; when the plugin is a
    symlink in YASARA's plg/, realpath resolves to the real working-dir copy. Falls back
    to cwd (plg/) for a copy-deploy. Returns (path, ok)."""
    cands = []
    for getter in (lambda: os.path.dirname(os.path.realpath(__file__)),
                   os.getcwd,
                   lambda: os.path.dirname(__file__)):
        try:
            cands.append(getter())
        except Exception:
            pass
    for d in cands:
        p = os.path.join(d, 'Tunneler_meshwrite.py')
        if os.path.exists(p):
            return p, True
    return '', False

_MESHWORKER, _MESHWORKER_OK = _find_meshworker()
_WRITE_LAST = ['idle']   # actual mode of the most recent .obj write (for the build message)
_WRITE_ERR = ['']        # reason the last parallel write fell back (for diagnostics)

def _write_obj_parallel(path, sections, nproc=_WRITE_NPROC):
    """Write an .obj at `path` from `sections` (list of (array, rowfmt)) using up to
    `nproc` worker subprocesses that format row-slices in parallel; concatenate the parts
    in order. Returns True on success, False (writing nothing) on any problem so the
    caller can fall back to the serial writer. Temp files live beside `path`."""
    import subprocess, base64
    tmp, procs, parts = [], [], []
    per = max(2, nproc // max(1, len(sections)))
    _WRITE_ERR[0] = ''
    try:
        for si, (arr, fmt) in enumerate(sections):
            npy = f'{path}.s{si}.npy'; np.save(npy, arr); tmp.append(npy)
            n = len(arr); fb = base64.b64encode(fmt.encode()).decode()
            bnd = [(n * j) // per for j in range(per + 1)]
            for j in range(per):
                if bnd[j] == bnd[j + 1]:
                    continue
                out = f'{path}.s{si}.{j}'; tmp.append(out); parts.append(out)
                procs.append(subprocess.Popen(
                    [sys.executable, _MESHWORKER, npy, str(bnd[j]), str(bnd[j + 1]), fb, out],
                    stderr=subprocess.PIPE))
        bad = None
        for p in procs:
            err = p.communicate()[1]
            if p.returncode != 0 and bad is None:
                bad = f'rc={p.returncode} {(err or b"").decode("utf-8", "replace").strip()[-160:]}'
        if bad is not None:
            _WRITE_ERR[0] = bad
            return False
        if not all(os.path.exists(x) for x in parts):
            _WRITE_ERR[0] = 'part file(s) missing'
            return False
        with open(path, 'wb') as f:
            for x in parts:
                with open(x, 'rb') as g:
                    shutil.copyfileobj(g, f)
        return True
    except Exception as e:
        _WRITE_ERR[0] = repr(e)[:160]
        return False
    finally:
        for x in tmp:
            try:
                os.remove(x)
            except OSError:
                pass

def _load_sphere_mesh(centers, radius, color, alpha, level, path=None, reuse=False,
                      normals=True):
    """Draw many equal-radius spheres of a SINGLE colour as one polygon-mesh object,
    built in numpy and imported via LoadWOb in a single call. This replaces ~2 YASARA
    calls per sphere (ShowSphere+PosObj) with one file load -> ~15x faster at large
    tunnels. Returns the new object number.

    LoadWOb quirks handled here (validated against ShowSphere, see project notes):
      * it applies T=(x,y,-z) to file coords and offsets by the object position, so we
        pre-flip z on the centres and template and PosObj(0,0,0) -> world == centres;
      * the z-reflection reverses triangle winding, so we reverse the face order, else
        YASARA shades the spheres' dark interiors (they render near-black);
      * open='No' culls back faces -> matches ShowSphere's transparency density so the
        alpha slider maps 1:1 (no remap needed).

    `normals=False` omits the per-vertex normals (and the 'f a//a ...' normal refs): it
    ~halves the .obj write time and file size for big tunnels, at the cost of flat/faceted
    sphere shading (YASARA falls back to per-face normals). Winding/backface-cull/alpha
    are unaffected. Used only above a size threshold where the write dominates and the
    spheres are too small/dense for the faceting to show.

    Geometry caching: `path`, when given, is a stable file to write the mesh to and
    keep (not a throwaway temp). If `reuse` is True and that file already exists, the
    write is skipped entirely and the cached .obj is re-loaded with the current colour
    and alpha -- geometry is independent of both, so an alpha (or tunnel-mode colour)
    change costs only the LoadWOb parse, not the dominant file-write. With no `path`
    the old throwaway-temp behaviour is kept."""
    own_temp = path is None
    if own_temp:
        fd, path = tempfile.mkstemp(suffix='.obj', prefix='ts'); os.close(fd)
    try:
        if not (reuse and os.path.exists(path)):
            Vt, Ft = _icosphere(level)
            centers = np.asarray(centers, float).reshape(-1, 3)
            C = centers.copy(); C[:, 2] *= -1
            Vr = Vt.copy(); Vr[:, 2] *= -1          # z-reflect template to cancel LoadWOb's T
            Fr = Ft[:, ::-1]                        # reverse winding (z-reflection flipped it)
            n = len(C); vpr = len(Vr)
            verts = (C[:, None, :] + Vr[None, :, :] * radius).reshape(-1, 3)
            faces = (Fr[None, :, :] + 1 + (np.arange(n) * vpr)[:, None, None]).reshape(-1, 3)
            if normals:
                norms = np.tile(Vr, (n, 1))
                face6 = np.repeat(faces, 2, axis=1)       # 'f a//a b//b c//c' needs each index twice
                sections = [(verts, 'v %.3f %.3f %.3f\n'),
                            (norms, 'vn %.3f %.3f %.3f\n'),
                            (face6, 'f %d//%d %d//%d %d//%d\n')]
            else:
                sections = [(verts, 'v %.3f %.3f %.3f\n'),
                            (faces, 'f %d %d %d\n')]       # no normals -> flat shading
            # Big meshes: fan the (CPU-bound) formatting across worker processes; small
            # ones and any failure fall back to the single-threaded writer.
            done = False
            if not _MESHWORKER_OK:
                _WRITE_LAST[0] = 'single-thread (worker not found)'
            elif len(verts) < _PARALLEL_WRITE_MIN_VERTS:
                _WRITE_LAST[0] = 'single-thread (small)'
            else:
                done = _write_obj_parallel(path, sections)
                _WRITE_LAST[0] = (f'multithread x{_WRITE_NPROC}' if done
                                  else f'single-thread (parallel failed: {_WRITE_ERR[0]})')
            if not done:
                with open(path, 'wb') as f:
                    for arr, fmt in sections:
                        _write_obj_rows(f, arr, fmt)
        obj = LoadWOb(path, color=color, alpha=alpha, open='No')[0]
        PosObj(obj, 0, 0, 0)
        return obj
    finally:
        if own_temp and os.path.exists(path):
            os.remove(path)


def _load_polygon_mesh(verts, faces, norms, color, alpha):
    """LoadWOb an arbitrary triangle mesh given in global coords as one single-colour
    object, applying the same LoadWOb compensation as _load_sphere_mesh (pre-flip z,
    reverse winding, open='No', PosObj(0,0,0)). Returns the object number."""
    V = verts.copy(); V[:, 2] *= -1
    N = norms.copy(); N[:, 2] *= -1
    F = faces[:, ::-1] + 1
    fd, path = tempfile.mkstemp(suffix='.obj', prefix='ts'); os.close(fd)
    try:
        with open(path, 'wb') as f:
            _write_obj_rows(f, V, 'v %.3f %.3f %.3f\n')
            _write_obj_rows(f, N, 'vn %.3f %.3f %.3f\n')
            _write_obj_rows(f, np.repeat(F, 2, axis=1), 'f %d//%d %d//%d %d//%d\n')
        obj = LoadWOb(path, color=color, alpha=alpha, open='No')[0]
        PosObj(obj, 0, 0, 0)
        return obj
    finally:
        if os.path.exists(path):
            os.remove(path)


# ---- Cross-section overlay (Tab-3 'section' toggle) ---------------------------
# Draw the tunnel's cross-section at the cutting plane as a thin filled + outlined
# mesh lying on the plane, rebuilt live as the diameter slider moves. Colours,
# alpha and rim width are cosmetic knobs.
_XSEC_FILL_COL = '00e0ff'   # translucent cyan fill
_XSEC_FILL_ALPHA = 45
_XSEC_EDGE_COL = '00e0ff'   # bright cyan rim
_XSEC_EDGE_ALPHA = 100
_XSEC_EDGE_W = 0.35         # rim thickness (A)


def _xsec_mesh_load(shape2d, u, v, nrm, c, color, alpha, simplify_tol):
    """Triangulate a shapely (Multi)Polygon given in plane-2D (u,v) coords, lift each
    vertex to 3D global via  p = x*u + y*v + c*nrm, and LoadWOb it as one flat
    single-colour mesh (reusing _load_polygon_mesh). Returns the object number, or
    None if nothing triangulable. Delaunay-of-vertices + inside-filter keeps the
    triangulation inside concave outlines / holes (no Steiner points)."""
    from shapely.ops import triangulate
    geoms = list(shape2d.geoms) if shape2d.geom_type.startswith('Multi') else [shape2d]
    verts = []
    faces = []
    for g in geoms:
        if simplify_tol:
            g = g.simplify(simplify_tol)
        if g.is_empty or g.area <= 0:
            continue
        for t in triangulate(g):
            if not g.contains(t.representative_point()):
                continue
            base = len(verts)
            for x, y in list(t.exterior.coords)[:3]:
                verts.append(x * u + y * v + c * nrm)
            faces.append([base, base + 1, base + 2])
    if not faces:
        return None
    verts = np.array(verts, dtype=float)
    faces = np.array(faces, dtype=int)
    norms = np.tile(nrm, (len(verts), 1))
    return _load_polygon_mesh(verts, faces, norms, _ycolor(color), alpha)


def _build_xsec_object(shapes2d, u, v, plane_origin, objname, anchor_obj, fill_alpha):
    """Build the filled + outlined cross-section overlay object `objname` from a list
    of shapely polygons in plane-2D coords (one per cluster/lobe). Returns the object
    number, or None if nothing was drawn. `fill_alpha` sets the translucency of the fill
    (driven by the same alpha control as the rectangular plane); the rim stays opaque so
    the outline is always legible.

    `anchor_obj` is the tunnel object the overlay belongs to. LoadWOb meshes are baked
    in the current global (screen) frame and do NOT rotate with the scene on their own
    (verified: a raw mesh stays put while atoms rotate), so we TransferObj the finished
    meshes onto `anchor_obj` with Local='Fix' -- this adopts the tunnel's coordinate
    frame while keeping them on-screen where built, so they rotate with the tunnel
    points (their cross-section) from then on."""
    nrm = np.cross(u, v)
    n_len = np.linalg.norm(nrm)
    if n_len == 0:
        return None
    nrm = nrm / n_len
    c = float(np.dot(nrm, plane_origin))
    objs = []
    for sh in shapes2d:
        if sh is None or sh.is_empty:
            continue
        fill = _xsec_mesh_load(sh, u, v, nrm, c, _XSEC_FILL_COL, fill_alpha, 0.25)
        if fill is not None:
            objs.append(fill)
        rim = _xsec_mesh_load(sh.boundary.buffer(_XSEC_EDGE_W / 2), u, v, nrm, c,
                              _XSEC_EDGE_COL, _XSEC_EDGE_ALPHA, 0.05)
        if rim is not None:
            objs.append(rim)
    if not objs:
        return None
    # Do NOT JoinObj the sub-meshes: the translucent fill (alpha 45) and the opaque
    # rim (alpha 100) are different transparency types, and YASARA refuses to join
    # meshes of different type ("Objects N and M differ significantly"). Instead give
    # every sub-mesh the same name -- multiple objects can share a name and all the
    # cleanup is by name pattern (NNN_xsec / ???_xsec), so a single object is not needed.
    NameObj(' '.join(str(o) for o in objs), objname)
    # Adopt the tunnel's coordinate frame so the overlay rotates with the scene.
    TransferObj(objname, anchor_obj, 'Fix')
    return objs[0]


def _skimage_missing():
    """True if scikit-image (needed for the MergeSph shape's marching cubes) is not
    importable -- lets the GUI show a helpful message instead of crashing."""
    import importlib.util
    return importlib.util.find_spec('skimage') is None


def _union_shape_mesh(centers, colors, radius, alpha, voxel=0.3, smooth=0.8,
                      geom_cache=None, geom_key=None, geom_frame=None):
    """'Merged spheres' surface: the union of balls of `radius` around `centers`,
    extracted with marching cubes over a signed-distance grid (dist-to-nearest-centre
    minus radius) built with cKDTree. Each surface vertex takes the colour of its
    nearest centre, so it follows the tunnel/distance colouring; triangles are grouped
    by colour into per-colour LoadWOb meshes and joined into one object (returned), or
    None if the selection yields no surface. Cheap: ~0.3-0.5s for a several-k-point
    tunnel. `smooth` Gaussian-blurs the grid for a rounder (metaball-like) blob; `voxel`
    trades surface detail against cost.

    Geometry caching: the marching-cubes step (the expensive part) depends only on the
    centres + radius, NOT on alpha or the colouring. When `geom_cache` (a dict) and
    `geom_key` are given, the (verts, faces, norms) are cached under that key and reused,
    so an alpha change (rebuild) OR a colour-mode change skips marching cubes entirely --
    only the per-colour split + LoadWOb (with the current colours/alpha) re-runs. The
    `centers` are GLOBAL (screen-frame) coords, which change when the structure is
    rotated/moved, so the cache entry is tagged with `geom_frame` (the source object's
    position+orientation) and reused only while that is unchanged -- otherwise the mesh
    would be baked at a stale orientation and appear offset from the other reps."""
    centers = np.asarray(centers, float).reshape(-1, 3)
    colors = np.asarray(colors)
    if len(centers) == 0:
        return None
    cached = geom_cache.get(geom_key) if geom_cache is not None else None
    if cached is not None and cached[0] == geom_frame:
        verts, faces, norms = cached[1], cached[2], cached[3]   # reuse cached marching-cubes geometry
    else:
        from skimage import measure                       # in the venv; imported lazily
        margin = radius + 2 * voxel
        lo = centers.min(0) - margin; hi = centers.max(0) + margin
        ax = [np.arange(lo[d], hi[d], voxel) for d in range(3)]
        gx, gy, gz = np.meshgrid(*ax, indexing='ij')
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
        sdf = (cKDTree(centers).query(pts)[0] - radius).reshape(gx.shape)
        if smooth > 0:
            from scipy.ndimage import gaussian_filter
            sdf = gaussian_filter(sdf, smooth)
        try:
            verts, faces, norms, _ = measure.marching_cubes(sdf, level=0.0, spacing=(voxel, voxel, voxel))
        except (ValueError, RuntimeError):
            return None
        verts = verts + lo
        # Our SDF is negative *inside* the balls, positive outside; marching_cubes'
        # default gradient_direction='descent' assumes the object is the higher-valued
        # region, so it returns inward-pointing normals -> the shape gets lit from
        # behind (dark/desaturated) while every other rep lights outward. Flip to
        # outward so shading matches the sphere reps and the rest of the scene.
        norms = -norms
        if geom_cache is not None:
            geom_cache[geom_key] = (geom_frame, verts, faces, norms)
    # colour the (possibly cached) geometry by the CURRENT colours. `colors` may hold hex
    # strings (cmap palette) or ints (hue/tunnel) -- keep them as-is (no int cast) and
    # coerce per-mesh via _ycolor, so LoadWOb (RGBCOLOR) gets a valid colour either way.
    tree = cKDTree(centers)
    fcol = colors[tree.query(verts)[1]][faces[:, 0]]   # face colour = nearest-centre colour of its 1st vertex
    objs = []
    for cv in np.unique(fcol):
        fsub = faces[fcol == cv]
        used = np.unique(fsub)
        loc = np.searchsorted(used, fsub)              # remap to a compact per-colour submesh
        objs.append(_load_polygon_mesh(verts[used], loc, norms[used], _ycolor(cv), alpha))
    NameObj(' '.join(str(x) for x in objs), 'mshape')
    j = ListObj('mshape')[0]
    JoinObj('mshape', j, center='No')
    return j


def tunneler_dialog():
    """Build and run the 3-tab Tunneler tkinter dialog.

    This is one giant closure: ~70 nested functions share state through local
    variables (tk Vars, widget references, etc.). The function does not return
    until the user clicks Exit.

    Sections (search for '# ---' banners):
      - State helpers (target, get_config, convert_status, switch_status)
      - Tab 1 — Create Tunnels (widgets + callbacks)
      - Tab 2 — Appearance (widgets + callbacks)
      - Tab 3 — Inspect Tunnel (widgets + callbacks)
      - Cross-section / diameter analysis
      - Dialog mainloop
    """
    Console("OFF")

    # --------------------------------------------------------
    #  STATE HELPERS — Determine current target, read config
    # --------------------------------------------------------

    def target():
        """Return the object number (as string) of the protein that has tunnels, or None."""
        Console("OFF")
        if PairObj('All', 'ball_spacing') != [] and ListObj('?Cl??????? ??Cl???????') != []:
            return(re.findall(r'^\d+', ListObj('?Cl??????? ??Cl???????', format='OBJNAME')[0])[0])
        else:
            return(None)
        
    def forget_crosssection():
        """Remove all cross-section widgets from Tab 3 (when switching to 'All')."""
        diamter_height_scale.place_forget()
        diam_up.place_forget()
        diam_down.place_forget()
        reset_ax.place_forget()
        adjust_ax.place_forget()
        make_path.place_forget()
        cut_axis_alpha_spin.place_forget()
        cut_axis_alpha_label.place_forget()
        cut_points_button.place_forget()
        plane_radio.place_forget()
        section_radio.place_forget()
        rough_path_button.place_forget()
        axis_button.place_forget()
        expose_path_button.place_forget()
        canvas_widget.place_forget()

    def place_crosssection():
        """Show all cross-section widgets on Tab 3 (when a specific tunnel is selected)."""
        diamter_height_scale.place(anchor="nw", x=19, y=220, width=115)
        diam_up.place(anchor="nw", x=134, y=218)
        diam_down.place(anchor="nw", x=0, y=218)
        axis_button.place(anchor="nw", x=0, y=245)
        adjust_ax.place(anchor="nw", x=50, y=243)
        reset_ax.place(anchor="nw", x=103, y=243)
        cut_axis_alpha_label.place(anchor="nw", x=0, y=267)
        cut_axis_alpha_spin.place(anchor="nw", x=40, y=266)
        cut_points_button.place(anchor="nw", x=0, y=289)
        plane_radio.place(anchor="nw", x=0, y=311)
        section_radio.place(anchor="nw", x=62, y=311)
        make_path.place(anchor="nw", x=208, y=0)
        rough_path_button.place(anchor="nw", x=250, y=-2)
        expose_path_button.place(anchor="nw", x=250, y=14)
        canvas_widget.place(anchor="nw", x=155, y=215, width=tnl_dia_canv_width, height=tnl_dia_canv_height)


    def Recluster():
        """Re-run DBSCAN clustering on the current tunnel points (Tab 2 action)."""
        Console("OFF")
        # Progress-bar popup, like the tunnel/sphere builds. The work is all on the main
        # thread (YASARA object creation), so the bar is pumped manually via _bump():
        # milestones for the fixed phases plus a live 15->80% ramp fed by the per-cluster
        # progress_cb inside cluster_tunnel_points_dbscan (the visibly slow part).
        global progress_window, progress_var, percent_label
        progress_window = tk.Toplevel(root)
        progress_window.title("Reclustering")
        progress_window.lift()
        progress_window.attributes("-topmost", True)
        progress_var = tk.IntVar()
        ttk.Progressbar(progress_window, orient="horizontal", length=200,
                        mode="determinate", variable=progress_var, maximum=100).pack(padx=5, pady=20)
        percent_label = ttk.Label(progress_window, text="0%")
        percent_label.pack(pady=5)
        _attach_elapsed_timer(progress_window)
        progress_window.update()

        def _bump(pct, note=''):
            try:
                progress_var.set(int(pct))
                percent_label.config(text=f'{note} ({int(pct)}%)' if note else f'{int(pct)}%')
                progress_window._elapsed_manual()
                progress_window.update_idletasks()
            except tk.TclError:
                pass

        _bump(5, 'preparing')
        tar = target()
        # check if the user used the hide surface atom slider to hide some points
        if ListObj(f'{tar}excl_pts') != [] or CountAtom(f'obj {tar}Cl???????') > CountAtom(f'obj {tar}Cl??????? visible'):
            exclude_hidden = exclude_chk.get()
        
            if exclude_hidden:
                hidden_points = DuplicateAtom(f'obj {tar}Cl??????? !visible')
                if hidden_points != []:
                    JoinObj(" ".join(str(i) for i in hidden_points), hidden_points[0])
                    NameObj(hidden_points[0], f'{tar}excl_pts')
                    DelAtom(f'obj {tar}Cl??????? !visible')
                    ColorObj(hidden_points[0], 'white')
                    SwitchObj(f'{tar}excl_pts', 'off')
                    ShowObj(f'{tar}excl_pts')

            else:
                # renaming makes its name match downstream
                NameObj(f'{tar}excl_pts', f'{tar}C00000000')
                SegObj(f'{tar}roughsurf', '.')
                ShowObj(f'{tar}Cl???????')
            
        get_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Tunneler_config.ini'))
        points = np.array(PosAtom(f'Obj {target()}Cl???????', coordsys='global')).reshape(-1,3)
        _bump(15, 'clustering')
        # Use the LIVE slider values for the clustering thresholds so Recluster
        # reflects the current settings (min cluster volume + connect cutoff).
        # ball_spacing stays the STORED value the cloud was generated with:
        # Recluster re-thresholds the existing points, it does not regenerate them.
        # progress_cb ramps the bar 15->80% as per-cluster objects are built.
        cluster_tunnel_points_dbscan(tar, points, min_vol_scale_chk.get(), float(ball_spacing),
                                     connect_cut_scale_chk.get(), recluster=True,
                                     progress_cb=lambda f: _bump(15 + 65 * f, 'building clusters'))
        _bump(82, 'fixing structure')
        SwitchObj(f'{tar}Cl????????', "OFF")
        SwitchObj(ListObj(f'{tar}Cl???????')[:5], "ON")
        transf_and_fix_ss(tar)
        DelObj('???_sphere ???_shape')
        _sph_cache_clear()     # new clusters -> any stashed sphere sets are stale
        _sph_obj_cache_clear() # ... and the point positions changed -> drop .obj geometry cache
        _shape_cache_clear()   # ... and any stashed shape (surface) sets
        _xsec_cache_clear()    # ... and the cached cross-section positions
        _bump(90, 'finalizing')
        # Recluster recreated the clusters (freshly coloured by tunnel) and invalidated
        # any distance colouring. Drop the whole distance cache (selection + colour +
        # signature) and reflect the reality in the GUI: select the 'Tunnel' colour mode
        # and reset the distance atom selector, so it no longer claims a stale reference.
        PairObj(tar, 'dist_sel', '')
        PairObj(tar, 'dist_col', '')
        PairObj(tar, 'dist_sig', '')
        radio_col_var.set('tunnel')
        button18.configure(text='select')
        _bump(93, 'computing distances')
        # Recluster changed the cluster set -> refresh the slider's precomputed distances.
        _precompute_surf_dist(tar)
        _bump(100, 'done')
        try:
            progress_window.destroy()
        except tk.TclError:
            pass
        Wait(1)
        Console("hidden")



    def convert_status(value):
        """Convert a boolean/int checkbox value to YASARA 'ON'/'OFF' string."""
        Console("OFF")
        return "ON" if value == 1 else "OFF"

    def Tunnelonoff():
        """Toggle visibility of all tunnel cluster objects."""
        Console("OFF")
        SwitchObj(f'{target()}Cl???????', convert_status(tunnel_chk.get()))
        Wait(1)
        Console("hidden")

    def Targetonoff():
        """Toggle visibility of the protein target object."""
        Console("OFF")
        SwitchObj(target(), convert_status(target_chk.get()))
        Wait(1)
        Console("hidden")


    def Nonprot():
        """Toggle display of non-protein residues (ligands, cofactors, etc.)."""
        Console("OFF")
        tar = target()
        if ListObj(f'{tar}NonProt', format='OBJNUM') == []:
            new = DuplicateObj(tar)[0]
            SwitchObj(new, 'ON')
            HideObj(new)
            HideSecStrObj(new)
            ShowRes(f'Obj {new} res !protein and !hoh')
            NameObj(new, f'{tar}NonProt')
        else:
            SwitchObj(f'{tar}NonProt', convert_status(nonprot_chk.get()))       
        Wait(1)
        Console("hidden")

    def H2O():
        """Toggle display of water molecules."""
        Console("OFF")
        tar = target()
        if ListObj(f'{tar}H2O', format='OBJNUM') == []:
            new = DuplicateObj(tar)[0]
            SwitchObj(new, 'ON')
            HideObj(new)
            HideSecStrObj(new)
            ShowRes(f'Obj {new} res hoh')
            NameObj(new, f'{tar}H2O')
        else:
            SwitchObj(f'{tar}H2O', convert_status(h2o_chk.get()))       
        Wait(1)
        Console("hidden")

    def Surf(*args, mind_console=True):
        """Create or update the molecular surface visualization object."""
        if mind_console:
            Console("OFF")
        tar = target()
        if ListObj(f'{tar}Surf') == []:
            new = DuplicateObj(tar)[0]
            SwitchObj(new, 'ON')
            HideObj(new)
            HideSecStrObj(new)
            NameObj(new, f'{tar}Surf')
            on_cut()
            return
        if surf_col.get() == 'choose..':
            col = ShowWin("ColorSelection","Select tunnel residues color", "Bow","Background","100")[0]
        elif surf_col.get() == 'outcol':
            col = 'white'
        else:
            col = surf_col.get()
        if surf_incol.get() == 'choose..':
            incol = ShowWin("ColorSelection","Select tunnel residues color", "Bow","Background","100")[0]
        elif surf_incol.get() == 'as outside':
            incol = 'atomcol'
        elif surf_incol.get() == 'incol':
            incol = '000001'
        elif surf_incol.get() == 'black':
            incol = '000001'
        else:
            incol = surf_incol.get()
        ColorRes(f'obj {tar}Surf', col)
        HideSurfObj(f'{tar}Surf')
        ShowSurfRes(f'obj {tar}Surf res protein', 'molecular', outcol='atomcol', outalpha=surf_col_alpha_chk.get(), incol=incol, inalpha=surf_incol_alpha_chk.get())
        SwitchObj(f'{tar}Surf', convert_status(surf_chk.get()))   
        Wait(1)
        if mind_console:
            Console("hidden")

    def _hide_atoms_chunked(atomnums, size=4000):
        """Hide a large list of atom numbers in bounded-size selection strings (one giant
        selection can exceed YASARA's parser limits)."""
        atomnums = [str(a) for a in atomnums]
        for i in range(0, len(atomnums), size):
            HideAtom('Atom ' + ' '.join(atomnums[i:i + size]))

    def _perf_hide_interior(tar, on, bspacing):
        """Points/balls rep: hide (on) / reveal (off) each tunnel's buried interior atoms.
        Non-destructive -- the atoms stay in the object, so pathfinding / cross-section
        still see them (they select all atoms regardless of visibility); they're just
        not drawn."""
        for o in ListObj(f'{tar}Cl???????'):
            if on and bspacing:
                p = np.array(PosAtom(f'obj {o}', coordsys='global')).reshape(-1, 3)
                atoms = np.array(ListAtom(f'obj {o}'))
                interior = atoms[~_shell_mask(p, bspacing)]
                ShowAtom(f'obj {o}')
                if len(interior):
                    _hide_atoms_chunked(interior.tolist())
            else:
                ShowAtom(f'obj {o}')

    def improve_performance():
        """Toggle non-destructive performance mode for the current scene.

        ON  -> opaque tunnels render as their surface shell only: sphere meshes rebuild
               shell-only (far fewer triangles) and points/balls hide interior atoms;
               stashed inactive-mode meshes are freed. The '...Cl...' cloud is untouched,
               so A* / cross-section / volume stay exact.
        OFF -> full detail restored.
        """
        Console("OFF")
        tar = target()
        if tar is None:
            Console("hidden"); return
        on = not perf_var.get()
        perf_var.set(on)
        perf_button.configure(text='Restore detail' if on else 'Improve performance')
        rep = radio_var.get()
        bspacing = _ball_spacing_for(tar)

        # measure how much is buried, for the report
        total = culled = 0
        if on and bspacing:
            for o in ListObj(f'{tar}Cl???????'):
                p = np.array(PosAtom(f'obj {o}', coordsys='global')).reshape(-1, 3)
                total += len(p); culled += int((~_shell_mask(p, bspacing)).sum())

        # reclaim memory: drop the stashed inactive colour-/shape-mode meshes (rebuilt on demand)
        if on:
            DelObj('sphT??? sphD??? shpV??? shpA??? shpM???')

        # apply to the active rep (mesh rebuilds honour perf via _sphere_cull_on())
        if rep in ('points', 'balls'):
            _perf_hide_interior(tar, on, bspacing)
        elif rep == 'spheres' and alpha_chk.get() >= 100:
            show_progress_spheres(new=True)      # only opaque spheres change (cull on/off)

        if on:
            if rep == 'spheres' and alpha_chk.get() < 100:
                ShowMessage('Performance mode ON, but spheres are transparent -- culling the '
                            'interior would show through, so the mesh is unchanged. Make '
                            'spheres opaque to lighten them.')
            elif rep == 'shape':
                ShowMessage('Performance mode ON: freed cached meshes. The Shape rep is '
                            'already a compact surface, so its geometry is unchanged.')
            elif bspacing and total:
                ShowMessage(f'Performance mode ON: {culled:,} of {total:,} buried points '
                            f'({100 * culled / total:.0f}%) dropped from rendering. '
                            f'Pathfinding / cross-section data unchanged.')
            else:
                ShowMessage('Performance mode ON.')
        else:
            ShowMessage('Performance mode OFF -- full detail restored.')
        Wait(30)
        HideMessage()
        Wait(1)
        Console("hidden")

    def Exit():
        """Save an exit snapshot, cleaning up first. The derived sphere/shape MESH objects
        can be hundreds of MB of triangles and make SaveSce slow, so drop them before
        saving -- they rebuild from the clusters on demand. The tunnels are kept as points
        (preserving which were visible) so a reopened scene still shows them."""
        Console("OFF")
        tar = target()
        if tar is None:
            Console("hidden")
            return
        # tunnels currently visible, from whichever rep is active (clusters when
        # points/balls, meshes when spheres/shapes)
        on_nums = set(f'{o:03d}' for o, v in
                      zip(ListObj(f'{tar}Cl???????'), SwitchObj(f'{tar}Cl???????')) if v == 'On')
        on_nums |= set(nm.split('_')[0] for nm, v in
                       zip(NameObj('???_Sphere ???_shape'), SwitchObj('???_Sphere ???_shape')) if v == 'On')
        DelObj('???_sphere ???_shape sphT??? sphD??? shpV??? shpA??? shpM???')
        for o in ListObj(f'{tar}Cl???????'):
            SwitchObj(o, 'On' if f'{o:03d}' in on_nums else 'Off')
        Console("hidden")
        SaveSce(f'{NameObj(tar)[0]}_tunnels_exit.sce')

    def Balls():
        """Switch tunnel display to ball-stick mode."""
        Console("OFF")
        on_spheres = [x for x,y in zip(NameObj('???_Sphere'), SwitchObj('???_Sphere')) if y == 'On']
        if len(on_spheres) == 0:
            on_spheres = [x for x,y in zip(NameObj('???_shape'), SwitchObj('???_shape')) if y == 'On']
        add_tunnels = [match.group() for x in on_spheres for match in [re.search('[0-9]+', x)] if match]
        SwitchObj(" ".join(add_tunnels), 'on')
        SwitchObj(" ".join(on_spheres), 'off')
        SwitchObj('???_shape', 'off')
        BallAtom(f'Obj {target()}Cl???????')
        Wait(1)
        Console("hidden")

    def Points():
        """Switch tunnel display to stick (point) mode."""
        Console("OFF")
        on_spheres = [x for x,y in zip(NameObj('???_Sphere'), SwitchObj('???_Sphere')) if y == 'On']
        if len(on_spheres) == 0:
            on_spheres = [x for x,y in zip(NameObj('???_shape'), SwitchObj('???_shape')) if y == 'On']
        add_tunnels = [match.group() for x in on_spheres for match in [re.search('[0-9]+', x)] if match]
        SwitchObj(" ".join(add_tunnels), 'on')
        SwitchObj(" ".join(on_spheres), 'off')
        SwitchObj('???_shape', 'off')
        StickAtom(f'Obj {target()}Cl???????')
        Wait(1)
        Console("hidden")

    def build_sphere_objects(progress=False, force_show=False):
        """(Re)build all tunnel sphere-mesh objects from the clusters' CURRENT colours.

        Mesh spheres bake their colour in at load time (LoadWOb) and can't be recoloured
        in place, so a colour-mode change rebuilds them. The point positions are unchanged
        but the per-colour grouping is not, so it is a full (fast, ~seconds) mesh rebuild.
        Assumes the console is already OFF (callers manage it).

        force_show=True makes every rebuilt tunnel visible regardless of its cluster's
        current on/off state -- used when rebuilding for a colour switch, where the
        clusters are hidden (spheres are the active view) so their state would wrongly
        hide the new spheres."""
        # A rebuild (e.g. an alpha/size slider release) happens while spheres are the
        # active view, so the clusters are hidden and their on/off state is a misleading
        # 'Off'. Read the intended visibility from the spheres we're about to replace
        # instead (keyed by the 3-digit tunnel prefix), falling back to the cluster state
        # on first build.
        #
        # Anti-flicker: rather than delete ALL old spheres up front (leaving the scene
        # blank while everything rebuilds), each tunnel's old mesh is kept visible until
        # its replacement is loaded, then swapped in place (see the per-tunnel tail).
        prev_on = {nm[:3]: st for nm, st in
                   zip(NameObj('???_Sphere'), SwitchObj('???_Sphere'))}
        tunnel_objs = ListObj(f'{target()}Cl???????')
        # Performance mode shell-cull: precompute each tunnel's surface positions +
        # mask ONCE so the triangle budget, the progress bar and the per-tunnel loop
        # all use the culled counts (and no PosAtom is fetched twice). The source
        # cluster is never modified -- only these local arrays are filtered.
        cull = _sphere_cull_on()
        bspacing = _ball_spacing_for(target()) if cull else None
        cull_pos, cull_mask = {}, {}
        if cull and bspacing:
            for targetobj in tunnel_objs:
                pc = np.array(PosAtom(f'obj {targetobj}', coordsys='global')).reshape(-1, 3)
                cull_pos[targetobj] = pc
                cull_mask[targetobj] = _shell_mask(pc, bspacing)
            total_spheres = int(sum(m.sum() for m in cull_mask.values()))
        else:
            total_spheres = CountAtom(f'Obj {target()}Cl???????')
        # Pick the smoothest sphere tessellation whose total triangle count stays
        # within a safe budget. Faces per ShowSphere-equivalent level: 0->20, 1->80,
        # 2->320, 3->1280. YASARA OOMs loading a mesh well beyond ~20M triangles, so
        # small tunnels get round L3/L2 spheres while only very large ones step down.
        _FACES = {0: 20, 1: 80, 2: 320, 3: 1280}
        sphere_level = next((lv for lv in (3, 2, 1, 0)
                             if total_spheres * _FACES[lv] <= 20000000), 0)
        _roundness = {0: 'low', 1: 'reduced', 2: 'high', 3: 'maximum'}[sphere_level]
        # The .obj write dominates the build for big tunnels; the 'fast' checkbox drops
        # per-vertex normals to ~halve it (flat shading via YASARA's per-face normals --
        # brighter/flatter, best on the small dense spheres of a large tunnel).
        smooth_normals = not fast_chk.get()
        ShowMessage(f"Creating {total_spheres:,} spheres at {_roundness} roundness "
                    f"(level {sphere_level}/3"
                    f"{'' if smooth_normals else ', flat shading (fast)'}).")
        Wait(1)
        radius = rad_chk.get() / 100 * 2.5
        alpha = alpha_chk.get()
        # Geometry cache: the .obj files depend on positions+radius+level (+colour
        # partition in distance mode), NOT on alpha. If those are unchanged since the
        # last build, reuse the files on disk and skip the (dominant) write -- only the
        # LoadWOb parse re-runs, applying the new alpha/colour. Positions change only on
        # re-predict/recluster, which call _sph_obj_cache_clear() to reset this.
        mode = radio_col_var.get()
        mode_tag = 'd' if mode == 'distance' else 't'
        geo_sig = _sph_geo_sig(mode, sphere_level)
        reuse = (geo_sig == _sph_state.get('geo'))
        _shade = 'smooth' if smooth_normals else 'flat/fast'
        _vpr = len(_icosphere(sphere_level)[0])
        _WRITE_LAST[0] = 'reusing cached geometry' if reuse else 'idle'
        done = 0
        for targetobj in tunnel_objs:
            col = _cluster_colors(targetobj)   # band-sourced (uniform type) in distance mode
            # Performance mode: build only the visible surface shell of an opaque tunnel
            # (buried interior spheres add triangles but are never seen). Reuse the
            # precomputed positions + mask; the source cluster keeps all its atoms.
            if cull and bspacing:
                p, col = cull_pos[targetobj][cull_mask[targetobj]], col[cull_mask[targetobj]]
            else:
                p = np.array(PosAtom(f'obj {targetobj}', coordsys='global')).reshape(-1, 3)
            ns = len(p)
            # intended write mode, shown live during the (possibly slow) build; the
            # 'built' message afterwards reports what actually happened.
            if reuse:
                wmode = 'reusing cache'
            elif not _MESHWORKER_OK:
                wmode = 'single-thread (worker not found)'
            elif ns * _vpr < _PARALLEL_WRITE_MIN_VERTS:
                wmode = 'single-thread (small)'
            else:
                wmode = f'multithread ({_WRITE_NPROC}w)'
            ShowMessage(f"Creating {ns:,} spheres of tunnel {NameObj(targetobj)[0]} at "
                        f"{_roundness} roundness (level {sphere_level}/3, {_shade}, {wmode}"
                        f"{', shell only' if (cull and bspacing) else ''})")
            Wait(1)
            on_off = 'On' if force_show else prev_on.get(f'{targetobj:03d}', SwitchObj(targetobj)[0])
            SwitchObj(targetobj, 'Off')
            DelObj('sphere')   # clear stray temp meshes only; the OLD tunnel mesh stays until swapped

            # One polygon-mesh object per distinct colour (LoadWOb takes a single
            # colour), each renamed 'sphere'; grouping keeps the whole build to a
            # handful of file loads instead of ~2 YASARA calls per point.
            order = np.argsort(col, kind='stable')
            p_s, col_s = p[order], col[order]
            uniq, starts = np.unique(col_s, return_index=True)
            bounds = list(starts) + [len(col_s)]
            for i, cv in enumerate(uniq):
                fpath = os.path.join(_SPH_OBJ_DIR, f'sph_{mode_tag}_{targetobj:03d}_{i}.obj')
                o = _load_sphere_mesh(p_s[bounds[i]:bounds[i + 1]], radius, _ycolor(cv),
                                      alpha, sphere_level, path=fpath, reuse=reuse,
                                      normals=smooth_normals)
                NameObj(o, 'sphere')
                if progress:
                    frac = (done + bounds[i + 1]) / total_spheres * 100
                    progress_var.set(frac)
                    percent_label.config(text=f'{frac:.0f}%')
                    # build runs on the main thread (see show_progress_spheres); tick the
                    # elapsed clock manually (after() is blocked) and repaint without
                    # processing input events (no reentrancy).
                    try:
                        progress_window._elapsed_manual()
                        progress_window.update_idletasks()
                    except Exception: pass
            done = done + len(p)

            # Join this tunnel's per-colour meshes into one object (center='No' avoids
            # the O(N^2) re-centering the default Center=Yes would do on each join).
            jobj = ListObj('sphere')[0]
            JoinObj('sphere', jobj, center='No')
            SwitchObj(jobj, on_off)
            # Swap in place: the new mesh (still named 'sphere') is loaded and visible, so
            # only now delete this tunnel's OLD mesh and give the new one its name. Other
            # tunnels keep their old meshes shown -> no blank flash.
            DelObj(f'{targetobj:03d}_sphere')
            NameObj(jobj, f'{targetobj:03d}_sphere')
        # the .obj files now on disk match this signature; a later build with the same
        # signature (e.g. only alpha changed) can reuse them.
        _sph_state['geo'] = geo_sig

    def Spheres(new=False, progress=False):
        """Switch tunnel display to sphere mode (one mesh sphere per tunnel point)."""
        Console("OFF")   # load-bearing: with the console ON every LoadWOb/PosObj prints,
                         # making the build dramatically slower (see build_sphere_objects)
        if ListObj('???_Sphere') != [] and not new:
            on_tunnels = [x for x,y in zip(ListObj(f'{target()}Cl???????'), SwitchObj(f'{target()}Cl???????')) if y == 'On']
            if len(on_tunnels) == 0:
                on_tunnels = [int(x[2:3]) for x,y in zip(NameObj('???_shape'), SwitchObj('???_shape')) if y == 'On']
            SwitchObj(" ".join([str(f'{x:03d}') + '_Sphere' for x in on_tunnels]), "on")
        else:
            _sph_cache_clear()                       # a fresh build invalidates cached sets
            build_sphere_objects(progress)
            _sph_state['mode'] = radio_col_var.get()   # remember what colouring this set has
            _sph_state['sig'][_sph_state['mode']] = _sph_sig(_sph_state['mode'])

        SwitchObj(f'{target()}Cl???????', 'off')
        SwitchObj('???_shape', 'off')
        HideMessage()
        Wait(1)
        Console("hidden")

    # --- Sphere colour-mode cache -------------------------------------------------
    # Mesh spheres bake their colour in (LoadWOb), so switching Colour by Tunnel<->
    # Distance needs a different mesh set. Rather than rebuild every time, keep the
    # inactive mode's set hidden and renamed (sphT###/sphD### -- deliberately NOT
    # matching the plugin's '???_sphere' selectors) and just swap, rebuilding only when
    # the geometry/colour settings for that mode actually changed.
    # 'mode' = colouring of the visible set; 'geo' = signature of the .obj files
    # currently on disk (see _sph_geo_sig / the geometry cache in build_sphere_objects).
    _sph_state = {'mode': None, 'sig': {}, 'geo': None}
    _SPH_SUF = {'tunnel': 'T', 'distance': 'D'}

    # Cross-section slider cache: a tunnel's point positions + atom numbers do NOT
    # change while the diameter slider is dragged, so fetch them from YASARA once per
    # tunnel selection instead of on every tick (PosAtom/ListAtom over all N points
    # were the per-move O(N) hotspots at dense spacing). 'cutp' tracks the atoms marked
    # as the 'cutp' segment last move, so SegAtom can be reset incrementally (O(k))
    # rather than re-stamping all N points each move. Cleared on new prediction,
    # recluster, and tunnel re-selection (see _xsec_cache_clear call-sites).
    #
    # 'frame' guards against scene ROTATION/MOVE: the positions are GLOBAL (screen)
    # coords, which change when the user rotates the scene. If we kept stale positions,
    # the next slider move would test them against the freshly-rotated cutting plane and
    # select a garbage 'near' set (points fanning off the plane). We re-fetch whenever the
    # tunnel object's PosOriObj signature changes -- a cheap O(1) check per move, O(N)
    # refetch only on an actual frame change.
    _xsec_cache = {'tnl': None, 'frame': None, 'pos': None, 'atoms': None, 'cutp': None}

    def _xsec_cache_clear():
        _xsec_cache.update(tnl=None, frame=None, pos=None, atoms=None, cutp=None)

    def _xsec_get(tnl_name):
        """(positions Nx3, atom-numbers) for a tunnel, cached across slider moves but
        refetched when the tunnel's coordinate frame (rotation/position) changes."""
        frame = _obj_frame_sig([ListObj(tnl_name)[0]])
        if (_xsec_cache['tnl'] != tnl_name or _xsec_cache['frame'] != frame
                or _xsec_cache['pos'] is None):
            _xsec_cache['tnl'] = tnl_name
            _xsec_cache['frame'] = frame
            _xsec_cache['pos'] = np.array(PosAtom(f'Obj {tnl_name}', coordsys='global')).reshape(-1, 3)
            _xsec_cache['atoms'] = np.array(ListAtom(f'Obj {tnl_name}'))
            _xsec_cache['cutp'] = None   # nothing marked yet for this frame
        return _xsec_cache['pos'], _xsec_cache['atoms']
    # When set, a sphere rebuild inside recolor_spheres_for_mode drives the progress popup
    # created by _recolor_with_progress (used for the slow distance recolour in sphere mode).
    _recolor_prog = {'on': False}

    def _sph_sig(mode):
        """Signature of a sphere set: rebuild is needed whenever this changes. In distance
        mode this includes the palette + handle window (they change the baked-in colours),
        so switching palette forces a rebuild instead of a false 'already correct' hit."""
        base = f'r{rad_chk.get()}|a{alpha_chk.get()}'
        if mode == 'tunnel':
            return f'tunnel|{base}|s{step_entry.get()}'
        tar = target()
        return (f'distance|{base}|{PairObj(tar, "dist_sel")}|{PairObj(tar, "dist_sig")}'
                f'|{dist_palette_var.get()}|{dist_grad["t0"]:.3f}|{dist_grad["t1"]:.3f}')

    def _obj_frame_sig(objs):
        """Compact signature of objects' current position+orientation (PosOriObj). Mesh
        geometry is baked in GLOBAL (screen-frame) coords, which change when the user
        rotates or moves the structure, so a cache is only valid while this is unchanged;
        otherwise the reused mesh is baked at a stale orientation and appears offset."""
        return ';'.join(str(o) + ':' + ','.join(f'{v:.2f}' for v in PosOriObj(o)) for o in objs)

    def _sph_geo_sig(mode, level):
        """Signature of the on-disk .obj GEOMETRY. Excludes alpha (only a LoadWOb arg)
        and, for tunnel mode, the colour step (one block per tunnel regardless), so an
        alpha or tunnel-colour change is a cache HIT that skips the dominant file write.
        Radius and tessellation level DO change vertex data, so they're included, as does
        the object frame (globals go stale on rotate/move -- see _obj_frame_sig) and the
        'fast' flag (flat vs smooth changes what's written to the .obj)."""
        base = f'r{rad_chk.get()}|L{level}|f{int(fast_chk.get())}|c{int(_sphere_cull_on())}'
        frame = _obj_frame_sig(ListObj(f'{target()}Cl???????'))
        if mode == 'tunnel':
            return f'tunnel|{base}|{frame}'
        tar = target()
        # Distance-mode sphere .obj files are grouped BY COLOUR, and the palette/window
        # decide both the colours AND the argsort order that indexes those files -- so
        # unlike the point cache (which keys on the palette-independent bands via dist_sig),
        # the geometry cache MUST include the palette + handle window, or a palette switch
        # would reuse a file whose colour-group no longer matches.
        return (f'distance|{base}|{PairObj(tar, "dist_sel")}|{PairObj(tar, "dist_sig")}'
                f'|{dist_palette_var.get()}|{dist_grad["t0"]:.3f}|{dist_grad["t1"]:.3f}|{frame}')

    def _sph_cache_clear():
        """Invalidate the colour-mode stash (does NOT touch the .obj geometry files,
        which survive a rebuild so an alpha/colour change can reuse them)."""
        DelObj('sphT??? sphD???')
        _sph_state['mode'] = None
        _sph_state['sig'] = {}

    def _sph_obj_cache_clear():
        """Drop the on-disk .obj geometry cache. Called only when the point positions
        themselves change (re-predict / recluster) -- radius/alpha/colour changes reuse
        the files instead. Crash leftovers are handled by the load-time dir wipe."""
        shutil.rmtree(_SPH_OBJ_DIR, ignore_errors=True)
        os.makedirs(_SPH_OBJ_DIR, exist_ok=True)
        _sph_state['geo'] = None

    def recolor_spheres_for_mode(mode):
        """Called when the colour mode changes to `mode`. If spheres are the active view,
        show a cached set for that mode when its settings are unchanged, else rebuild.
        If spheres aren't shown, drop stale sets so the next Spheres build is fresh."""
        if radio_var.get() != 'spheres':
            DelObj('???_sphere sphT??? sphD???')
            _sph_state['mode'] = None
            _sph_state['sig'] = {}
            return
        Console("OFF")
        cur = _sph_state['mode']
        tsig = _sph_sig(mode)
        if cur == mode:
            if _sph_state['sig'].get(mode) == tsig:
                HideMessage(); Wait(1); Console("hidden"); return   # already correct
            DelObj('???_sphere')
            build_sphere_objects(force_show=True, progress=_recolor_prog['on'])
        else:
            if cur is not None and ListObj('???_sphere') != []:     # stash outgoing set
                for o in ListObj('???_sphere'):
                    num = NameObj(o)[0].split('_')[0]
                    NameObj(o, f'sph{_SPH_SUF[cur]}{num}'); SwitchObj(o, 'Off')
            if _sph_state['sig'].get(mode) == tsig and ListObj(f'sph{_SPH_SUF[mode]}???') != []:
                for o in ListObj(f'sph{_SPH_SUF[mode]}???'):        # CACHE HIT: reuse
                    num = NameObj(o)[0][3 + len(_SPH_SUF[mode]):]
                    NameObj(o, f'{num}_sphere'); SwitchObj(o, 'On')
            else:
                DelObj(f'sph{_SPH_SUF[mode]}???')                    # stale -> rebuild
                build_sphere_objects(force_show=True, progress=_recolor_prog['on'])
        _sph_state['mode'] = mode
        _sph_state['sig'][mode] = tsig
        SwitchObj(f'{target()}Cl???????', 'off')
        SwitchObj('???_shape', 'off')
        HideMessage(); Wait(1); Console("hidden")

    def _dist_bands(disto, mind, maxd):
        """Quantise distances to band indices 0..DIST_BANDS-1 (normalised to [mind,maxd]).
        Bands are palette/window-INDEPENDENT -- they're what gets cached in SegAtom, so a
        palette or handle change just remaps bands->colour without recomputing distances."""
        d = np.asarray(disto, float)
        span = (maxd - mind) or 1.0
        tn = np.clip((d - mind) / span, 0.0, 1.0)
        return np.rint(tn * (DIST_BANDS - 1)).astype(int).tolist()

    def _band_color(b):
        """Map a band index to the current palette's colour (native int or 'rrggbb' hex),
        applying the handle sub-window [t0,t1]."""
        t0, t1 = dist_grad['t0'], dist_grad['t1']
        u = t0 + (b / (DIST_BANDS - 1)) * (t1 - t0)
        return _palette_color(dist_palette_var.get(), max(0.0, min(1.0, u)))

    def _cluster_colors(targetobj):
        """Per-point colour specs for a cluster's atoms, used to bake the sphere/shape
        meshes. In distance mode we derive them from the cached BANDS via _band_color -- a
        uniform-typed array (all hex for a cmap palette, all int for hue) -- rather than
        reading ColorAtom back (which returns YASARA's snapped hex/gray-circle MIX and would
        break grouping/int-casts). Otherwise (tunnel mode) the atoms' own int colours."""
        if radio_col_var.get() == 'distance':
            segs = SegAtom(f'obj {targetobj}')
            if segs and all(s[:1] == 'c' and s[1:].isdigit() for s in segs):
                return np.array([_band_color(int(s[1:])) for s in segs], dtype=object)
        return np.array(ColorAtom(f'obj {targetobj}'))

    def group_and_color(atomlist, collist, mind_console=True):
        """Batch-colour atoms grouped by their distance BAND (far faster than per-atom).
        collist holds band indices (as stored in SegAtom 'c<band>'); each is mapped to the
        current palette's colour via _band_color, so a palette/window change recolours
        instantly from the cached bands, no distance recompute.

        Drives the recolour progress popup (per band group) when it's active -- this is the
        dominant cost of a points/balls recolour. Skipped in sphere mode, where the sphere
        BUILD drives the bar instead (so it doesn't get filled twice)."""
        if mind_console:
            Console("OFF")
        from collections import defaultdict
        grouped_items = defaultdict(list)
        for atoms, band in zip(atomlist, collist):
            grouped_items[band].append(atoms)
        tick = _recolor_prog['on'] and radio_var.get() != 'spheres'
        items = list(grouped_items.items())
        n = len(items)
        for k, (band, atoms) in enumerate(items):
            ColorAtom(" ".join(str(x) for x in atoms), _band_color(int(band)))
            if tick and n:
                try:
                    pct = (k + 1) / n * 100
                    progress_var.set(pct)
                    percent_label.config(text=f'{pct:.0f}%')
                    progress_window._elapsed_manual()
                    progress_window.update_idletasks()
                except Exception:
                    pass

          
    def Colorbytunneldist(shapes=True, prompt_if_unset=True):
        """Color tunnel points by their distance to a user-selected reference atom/center.

        prompt_if_unset: when True (radiobutton / select button) and no reference atom
        has been chosen yet, pop the atom picker. When False (the 'calc per tunnel'
        toggle) just do nothing if nothing is selected -- toggling the scaling mode
        must never prompt for an atom.
        """
        Console("OFF")
        tar = target()
        objs = [str(x) for x in ListObj(f'{target()}Cl???????')]

        # if the distance center is the same as before, we can reuse the color stored in SegAtom
        dist_sel = PairObj(tar, 'dist_sel')
        if dist_sel == []:
            if not prompt_if_unset:
                Console("hidden")
                return
            atms = SelectDistAtom(win=True)
            dist_sel = PairObj(tar, 'dist_sel')
        else:
            atms = SelectDistAtom(win=False)
        # Capture the object's key-value pairs AFTER the reference atom is picked, so a
        # freshly-selected dist_sel is included. The recompute 'dance' below (JoinObj/
        # DelObj on the target object) wipes the object's pairs and the restore at the
        # end re-adds save_pairs; capturing before selection lost a just-picked dist_sel
        # (the radiobutton first-time path), so the next 'calc per tunnel' toggle saw no
        # selection and silently did nothing.
        save_pairs = PairObj(tar)
        dist_col = PairObj(tar, 'dist_col')
        # What gets cached in SegAtom is the distance BAND per atom, which depends only on
        # the reference atom (dist_sel==dist_col) and the scaling mode ('calc per tunnel'
        # normalises per tunnel vs globally) -- NOT on the palette or handle window (those
        # are applied at band->colour time). So the signature keys on per_tunnel only, and
        # a palette/handle change takes the fast cache path below (remap, no recompute).
        per_tunnel = pertun_chk.get()
        cur_sig = f"{int(per_tunnel)}"
        if dist_col != [] and dist_sel == dist_col and PairObj(tar, 'dist_sig') == [cur_sig]:
            atomlist = ListAtom(f'obj {tar}Cl???????')
            collist = [x[1:] for x in SegAtom(f'obj {tar}Cl???????')]
            group_and_color(atomlist, collist)
            HideMessage()
            Wait(1)
            if ListObj('???_shape') != []:
                if shape_surf_option.get() == 'MergeSph':
                    Shapes(new=True)   # merged-sphere shape is a mesh (no atoms) -> rebuild
                else:
                    _color_shapes_from_clusters()   # colours (auto-rebuilds if out of sync)
            recolor_spheres_for_mode('distance')   # mesh spheres bake in colour -> swap/rebuild set
            Console("hidden")
            return

        # if the distance center is new, we calculate all distances
        ShowMessage('Coloring by distance')
        Wait(1)
        start_time = time.perf_counter()
        from collections import defaultdict

        if len(atms) > 1:
            ShowMessage('Creating center helper object')
            Wait(1)
            c = DuplicateAtom(" ".join([str(x) for x in atms]))
            JoinObj(" ".join(str(i) for i in c), c[0])
            cx,cy,cz = PosAtom("obj " + str(c[0]), mean=True, coordsys='global')
            cen = BuildAtom("C")
            NameObj(cen, 'CenterHlp')
            PosAtom("obj " + str(cen), x = cx,y = cy, z = cz, coordsys='global')
            DelObj(c[0])
            center=str(ListAtom('Obj ' + str(cen), format='ATOMNUM')[0])
        else:
            center = " ".join(atms)

        # if calculating for all points, check min and max distance first
        if not per_tunnel:
            ShowMessage('Getting min and max distance')
            Wait(1)
            
            atm_obj = ListObj('atom ' + center, format='OBJNUM')[0]
            mind = ListAtom(f'obj {" ".join(objs)} with minimum distance from atom {center}')[0]
            x = DuplicateAtom(mind)[0]
            SwapAtom('obj ' + str(x), 'Du')
            JoinObj(x, atm_obj)
            mind = Distance(f'Obj {atm_obj} element Du', f'Obj {atm_obj} atom {center}')[0]
            DelAtom(f'Obj {atm_obj} element Du')
            maxd = ListAtom(f'obj {" ".join(objs)} with maximum distance from atom {center}')[0]
            x = DuplicateAtom(maxd)[0]
            SwapAtom('obj ' + str(x), 'Du')
            JoinObj(x, atm_obj)
            maxd = Distance(f'Obj {atm_obj} element Du', f'Obj {atm_obj} atom {center}')[0]
            DelAtom(f'Obj {atm_obj} element Du')

        dist_target = ListObj(f'atom {center}', format='OBJNUM')[0]
        for tunnel in objs:
            tname = ListObj(tunnel, format='OBJNAME')[0]

            n = DuplicateObj(tunnel)[0]
            natoms = CountAtom("obj " + str(n))

            SwapAtom(f'Obj {n}', "Du")
            JoinObj(n, dist_target)
            if natoms > 1000:
                ShowMessage(f'Obj {tunnel}: getting distances of {natoms} points')
                Wait(1)

            disto = Distance(f'obj {dist_target} element Du', center)

            new = DuplicateAtom(f'obj {dist_target} element Du')
            DelAtom(f'obj {dist_target} element Du')
            NameObj(new, 'coltunnel')
            on_off = SwitchObj(tunnel)[0]
            SwitchObj(new, on_off)
            DelObj(tunnel)
            RenumberObj(n, tunnel)
            NameObj(tunnel, tname)
            SwapAtom(f'obj {tunnel}', 'H', rename=False)

            atomlist = ListAtom(f'obj {tunnel}')

            if natoms > 1000:
                ShowMessage(f'Obj {tunnel}: Coloring {natoms:,} points')
                Wait(1)

            if per_tunnel:
                mind = min(disto)
                maxd = max(disto)
            
            all_bands = _dist_bands(disto, mind, maxd)

            # Group atoms by their distance BAND and issue one ColorAtom + one SegAtom per
            # band instead of per atom. Batched (per-atom ColorAtom degrades superlinearly
            # on large tunnels -- each single-atom selection is re-resolved against the whole
            # object, ~558s vs ~0.4s at 488k points), and bounded at DIST_BANDS groups. The
            # SegAtom stores the band ('c<band>', palette-independent) so an unchanged
            # reference is reused via the fast cache path at the top of this function -- a
            # palette/handle change just remaps bands->colour there, no distance recompute.
            grouped = defaultdict(list)
            for atom, band in zip(atomlist, all_bands):
                grouped[band].append(atom)
            for band, atoms in grouped.items():
                sel = " ".join(str(x) for x in atoms)
                ColorAtom(sel, _band_color(band))
                SegAtom(sel, f'c{band}')

        if ListObj('???_shape') != []:
            if shape_surf_option.get() == 'MergeSph':
                Shapes(new=True)   # merged-sphere shape is a mesh (no atoms) -> rebuild
            else:
                _color_shapes_from_clusters()   # colours (auto-rebuilds if out of sync)

        for i in range(0, len(save_pairs), 2):
            PairObj(tar, save_pairs[i], save_pairs[i + 1])

        PairObj(tar, 'dist_col', dist_sel[0])
        PairObj(tar, 'dist_sig', cur_sig)
        DelObj('CenterHlp')
        recolor_spheres_for_mode('distance')   # mesh spheres bake in colour -> swap/rebuild set
        HideMessage()
        Wait(1)
        Console("hidden")


    def SecStr(*args):
        """Toggle secondary structure visualization (ribbon/cartoon/tube/trace)."""
        Console("OFF")
        tar = target()
        if ss_chk.get():
            SwitchObj(tar, 'OFF')
            DelObj(f'{tar}SS')
            new = DuplicateObj(tar)[0]
            SwitchObj(new, 'ON')
            NameObj(new, f'{tar}SS')
            HideObj(new)
            if ss_style.get() != 'Trace':
                ShowSecStrObj(new, ss_style.get())
            else:
                HideSecStrObj(new)
                ShowTrace(f'obj {new} atom CA')
                HideAtom(f'obj {new}')
                ShowAtom(f'obj {new} atom CA')
                BallStickAtom(f'obj {new} atom CA')
            ss_col_change()
        else:
            SwitchObj(f'{tar}SS', 'OFF')
  
        Wait(1)
        Console("hidden")


    # Variable to control the loop
    continue_loop = True

    def switch_status(obj):
        Console("OFF")
        if obj == None:
            return None
        # Query the visibility with NO visibility argument -- SwitchObj(sel) just
        # returns the ['On'/'Off', ...] states. Do NOT pass 'OnOff': that is not a
        # query, it tells YASARA to step through the objects switching them on and off
        # sequentially as a never-ending animation, which keeps running even after the
        # plugin window is closed (it is a YASARA-side animation, not plugin code).
        stat = SwitchObj(obj)
        bool_stat = [True if x == 'On' else False for x in stat]
        return any(bool_stat)

    def on_cancel():
        Console("OFF")
        nonlocal continue_loop
        Exit()
        continue_loop = False
        root.destroy()

    # --------------------------------------------------------
    #  DIALOG SETUP — Root window, notebook, config
    # --------------------------------------------------------

    root = tk.Tk()
    root.title("Tunneler Customization Menu")
    root.attributes("-topmost", True)
    root.geometry(f"+{root.winfo_x()}+{int(root.winfo_y() +55)}")
    # Route the window-manager close (X button) through the same clean shutdown as the
    # Exit button, so it saves the scene and lets the plugin end properly instead of
    # yanking the window out from under the mainloop.
    root.protocol("WM_DELETE_WINDOW", on_cancel)
    # macOS Cmd-Q hits Tk's Apple-menu Quit, which tears down the Tcl interpreter before
    # atexit/WM_DELETE_WINDOW can run -> route it through on_cancel too. Harmless (and a
    # no-op) on platforms without the mac-specific command.
    try:
        root.createcommand('::tk::mac::Quit', on_cancel)
    except tk.TclError:
        pass

    initializing = True

    # --- Performance mode (non-destructive) ------------------------------------
    # When on, opaque renders drop the buried interior of each tunnel: sphere
    # meshes are built from the surface shell only (far fewer triangles) and the
    # points/balls rep hides interior atoms. The source '...Cl...' cloud is never
    # modified, so pathfinding / cross-section / volume (which read all atoms)
    # stay exact. alpha_chk / radio_var are defined later but resolved lazily.
    perf_var = tk.BooleanVar(value=False)

    def _sphere_cull_on():
        """Shell-cull sphere meshes only when perf mode is on AND they're opaque
        (a transparent tube would reveal the hollow interior)."""
        return perf_var.get() and alpha_chk.get() >= 100

    def _ball_spacing_for(tar):
        """Stored grid spacing of the tunnel cloud, or None if unavailable."""
        bs = PairObj(tar, 'ball_spacing') or PairObj('All', 'ball_spacing')
        try:
            return float(bs[0])
        except (IndexError, ValueError, TypeError):
            return None

    # Create the Notebook widget
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill='both', padx=0)


    # get previous settings
    def get_config(config_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Tunneler_config.ini')):
        """Load tunnel parameters from the INI config file into global variables."""
        # `global` must cover BOTH branches: the else-branch previously assigned
        # module-local names that vanished, leaving the globals unset (NameError
        # on a fresh install with no config yet).
        global ignore_surface, ball_spacing, max_ball_protein, surf_con_prev, keep_surf_points, mds, min_vol, connect_cut, build_pol, prog
        if os.path.exists(config_file):
            config = ConfigParser()
            config.read(config_file)
            v = config['Variables']
            ignore_surface = float(v['ignore_surface'])
            ball_spacing = float(v['ball_spacing'])
            max_ball_protein = float(v['max_ball_protein'])
            surf_con_prev = float(v['surf_con_prev'])
            # getboolean parses 'True'/'False' strings; plain bool('False') is
            # always True, which silently forced these flags on.
            keep_surf_points = config.getboolean('Variables', 'keep_surf_points')
            mds = int(v['mds'])
            min_vol = float(v['min_vol'])
            connect_cut = int(v['connect_cut'])
            build_pol = config.getboolean('Variables', 'build_pol')
            prog = str(v['prog'])
        else:
            ignore_surface, ball_spacing, max_ball_protein, surf_con_prev, keep_surf_points, mds, min_vol, connect_cut, build_pol, prog = 3.8, 0.33, 2.8, 2.7, False, 0, 5, 1, True, 'vis'
    
    get_config()
    style = ttk.Style()
    current_theme = root.tk.call("ttk::style", "theme", "use")
    style.theme_create( "MyStyle", parent=current_theme, settings={
            "TNotebook.Tab": {"configure": {"padding": [10, 10, 10, 10] },},
            "TNotebook": {"configure": {"tabmargins": [5, 0, 5, 5] } },
            })
    style.theme_use("MyStyle")

    # --------------------------------------------------------
    #  TAB 1 — CREATE TUNNELS
    #  Widgets: parameter sliders, target dropdown, residue
    #  exclusion listbox, find-tunnels button
    #  Callbacks: run_tun, show_progress_tunneler, reset
    # --------------------------------------------------------
    tab1_mktun = ttk.Frame(notebook)
    tab1_mktun.configure(height=375, width=310)  # Set dimensions as needed
    notebook.add(tab1_mktun, text='Create Tunnels', padding=0)  # Add tab1_mktun as the second tab

    def update_label(var, label, n=1):
        """Update the label with the value of the variable."""
        # The value is retrieved from the passed variable and set on the passed label.
        label.config(text=f"{var.get():.{n}f}")

 
    ign_surf_scale_chk = tk.DoubleVar(value=ignore_surface)
    ign_surf_value_label = tk.Label(tab1_mktun, text=f"{ign_surf_scale_chk.get():.1f}")
    ign_surf_scale = ttk.Scale(tab1_mktun, from_=0, to=10, orient="horizontal", variable=ign_surf_scale_chk,
                            command=lambda value, var=ign_surf_scale_chk, label=ign_surf_value_label: update_label(var, label))
    ign_surf_label = tk.Label(tab1_mktun, text=f"Ignore surface up to (\u212B)")
    update_label(ign_surf_scale_chk, ign_surf_value_label)


    surf_con_scale_chk = tk.DoubleVar(value=surf_con_prev)
    surf_con_value_label = tk.Label(tab1_mktun, text=f"{surf_con_scale_chk.get():.1f}")
    surf_con_scale = ttk.Scale(tab1_mktun, from_=0, to=10, orient="horizontal", variable=surf_con_scale_chk,
                            command=lambda value, var=surf_con_scale_chk, label=surf_con_value_label: update_label(var, label))
    surf_con_label = tk.Label(tab1_mktun, text=f"Prevent surface connect (\u212B)")
    update_label(surf_con_scale_chk, surf_con_value_label)


    prot_space_scale_chk = tk.DoubleVar(value=max_ball_protein)
    prot_space_value_label = tk.Label(tab1_mktun, text=f"{prot_space_scale_chk.get()}")
    prot_space_scale = ttk.Scale(tab1_mktun, from_=0, to=5, orient="horizontal", variable=prot_space_scale_chk,
                            command=lambda value, var=prot_space_scale_chk, label=prot_space_value_label: update_label(var, label, 1))
    prot_space_label = tk.Label(tab1_mktun, text=f"Ball-protein distance (\u212B)")
    update_label(prot_space_scale_chk, prot_space_value_label, 2)


    min_vol_scale_chk = tk.DoubleVar(value=min_vol)
    min_vol_value_label = tk.Label(tab1_mktun, text=f"{min_vol_scale_chk.get():.0f}")
    min_vol_scale = ttk.Scale(tab1_mktun, from_=0, to=200, orient="horizontal", variable=min_vol_scale_chk,
                            command=lambda value, var=min_vol_scale_chk, label=min_vol_value_label: update_label(var, label, 0))
    min_vol_label = tk.Label(tab1_mktun, text=f"Minimum cluster volume (\u212B\u00b3)")
    update_label(min_vol_scale_chk, min_vol_value_label, 0)


    num_md_scale_chk = tk.IntVar(value=mds)
    num_md_value_label = tk.Label(tab1_mktun, text=f"{num_md_scale_chk.get()}")
    num_md_scale = ttk.Scale(tab1_mktun, from_=0, to=10, orient="horizontal", variable=num_md_scale_chk,
                            command=lambda value, var=num_md_scale_chk, label=num_md_value_label: update_label(var, label, 0))
    num_md_label = tk.Label(tab1_mktun, text=f"Number of MD simulations")
    update_label(num_md_scale_chk, num_md_value_label, 0)


    ball_spacing_scale_chk = tk.DoubleVar(value=ball_spacing)
    ball_spacing_value_label = tk.Label(tab1_mktun, text=f"{ball_spacing_scale_chk.get()}")
    ball_spacing_scale = ttk.Scale(tab1_mktun, from_=0.15, to=1.7, orient="horizontal", variable=ball_spacing_scale_chk,
                            command=lambda value, var=ball_spacing_scale_chk, label=ball_spacing_value_label: update_label(var, label, 2))
    ball_spacing_label = tk.Label(tab1_mktun, text=f"Ball spacing")
    update_label(ball_spacing_scale_chk, ball_spacing_value_label, 2)
    
    def on_scale_change(var, scale, reset=False):
        if not reset:
            # Round the scale's current value to the nearest integer
            new_value = round(scale.get())
            # Update the variable and the scale's position
            var.set(new_value)
            scale.set(new_value)
        else:
            var.set(1)
            scale.set(1)


    connect_cut_scale_chk = tk.IntVar(value=connect_cut)
    connect_cut_value_label = tk.Label(tab1_mktun, text=f"{connect_cut_scale_chk.get()}")
    connect_cut_scale = ttk.Scale(tab1_mktun, from_=1, to=5, orient="horizontal", variable=connect_cut_scale_chk,
                            command=lambda value, var=connect_cut_scale_chk, label=connect_cut_value_label: update_label(var, label, 0))
    connect_cut_label = tk.Label(tab1_mktun, text=f"Connect cutoff (\u00D7 ball sp.)")
    update_label(connect_cut_scale_chk, connect_cut_value_label, 0)
    connect_cut_scale_chk.trace_add("write", lambda *args: on_scale_change(connect_cut_scale_chk, connect_cut_scale))

    # place all the sliders of tab 1
    ign_surf_label.place(anchor="nw", x=0, y=0)
    ign_surf_value_label.place(anchor="nw", x=160, y=17)
    ign_surf_scale.place(anchor="nw", x=0, y=18, width=162)
    surf_con_label.place(anchor="nw", x=0, y=46)
    surf_con_value_label.place(anchor="nw", x=160, y=63)
    surf_con_scale.place(anchor="nw", x=0, y=64, width=162)
    prot_space_label.place(anchor="nw", x=0, y=92)
    prot_space_value_label.place(anchor="nw", x=160, y=109)
    prot_space_scale.place(anchor="nw", x=0, y=110, width=162)
    min_vol_label.place(anchor="nw", x=0, y=138)
    min_vol_value_label.place(anchor="nw", x=160, y=155)
    min_vol_scale.place(anchor="nw", x=0, y=156, width=162)
    num_md_label.place(anchor="nw", x=0, y=184)
    num_md_value_label.place(anchor="nw", x=160, y=201)
    num_md_scale.place(anchor="nw", x=0, y=202, width=162)
    ball_spacing_label.place(anchor="nw", x=0, y=230)
    ball_spacing_value_label.place(anchor="nw", x=160, y=247)
    ball_spacing_scale.place(anchor="nw", x=0, y=248, width=162)
    connect_cut_label.place(anchor="nw", x=0, y=276)
    connect_cut_value_label.place(anchor="nw", x=160, y=293)
    connect_cut_scale.place(anchor="nw", x=0, y=294, width=162)


    separator1 = ttk.Separator(tab1_mktun)
    separator1.configure(orient="vertical")
    separator1.place(anchor="nw", height=300, width=2, x=193, y=0)

    target_label = tk.Label(tab1_mktun, text=f"Target object:")
    target_label.place(anchor="nw", x=200, y=0)

    target_options_list = ListObj('All', format='OBJNUM: OBJNAME')
    target_option = tk.StringVar(value=target_options_list[0])  # Setting default value to 'select'

    dropdown = ttk.OptionMenu(tab1_mktun, target_option, target_option.get(), *target_options_list)
    dropdown.place(anchor="nw", width=97, height=27, x=205, y=22)

    def get_tnl_name():
        """Return the YASARA object name of the currently selected tunnel (or wildcard for 'All')."""
        if tnl_insp_option.get() != 'All':
            targ = target()
            tnl_objnum = re.findall(r"\d+(?=:)", tnl_insp_option.get())[0]
            return NameObj(tnl_objnum)[0]
        else:
            if target() != None:
                return f'{target()}Cl???????'
            else:
                return None

    def target_changed(*args):
        Console("OFF")
        cur_target = re.findall(r"\d+(?=:)", target_option.get())[0]
        if ListRes(f'obj {cur_target} res HOH') != []:
            items = ['HOH']
        else:
            items = []
        [items.append(x) for x in ListRes(f'Obj {cur_target} res !protein and !hoh', format='RESNAME RESNUM')]
        listbox.delete(0, tk.END)
        for item in items:
            listbox.insert(tk.END, item)
        if ListRes(f'obj {cur_target} res HOH') != []:
            listbox.selection_set(0)

    # Link the function to the variable, so it gets called when the selection changes
    target_option.trace_add("write", target_changed)

    def on_sel_exclude_res(event):
        Console("OFF")
        selected_indices = listbox.curselection()
        selected_res = " ".join([listbox.get(i) for i in selected_indices])
        ShowRes(selected_res)
        UnselectAll()
        SelectRes(selected_res)
        Wait(1)
        Console("hidden")

    res_label = tk.Label(tab1_mktun, text=f"Exclude residues:")
    res_label.place(anchor="nw", x=200, y=51)


    listbox = tk.Listbox(tab1_mktun, selectmode='multiple')
    listbox.place(x=208, y=75, height=85, width=90)
    scrollbar = tk.Scrollbar(listbox, command=listbox.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    listbox.config(yscrollcommand=scrollbar.set, borderwidth=0.1)
    target_changed()

    listbox.bind('<<ListboxSelect>>', on_sel_exclude_res)

    polygon_chk = tk.BooleanVar(value=build_pol)  # restored from config (get_config), so the checkbox round-trips
    polygon_button = ttk.Checkbutton(tab1_mktun)
    polygon_button.configure(text='Build polygon', variable=polygon_chk)
    polygon_button.place(anchor="nw", x=200, y=175)

    # # Radio button variable
    show_prog_var = tk.StringVar()
    show_prog_var.set('vis')

    show_prog_var_radio = ttk.Radiobutton(tab1_mktun)
    show_prog_var_radio.configure(value='vis', variable=show_prog_var, text="Visualize steps")
    show_prog_var_radio.place(anchor="nw", x=200, y=202)

    show_prog_var_radio = ttk.Radiobutton(tab1_mktun)
    show_prog_var_radio.configure(value='fast', variable=show_prog_var, text="No steps (fast)")
    show_prog_var_radio.place(anchor="nw", x=200, y=223)

    show_prog_var_radio = ttk.Radiobutton(tab1_mktun)
    show_prog_var_radio.configure(value='wait', variable=show_prog_var, text="Debug (Wait)")
    show_prog_var_radio.place(anchor="nw", x=200, y=244)


    def show_progress_tunneler():
        """Create a progress window and run the Tunneler pipeline (called in a thread)."""
        Console("OFF")
        global progress_window, progress_var, percent_label, initializing
        initializing = True
        progress_window = tk.Toplevel(root)
        progress_window.title("Creating tunnels")
        progress_window.lift()
        progress_window.attributes("-topmost", True) 
        progress_var = tk.IntVar()
        progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=200, mode="determinate", variable=progress_var, maximum=100)
        progress_bar.pack(padx=5, pady=5)
        percent_label = ttk.Label(progress_window, text="0%")  # Initial text for the label
        percent_label.pack(pady=5)

        _attach_elapsed_timer(progress_window)

        # A fresh prediction rebuilds the clusters -> drop any stashed sphere/shape cache
        # sets (objects + state), else stale shpV/A/M### / sphT/D### objects linger and a
        # stale signature could cause a false cache hit.
        _sph_cache_clear()
        _sph_obj_cache_clear()   # new positions -> drop .obj geometry cache
        _shape_cache_clear()
        _xsec_cache_clear()      # new points -> drop cached cross-section positions
        # Also wipe any leftover axis/slice objects from a previous prediction: they
        # carry stale coordinates and, worse, a lingering NNN_axis would be taken for
        # the new tunnel's axis by on_diameter's existence check (feeding old In/Out
        # positions into the slice).
        DelObj('???_axis ???_slice ???_xsec')

        Tunneler(target=re.findall(r"\d+(?=:)", target_option.get())[0], ignore_res=[listbox.get(i) for i in listbox.curselection()],
                 ignore_surface=ign_surf_scale_chk.get(), 
                 ball_spacing=ball_spacing_scale_chk.get(),
                 max_ball_protein=prot_space_scale_chk.get(), 
                 surf_con_prev=surf_con_scale_chk.get(), 
                 keep_surf_points=False, 
                 mds=num_md_scale_chk.get(), 
                 min_vol=min_vol_scale_chk.get(), 
                 connect_cut=connect_cut_scale_chk.get(), 
                 build_pol=polygon_chk.get(),
                 prog=show_prog_var.get(),
                 progress_var=progress_var,
                 percent_label=percent_label)
        progress_window.destroy()

        if target():
            notebook.add(tab2_appear, text = 'Appearance')
            notebook.add(tab3_inspect, text = 'Inspect Tunnel')
            notebook.select(tab2_appear)
            tnl_insp_options_list = ['All']
            for x in ListObj(f'{target()}Cl???????', format='OBJNUM: OBJNAME'):
                tnl_insp_options_list.append(x)
            update_option_menu(tab3_inspect, tnl_insp_option, tnl_insp_options_list, current_value='All')
            # Precompute surface-point distances now (cheap KDTree) so the
            # Surface-points slider in the Appearance tab is instant from the first drag.
            _precompute_surf_dist(target())
        initializing = False
        Wait(1)
        Console("hidden")

    def run_tun(*args):
        Console("OFF")
        UnselectAll()
        threading.Thread(target=show_progress_tunneler).start()


    run_tun_button = ttk.Button(tab1_mktun)
    run_tun_button.configure(style='Toolbutton', text='    Find tunnels    ', command=run_tun)
    run_tun_button.place(anchor="nw",  width=111,height=33, x=200, y=272)

    def reset():
        Console("OFF")
        ign_surf_scale_chk.set(3.8), 
        ball_spacing_scale_chk.set(0.33),
        prot_space_scale_chk.set(2.8), 
        surf_con_scale_chk.set(2.7), 
        num_md_scale_chk.set(0), 
        min_vol_scale_chk.set(5),
        connect_cut_scale_chk.set(1), 
        polygon_chk.set(True),
        show_prog_var.set('vis')
        update_label(ign_surf_scale_chk, ign_surf_value_label, 1)
        update_label(ball_spacing_scale_chk, ball_spacing_value_label, 2)
        update_label(prot_space_scale_chk, prot_space_value_label, 2)
        update_label(surf_con_scale_chk, surf_con_value_label, 1)
        update_label(num_md_scale_chk, num_md_value_label, 0)
        update_label(min_vol_scale_chk, min_vol_value_label, 0)
        on_scale_change(connect_cut_scale_chk, connect_cut_scale, reset=True)

    reset_button = ttk.Button(tab1_mktun)
    reset_button.configure(text = 'Reset to default', command=reset)
    reset_button.place(anchor="nw", x=10, y=315)

    # --------------------------------------------------------
    #  TAB 2 — APPEARANCE
    #  Sections: Show/Hide, display mode (Points/Balls/Spheres/Shape),
    #  Color by (tunnel / distance), Actions (improve performance, recluster),
    #  Surface points slider
    #  Callbacks: Tunnelonoff, Targetonoff, Balls, Points, Spheres,
    #    Shapes, Colorbytunnel, Colorbytunneldist, improve_performance,
    #    Recluster, ml_outside_points, SecStr, Surf, on_cut
    # --------------------------------------------------------
    tab2_appear = ttk.Frame(notebook)
    notebook.add(tab2_appear, text='Appearance', padding=0) 
    tab2_appear.configure(height=375, width=310)

    ## show/hide section
    separator1 = ttk.Separator(tab2_appear)
    separator1.configure(orient="horizontal")
    separator1.place(anchor="nw", height=2, width=242, x=60, y=7)

    label2 = ttk.Label(tab2_appear)
    label2.configure(
        font="TkSmallCaptionFont",
        foreground="#797979",
        text='Show/Hide')
    label2.place(anchor="nw", x=0, y=0)

    if target() != None:
        targ_switch = switch_status(f'{target()}Cl???????')
    else:
        targ_switch = False
    tunnel_chk = tk.BooleanVar(value=targ_switch)  # Set to True for prechecked
    checkbutton4 = ttk.Checkbutton(tab2_appear)
    checkbutton4.configure(text='Tunnels', variable=tunnel_chk, command=Tunnelonoff)
    checkbutton4.place(anchor="nw", x=0, y=16)

    target_chk = tk.BooleanVar(value=targ_switch)  # Set to True for prechecked
    checkbutton1 = ttk.Checkbutton(tab2_appear)
    checkbutton1.configure(text='Target', variable=target_chk, command=Targetonoff)
    checkbutton1.place(anchor="nw", x=73, y=16)

    radio_var = tk.StringVar()
    radio_var.set('points')

    radiobutton4 = ttk.Radiobutton(tab2_appear)
    radiobutton4.configure(text='Points', variable=radio_var, value="points", command=Points)
    radiobutton4.place(anchor="nw", x=3, y=65)

    radiobutton5 = ttk.Radiobutton(tab2_appear)
    radiobutton5.configure(text='Balls', variable=radio_var, value="balls", command=Balls)
    radiobutton5.place(anchor="nw", x=3, y=85)

    separator8 = ttk.Separator(tab2_appear)
    separator8.configure(orient="horizontal")
    separator8.place(anchor="nw", height=2, width=9, x=78, y=116)
 
    separator6 = ttk.Separator(tab2_appear)
    separator6.configure(orient="vertical")
    separator6.place(anchor="nw", height=33, width=2, x=85, y=95)

    def show_progress_spheres(new=True):
        Console("OFF")
        global progress_window, progress_var, percent_label
        progress_window = tk.Toplevel(root)
        progress_window.title("Creating spheres")
        progress_window.lift()
        progress_window.attributes("-topmost", True) 
        progress_var = tk.IntVar()

        progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=200, mode="determinate", variable=progress_var, maximum=100)
        progress_bar.pack(padx=5, pady=30)

        percent_label = ttk.Label(progress_window, text="0%")  # Initial text for the label
        percent_label.pack(pady=30)

        _attach_elapsed_timer(progress_window)

        # Run the sphere build on the MAIN thread, NOT a background thread: dispatching
        # YASARA commands from a worker thread costs ~20s of GIL scheduling overhead at
        # ~70k spheres (measured: between-call time 20.3s threaded vs 0.14s main-thread).
        # The build loop keeps the progress bar live via progress_window.update_idletasks().
        progress_window.update()
        on_new_sphere(new, True)


    def on_new_sphere(new=False, progress=False):
        Console("OFF")
        progress_var.set(1)
        radio_var.set('spheres')
        Spheres(new=new, progress=progress)
        progress_window.destroy()

    radiobutton6 = ttk.Radiobutton(tab2_appear)
    radiobutton6.configure(text='Spheres', variable=radio_var, value="spheres", command=lambda: show_progress_spheres(new=False))
    radiobutton6.place(anchor="nw", x=3, y=105)

    def _make_debounced(fn, delay=300):
        """Return a trigger that runs fn() once, `delay` ms after the LAST call. Rapid
        repeats (mousewheel scrubbing, typing) coalesce into a single deferred call, so an
        expensive apply (a sphere/MergeSph rebuild) fires once per burst, not per notch."""
        st = {'id': None}
        def trigger(*_):
            if st['id'] is not None:
                try:
                    root.after_cancel(st['id'])
                except Exception:
                    pass
            st['id'] = root.after(delay, fn)
        return trigger

    def _numeric_spinbox(parent, var, lo, hi, apply_fn, x, y, width=55, wheel=10, delay=300):
        """Compact integer Spinbox bound to IntVar `var`, clamped to [lo, hi]. The arrow
        buttons and typing give +/-1 fine control; the MOUSEWHEEL jumps by `wheel` (a
        bigger step so scrubbing a 1-100 range isn't tedious) -- the two are independent.
        Value changes DEBOUNCE-apply via apply_fn so a scroll/type burst triggers one
        rebuild. `var` is kept a valid int at all times so the many existing `var.get()`
        readers stay safe."""
        def _valid(proposed):
            # allow empty (mid-edit) and any in-range digit string; below-lo is fine while
            # typing (e.g. '4' on the way to '40'), _commit clamps it up afterwards
            return proposed == '' or (proposed.isdigit() and int(proposed) <= hi)
        vcmd = (root.register(_valid), '%P')
        sb = ttk.Spinbox(parent, from_=lo, to=hi, increment=1, width=4,
                         validate='key', validatecommand=vcmd, textvariable=var)
        sb.place(anchor="nw", x=x, y=y, width=width)
        trig = _make_debounced(apply_fn, delay)
        def _clamp():
            try:
                v = int(var.get())
            except (tk.TclError, ValueError):
                v = lo
            var.set(min(hi, max(lo, v)))
        def _commit(*_):
            _clamp(); trig()
        def _wheel(e):
            try:
                v = int(var.get())
            except (tk.TclError, ValueError):
                v = lo
            up = getattr(e, 'delta', 0) > 0 or getattr(e, 'num', 0) == 4
            var.set(min(hi, max(lo, v + (wheel if up else -wheel))))
            trig()
            return 'break'   # don't let the wheel also scroll a parent container
        sb.configure(command=_commit)          # arrow buttons
        sb.bind('<Return>', _commit)            # typed value -> apply
        sb.bind('<FocusOut>', _commit)
        sb.bind('<MouseWheel>', _wheel)         # macOS / Windows
        sb.bind('<Button-4>', _wheel)           # Linux up
        sb.bind('<Button-5>', _wheel)           # Linux down
        return sb

    def _on_sphere_alpha_release(*_):
        """Apply a sphere-alpha change (debounced). Mesh spheres can't be re-alpha'd in
        place, so this rebuilds -- but alpha is excluded from the geometry signature, so
        the cached .obj files are reused and only the (fast) LoadWOb re-runs. No-op unless
        spheres are the active view."""
        if radio_var.get() != 'spheres' or ListObj('???_Sphere') == []:
            return
        Spheres(new=True)

    alpha_chk = tk.IntVar()
    alpha_spin = _numeric_spinbox(tab2_appear, alpha_chk, 1, 100,
                                  _on_sphere_alpha_release, x=125, y=87)
    alpha_chk.set(19)

    def _on_sphere_size_release(*_):
        """Apply a size change (debounced). The size control is shared: it's the ball
        radius for both Spheres and the MergeSph shape (VdW/accessible ignore it). Size
        changes every vertex, so the geometry is fully rebuilt (spheres: rewrite .obj via
        the progress window; MergeSph: recompute marching cubes). No-op for reps that don't
        use size."""
        rv = radio_var.get()
        if rv == 'spheres' and ListObj('???_Sphere') != []:
            show_progress_spheres()
        elif rv == 'shape' and shape_surf_option.get() == 'MergeSph' and ListObj('???_shape') != []:
            Shapes(new=True)   # MergeSph radius changed -> marching-cubes recompute

    rad_chk = tk.IntVar()
    rad_spin = _numeric_spinbox(tab2_appear, rad_chk, 4, 100,
                                _on_sphere_size_release, x=125, y=108)
    rad_chk.set(18)

    label1 = ttk.Label(tab2_appear)
    label1.configure(text = 'alpha')
    label1.place(anchor="nw", x=90, y=88)

    label4 = ttk.Label(tab2_appear)
    label4.configure(text = 'size')
    label4.place(anchor="nw", x=90, y=109)

    fast_chk = tk.BooleanVar(value=False)   # off = smooth (per-vertex normals) by default
    def _on_fast_toggle(*_):
        """Toggle flat (fast) vs smooth sphere shading. It changes what's written to the
        .obj (per-vertex normals), so it invalidates the geometry cache via _sph_geo_sig
        and the rebuild rewrites. Only acts while spheres are the active view."""
        if radio_var.get() == 'spheres' and ListObj('???_Sphere') != []:
            show_progress_spheres()
    fast_check = ttk.Checkbutton(tab2_appear, text='fast', variable=fast_chk,
                                 command=_on_fast_toggle)
    fast_check.place(anchor="nw", x=190, y=96)

    # --- Shape (surface) type cache ---------------------------------------------
    # Building a shape (esp. MergeSph, which runs marching cubes) is worth caching:
    # switching between VdW / accessible / MergeSph stashes the inactive set hidden
    # (renamed shpV###/shpA###/shpM###, not matching '???_shape') and swaps it back
    # instead of rebuilding, as long as its signature is unchanged. MergeSph bakes its
    # colour into the mesh, so its signature includes the colour state; VdW/accessible
    # recolour cheaply in place, so theirs excludes colour and they are just recoloured
    # on swap-in.
    _shape_state = {'surf': None, 'sig': {}}
    _SHAPE_SUF = {'VdW': 'V', 'accessible': 'A', 'MergeSph': 'M'}
    # MergeSph marching-cubes geometry, keyed by (tunnel obj, radius). Reused across
    # alpha/colour changes (they don't affect the geometry); dropped when positions
    # change (re-predict / recluster call _shape_cache_clear).
    _mrg_geom_cache = {}

    def _shape_sig(surf):
        # VdW/accessible are dynamic surfaces whose alpha is changed in place via
        # ColorSurfObj (see _apply_shape_alpha), so alpha is EXCLUDED from their
        # signature -- an alpha change must not invalidate the cache / force a rebuild.
        # MergeSph bakes alpha into its mesh at LoadWOb time, so it stays in the sig
        # (the rebuild it triggers reuses the cached marching-cubes geometry).
        if surf == 'MergeSph':
            sig = f'{surf}|a{shape_alpha_chk.get()}|r{rad_chk.get()}|c{radio_col_var.get()}'
            if radio_col_var.get() == 'distance':
                tar = target()
                sig += (f'|{PairObj(tar, "dist_sel")}|{PairObj(tar, "dist_sig")}'
                        f'|{dist_palette_var.get()}|{dist_grad["t0"]:.3f}|{dist_grad["t1"]:.3f}')
            else:
                sig += f'|s{step_entry.get()}'
            return sig
        return f'{surf}'

    def _shape_cache_clear():
        DelObj('shpV??? shpA??? shpM???')
        _shape_state['surf'] = None
        _shape_state['sig'] = {}
        _mrg_geom_cache.clear()   # positions changed -> cached marching-cubes geometry is stale

    def _apply_shape_alpha(surf=None):
        """Re-apply the current shape alpha to the visible VdW/accessible surfaces IN
        PLACE (ColorSurfObj on a dynamic surface -> no surface recompute, ~instant).
        MergeSph alpha is baked into its mesh, so it's a no-op here (handled by a
        rebuild that reuses the cached marching-cubes geometry)."""
        if surf is None:
            surf = shape_surf_option.get()
        if surf == 'MergeSph':
            return
        a = shape_alpha_chk.get()
        for o in ListObj('???_shape'):
            ColorSurfObj(o, surf, outcol='atomcol', outalpha=a, incol='atomcol', inalpha=a)

    def _recolor_active_shape():
        """Recolour the visible atom-based ???_shape set (VdW/accessible) to the current
        colour mode. Colours ONLY the shapes -- deliberately NOT via Colorbytunnel, which
        also recolours clusters and (via recolor_spheres_for_mode) would drop the sphere
        cache. MergeSph is not recoloured here (its colour is baked into the mesh)."""
        if radio_col_var.get() == 'tunnel':
            try:
                step = int(step_entry.get())
            except (ValueError, TypeError):
                step = 25
            # colour by cluster POSITION (matches detection + balls/spheres), not obj number
            for i, objnum in enumerate(ListObj(f'{target()}Cl???????')):
                if ListObj(f'{objnum:03d}_shape') != []:
                    ColorObj(f'{objnum:03d}_shape', (i + 1) * step)
        else:
            _color_shapes_from_clusters()   # colours (auto-rebuilds if out of sync)

    def build_shape_objects(surf):
        """Build the ???_shape set for surface type `surf` from the current clusters."""
        tunnel_objs = ListObj(f'{target()}Cl???????')
        if surf == 'MergeSph':
            # Merged-spheres surface: numpy union-of-balls (marching cubes), coloured
            # per-vertex by nearest cluster point so it follows the current tunnel/
            # distance colouring (mesh has no atoms -> no post-recolor).
            radius = rad_chk.get() / 100 * 2.5
            for targetobj in tunnel_objs:
                ShowMessage(f"Creating merged-sphere shape of tunnel {NameObj(targetobj)[0]}")
                Wait(1)
                on_off = SwitchObj(targetobj)[0]
                p = np.array(PosAtom(f'obj {targetobj}', coordsys='global')).reshape(-1, 3)
                col = _cluster_colors(targetobj)   # band-sourced (uniform type) in distance mode
                j = _union_shape_mesh(p, col, radius, shape_alpha_chk.get(),
                                      geom_cache=_mrg_geom_cache,
                                      geom_key=(int(targetobj), rad_chk.get()),
                                      geom_frame=_obj_frame_sig([targetobj]))
                if j is None:
                    continue
                SwitchObj(j, on_off)
                NameObj(j, f'{targetobj:03d}_shape')
                PairObj(f'{targetobj:03d}_shape', 'surf', 'MergeSph')
        else:
            for targetobj in tunnel_objs:
                ShowMessage(f"Creating shape of tunnel {NameObj(targetobj)[0]}")
                Wait(1)
                newo = DuplicateObj(targetobj)[0]
                HideObj(newo)
                SwapAtom(f'obj {newo}', 'H')
                ShowSurfObj(newo, surf, outcol='atomcol', outalpha=shape_alpha_chk.get())
                HideObj(newo)
                NameObj(newo, f'{targetobj:03d}_shape')
                on_off = SwitchObj(targetobj)[0]
                SwitchObj(newo, on_off)
                PairObj(newo, 'surf', surf)
            # recolor because swapatom resets color, imperfect. change if swapatom keepcol becomes avail.
            _recolor_active_shape()

    _shape_sync = {'busy': False}

    def _color_shapes_from_clusters():
        """Colour the atom-based ???_shape set from the cluster point colours, paired 1:1.

        If the shape set has drifted out of sync with the clusters (their total atom counts
        differ -- e.g. stale shapes lingering after the clusters changed), AUTO-RECOVER by
        rebuilding the shapes fresh from the current clusters, instead of raising the old
        'delete shapes and recreate' error at the user. Re-entrancy-guarded: if even a fresh
        rebuild can't reconcile the counts, it degrades to a quiet no-op rather than looping
        (build_shape_objects calls back here to colour what it just built)."""
        tar = target()
        atomlist = ListAtom('obj ???_shape')
        collist = [x[1:] for x in SegAtom(f'obj {tar}Cl???????')]
        if len(atomlist) == len(collist):
            group_and_color(atomlist, collist, mind_console=False)
            return
        if _shape_sync['busy']:
            return   # a rebuild is already in progress and still doesn't match -> give up quietly
        _shape_sync['busy'] = True
        try:
            DelObj('???_shape')
            _shape_cache_clear()                          # drop any stashed (now-stale) shape sets
            build_shape_objects(shape_surf_option.get())  # rebuild fresh from clusters (+ recolours)
        finally:
            _shape_sync['busy'] = False

    def Shapes(*args, new=False):
        """Switch tunnel display to surface-shape mode (VdW / accessible / MergeSph),
        caching each surface type so switching between them swaps instead of rebuilding."""
        if radio_var.get() != 'shape':
            # the surface-type dropdown's trace fires whenever it changes, even when the
            # Shape representation isn't selected -- don't touch the current display then.
            return
        Console("OFF")
        surf = shape_surf_option.get()
        tsig = _shape_sig(surf)
        cur = _shape_state['surf']
        # per-tunnel visibility of the current shape set, to carry across swap/rebuild
        shapes_on = list(zip(ListObj('???_shape', format='OBJNAME'), SwitchObj('???_shape')))
        # Which tunnels are currently shown (from whichever representation is active), so
        # the shapes inherit that -- reading cluster visibility alone fails when coming
        # from Spheres (clusters are hidden while spheres are shown).
        on_nums = set(f'{o:03d}' for o, v in zip(ListObj(f'{target()}Cl???????'),
                                                 SwitchObj(f'{target()}Cl???????')) if v == 'On')
        if not on_nums:
            on_nums = set(nm.split('_')[0] for nm, v in
                          zip(NameObj('???_Sphere'), SwitchObj('???_Sphere')) if v == 'On')
        if not on_nums and shapes_on:
            on_nums = set(nm.split('_')[0] for nm, v in shapes_on if v == 'On')

        if not new and cur == surf and _shape_state['sig'].get(surf) == tsig and ListObj('???_shape') != []:
            # already showing the requested shape set: just switch on the visible tunnels
            on_tunnels = [x for x,y in zip(ListObj(f'{target()}Cl???????'), SwitchObj(f'{target()}Cl???????')) if y == 'On']
            if len(on_tunnels) == 0:
                on_tunnels = [int(x[2:3]) for x,y in zip(NameObj('???_Sphere'), SwitchObj('???_Sphere')) if y == 'On']
            SwitchObj(" ".join([str(f'{x:03d}') + '_shape' for x in on_tunnels]), "on")
        else:
            # stash the current active set under its surf (unless forced rebuild / same surf)
            if not new and cur is not None and cur != surf and ListObj('???_shape') != []:
                for o in ListObj('???_shape'):
                    num = NameObj(o)[0].split('_')[0]
                    NameObj(o, f'shp{_SHAPE_SUF[cur]}{num}'); SwitchObj(o, 'Off')
            else:
                DelObj('???_shape')
            # bring up the target surf: cached set if its signature still matches, else build
            if not new and _shape_state['sig'].get(surf) == tsig and ListObj(f'shp{_SHAPE_SUF[surf]}???') != []:
                for o in ListObj(f'shp{_SHAPE_SUF[surf]}???'):
                    num = NameObj(o)[0][3 + len(_SHAPE_SUF[surf]):]
                    NameObj(o, f'{num}_shape'); SwitchObj(o, 'On')
                if surf != 'MergeSph':
                    _recolor_active_shape()   # colour excluded from VdW/acc sig -> recolour now
                    _apply_shape_alpha(surf)  # alpha too excluded from sig -> re-alpha the stashed set now
            else:
                DelObj(f'shp{_SHAPE_SUF[surf]}???')   # stale cache for this surf
                if surf == 'MergeSph' and _skimage_missing():
                    ShowMessage("MergeSph shape needs the 'scikit-image' package in the YASARA "
                                "venv. Install it (pip install scikit-image) or rerun setup_venv.sh.")
                    Wait('Continuebutton')
                else:
                    build_shape_objects(surf)
                    _shape_state['sig'][surf] = tsig
            # show the new/swapped shapes for exactly the tunnels that were visible before
            for o in ListObj('???_shape'):
                num = NameObj(o)[0].split('_')[0]
                SwitchObj(o, 'On' if num in on_nums else 'Off')
            _shape_state['surf'] = surf

        SwitchObj(f'{target()}Cl??????? ???_sphere', 'off')
        HideMessage()
        Wait(1)
        Console("hidden")

    def _on_shape_alpha_release(*_):
        """Apply a shape-alpha slider change. VdW/accessible re-alpha in place (instant,
        no surface recompute); MergeSph rebuilds but reuses the cached marching-cubes
        geometry, so only the (cheap) LoadWOb re-runs. No-op unless a shape is shown."""
        if radio_var.get() != 'shape':
            return
        Console("OFF")
        if shape_surf_option.get() == 'MergeSph':
            Shapes(new=True)
        else:
            _apply_shape_alpha()
        Console("hidden")

    radiobutton7 = ttk.Radiobutton(tab2_appear)
    radiobutton7.configure(text='Shape', variable=radio_var, value="shape", command=lambda: Shapes())
    radiobutton7.place(anchor="nw", x=3, y=125)

    #  Tunnel AA backbone atom type
    shape_surf_option = tk.StringVar()
    shape_surf_option.set("molecular")
    shape_surf_dropdown = ttk.OptionMenu(tab2_appear, shape_surf_option, "VdW", "VdW", "accessible", "MergeSph")
    shape_surf_dropdown.place(anchor="nw", width=100, height=27, x=85, y=127)

    shape_surf_option.trace_add("write", Shapes)

    shape_alpha_chk = tk.IntVar()
    shape_alpha_spin = _numeric_spinbox(tab2_appear, shape_alpha_chk, 1, 100,
                                        _on_shape_alpha_release, x=200, y=127)
    shape_alpha_chk.set(80)


    # target -> (n_cluster_atoms, min_dist, max_dist). The per-atom distance itself
    # lives in each cluster atom's Property field. Cached so the precompute runs
    # once per cluster set. Cheap enough (KDTree) to run eagerly at the end of
    # detection/recluster (see _precompute_surf_dist calls) so the slider is instant
    # from the very first drag.
    surf_dist_cache = {}

    def _precompute_surf_dist(tar):
        """Precompute each cluster atom's distance to the roughsurf point cloud
        (KDTree, global coords) and store it bucketed in the atom Property field, so
        the Surface-points slider becomes a pure 'Property<threshold' hide with no
        per-move surface recompute. No-op if already done for the current cluster
        set. Does not change what is visible; only populates the Property field."""
        n_cl = CountAtom(f'obj {tar}Cl???????')
        cache = surf_dist_cache.get(tar)
        if n_cl == 0 or (cache is not None and cache[0] == n_cl):
            return
        Console("OFF")
        SupAtom(f'obj {tar}roughsurf', f'obj {tar}', match='Yes')
        rs_du = DuplicateObj(f'{tar}roughsurf')[0]
        DelAtom(f'obj {rs_du} element !Du')
        rs_pos = np.array(PosAtom(f'obj {rs_du} element Du', coordsys='global')).reshape(-1, 3)
        DelObj(rs_du)
        cl = np.array(ListAtom(f'obj {tar}Cl???????'))
        cl_pos = np.array(PosAtom(f'obj {tar}Cl???????', coordsys='global')).reshape(-1, 3)
        if len(cl) == 0 or len(rs_pos) == 0:
            return
        dist = cKDTree(rs_pos).query(cl_pos)[0]
        # Store distance (bucketed to 0.1 A) per atom: one grouped PropAtom per
        # bucket (~200 calls, independent of atom count; PropAtom takes a single
        # value, so a per-atom list can't be set at once).
        PropAtom(f'obj {tar}Cl???????', 99999)
        dr = np.round(dist, 1)
        for v in np.unique(dr):
            PropAtom('atom ' + ' '.join(map(str, cl[dr == v])), float(v))
        surf_dist_cache[tar] = (n_cl, float(dist.min()), float(dist.max()))

    def ml_outside_points(by=25.5):
        """Hide tunnel points near the protein surface (the 'Surface points' slider callback).

        Distances are precomputed into the atom Property field (see
        _precompute_surf_dist, run eagerly after detection/recluster), so moving the
        slider is just a fast 'Property<threshold' hide. Falls back to precomputing
        on demand if it has not run yet for this cluster set.
        """
        Console("OFF")
        tar = target()
        # Neither the sphere meshes nor the MergeSph/VdW shapes can be partially hidden
        # (they are monolithic LoadWOb/surface meshes, not per-atom), so a surface-points
        # change can't update them live -- drop to the native, per-atom-hideable ball rep.
        # Balls() reads which sphere/shape objects are currently *on* to know which tunnels
        # to show, so it must run BEFORE those meshes get switched off (it turns them off
        # itself). Doing a blanket SwitchObj('...','OFF') first would hide the shapes and
        # leave nothing visible -- which was the Shape-mode bug.
        if radio_var.get() in ('spheres', 'shape'):
            radio_var.set('balls')
            Balls()
        else:
            SwitchObj('???_Sphere ???_shape', 'OFF')
        _precompute_surf_dist(tar)
        cache = surf_dist_cache.get(tar)
        if cache is None:
            Wait(1); Console("hidden"); return
        _, min_dist, max_dist = cache
        cur_dist = min_dist + (max_dist - min_dist) * surf_pts_chk.get()
        ShowAtom(f'obj {tar}Cl???????')
        HideAtom(f'obj {tar}Cl??????? and Property<{cur_dist:.3f}')
        Wait(1)
        Console("hidden")

    surf_chk = tk.IntVar(value=switch_status(f'{target()}Surf'))  
    checkbutton5 = ttk.Checkbutton(tab2_appear)
    checkbutton5.configure(text='Surf', variable=surf_chk, command=Surf)
    checkbutton5.place(anchor="nw", x=0, y=40)

    surf_col = tk.StringVar()
    surf_col_drop = ttk.OptionMenu(tab2_appear, surf_col, "white", "white", "element", "restype", "Bfactor", "SecStr", "Occupancy", 'choose..', command=Surf)
    surf_col.set("outcol")
    surf_col_drop.place(anchor="nw", width=65, height=27, x=50, y=37)

    surf_col_alpha_chk = tk.IntVar()
    surf_col_alpha_scale = ttk.Scale(tab2_appear, from_=1, to=100)
    surf_col_alpha_scale.configure(orient="horizontal", state="normal", variable=surf_col_alpha_chk, command=Surf)
    surf_col_alpha_scale.place(anchor="nw", x=115, y=37, height=30, width=46)
    surf_col_alpha_chk.set(80)

    surf_incol = tk.StringVar()
    surf_incol_drop = ttk.OptionMenu(tab2_appear, surf_incol, "black", "as outside", "black", "white", 'choose..', command=Surf)
    surf_incol.set("incol")
    surf_incol_drop.place(anchor="nw", width=65, height=27, x=161, y=37)

    surf_incol_alpha_chk = tk.IntVar()
    surf_incol_alpha_scale = ttk.Scale(tab2_appear, from_=1, to=100)
    surf_incol_alpha_scale.configure(orient="horizontal", state="normal", variable=surf_incol_alpha_chk, command=Surf)
    surf_incol_alpha_scale.place(anchor="nw", x=226, y=37, height=30, width=46)
    surf_incol_alpha_chk.set(80)

    cut_surf_label = ttk.Label(tab2_appear)
    cut_surf_label.configure(text = 'Cut surf:')
    cut_surf_label.place(anchor="nw", x=266, y=16)

    def surf_obj(tar):
        sobj = DuplicateObj(tar)[0]
        SwitchObj(sobj, 'on')
        NameObj(sobj, f'{tar}Surf')
        ShowSurfRes(f'obj {sobj}', 'molecular')
        HideObj(sobj)
        HideSecStrObj(sobj)
        return sobj


    def on_cut(*args):
        Console('off')
        tar = target()
        DelObj(f'{tar}CutPlane')
        DelObj(f'{tar}Surf')
        cut = int(cut_surf_times.get())
        if cut > 0:
            for i in range(cut):
                sobj = surf_obj(tar)
                cobj = CutObj(sobj)[0]
                NameObj(cobj, f'{tar}CutPlane')
                SwitchObj(cobj, 'off')
                if i == 1:
                    RotateObj(cobj, 90)
                elif i == 2:
                    RotateObj(cobj, 0, 90)
        else:
            surf_obj(tar)
        Surf(mind_console=False)
        Wait(1)
        Console("hidden")


    cut_surf_times = tk.StringVar()
    cut_surf_times_drop = ttk.OptionMenu(tab2_appear, cut_surf_times, "2", "0", "1", "2", "3", command=on_cut)
    cut_surf_times.set("2")
    cut_surf_times_drop.place(anchor="nw", width=50, height=27, x=272, y=37)
    cut_surf_times.trace_add("write", on_cut)


    h2o_chk = tk.BooleanVar(value=switch_status(f'{target()}H2O'))  # Variable to track the checkbox status
    checkbutton6 = ttk.Checkbutton(tab2_appear)
    checkbutton6.configure(text='H\u2082O', variable=h2o_chk, command=H2O)
    checkbutton6.place(anchor="nw", x=136, y=16)

    # --- Color utilities (YASARA hue ↔ RGB conversion) ---
    import colorsys
    color_names = {
        "blue": 0,
        "magenta": 60,
        "red": 120,
        "yellow": 180,
        "green": 240,
        "cyan": 300,
        "gray": None,
    }

    def get_contrasting_text_color(hex_color):
        """Return black or white hex color for readable text on the given background."""
        Console("OFF")
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return '#000000' if luminance > 0.5 else '#FFFFFF'
    
    def hue_to_rgb(hue, grey='w'):
        """Convert a YASARA hue value (0-360+) to a hex RGB color string (#rrggbb)."""
        if isinstance(hue, str):
            hue_lower = hue.lower()
            if hue_lower in color_names:
                hue = color_names[hue_lower]
                if hue is None:  # Special handling for grey
                    return "#FFFFFF"  # white
            else:
                try:
                    # Attempt to convert string to a number
                    hue = float(hue)
                except ValueError:
                    ShowMessage(f'this is a bug, tried to use this hue: {hue}')
                    wc()
                    plugin.end()
 
        adjusted_hue = (int(hue) + 240) % 360
        r, g, b = colorsys.hsv_to_rgb(adjusted_hue / 360, 1, 1)
        hex_color = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        return hex_color

    def _palette_rgb(name, u):
        """'#rrggbb' for the strip preview at fraction u. 'hue' palettes match the rendered
        atoms exactly; cmaps show their TRUE colours while the atoms get YASARA's nearest
        displayable snap, so they can differ a little (mainly at dark extremes)."""
        kind, spec = DIST_PALETTES[name]
        if kind == 'hue':
            return hue_to_rgb(_palette_num(spec, u))
        return '#%02x%02x%02x' % _cmap_rgb255(spec, u)

    def _make_gradient_picker(parent, x, y, width, height, get_rgb, grad, on_change):
        """A slim in-tkinter strip that renders the CURRENTLY selected palette (see
        DIST_PALETTES) as a gradient, with two draggable handles that trim the sub-window
        [t0,t1] (fractions 0-1) of the palette actually used for the distance colouring.
        Default handles at the ends = full palette. `get_rgb(u)` returns the '#rrggbb' of the
        active palette at fraction u in [0,1]; `grad` is a {'t0','t1'} dict updated in place;
        the recolour `on_change` fires on handle release (caller debounces it).
        Returns repaint(), to call when the palette changes."""
        cv = tk.Canvas(parent, width=width, height=height, highlightthickness=1,
                       highlightbackground='#888888', cursor='sb_h_double_arrow')
        cv.place(anchor='nw', x=x, y=y)
        img = tk.PhotoImage(width=width, height=height)   # keep referenced vs GC
        cv._grad_img = img
        cv.create_image(0, 0, anchor='nw', image=img)
        # Bright bars along top+bottom edges bracket the selected [t0,t1] sub-window.
        # (Highlight rather than dim the rest: stipple fake-transparency is unreliable on
        # macOS Aqua Tk.) Then the two handle markers.
        # Top: an arrow from the near handle (t0) to the far handle (t1) -- shows the
        # gradient DIRECTION (which the two handle positions alone don't), so the swap
        # button visibly flips it. Drawn as a black halo + white arrow so it stays visible
        # over any palette colour. Bottom: a plain bracket over the selected window.
        bar_top_sh = cv.create_line(0, 3, 0, 3, fill='black', width=4,
                                    arrow='last', arrowshape=(8, 9, 4))
        bar_top = cv.create_line(0, 3, 0, 3, fill='white', width=2,
                                 arrow='last', arrowshape=(7, 8, 3))
        bar_bot = cv.create_line(0, height - 1, 0, height - 1, fill='white', width=3)
        h0_w = cv.create_line(0, 0, 0, height, fill='white', width=3)
        h0_b = cv.create_line(0, 0, 0, height, fill='black', width=1)
        h1_w = cv.create_line(0, 0, 0, height, fill='white', width=3)
        h1_b = cv.create_line(0, 0, 0, height, fill='black', width=1)

        def _fx(t):
            return max(0, min(width, int(t * width)))

        def _redraw():
            x0, x1 = _fx(grad['t0']), _fx(grad['t1'])
            lo, hi = min(x0, x1), max(x0, x1)
            cv.coords(bar_top_sh, x0, 3, x1, 3)   # black halo under the arrow
            cv.coords(bar_top, x0, 3, x1, 3)      # near(t0) -> far(t1)
            cv.coords(bar_bot, lo, height - 1, hi, height - 1)
            for item, xx in ((h0_w, x0), (h0_b, x0), (h1_w, x1), (h1_b, x1)):
                cv.coords(item, xx, 0, xx, height)

        def repaint():
            # Repaint the palette gradient (call on palette switch) and redraw the handles.
            for px in range(width):
                img.put(get_rgb(px / (width - 1)), to=(px, 0, px + 1, height))
            _redraw()
        repaint()

        drag = {'which': None}

        def _t_at(ex):
            return max(0.0, min(1.0, ex / width))

        def _press(e):
            # grab whichever handle is nearer the click, then move it there
            drag['which'] = 't0' if abs(e.x - _fx(grad['t0'])) <= abs(e.x - _fx(grad['t1'])) else 't1'
            _move(e)

        def _move(e):
            # Update only the strip visual while dragging -- the (expensive) recolour is
            # deferred to release so it doesn't fire on every motion event / mid-drag pause.
            if drag['which'] is None:
                return
            grad[drag['which']] = _t_at(e.x)
            _redraw()

        def _release(e):
            if drag['which'] is None:
                return
            drag['which'] = None
            on_change()   # recolour once, after a settle delay (caller debounces this)
        cv.bind('<Button-1>', _press)
        cv.bind('<B1-Motion>', _move)
        cv.bind('<ButtonRelease-1>', _release)
        return repaint


    ss_chk = tk.BooleanVar(value=switch_status(f'{target()}SS'))  # Variable to track the checkbox status
    checkbutton7 = ttk.Checkbutton(tab2_appear)
    checkbutton7.configure(text='SecStr', variable=ss_chk, command=SecStr)
    checkbutton7.place(anchor="nw", x=70, y=65)
    

    ss_col = tk.StringVar()
    ss_col.set("element")
    ss_col_drop = ttk.OptionMenu(tab2_appear, ss_col, "white", "white", "element", "restype", "Bfactor", "SecStr", "Occupancy", 'choose..')
    ss_col_drop.place(anchor="nw", width=85, height=27, x=140, y=61)

    ss_style = tk.StringVar()
    ss_style.set("ribbon")
    ss_style_drop = ttk.OptionMenu(tab2_appear, ss_style, "Ribbon", "Ribbon", "Cartoon", "Tube", "Trace", command=SecStr)
    ss_style_drop.place(anchor="nw", width=85, height=27, x=230, y=61)

    # Function to handle the selection change
    def ss_col_change(*args):
        Console("OFF")
        if ss_col.get() == 'choose..':
            col = ShowWin("ColorSelection","Select tunnel residues color", "Bow","Background","100")[0]
        else:
            col = ss_col.get()
        ColorObj(f'{target()}SS', col)
        Wait(1)
        Console("hidden")

    # Link the function to the variable, so it gets called when the selection changes
    ss_col.trace_add("write", ss_col_change)

    nonprot_chk = tk.BooleanVar(value=switch_status(f'{target()}NonProt'))  # Variable to track the checkbox status
    checkbutton8 = ttk.Checkbutton(tab2_appear)
    checkbutton8.configure(text='NonProt', variable=nonprot_chk, command=Nonprot)
    checkbutton8.place(anchor="nw", x=185, y=16)

    # # Radio button variable
    radio_col_var = tk.StringVar()
    radio_col_var.set('tunnel')

    def Colorbytunnel(shapes=True, mind_console=True):
        """Color each tunnel cluster a different hue (stepped by the 'step' entry value)."""
        if mind_console:
            Console("OFF")
        objs = ListObj(f'{target()}Cl???????')
        try:
            step = int(step_entry.get())
        except:
            ShowMessage('Invalid step size, you must use a number between 0 and 360. Defaulting to 25')
            Wait(30)
            step = 25
            step_entry.delete(0, tk.END)
            step_entry.insert(0, 25)
        # Colour by the cluster's POSITION i (0-based), matching detection's
        # `ColorObj(c, (i+1)*25)` -- NOT the object number, which drifts from the
        # detection colours (points/balls/spheres) and shifted the shape colouring.
        if not shapes:
            for i, objnum in enumerate(objs):
                ColorObj(objnum, (i +1) * step)
        else:
            for i, objnum in enumerate(objs):
                ColorObj(objnum, (i +1) * step)
                if shape_surf_option.get() != 'MergeSph':
                    ColorObj(str(f'{objnum:03d}') + '_shape', (i +1) * step)
            if shape_surf_option.get() == 'MergeSph' and ListObj('???_shape') != []:
                Shapes(new=True)   # merged-sphere shape is a mesh (no atoms) -> rebuild
        recolor_spheres_for_mode('tunnel')   # mesh spheres bake in colour -> swap/rebuild set
        HideMessage()
        Wait(1)
        if mind_console:
            Console("hidden")

    ### color by section
    label3 = ttk.Label(tab2_appear)
    label3.configure(
        font="TkSmallCaptionFont",
        foreground="#797979",
        text='Color by')
    label3.place(anchor="nw", x=0, y=145)

    separator4 = ttk.Separator(tab2_appear)
    separator4.configure(orient="horizontal")
    separator4.place(anchor="nw", height=2, width=250, x=50, y=151)
  
    separator2 = ttk.Separator(tab2_appear)
    separator2.configure(orient="vertical")
    separator2.place(anchor="nw", height=55, width=2, x=12, y=168)
 
    radiobutton14 = ttk.Radiobutton(tab2_appear)
    radiobutton14.configure(text='Tunnel', variable=radio_col_var, value="tunnel", command=Colorbytunnel)
    radiobutton14.place(anchor="nw", x=5, y=158)

    label14 = ttk.Label(tab2_appear)
    label14.configure(text='step')
    label14.place(anchor="nw", x=20, y=180)

    # Spinbox (compact: number + arrows + mousewheel) rather than a bare entry, and it
    # applies the colouring live instead of only when the Tunnel radio is re-clicked.
    # step is a hue increment 0-360; changing it re-runs Colorbytunnel (which reuses the
    # sphere .obj cache since colour step doesn't change the geometry).
    step_entry = ttk.Spinbox(tab2_appear, from_=0, to=360, increment=5, width=4)
    step_entry.place(anchor="nw", x=20, y=198, width=52)   # fits before the x=79 separator
    step_entry.delete(0, tk.END)
    step_entry.insert(0, '25')

    def _on_step_change(*_):
        """Live-apply the step value -- only while Tunnel colouring is the active mode
        and tunnels exist (else it takes effect when the user switches to Tunnel)."""
        if target() is None or radio_col_var.get() != 'tunnel':
            return
        Colorbytunnel()

    def _step_nudge(delta):
        try:
            v = int(step_entry.get())
        except (ValueError, TypeError):
            v = 25
        v = min(360, max(0, v + delta))
        step_entry.delete(0, tk.END)
        step_entry.insert(0, v)
        _on_step_change()

    def _step_wheel(e):
        # macOS/Windows use e.delta (sign), Linux uses Button-4/5 (e.num)
        _step_nudge(5 if (getattr(e, 'delta', 0) > 0 or getattr(e, 'num', 0) == 4) else -5)

    step_entry.configure(command=_on_step_change)   # fires on the arrow buttons
    step_entry.bind('<Return>', _on_step_change)     # typed value -> apply on Enter
    step_entry.bind('<FocusOut>', _on_step_change)   # ...or when focus leaves
    step_entry.bind('<MouseWheel>', _step_wheel)
    step_entry.bind('<Button-4>', _step_wheel)
    step_entry.bind('<Button-5>', _step_wheel)

    def on_colbydist():
        SelectDistAtom(win=True)
        Colorbytunneldist()
        radio_col_var.set('distance')

    def SelectDistAtom(win=False):
        """Prompt user to select reference atom(s) for distance-based coloring."""
        Console("OFF")
        selection = PairObj(target(), 'dist_sel')
        if win or selection == []:
            selection = ShowWin('AtomSelection', 'Select atom from which to calculate distance')[0]
            selection = " ".join([str(x) for x in ListAtom(selection)])
            PairObj(target(), 'dist_sel', selection)

        atms = [str(x) for x in ListAtom(selection)]

        if len(atms) > 1:
            atm_names = ListAtom(' '.join(str(x) for x in atms), format='ATOMNAME')
            if len(set(atm_names)) == 1:
                show_txt = f'{"".join(set(atm_names))}, {len(atms)} atom\'s center'         
            else:
                show_txt = str(len(atms)) + ' atom\'s center'
        elif len(atms) == 0:
            return
        else:
            show_txt = " ".join(atms)
        button18.configure(style="Toolbutton", text=show_txt, command=on_colbydist)
        Wait(1)
        return atms

  
    separator3 = ttk.Separator(tab2_appear)
    separator3.configure(orient="vertical")
    separator3.place(anchor="nw", height=55, width=2, x=79, y=168)
 
    radiobutton13 = ttk.Radiobutton(tab2_appear)
    radiobutton13.configure(text='Distance to', variable=radio_col_var, value="distance", command=lambda: Colorbytunneldist())
    radiobutton13.place(anchor="nw", x=72, y=158)

    button18 = ttk.Button(tab2_appear)
    button18.configure(text='select', command=on_colbydist)
    button18.place(anchor="nw", x=165, y=155)

    pertun_chk = tk.BooleanVar()  # Variable to track the checkbox status
    _pertun_busy = {'v': False}
    def on_pertun_toggle():
        # Recolour immediately when the scaling mode is toggled, but only while we
        # are already colouring by distance with a chosen reference -- never pop the
        # atom picker just because this flag changed.
        if radio_col_var.get() != 'distance' or PairObj(target(), 'dist_sel') == []:
            return
        # Colorbytunneldist pumps the event loop via its Wait(1) calls, so a rapid
        # second click can re-enter this handler mid-recolour and corrupt the cached
        # pairs. Guard against re-entrancy, and loop until the applied scaling matches
        # the checkbox's final state (the box's variable is toggled by tk before the
        # command fires, so a click parked behind the guard is still reflected here).
        if _pertun_busy['v']:
            return
        _pertun_busy['v'] = True
        try:
            want, guard = None, 0
            while want != pertun_chk.get() and guard < 20:
                want = pertun_chk.get()
                Colorbytunneldist(prompt_if_unset=False)
                guard += 1
        finally:
            _pertun_busy['v'] = False
    checkbutton7 = ttk.Checkbutton(tab2_appear)
    checkbutton7.configure(text='calc per tunnel', variable=pertun_chk, command=on_pertun_toggle)
    checkbutton7.place(anchor="nw", x=90, y=182)

    # Distance-gradient colouring: a palette (DIST_PALETTES: native-hue OR a real matplotlib
    # colormap sampled to hex) chosen from a dropdown, rendered on an inline strip whose two
    # handles trim the sub-window [t0,t1] used (default full = 0..1). Distances are cached as
    # palette-independent BANDS (see _dist_bands/_band_color), so palette/handle changes just
    # remap without recomputing distances.
    dist_palette_var = tk.StringVar(value='Rainbow')
    dist_grad = {'t0': 0.0, 't1': 1.0}   # handle sub-window over the palette

    def _recolor_with_progress():
        """Apply the distance recolour with a progress popup sized to how slow it'll be:
          - points/balls: the group_and_color loop is ~seconds only on a DENSE tunnel, so a
            determinate bar (ticked by group_and_color) shows only above a size threshold;
            small tunnels recolour fast -> no popup (would just flicker).
          - spheres: mesh rebuild, always a determinate bar (ticked by the sphere build).
          - MergeSph shape: mesh rebuild with no per-step hook -> a static 'please wait' popup.
        """
        global progress_window, progress_var, percent_label
        rep = radio_var.get()
        is_mergesph = (rep == 'shape' and shape_surf_option.get() == 'MergeSph')
        big = CountAtom(f'obj {target()}Cl???????') > _RECOLOR_POPUP_MIN
        if rep in ('points', 'balls') and not big:
            Colorbytunneldist(prompt_if_unset=False)   # fast enough -> no popup
            return
        progress_window = tk.Toplevel(root)
        progress_window.title('Recolouring')
        progress_window.lift(); progress_window.attributes('-topmost', True)
        if is_mergesph:
            ttk.Label(progress_window, text='Rebuilding shapes, please wait…').pack(padx=24, pady=18)
        else:
            progress_var = tk.IntVar()
            ttk.Progressbar(progress_window, orient='horizontal', length=200,
                            mode='determinate', variable=progress_var, maximum=100).pack(padx=10, pady=(14, 4))
            percent_label = ttk.Label(progress_window, text='0%'); percent_label.pack(pady=(0, 8))
            _attach_elapsed_timer(progress_window)
        progress_window.update()
        _recolor_prog['on'] = not is_mergesph
        try:
            Colorbytunneldist(prompt_if_unset=False)
        finally:
            _recolor_prog['on'] = False
            try:
                progress_window.destroy()
            except tk.TclError:
                pass

    def _apply_dist_colors():
        """Live-recolour when the palette/handles change -- only while colouring by distance
        with a reference already chosen (never pops the atom picker)."""
        if target() is None or radio_col_var.get() != 'distance' or PairObj(target(), 'dist_sel') == []:
            return
        _recolor_with_progress()
    # Fires ~350 ms after a handle is released (recolour is on release, not per drag-motion),
    # so the recolour reacts with a settle delay, not continuously during the drag.
    _apply_dist_debounced = _make_debounced(_apply_dist_colors, delay=350)

    # Dropdown sits right of the 'calc per tunnel' checkbox (free space on that row); the
    # full-width strip below renders the chosen palette, so no separate colour label needed.
    palette_drop = ttk.OptionMenu(tab2_appear, dist_palette_var, 'Rainbow', *DIST_PALETTES.keys())
    palette_drop.place(anchor="nw", x=192, y=181, width=110)

    _grad_repaint = _make_gradient_picker(tab2_appear, x=90, y=205, width=163, height=16,
                                          get_rgb=lambda u: _palette_rgb(dist_palette_var.get(), u),
                                          grad=dist_grad, on_change=_apply_dist_debounced)

    def _swap_gradient():
        # Reverse the gradient direction: swap the near/far handle positions, redraw (the
        # top arrow flips) and recolour. Common enough to warrant a one-click button.
        dist_grad['t0'], dist_grad['t1'] = dist_grad['t1'], dist_grad['t0']
        _grad_repaint()
        _apply_dist_debounced()
    # A tk.Label styled as a button, NOT ttk.Button: the macOS Aqua button clips its label
    # when forced into a small height, showing only a sliver. A Label renders text at any
    # size and honours bg/relief, so the arrow glyph is reliably visible.
    swap_btn = tk.Label(tab2_appear, text='↔', relief='raised', bd=2,
                        bg='#e6e6e6', fg='black', cursor='hand2', font=('TkDefaultFont', 13))
    swap_btn.place(anchor="nw", x=257, y=204, width=46, height=19)

    def _swap_press(_e):
        swap_btn.config(relief='sunken')

    def _swap_release(_e):
        swap_btn.config(relief='raised')
        _swap_gradient()
    swap_btn.bind('<ButtonPress-1>', _swap_press)
    swap_btn.bind('<ButtonRelease-1>', _swap_release)

    def _on_palette_change(*_):
        # New palette -> repaint the strip and reset the handles to the full extent, then
        # live-recolour (debounced) if we're currently colouring by distance.
        dist_grad['t0'], dist_grad['t1'] = 0.0, 1.0
        _grad_repaint()
        _apply_dist_debounced()
    dist_palette_var.trace_add('write', _on_palette_change)

    ### actions section
    label13 = ttk.Label(tab2_appear)
    label13.configure(
        font="TkSmallCaptionFont",
        foreground="#797979",
        text='Actions')
    label13.place(anchor="nw", x=0, y=266)

    separator9 = ttk.Separator(tab2_appear)
    separator9.configure(orient="horizontal")
    separator9.place(anchor="nw", height=2, width=255, x=45, y=274)

    perf_button = ttk.Button(tab2_appear)
    perf_button.configure(style="Toolbutton",
                          text='Restore detail' if perf_var.get() else 'Improve performance',
                          command=improve_performance)
    perf_button.place(anchor="nw", x=0, y=284)


    def create_tooltip(widget, text, delay=775):
        """Attach a hover tooltip to a tkinter widget."""
        tooltip_window = None
        after_id = None

        def show_tooltip(event):
            nonlocal tooltip_window, after_id
            def create_tooltip_window():
                nonlocal tooltip_window
                if tooltip_window:
                    return
                x, y, width, height = widget.bbox("insert")
                x = x + widget.winfo_rootx() + 20
                y = y + height + widget.winfo_rooty() + 20
                tooltip_window = tk.Toplevel(widget)
                tooltip_window.wm_overrideredirect(True)
                tooltip_window.wm_geometry(f"+{x}+{y}")
                label = tk.Label(tooltip_window, text=text, justify='left',
                                background='lightyellow', relief='solid', borderwidth=0.2,
                                font=("tahoma", "8", "normal"))
                label.pack(ipadx=1, ipady=26)
                tooltip_window.lift()
                tooltip_window.transient(widget.winfo_toplevel())

            # Schedule the tooltip to appear after a delay
            after_id = widget.after(delay, create_tooltip_window)

        def hide_tooltip(event):
            nonlocal tooltip_window, after_id
            if after_id:
                widget.after_cancel(after_id)
                after_id = None
            if tooltip_window:
                tooltip_window.destroy()
                tooltip_window = None

        widget.bind('<Enter>', show_tooltip)
        widget.bind('<Leave>', hide_tooltip)


    create_tooltip(perf_button, "Non-destructive speed-up for the current scene: opaque tunnels\n"
                                "render as their surface shell only (the buried interior is never\n"
                                "visible), so meshes have far fewer triangles and points/balls draw\n"
                                "fewer atoms. The tunnel data is untouched -- pathfinding, cross-\n"
                                "section and volume stay exact. Click again to restore full detail.")

    button2 = ttk.Button(tab2_appear)
    button2.configure(style="Toolbutton", text='Recluster', command=Recluster)
    button2.place(anchor="nw", x=0, y=312)
    
    exclude_chk = tk.BooleanVar()  # Variable to track the checkbox status
    checkbutton7 = ttk.Checkbutton(tab2_appear)
    checkbutton7.configure(text='only visible points', variable=exclude_chk)
    checkbutton7.place(anchor="nw", x=70, y=315)

    def on_rotsurf():
        Console("OFF")
        GrabObj(f'{target()}CutPlane')
        ShowMessage('Rotate surface cutplanes now. Press Continue to accept new rotation')
        Wait('continuebutton')
        GrabAll()
        HideMessage()
        Console("hidden")

    button5 = ttk.Button(tab2_appear)
    button5.configure(style="Toolbutton", text='Rotate surf. cut', command=on_rotsurf)
    button5.place(anchor="nw", x=206, y=312)

    ## surface points section
    label10 = ttk.Label(tab2_appear)
    label10.configure(
        font="TkSmallCaptionFont",
        foreground="#797979",
        text='Surface points')
    label10.place(anchor="nw", x=0, y=228)

    separator5 = ttk.Separator(tab2_appear)
    separator5.configure(orient="horizontal")
    separator5.place(anchor="nw", height=2, width=220, x=80, y=235)

    label11 = ttk.Label(tab2_appear)
    label11.configure(
        font="TkSmallCaptionFont",
        foreground="#919191",
        text='All')
    label11.place(anchor="nw", x=10, y=247)

    label12 = ttk.Label(tab2_appear)
    label12.configure(
        font="TkSmallCaptionFont",
        foreground="#919191",
        text='None')
    label12.place(anchor="nw", x=274, y=247)
  
    surf_pts_chk = tk.DoubleVar()  # Variable to track the checkbox status
    scale1 = ttk.Scale(tab2_appear)
    scale1.configure(orient="horizontal", state="normal", variable=surf_pts_chk, command=ml_outside_points)
    scale1.place(
        anchor="nw",
        relwidth=0.5,
        relx=0.0,
        width=80,
        x=30,
        y=245)

    # --- "Follow YASARA" window behaviour --------------------------------
    # When checked (default) the dialog rides forward with YASARA but sinks
    # behind other apps: we keep it -topmost only while an "ours" app (YASARA
    # or this dialog's own Tk process) is frontmost. Where the frontmost-app
    # query is unavailable (non-macOS or lsappinfo missing) we fall back to
    # plain always-on-top so the box never disappears behind YASARA.
    _follow_pid, _ = _macos_frontmost_app()          # probe once
    _follow_available = _follow_pid is not None
    _follow_own_pid = os.getpid()
    _follow_state = {'topmost': True, 'after_id': None}

    def _follow_is_ours(pid, name):
        return pid == _follow_own_pid or 'yasara' in (name or '').lower()

    def _set_topmost(on):
        if _follow_state['topmost'] != on:
            root.attributes('-topmost', on)
            _follow_state['topmost'] = on

    def _follow_tick():
        if not continue_loop:                        # dialog is closing -> stop the loop
            return
        try:
            if always_on_top_var.get():
                if not _follow_available:
                    _set_topmost(True)               # fallback: plain always-on-top
                else:
                    pid, name = _macos_frontmost_app()
                    if pid is not None:              # ignore transient query failures
                        _set_topmost(_follow_is_ours(pid, name))
        except tk.TclError:
            return                                   # root went away mid-tick
        except Exception:
            pass                                     # never let one bad tick kill the loop
        _follow_state['after_id'] = root.after(450, _follow_tick)

    def toggle_always_on_top():
        Console("OFF")
        if always_on_top_var.get():
            # Attached mode: pop to front (poll then manages the level) and drop the
            # standalone Dock/Cmd-Tab presence so the dialog rides with YASARA.
            _set_topmost(True)
            if _follow_available:
                _macos_set_accessory(True)
        else:
            # Independent mode: give it back its own switcher entry so it can be
            # reached with Cmd-Tab when it isn't floating on top.
            _set_topmost(False)
            if _follow_available:
                _macos_set_accessory(False)

    # Variable for the follow/always-on-top checkbox
    always_on_top_var = tk.BooleanVar(value=True)
    checkbutton21 = ttk.Checkbutton(root)
    _follow_label = 'Follow YASARA' if _follow_available else 'Keep dialog on top'
    checkbutton21.configure(text=_follow_label, variable=always_on_top_var, command=toggle_always_on_top)
    checkbutton21.place(anchor="nw", x=5, y=405)
    if _follow_available and always_on_top_var.get():
        _macos_set_accessory(True)                   # attached by default
    _follow_tick()                                   # start the poll loop

    button4 = ttk.Button(root)
    button4.configure(text='Exit', command=on_cancel)
    button4.place(anchor="nw", x=256, y=401)

    separator7 = ttk.Separator(root)
    separator7.configure(orient="horizontal")
    separator7.place(anchor="nw", height=2, width=310, x=5, y=400)


    # --------------------------------------------------------
    #  TAB 3 — INSPECT TUNNEL
    #  Sections: tunnel selector, amino acid display options,
    #  surface display, cross-section / diameter analysis,
    #  pathfinding
    #  Callbacks: inspect_changed, on_tnl_aas, on_tnl_aa_bb,
    #    on_tnl_aa_lab, on_tnl_aa_surf, on_diameter,
    #    draw_diameter_plot, on_make_path
    # --------------------------------------------------------
    tab3_inspect = ttk.Frame(notebook)
    tab3_inspect.configure(height=375, width=310)  # Set dimensions as needed
    notebook.add(tab3_inspect, text='Inspect Tunnel', padding=0)  # Add tab1_mktun as the second tab

    tnl_insp_label = tk.Label(tab3_inspect, text=f"Select:")
    tnl_insp_label.place(anchor="nw", x=2, y=1)

    zoomsteps = 10
    def inspect_changed(*args):
        """Callback when the tunnel selector dropdown changes. Zooms to the selected tunnel."""
        if 'initializing' in globals():
            global initializing
        elif 'initializing' not in locals():
            initializing = False
        if initializing:
            return
        Console("OFF")
        _xsec_cache_clear()   # different tunnel selected -> refetch its cross-section points
        if target() != None:
            targ = target()
            tnl_name = get_tnl_name()
            ShowObj(tnl_name)
            DelObj(f'???_slice ???_axis ???_xsec')
            if tnl_insp_option.get() != 'All':
                tnl_objnum = re.findall(r"\d+(?=:)", tnl_insp_option.get())[0]
                SwitchObj(f'{targ}Cl???????? ???_sphere ???_shape', 'OFF')
                SwitchObj(tnl_objnum, 'ON')
                ZoomAtom(f'Obj {tnl_name}?', zoomsteps)
                Wait(zoomsteps)
                CellAuto(1, 'cuboid', f'obj {tnl_objnum}')
                SwitchObj('SimCell', 'off')
                NameObj('SimCell', 'CntrOfRot')
                on_tnl_aas()
                place_crosssection()
            else:
                forget_crosssection()
                ZoomAtom('all', zoomsteps)
                DelObj('CntrOfRot')
                MarkAtom('none')
                SwitchObj(f'{targ}Cl???????? {targ}excluded {targ}Close2Surf {targ}Close2Prot', 'OFF')
                SwitchObj(ListObj(f'{targ}Cl???????')[:5], "ON")
            HideSurfObj(f'{targ}tnlAAsurf')
            Wait(1)
        Console("hidden")

    tnl_insp_options_list = ['All']
    if target() != None:
        for x in ListObj(f'{target()}Cl???????', format='OBJNUM: OBJNAME'):
            tnl_insp_options_list.append(x)

    tnl_insp_option = tk.StringVar(value='All')  # Set default value

    dropdown_insp = ttk.OptionMenu(tab3_inspect, tnl_insp_option, tnl_insp_option.get(), *tnl_insp_options_list)
    dropdown_insp.place(anchor="nw", width=150, height=27, x=52, y=0)
    tnl_insp_option.trace_add("write", inspect_changed)

    def update_option_menu(parent, variable, options, current_value=None):
        if hasattr(parent, 'dropdown_insp'):
            parent.dropdown_insp.destroy()

        parent.dropdown_insp = ttk.OptionMenu(parent, variable, current_value if current_value else options[0], *options)
        parent.dropdown_insp.place(anchor="nw", width=150, height=27, x=52, y=0)



    # row 2, separater show/hide
    label2 = ttk.Label(tab3_inspect)
    label2.configure(
        font="TkSmallCaptionFont",
        foreground="#797979",
        text='Show/Hide amino acids\' ...')
    label2.place(anchor="nw", x=0, y=26)

    separator1 = ttk.Separator(tab3_inspect)
    separator1.configure(orient="horizontal")
    separator1.place(anchor="nw", height=2, width=170, x=141, y=35)



    # row 3, AAs and color label
    def on_tnl_aas():
        """Toggle display of tunnel-lining amino acid sidechains."""
        Console("OFF")
        tnl_name = get_tnl_name()
        if tnl_name != None:
            SwitchObj(f'{tnl_name}A', convert_status(tnl_aas.get()))
            Wait(1)
            Console("hidden")


    tnl_aas = tk.BooleanVar(value=True)
    chkbox_tnl_aas = ttk.Checkbutton(tab3_inspect)
    chkbox_tnl_aas.configure(text='sidechains', variable=tnl_aas, command=on_tnl_aas)
    chkbox_tnl_aas.place(anchor="nw", x=0, y=45)


    #  Tunnel AA backbone atom type
    tnl_res_bbatmtyp_option = tk.StringVar()
    tnl_res_bbatmtyp_option.set("BallSticks")
    tnl_res_bbatmtyp_dropdown = ttk.OptionMenu(tab3_inspect, tnl_res_bbatmtyp_option, "BallSticks", "Sticks", "BallSticks", "Balls")
    tnl_res_bbatmtyp_dropdown.place(anchor="nw", width=85, height=27, x=100, y=68)

    def tnl_res_bbatmtyp_changed(*args):
        if initializing:
            return
        Console("OFF")
        tnl_name = get_tnl_name()
        if tnl_name != None:
            if tnl_res_bbatmtyp_option.get() == 'Sticks':
                StickAtom(f'obj {tnl_name}A atom backbone')
            elif tnl_res_bbatmtyp_option.get() == 'BallSticks':
                BallStickAtom(f'obj {tnl_name}A atom backbone')
            else:
                BallAtom(f'obj {tnl_name}A atom backbone')
        Console("hidden")

    tnl_res_bbatmtyp_option.trace_add("write", tnl_res_bbatmtyp_changed)

    #  Tunnel AAs atom type
    def tnl_res_atmtyp_changed(*args):
        if initializing:
            return
        tnl_name = get_tnl_name()
        if tnl_name != None:
            Console("OFF")
            tnl_aas_name = f'{tnl_name}A'
            if tnl_res_atmtyp_option.get() == 'Sticks':
                StickObj(tnl_aas_name)
            elif tnl_res_atmtyp_option.get() == 'BallSticks':
                BallStickObj(tnl_aas_name)
            else:
                BallObj(tnl_aas_name)
            tnl_res_bbatmtyp_changed()
            Console("hidden")

    tnl_res_atmtyp_option = tk.StringVar()
    tnl_res_atmtyp_option.set("element")
    tnl_res_atmtyp_dropdown = ttk.OptionMenu(tab3_inspect, tnl_res_atmtyp_option, "Sticks", "Sticks", "BallSticks", "Balls")
    tnl_res_atmtyp_dropdown.place(anchor="nw", width=85, height=27, x=100, y=43)
    tnl_res_atmtyp_option.trace_add("write", tnl_res_atmtyp_changed)

    tnl_res_col_label = tk.Label(tab3_inspect, text=f"Color atoms:")
    tnl_res_col_label.place(anchor="nw", x=200, y=46)

    tnl_res_atmtyp_changed()


    # row 4, AA bbs and color dropdown

    def on_tnl_aa_bb():
        """Toggle display of backbone atoms in tunnel-lining residues."""
        Console("OFF")
        tnl_name = get_tnl_name()
        if tnl_name != None:
            if tnl_aa_bb.get():
                ShowAtom(f'obj {tnl_name}A atom C N O')
            else:
                HideAtom(f'obj {tnl_name}A atom C O or res !pro atom N')
            Wait(1)
        Console("hidden")

    tnl_aa_bb = tk.BooleanVar(value=False)
    chkbox_tnl_aa_bb = ttk.Checkbutton(tab3_inspect)
    chkbox_tnl_aa_bb.configure(text='backbones', variable=tnl_aa_bb, command=on_tnl_aa_bb)
    chkbox_tnl_aa_bb.place(anchor="nw", x=0, y=70)


    tnl_res_bbatmtyp_changed()

    tnl_res_col_option = tk.StringVar()
    tnl_res_col_option.set("element")
    tnl_res_col_dropdown = ttk.OptionMenu(tab3_inspect, tnl_res_col_option, "element", "element", "restype", "Bfactor", "SecStr", "Occupancy", "Distance to tunnel", "choose..")
    tnl_res_col_dropdown.place(anchor="nw", width=115, height=27, x=200, y=63)

    # Tunnel aa color selection
    def col_by_dist_to_tun(tunnel):
        Console("OFF")
        if tnl_aa_surf.get():
            SwitchObj(f'{target()}tnlAAsurf', 'OFF')
        tnl_aa_obj = ListObj(NameObj(tunnel)[0] + 'A')[0]
        tnl_aa_atms = ListAtom(f'obj {tnl_aa_obj}')
        tnl_points = ListObj(tunnel)[0]

        TransferObj(tnl_aa_obj, tnl_points, 'fix')

        min_color = 100
        max_color = 360

        disto = [round(Distance(x, ListAtom(f'obj {tnl_points} with minimum distance from {x}')[0])[0],2) for x in tnl_aa_atms]
        all_cols = rescale_floats_to_range(disto, int(min_color), int(max_color))

        for i in range(len(tnl_aa_atms)):
            ColorAtom(tnl_aa_atms[i], int(all_cols[i]))

        if tnl_surf_col_option.get() == 'atomcol':
            for i in range(len(tnl_aa_atms)):
                ColorAtom(f'obj {target()}tnlAAsurf with distance < 0.1 from {tnl_aa_atms[i]}', int(all_cols[i]))
        if tnl_aa_surf.get():
            SwitchObj(f'{target()}tnlAAsurf', 'On')
        Wait(1)
        Console("hidden")

    # Function to handle the selection change
    def tnl_res_col_changed(*args):
        Console("OFF")
        tnl_name = get_tnl_name()
        if tnl_name != None:
            tnl_aas_name = f'{tnl_name}A'
            if tnl_res_col_option.get() != "Distance to tunnel" and tnl_res_col_option.get() != "choose..":
                ColorObj(tnl_aas_name, tnl_res_col_option.get())
                if tnl_surf_col_option.get() == 'atomcol':
                    ColorObj(f'{target()}tnlAAsurf', tnl_res_col_option.get())
            else:
                if tnl_res_col_option.get() == "Distance to tunnel":
                    for obj in ListObj(tnl_name, format='OBJNAME'):
                        col_by_dist_to_tun(obj)
                else:
                    col = ShowWin("ColorSelection","Select tunnel residues color", "Bow","Background","100")[0]
                    ColorObj(tnl_aas_name, col)
                    if tnl_surf_col_option.get() == 'atomcol':
                        ColorObj(f'{target()}tnlAAsurf', col)
            Wait(1)
        Console("hidden")

    # Link the function to the variable, so it gets called when the selection changes
    tnl_res_col_option.trace_add("write", tnl_res_col_changed)



    # row 5, AA label, size, col
    def on_tnl_aa_lab(*args):
        if initializing:
            return
        Console("OFF")
        tnl_name = get_tnl_name()
        if tnl_name != None:
            UnlabelAtom(f'obj {tnl_name}A')
            if tnl_aa_lab.get():
                if tnl_res_lab_col_option.get() == 'auto':
                    ca_list = ListAtom(f'obj {tnl_name}A atom CA')
                    for ca in ca_list:
                        ca_col = ColorAtom(ca)[0]
                        LabelAtom(ca, 'RESNAME1RESNUM', tnl_aa_lab_size.get(), get_contrasting_text_color(hue_to_rgb(str(ca_col)))[1:])
                else:
                    LabelAtom(f'obj {tnl_name}A atom CA', 'RESNAME1RESNUM', tnl_aa_lab_size.get(), tnl_res_lab_col_option.get())
            Wait(1)
        Console("hidden")

    tnl_aa_lab = tk.BooleanVar(value=False)
    chkbox_tnl_aa_lab = ttk.Checkbutton(tab3_inspect)
    chkbox_tnl_aa_lab.configure(text='label', variable=tnl_aa_lab, command=on_tnl_aa_lab)
    chkbox_tnl_aa_lab.place(anchor="nw", x=0, y=95)

    tnl_aa_lab_col_label = tk.Label(tab3_inspect, text=f"Color:")
    tnl_aa_lab_col_label.place(anchor="nw", x=200, y=94)

    tnl_res_lab_col_option = tk.StringVar()
    tnl_res_lab_col_option.set("black")
    tnl_res_lab_col_dropdown = ttk.OptionMenu(tab3_inspect, tnl_res_lab_col_option, "black", "black", "white", "auto")
    tnl_res_lab_col_dropdown.place(anchor="nw", width=70, height=27, x=245, y=93)

    tnl_res_lab_col_option.trace_add("write", on_tnl_aa_lab)


    def new_tnl_aa_lab_size(var, label, n=2):
        if initializing:
            return
        tnl_name = get_tnl_name()
        if tnl_name != None:
            Console('off')        
            UnlabelAtom(f'obj {tnl_name}A')
            on_tnl_aa_lab()
            label.config(text=f"{var.get():.{n}f}")
            Console("hidden")

    tnl_aa_lab_size = tk.DoubleVar(value=0.22)
    tnl_aa_lab_size_value_label = tk.Label(tab3_inspect, text=f"{tnl_aa_lab_size.get():.12}")
    tnl_aa_lab_size_value_label.place(anchor="nw", x=150, y=94)
    tnl_aa_lab_size_scale = ttk.Scale(tab3_inspect, from_=0.2, to=0.8, orient="horizontal", variable=tnl_aa_lab_size,
                            command=lambda value, var=tnl_aa_lab_size, label=tnl_aa_lab_size_value_label: new_tnl_aa_lab_size(var, label))
    tnl_aa_lab_size_scale.place(anchor="nw", x=53, y=95, width=95)
    new_tnl_aa_lab_size(tnl_aa_lab_size, tnl_aa_lab_size_value_label)


    #  Tunnel AA ss style
    def tnl_res_ssstyle_changed(*args):
        Console("OFF")
        tnl_name = get_tnl_name() + "A"
        if tnl_name != None:
            if tnl_aa_ss.get():
                if tnl_res_ssstyle_option.get() != 'Trace':
                    ShowSecStrObj(tnl_name, tnl_res_ssstyle_option.get())
                    HideTrace(f'obj {tnl_name} atom CA')
                    tnl_res_atmtyp_changed()
                else:
                    HideSecStrObj(tnl_name)
                    ShowTrace(f'obj {tnl_name} atom CA')
                    BallStickAtom(f'obj {tnl_name} atom CA')
            else:
                HideSecStrObj(tnl_name)
                HideTrace(f'obj {tnl_name} atom CA')
        Console("hidden")



    # row 6, ss
    tnl_aa_ss = tk.BooleanVar(value=True)
    chkbox_tnl_aa_ss = ttk.Checkbutton(tab3_inspect)
    chkbox_tnl_aa_ss.configure(text='SecStr', variable=tnl_aa_ss, command=tnl_res_ssstyle_changed)
    chkbox_tnl_aa_ss.place(anchor="nw", x=0, y=120)

    tnl_res_ssstyle_option = tk.StringVar()
    tnl_res_ssstyle_option.set("Ribbon")
    tnl_res_ssstyle_dropdown = ttk.OptionMenu(tab3_inspect, tnl_res_ssstyle_option, "Ribbon", "Ribbon", "Cartoon", "Tube", "Trace")
    tnl_res_ssstyle_dropdown.place(anchor="nw", width=100, height=27, x=65, y=117)
    tnl_res_ssstyle_option.trace_add("write", tnl_res_ssstyle_changed)



    # row 7, surf
    tnl_aa_surf = tk.BooleanVar(value=False)

    def on_tnl_aa_surf(*args, dist=None):
        """Toggle molecular surface display around tunnel-lining residues."""
        if initializing:
            return
        tar = target()
        if tar != None:
            Console("OFF")
            tnl_name = get_tnl_name()
            if tnl_name != None and tnl_aa_surf.get():
                if ListObj(f'{tar}tnlAAsurf') == []:
                    new = DuplicateObj(tar)[0]
                    HideObj(new)
                    HideSecStrObj(new)
                    MoveObj(new, x=0.01)
                    NameObj(new, f'{tar}tnlAAsurf')
                if dist == None:
                    max1 = ListAtom(f'obj {tnl_name}A with maximum distance from obj {tnl_name}')[0]
                    max2 = ListAtom(f'obj {tnl_name} with minimum distance from {max1}')[0]

                    # Ensure each 'A' companion shares its cluster object's coordinate
                    # system before the cross-object Distance() below (else YASARA
                    # raises error 467). A wildcard PairObj only returns entries for
                    # objects that HAVE the key, so an unset object is simply absent
                    # from the list — an empty/short list means "not all fixed yet".
                    # Compare the fixed-count against the number of 'A' objects rather
                    # than using all(), which is True on the empty (nothing-fixed) list.
                    n_a_objs = len(ListObj(f'{tar}Cl???????A'))
                    n_fixed = sum(1 for x in PairObj(f'{tar}Cl???????A', 'fix') if x == 'True')
                    if n_fixed < n_a_objs:
                        ShowMessage('Aligning coordinate systems, please wait.')
                        Wait(1)
                        transf_and_fix_ss(tar)
                        HideMessage()
                    maxd = Distance(max1, max2)[0]

                    try:
                        mind = float(PairObj(tar, 'max_ball_protein')[0])
                    except IndexError:
                        mind = float(PairObj('All', 'max_ball_protein')[0])

                    dist = round(mind + (tnl_aa_surf_dist.get() / 100.0) * (maxd - mind), 2)

                # The surf-dup carries only the surface (its atoms are hidden). If the
                # object itself is switched off, ShowSurfAtom draws a surface nobody can
                # see — so make sure it is switched on whenever the surface is enabled.
                SwitchObj(f'{tar}tnlAAsurf', 'on')
                AddEnvRes(f'obj {tar}tnlAAsurf res protein')
                HideSurfObj(f'{tar}tnlAAsurf')
                surf_atms = " ".join(str(x) for x in ListAtom(f'obj {tar}tnlAAsurf res protein with distance < 0.1 from obj {tnl_name}A'))

                ShowSurfAtom(f'{surf_atms} with distance < {dist} from obj {tnl_name}', tnl_surfstyle_option.get(), outcol='atomcol', outalpha=tnl_aa_surf_alpha.get())
            else:
                HideSurfObj(f'obj {tar}tnlAAsurf')

            Wait(1)
            Console("hidden")
 
    def new_tnl_aa_surf_alpha(var, label, n=0):
        on_tnl_aa_surf()
        label.config(text=f"{var.get():.{n}f}")

    chkbox_tnl_aa_surf = ttk.Checkbutton(tab3_inspect)
    chkbox_tnl_aa_surf.configure(text='surface', variable=tnl_aa_surf, command=on_tnl_aa_surf)
    chkbox_tnl_aa_surf.place(anchor="nw", x=0, y=145)

    tnl_surfstyle_option = tk.StringVar()
    tnl_surfstyle_option.set("molecular")
    tnl_surfstyle_dropdown = ttk.OptionMenu(tab3_inspect, tnl_surfstyle_option, "molecular", "molecular", "VdW", "accessible")
    tnl_surfstyle_dropdown.place(anchor="nw", width=110, height=27, x=75, y=142)
    tnl_surfstyle_option.trace_add("write", on_tnl_aa_surf)

    tnl_surf_col_label = tk.Label(tab3_inspect, text=f"Color:")
    tnl_surf_col_label.place(anchor="nw", x=187, y=143)

    def on_tnl_surf_col(*args):
        if tnl_surf_col_option.get() == 'atomcol':
            col = tnl_res_col_option.get()
        elif tnl_surf_col_option.get() == 'element':
            col = 'element'
        else:
            col = ShowWin("ColorSelection","Select tunnel surface color", "Bow","Background","100")[0]
        if col == 'choose..':
            col = ColorAtom(f'obj {get_tnl_name()}A')[0]
        elif col == 'Distance to tunnel':
            obj = get_tnl_name()
            col = ColorAtom(f'obj {obj}A with minimum distance from obj {obj}')[0]
        ColorAtom(f'obj {target()}tnlAAsurf', col)
        on_tnl_aa_surf()
        Wait(1)

    tnl_surf_col_option = tk.StringVar()
    tnl_surf_col_option.set("black")
    tnl_surf_col_dropdown = ttk.OptionMenu(tab3_inspect, tnl_surf_col_option, "atomcol", "atomcol", 'element', "choose...")
    tnl_surf_col_dropdown.place(anchor="nw", width=83, height=27, x=232, y=142)

    tnl_surf_col_option.trace_add("write", on_tnl_surf_col)


    tnl_aa_surf_alpha_label = tk.Label(tab3_inspect, text=f"alpha:")
    tnl_aa_surf_alpha_label.place(anchor="nw", x=0, y=165)

    tnl_aa_surf_alpha = tk.DoubleVar(value=75)
    tnl_aa_surf_alpha_value_label = tk.Label(tab3_inspect, text=f"{tnl_aa_surf_alpha.get():.0f}")
    tnl_aa_surf_alpha_value_label.place(anchor="nw", x=288, y=166)
    tnl_aa_surf_alpha_scale = ttk.Scale(tab3_inspect, from_=0, to=100, orient="horizontal", variable=tnl_aa_surf_alpha,
                            command=lambda value, var=tnl_aa_surf_alpha, label=tnl_aa_surf_alpha_value_label: new_tnl_aa_surf_alpha(var, label))
    tnl_aa_surf_alpha_scale.place(anchor="nw", x=45, y=167, width=240)
    new_tnl_aa_surf_alpha(tnl_aa_surf_alpha, tnl_aa_surf_alpha_value_label)

    tnl_aa_surf_dist_label = tk.Label(tab3_inspect, text=f"dist:")
    tnl_aa_surf_dist_label.place(anchor="nw", x=0, y=184)


    def new_tnl_aa_surf_dist(var, label, n=1):
        on_tnl_aa_surf()
        label.config(text=f"{var.get():.{n}f}")

    tnl_aa_surf_dist = tk.DoubleVar(value=0.5)
    tnl_aa_surf_dist_value_label = tk.Label(tab3_inspect, text=f"{tnl_aa_surf_dist.get():.0f}")
    tnl_aa_surf_dist_value_label.place(anchor="nw", x=288, y=184)
    tnl_aa_surf_dist_scale = ttk.Scale(tab3_inspect, from_=0, to=100, orient="horizontal", variable=tnl_aa_surf_dist,
                            command=lambda value, var=tnl_aa_surf_dist, label=tnl_aa_surf_dist_value_label: new_tnl_aa_surf_dist(var, label))
    tnl_aa_surf_dist_scale.place(anchor="nw", x=30, y=186, width=255)
    new_tnl_aa_surf_dist(tnl_aa_surf_dist, tnl_aa_surf_dist_value_label)


    ## Tunnel diameter section
    separator1 = ttk.Separator(tab3_inspect)
    separator1.configure(orient="horizontal")
    separator1.place(anchor="nw", height=2, width=242, x=70, y=212)

    label2 = ttk.Label(tab3_inspect)
    label2.configure(
        font="TkSmallCaptionFont",
        foreground="#797979",
        text='Tunnel crosssection')
    label2.place(anchor="nw", x=0, y=205)

    dpi = 52  
    tnl_dia_canv_width = 155
    tnl_dia_canv_height = 130
    figsize_inches = (tnl_dia_canv_width / dpi, tnl_dia_canv_height / dpi)  # Convert pixel dimensions to inches

    fig = Figure(dpi=dpi)
    ax = fig.add_subplot(111)

    # Update the figure size
    fig.set_size_inches(figsize_inches[0], figsize_inches[1], forward=True)

    # Now calculate plot_width and plot_height based on figure size and dpi
    plot_width, plot_height = figsize_inches[0] * dpi, figsize_inches[1] * dpi


    def make_axis(tnl_name):
        """Create a PCA-based principal axis for a tunnel as a YASARA object.

        Computes the tunnel's principal axis (PCA) directly from the tunnel
        points' GLOBAL coordinates and places two marker atoms at the extreme
        points along that axis (inner 'In' / outer 'Out'), with a black arrow
        between them.

        Returns (ext_center_atom, ext_outer_atom) — atom numbers of the two endpoints.

        NOTE: this used to save the tunnel surface to a .obj, reload the vertices
        via a centred PDB and TransferObj them back onto the tunnel. That round-trip
        (plus a "rotate 180 deg around y for transfer to work" hack) mis-restored the
        global frame, so the arrow appeared shifted in space. The tunnel points are
        already in the scene at known global positions, so we read them directly and
        skip the whole fragile export/re-import.
        """
        DelObj('???_axis ???_slice ???_xsec')

        # Tunnel points in global coordinates; the axis lives in the same frame.
        pts = np.array(PosAtom(f'obj {tnl_name}', coordsys='global')).reshape(-1, 3)
        axis = find_principal_axis(pts)
        p1, p2 = find_extreme_points(pts, axis)   # two extreme tunnel points (global)

        # Two marker atoms placed directly at the extremes (global coords).
        n = BuildAtom('C', copies=2)              # one object, two overlaid atoms
        endatoms = ListAtom(f'obj {n}')
        PosAtom(f'atom {endatoms[0]}', x=p1[0], y=p1[1], z=p1[2], coordsys='global')
        PosAtom(f'atom {endatoms[1]}', x=p2[0], y=p2[1], z=p2[2], coordsys='global')

        # Inner = closer to the protein centre, outer = farther (tunnel mouth).
        cx, cy, cz = PosAtom(f"obj {target()}", mean=True, coordsys='global')
        cen = BuildAtom("C")
        PosAtom(f"obj {cen}", x=cx, y=cy, z=cz, coordsys='global')
        ext_center = ListAtom(f'obj {n} with minimum distance from obj {cen}')[0]
        ext_outer = ListAtom(f'obj {n} with maximum distance from obj {cen}')[0]
        NameAtom(ext_center, 'In')
        NameAtom(ext_outer, 'Out')
        DelObj(f'{cen}')

        # show arrow between extremes
        ShowArrow('atatom', ext_outer, 'atatom',  ext_center, color='black')
        StickObj(n)
        ColorObj(n, 'black')
        NameObj(n, f'{ListObj(tnl_name)[0]:03d}_axis')
        return ext_center, ext_outer

    def draw_diameter_plot(tnl_name, slice_obj, fig, ax, on_canvas=True, only_area=True):
        """Draw the cross-section plot for a tunnel at the current slice position.

        Projects tunnel points near the cutting plane into 2D, clusters them,
        calculates areas (and optionally inscribed circles), and renders the
        plot either on the embedded canvas or in a separate matplotlib window.
        """
        if on_canvas:
            ax.clear()
        else:
            fig, ax = plt.subplots()

        plane_points_pos = np.array(PosAtom(f'obj {slice_obj}', coordsys='global')).reshape(-1,3)
        # Tunnel coords + atom numbers are cached across slider moves (they don't
        # change during a drag) -> no per-move O(N) PosAtom/ListAtom round-trip.
        tnl_points_pos, tnl_points = _xsec_get(tnl_name)
        ball_spacing = float(PairObj(target(), 'ball_spacing')[0])
        # Slab half-thickness for "points near the cutting plane". The old value
        # (ball_spacing / 800 ~ 0.0004 A) was far thinner than the point grid, so
        # it caught ~zero points and the cross-section plot was always blank.
        # ball_spacing / 2 catches ~one monolayer per plane position (adjacent
        # slices tile without gaps) -> the buffered-circle union approximates the
        # true cross-sectional area.
        threshold = ball_spacing / 2
        near_plane, near_indices, not_near_plane, not_near_indices = find_points_near_plane(tnl_points_pos, plane_points_pos, distance_threshold=threshold)
        if near_indices.size > 0:
            near_atoms = tnl_points[near_indices]
            # Mark the 'cutp' segment incrementally: reset only what we marked last
            # move (O(k)), not all N points. First touch of a tunnel does one full
            # reset to clear any stale marks, then stays O(k) thereafter.
            prev_cutp = _xsec_cache['cutp']
            if prev_cutp is None:
                SegAtom(tnl_points, '.')
            elif len(prev_cutp):
                SegAtom(prev_cutp, '.')
            SegAtom(near_atoms, 'cutp')
            _xsec_cache['cutp'] = near_atoms
            if cut_points_chk.get():
                # Hide the whole object cheaply, then show only the near points (O(k)),
                # rather than enumerating the huge not-near set in one HideAtom string.
                HideObj(tnl_name)
                ShowAtom(near_atoms)

            near_plane_points_projected, original_indices, plane_origin, u, v = project_points_onto_plane(near_plane, plane_points_pos)

            if len(near_plane_points_projected) > 0:
                # Apply DBSCAN clustering to the near plane points
                eps = 1.4
                min_samples = 1  # Minimum samples for a core point, this could be adjusted based on your point density
                cluster_labels = cluster_points_with_dbscan(near_plane_points_projected, eps, min_samples)

                # You can now separate the points by clusters based on the labels
                unique_labels = set(cluster_labels)
                clusters = {label: near_plane_points_projected[cluster_labels == label] for label in unique_labels if label != -1}
                # Initialize variables to track the min and max bounds of all clusters
                all_data_x_min = float('inf')
                all_data_x_max = float('-inf')
                all_data_y_min = float('inf')
                all_data_y_max = float('-inf')

                cluster_shapes = {}
                total_area = 0
                # Collect per-cluster shapes for the optional 3D cross-section overlay
                # (built only for the live preview, not the detailed separate window).
                xsec_build = on_canvas and xsec_mode.get() == 'section'
                xsec_shapes = []

                if not only_area:
                    StickAtom(tnl_points)

                # Loop over clusters to calculate bounds
                for label, cluster_points in clusters.items():
                    area, merged_shape = calculate_area_of_points(cluster_points, ball_spacing * 2, radius=0.75)
                    # print(f'cluster {label} has area {area}')
                    x, y = merged_shape.exterior.xy
                    if not only_area:
                        try:
                            max_circle_center, max_circle_radius = find_maximum_inscribed_circle(merged_shape)
                            if max_circle_center is not None:
                                # Find the index of the closest original 3D point
                                closest_point_index = find_closest_point_index(near_plane_points_projected, max_circle_center)
                                original_point_index = original_indices[closest_point_index]
                                correct_index = near_indices[original_point_index]
                                descriptor = int(tnl_points[correct_index])
                                BallAtom(descriptor)
                        except AttributeError:
                            continue
                    total_area += area
                    cluster_shapes[label] = (x, y, area)
                    if xsec_build:
                        xsec_shapes.append(merged_shape)
                    # Update the bounds for all clusters
                    all_data_x_min = min(all_data_x_min, min(x))
                    all_data_x_max = max(all_data_x_max, max(x))
                    all_data_y_min = min(all_data_y_min, min(y))
                    all_data_y_max = max(all_data_y_max, max(y))

                # Build the 3D cross-section overlay (filled + outline) on the plane.
                if xsec_build:
                    _build_xsec_object(xsec_shapes, u, v, plane_origin,
                                       f'{ListObj(tnl_name)[0]:03d}_xsec',
                                       ListObj(tnl_name)[0], cut_axis_alpha.get())

                # Set the axis limits after determining the bounds for all clusters
                all_data_width = all_data_x_max - all_data_x_min
                all_data_height = all_data_y_max - all_data_y_min
                max_data_extent = max(all_data_width, all_data_height)
                fig_aspect_ratio = plot_width / plot_height

                if np.isnan(max_data_extent) or not np.isfinite(max_data_extent):
                    return

                if all_data_width > all_data_height:
                    new_y_half_extent = max_data_extent / fig_aspect_ratio / 2
                    ax.set_ylim([all_data_y_min - new_y_half_extent, all_data_y_max + new_y_half_extent])
                    ax.set_xlim([all_data_x_min, all_data_x_max])
                else:
                    new_x_half_extent = max_data_extent * fig_aspect_ratio / 2
                    ax.set_xlim([all_data_x_min - new_x_half_extent, all_data_x_max + new_x_half_extent])
                    ax.set_ylim([all_data_y_min, all_data_y_max])

                # Plot the clusters now with the updated axis limits
                for label, (x, y, area) in cluster_shapes.items():
                    ax.fill(x, y, color='black', label=f'Cluster {label}')
                    ax.plot(x, y, color='cyan', linewidth=0.75, label=f'Cluster {label}')
                    # Add labels only in the separate window
                    if not on_canvas:
                        # Calculate the centroid of the cluster
                        centroid_x = sum(x) / len(x)
                        centroid_y = sum(y) / len(y)

                        # Place the text annotation near the centroid
                        ax.text(centroid_x, centroid_y, f"{area:.2f} \u212B\u00b2", 
                                ha='center', va='center', color='red', fontsize=8.5, fontweight='bold')


                # Set the aspect ratio and other plot properties
                ax.set_aspect('equal', adjustable='datalim')
                ax.set_title(f"total area ({diamter_height.get():.1f}): {total_area:.2f} \u212B\u00b2")
                ax.grid(True, linewidth=0.3, color='gray')


                # Move ticks to the bottom and left spines
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')

                # Redraw the canvas (draw_idle coalesces rapid slider-driven redraws)
                if on_canvas:
                    canvas.draw_idle()
                else:
                    plt.show()

            else:
                ShowMessage('Unexpected state: near_indices and projected points mismatch.')
                wc()
        else:
            ax.clear()
            ax.set_title(f"tunnel crosssection area: 0 \u212B\u00b2")
            ax.axis('off')
            canvas.draw_idle()


    def on_diameter(*args):
        """Callback for cross-section height slider. Builds cutting plane and draws the plot."""
        if initializing:
            return
        Console('OFF')
        if tnl_insp_option.get() != 'All':
            tnl_name = get_tnl_name()

            # if not existant, make axis:
            if ListObj(f'{ListObj(tnl_name)[0]:03d}_axis') == []:
                ext_center, ext_outer = make_axis(tnl_name)
            else:
                DelObj
                ext_center = ListAtom('obj ' + f'{ListObj(tnl_name)[0]:03d}_axis ' + 'atom In')[0]
                ext_outer = ListAtom('obj ' + f'{ListObj(tnl_name)[0]:03d}_axis ' + 'atom Out')[0]

            DelObj(f'{ListObj(tnl_name)[0]:03d}_slice {ListObj(tnl_name)[0]:03d}_xsec')

            # Create square cutting through tunnel along axis
            point1 = PosAtom(ext_outer, coordsys='global')
            point2 = PosAtom(ext_center, coordsys='global')
            
            side_length = CUTTING_PLANE_SIDE_LENGTH

            position = diamter_height.get() / 100
            vertices = square_vertices(point1, point2, side_length, position)

            # display a black square to indicate slice
            for i in range(4):
                n= BuildAtom('du')
                PosAtom(f'Obj {n}', *vertices[i], coordsys='global')
                if i == 0: 
                    slice_obj = n
                else:
                    JoinObj(n, slice_obj)
            NameObj(slice_obj, f'{ListObj(tnl_name)[0]:03d}_slice')
            HideObj(slice_obj)
            # slice_obj is always built (draw_diameter_plot needs it to define the plane),
            # but the visible black rectangle is only drawn in 'plane' mode -- in 'section'
            # mode the filled cross-section overlay stands in for it.
            if xsec_mode.get() == 'plane':
                ShowPolygonAtoms('black', cut_axis_alpha.get(), 4, *ListAtom(f'obj {slice_obj}'))

            draw_diameter_plot(tnl_name, slice_obj, fig, ax)

        else:
            ax.clear()
            ShowMessage('Select a tunnel first')
            Wait(25)
            HideMessage()            
        Wait(1)
        Console("hidden")


    canvas = FigureCanvasTkAgg(fig, master=tab3_inspect)  
    canvas_widget = canvas.get_tk_widget()

    diamter_height = tk.DoubleVar(value=0.5)
    diamter_height_scale = ttk.Scale(tab3_inspect, from_=0, to=100, orient="horizontal", variable=diamter_height, command=on_diameter)

    def on_axis(*args):
        Console('off')
        SwitchObj(f'???_axis', convert_status(axis_chk.get())) 
        Wait(1)   
        Console("hidden")  

    axis_chk = tk.BooleanVar(value=True)  
    axis_button = ttk.Checkbutton(tab3_inspect)
    axis_button.configure(text='axis', variable=axis_chk, command=on_axis)


    def on_cut_points(*args):
        Console('off')
        if not cut_points_chk.get():
            ShowObj(get_tnl_name())
        else:
            if ListAtom(f'obj {get_tnl_name()} segment cutp') != []:
                HideObj(get_tnl_name())
                ShowAtom(f'obj {get_tnl_name()} segment cutp')
            else:
                on_diameter()
        Console("hidden")  

    cut_points_chk = tk.BooleanVar(value=True)
    cut_points_button = ttk.Checkbutton(tab3_inspect)
    cut_points_button.configure(text='only cut pts', variable=cut_points_chk, command=on_cut_points)

    # The cutting plane can be shown EITHER as the rectangular plane OR as the filled
    # cross-section overlay -- not both (they represent the same slice). Radio choice.
    def on_xsec_mode(*args):
        """Switch between the rectangular plane and the cross-section overlay."""
        Console('off')
        DelObj('???_xsec')          # drop the overlay; on_diameter rebuilds per the mode
        on_diameter()               # rebuild slice: shows plane or section as selected
        Console('hidden')

    xsec_mode = tk.StringVar(value='plane')
    plane_radio = ttk.Radiobutton(tab3_inspect, text='plane', variable=xsec_mode,
                                  value='plane', command=on_xsec_mode)
    section_radio = ttk.Radiobutton(tab3_inspect, text='section', variable=xsec_mode,
                                    value='section', command=on_xsec_mode)


    # Alpha for the cutting-plane display: drives BOTH the rectangular plane and the
    # cross-section fill (whichever the plane/section radio selects). Compact debounced
    # Spinbox, like the sphere/shape alpha controls.
    cut_axis_alpha_label = tk.Label(tab3_inspect, text=f"alpha")
    cut_axis_alpha = tk.IntVar(value=90)
    cut_axis_alpha_spin = _numeric_spinbox(tab3_inspect, cut_axis_alpha, 1, 100,
                                           on_diameter, x=40, y=266)
    # _numeric_spinbox self-places on creation; the rest of the cross-section
    # widgets start hidden and appear only via place_crosssection(), so forget it
    # now to match -- otherwise it leaks through in the default 'All' state.
    cut_axis_alpha_spin.place_forget()

    def on_cut_detail():
        Console('off')
        if tnl_insp_option.get() != 'All' and len(ListObj('???_slice')) > 0:
            draw_diameter_plot(get_tnl_name(), ListObj('???_slice')[0], fig, ax, on_canvas=False)
        else:
            ax.clear()
            ShowMessage('Select and slice a tunnel first')
            Wait(25)
            HideMessage()            
        Wait(1)
        Console("hidden")

    # Clicking the preview plot itself opens the detailed (separate-window) plot,
    # so no separate 'Detailed Plot' button is needed.
    canvas_widget.configure(cursor='hand2')
    canvas_widget.bind('<Button-1>', lambda e: on_cut_detail())



    
    def plot_radius_vs_height(circles, only_area=True):
        """
        Plot the radius of the maximum inscribed circles and area as a function of the height.

        :param circles: List of tuples containing (height, descriptor, max_circle_center, max_circle_radius, area).
        :param only_area: Boolean flag to indicate if only the area should be plotted.
        """
        height_to_radii = {}
        height_to_areas = {}

        for height, descriptor, center, radius, area in circles:
            if height not in height_to_radii:
                height_to_radii[height] = []
                height_to_areas[height] = []
            height_to_radii[height].append(radius)
            height_to_areas[height].append(area)

        heights = []
        radii = []
        areas = []

        for height in sorted(height_to_radii.keys()):
            for radius in height_to_radii[height]:
                heights.append(float(height))
                radii.append(radius)
            for area in height_to_areas[height]:
                areas.append(area)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Set the x-ticks every 5 steps using numpy.arange
        ax1.set_xticks(np.arange(0, 100, 5))

        # Enable gridlines
        ax1.grid(True, which='both')

        # If not only_area, add second axis
        if not only_area:
            color = 'tab:blue'
            ax1.set_xlabel('Height')
            ax1.set_ylabel('Maximum Radius of Inscribed Circle', color=color)
            ax1.plot(heights, radii, 'o', linestyle='None', color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:red'
            ax2.set_ylabel('Area', color=color)  # we already handled the x-label with ax1
            ax2.plot(heights, areas, 'o', linestyle='None', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        else:
            color = 'tab:green'
            ax1.set_xlabel('Height')
            ax1.set_ylabel('Area', color=color)
            ax1.plot(heights, areas, 'o', linestyle='None', color=color)
            ax1.tick_params(axis='y', labelcolor=color)

        # Adjust the x-tick labels using numpy.arange
        plt.xticks(np.arange(0, 100, 5))

        # Display the plot
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title('Maximum Radius of Inscribed Circle and Area vs. Height' if not only_area else 'Area vs. Height')
        plt.grid(True)
        plt.show()



    def on_make_path():
        """Find and display the shortest A* path through the selected tunnel."""
        Console('Off')
        ShowObj(get_tnl_name())

        tnl_name = get_tnl_name()  # Replace with the actual tunnel name
        if ListObj(f'{ListObj(tnl_name)[0]:03d}_axis') == []:
            ext_center, ext_outer = make_axis(tnl_name)
        else:
            ext_center = ListAtom('obj ' + f'{ListObj(tnl_name)[0]:03d}_axis ' + 'atom In')[0]
            ext_outer = ListAtom('obj ' + f'{ListObj(tnl_name)[0]:03d}_axis ' + 'atom Out')[0]

        ext_center = ListAtom(f'Obj {tnl_name} with minimum distance from atom {ext_center}')[0]
        ext_outer = ListAtom(f'Obj {tnl_name} with minimum distance from atom {ext_outer}')[0]
        
        HideArrowAtom(f'obj {tnl_name}')
        NameAtom(f'obj {tnl_name}', 'UNL')
        shortest_path_points = find_shortest_path(tnl_name, ball_spacing * 1.02, ball_spacing * connect_cut * 1.021, ext_center, ext_outer, coarse_only=rough_path_chk.get())
        
        if shortest_path_points:
            NameAtom(shortest_path_points, '_SP')
            for i in range(len(shortest_path_points)):
                if i > 0:
                    ShowArrow('atatom', int(shortest_path_points[i]), 'atatom', int(shortest_path_points[i-1]), 0.2, 0, ColorAtom(int(shortest_path_points[i]))[0])
            expose_path()
        else:
            ShowMessage('No path found for this tunnel.')
            Wait('continuebutton')
            HideMessage()
            
        Console('Hidden')


    make_path = ttk.Button(tab3_inspect)
    make_path.configure(text='Path', style='Toolbutton', command=on_make_path)
    
    def expose_path():
        Console('off')
        tnl_name = get_tnl_name()
        SwitchObj(tnl_name, 'ON')
        ShowObj(tnl_name)
        if expose_path_chk.get():
            HideAtom(f'obj {tnl_name} atom UNL')

        Console("hidden")  
        
    rough_path_chk = tk.BooleanVar(value=True)  
    rough_path_button = ttk.Checkbutton(tab3_inspect)
    rough_path_button.configure(text='coarse', variable=rough_path_chk)

    expose_path_chk = tk.BooleanVar(value=True)  
    expose_path_button = ttk.Checkbutton(tab3_inspect)
    expose_path_button.configure(text='expose', variable=expose_path_chk, command = expose_path)


    def on_adjust():
        Console("OFF")
        GrabObj(f'???_axis ???_slice')
        CenterObj(f'???_axis')
        ShowMessage('Rotate axis now. Press Continue to accept new rotation')
        Wait('continuebutton')
        GrabAll()
        HideMessage()
        on_diameter()
        Console("hidden")        

    adjust_ax = ttk.Button(tab3_inspect)
    adjust_ax.configure(text='Adjust', style='Toolbutton', command=on_adjust)

    def on_reset():
        Console('off')
        DelObj('???_axis')
        if tnl_insp_option.get() != 'All':
            make_axis(get_tnl_name())
            on_diameter()
        else:
            ShowMessage('Select a tunnel first.')
            Wait(20)
            HideMessage()
        Console("hidden")

    reset_ax = ttk.Button(tab3_inspect)
    reset_ax.configure(text='Reset', style='Toolbutton', command=on_reset)

    def on_diam_up():
        diamter_height.set(diamter_height.get() + 0.5)
        on_diameter()

    def on_diam_down():
        diamter_height.set(diamter_height.get() - 0.5)
        on_diameter()

    diam_down = ttk.Button(tab3_inspect)
    diam_down.configure(text='<', style='Toolbutton', command=on_diam_down)

    diam_up = ttk.Button(tab3_inspect)
    diam_up.configure(text='>', style='Toolbutton', command=on_diam_up)

    # --------------------------------------------------------
    #  DIALOG MAINLOOP
    # --------------------------------------------------------

    # Hide Tab 2 and Tab 3 if no tunnels exist yet
    if target() == None:
        notebook.tab(1, state='hidden')
        notebook.tab(2, state='hidden')

    Console("hidden")
    initializing = False

    root.mainloop()


# Tell YASARA the plugin has finished, exactly once, on every exit path. Without this
# YASARA reports "The plugin stopped without notice ... forgot to call plugin.end". The
# atexit backstop covers exits that don't return through the loop below (e.g. Cmd-Q,
# window close), while the explicit call handles the normal case; the guard prevents a
# double call. Errors are swallowed -- at this point the script is done regardless.
import atexit as _atexit
_plugin_ended = []
def _end_plugin_once():
    if _plugin_ended:
        return
    _plugin_ended.append(True)
    try:
        plugin.end()
    except BaseException:
        pass
_atexit.register(_end_plugin_once)

# Keep showing the dialog until the user clicks "Cancel"
while tunneler_dialog():
    pass

_end_plugin_once()

