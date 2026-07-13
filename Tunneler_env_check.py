"""Environment / dependency checker for the Tunneler YASARA plugin.

Runs at plugin startup, BEFORE the heavy third-party imports, so that a broken
Python environment produces a clear diagnosis and a working repair path instead
of a raw ImportError crash deep inside matplotlib/tkinter.

It answers, in order:
  1. Which Python is YASARA actually running?  (sys.executable / version)
  2. Is it YASARA's bundled "epy" Python?  -> that one can't pip-install; the
     user needs to point yasara.ini's PythonPath at a real Python instead.
  3. Does that Python have every dependency, and does each one actually WORK?
     A plain `import matplotlib` is not enough: under Tcl/Tk 9 an old matplotlib
     imports fine but its tkagg backend fails ("Failed to load Tcl_SetVar").
     So every dependency has a *functional probe*, and results are classified
     as ok / missing / broken.
  4. If something is wrong, is it because the Python is undesirable (epy), or a
     genuine missing/outdated package that we can install/upgrade?

The core functions use only the standard library, so this module is safe to
import before numpy/matplotlib/etc. exist, and is unit-testable headlessly.
YASARA is only touched by the thin GUI wrapper `ensure_dependencies()`.
"""
import sys
import os
import subprocess
import importlib


# (pip name, import name, functional probe). The probe must exercise the part of
# the package the plugin actually relies on -- that is what catches "installed
# but broken" cases that a bare import would miss.
DEPENDENCIES = [
    ("numpy",        "numpy",      "import numpy; numpy.zeros(3).sum()"),
    ("scipy",        "scipy",      "from scipy.spatial import ConvexHull, Delaunay, Voronoi"),
    ("scikit-learn", "sklearn",    "from sklearn.cluster import DBSCAN"),
    ("shapely",      "shapely",    "from shapely.geometry import Point, Polygon"),
    # The critical one: import the tkagg backend, not just matplotlib.
    ("matplotlib",   "matplotlib", "from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg"),
]


# --------------------------------------------------------------------------
#  Core (standard-library only; no YASARA, no third-party imports at module
#  load time -- everything heavy is probed lazily inside functions).
# --------------------------------------------------------------------------

def is_bundled_python(executable=None):
    """True if we appear to be running YASARA's bundled ('epy') Python.

    That interpreter is a stripped build that cannot reliably pip-install, so
    detecting it lets us give the right advice (fix yasara.ini PythonPath)
    instead of trying to install into it.
    """
    exe = (executable or sys.executable or "").lower()
    if not exe:
        return False
    # Bundled interpreter lives inside the YASARA application/installation tree.
    return (("yasara" in exe and ".app" in exe)
            or "/yasara/epy" in exe
            or os.sep + "epy" + os.sep in exe)


def _probe_inprocess(code):
    """Run a functional probe in THIS process. Returns (ok, error_str)."""
    try:
        exec(code, {})
        return True, None
    except Exception as e:               # ImportError, or a backend load failure
        return False, "%s: %s" % (type(e).__name__, e)


def _probe_subprocess(code, executable=None):
    """Run a functional probe in a FRESH interpreter (reflects a just-installed
    package without the current process's cached modules). Returns (ok, err)."""
    exe = executable or sys.executable
    try:
        r = subprocess.run([exe, "-c", code], capture_output=True, text=True)
        if r.returncode == 0:
            return True, None
        return False, (r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout).strip() else "exit %d" % r.returncode
    except Exception as e:
        return False, "%s: %s" % (type(e).__name__, e)


def classify(import_name, code, subprocess_=False):
    """Classify one dependency as 'ok' / 'missing' / 'broken'.

    missing = the base module can't be imported at all.
    broken  = it imports, but the functional probe fails (e.g. wrong version,
              incompatible backend).
    """
    probe = _probe_subprocess if subprocess_ else _probe_inprocess
    if subprocess_:
        # In a subprocess we can't cheaply separate missing vs broken up front;
        # run the probe and inspect the error.
        ok, err = probe(code)
        if ok:
            return "ok", None
        low = (err or "").lower()
        if "modulenotfounderror" in low or "no module named" in low:
            return "missing", err
        return "broken", err
    # In-process: first check importability of the base module.
    try:
        importlib.import_module(import_name)
    except ImportError as e:
        return "missing", "%s: %s" % (type(e).__name__, e)
    ok, err = probe(code)
    return ("ok", None) if ok else ("broken", err)


def check(dependencies=DEPENDENCIES, subprocess_=False):
    """Return a report dict describing the environment and each dependency."""
    report = {
        "executable": sys.executable,
        "version": sys.version.split()[0],
        "bundled": is_bundled_python(),
        "deps": {},
    }
    for pip_name, import_name, code in dependencies:
        status, err = classify(import_name, code, subprocess_=subprocess_)
        report["deps"][import_name] = {"pip": pip_name, "status": status, "error": err}
    report["missing"] = [d["pip"] for d in report["deps"].values() if d["status"] == "missing"]
    report["broken"] = [d["pip"] for d in report["deps"].values() if d["status"] == "broken"]
    report["ok"] = not report["missing"] and not report["broken"]
    return report


def pip_install(pip_args, executable=None, upgrade=False):
    """Install/upgrade packages with the pip that belongs to THIS interpreter.

    Uses `python -m pip` (NOT `pip3`, which is not a runnable module). If the
    target interpreter is externally managed (PEP 668, e.g. Homebrew system
    Python), retries once with --break-system-packages. Returns (ok, log)."""
    exe = executable or sys.executable
    base = [exe, "-m", "pip", "install"]
    if upgrade:
        base.append("--upgrade")
    try:
        r = subprocess.run(base + list(pip_args), capture_output=True, text=True)
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0 and "externally-managed" in out.lower():
            r = subprocess.run(base + ["--break-system-packages"] + list(pip_args),
                               capture_output=True, text=True)
            out += "\n[retry --break-system-packages]\n" + (r.stdout or "") + (r.stderr or "")
        return r.returncode == 0, out
    except Exception as e:
        return False, "Could not run pip: %s: %s" % (type(e).__name__, e)


def repair(report, dependencies=DEPENDENCIES):
    """Install missing and upgrade broken packages, then re-verify in a fresh
    interpreter. Returns (ok, log_lines, new_report)."""
    log = []
    if report["missing"]:
        ok, out = pip_install(report["missing"])
        log.append("install %s -> %s" % (report["missing"], "OK" if ok else "FAILED"))
        if not ok:
            log.append(out[-800:])
    if report["broken"]:
        ok, out = pip_install(report["broken"], upgrade=True)
        log.append("upgrade %s -> %s" % (report["broken"], "OK" if ok else "FAILED"))
        if not ok:
            log.append(out[-800:])
    # Verify in a fresh interpreter so we see the newly installed versions.
    new = check(dependencies, subprocess_=True)
    return new["ok"], log, new


def format_diagnosis(report):
    """Human-readable multi-line summary of a report (for console/logs)."""
    lines = ["Tunneler environment check:",
             "  Python : %s" % report["executable"],
             "  Version: %s" % report["version"]]
    if report["bundled"]:
        lines.append("  WARNING: this looks like YASARA's bundled Python (cannot pip-install).")
    for name, d in report["deps"].items():
        mark = {"ok": "ok", "missing": "MISSING", "broken": "BROKEN"}[d["status"]]
        lines.append("  %-12s %s%s" % (name, mark, (" (%s)" % d["error"]) if d["error"] else ""))
    return "\n".join(lines)


# --------------------------------------------------------------------------
#  GUI wrapper (imports YASARA lazily; only used inside the running plugin).
# --------------------------------------------------------------------------

def ensure_dependencies(dependencies=DEPENDENCIES, interactive=True):
    """Plugin startup gate. Returns one of:
        'ok'               -> all good, safe to proceed with heavy imports.
        'repaired-restart' -> packages were fixed; the user must relaunch the
                              plugin (the current process still holds the old,
                              broken modules).
        'abort'            -> environment cannot proceed; caller should stop.

    `interactive=False` auto-repairs without prompting (for scripted/headless
    use); `True` uses YASARA's ShowWin/ShowMessage dialogs.
    """
    report = check()
    if report["ok"]:
        return "ok"                       # happy path: no YASARA UI needed

    print(format_diagnosis(report))

    if interactive:
        # Imported lazily and only when we actually need to show a dialog, so
        # the happy path (and headless use) never depends on YASARA.
        from yasara import ShowWin, ShowMessage

    # Case 1: wrong interpreter (YASARA's bundled epy) -- installing won't help.
    if report["bundled"]:
        if interactive:
            ShowWin("Custom", "Wrong Python", 640, 300,
                    "Text", 20, 45, "YASARA is running its bundled Python, which cannot install packages:",
                    "Text", 20, 70, report["executable"][:78],
                    "Text", 20, 105, "To use a real Python with the scientific packages:",
                    "Text", 20, 130, "1. In yasara.ini set:  PythonPath /path/to/python3",
                    "Text", 20, 152, "2. Plugin header must have:  # PLATFORMS: ExternalPython",
                    "Text", 20, 174, "3. (macOS) rename the bundled epy folder to epy_disable",
                    "Text", 20, 196, "Then restart YASARA.",
                    "Button", 300, 250, "OK")
        return "abort"

    # Case 2: genuine missing/outdated packages on a usable interpreter.
    missing = report["missing"]
    broken = report["broken"]
    if not interactive:
        ok, log, new = repair(report, dependencies)
        print("\n".join(log))
        return "repaired-restart" if ok else "abort"

    detail = []
    if missing:
        detail.append("Missing: " + ", ".join(missing))
    if broken:
        detail.append("Installed but not working (will upgrade): " + ", ".join(broken))
    choice = ShowWin("Custom", "Missing / broken packages", 640, 260,
                     "Text", 20, 45, "Python: " + report["executable"][:78],
                     "Text", 20, 75, detail[0] if detail else "",
                     "Text", 20, 97, detail[1] if len(detail) > 1 else "",
                     "Text", 20, 130, "Install / upgrade them into this Python now?",
                     "Button", 300, 205, "No",
                     "Button", 400, 205, "Yes")[0]
    if choice != "Yes":
        ShowMessage("Tunneler needs numpy, scipy, scikit-learn, shapely and "
                    "matplotlib. Install them into %s and retry." % report["executable"])
        return "abort"

    ok, log, new = repair(report, dependencies)
    print("\n".join(log))
    if ok:
        ShowWin("Custom", "Packages fixed", 600, 200,
                "Text", 20, 55, "Dependencies installed successfully.",
                "Text", 20, 85, "Please start the Tunneler plugin again.",
                "Button", 280, 150, "OK")
        return "repaired-restart"
    else:
        ShowWin("Custom", "Install failed", 640, 300,
                "Text", 20, 45, "Automatic install did not fully succeed.",
                "Text", 20, 75, "Still broken/missing: " + ", ".join(new["missing"] + new["broken"]),
                "Text", 20, 110, "Try manually, then restart YASARA:",
                "Text", 20, 135, "%s -m pip install --upgrade %s"
                % (os.path.basename(report["executable"]),
                   " ".join(d[0] for d in dependencies)),
                "Button", 300, 250, "OK")
        return "abort"
