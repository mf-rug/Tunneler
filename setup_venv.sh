#!/bin/bash
# Reproducibly build the dedicated Python venv that YASARA's ExternalPython
# plugins use (Tunneler, FoldSeek, bioprodict). Isolates the plugin Python from
# Homebrew system-python churn (the recurring source of dependency breakage).
#
# After running, point yasara.ini at it:
#     PythonPath <VENV>/bin/python
# and restart YASARA.
set -euo pipefail

# --- configuration ---------------------------------------------------------
VENV="${1:-$HOME/.yasara-venv}"
# Base interpreter to build the venv from (must have a working tkinter/Tcl-Tk).
BASE_PY="${BASE_PY:-/opt/homebrew/opt/python@3.12/bin/python3.12}"
# YASARA's Python module dir — provides `import yasara`. A GUI-launched YASARA
# does NOT inherit your shell PYTHONPATH, so the venv must find it on its own.
YASARA_PYM="${YASARA_PYM:-/Applications/YASARA.app/Contents/yasara/pym}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "Building venv at: $VENV  (base: $BASE_PY)"
# Clean rebuild: remove any prior venv first (safe -- this dir is a dedicated,
# disposable venv). Guard against obviously wrong paths.
case "$VENV" in
  "$HOME"/.yasara-venv|"$HOME"/*venv*|/private/tmp/*|/tmp/*) rm -rf "$VENV" ;;
  *) echo "Refusing to rm unexpected VENV path: $VENV" >&2; exit 1 ;;
esac
# --copies (real python binary, not a symlink): YASARA realpath-resolves the
# PythonPath before exec, and a symlinked venv/bin/python resolves back to the
# base interpreter -> the venv silently does NOT activate. A real copied binary
# has nothing to resolve away, so the venv activates no matter how it's invoked.
"$BASE_PY" -m venv --copies "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r "$HERE/requirements.txt"

# Make `import yasara` work regardless of environment inheritance by dropping a
# .pth into the venv's site-packages that adds YASARA's pym/ dir to sys.path.
SP="$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
echo "$YASARA_PYM" > "$SP/yasara_pym.pth"
echo "Added yasara .pth -> $SP/yasara_pym.pth  ($YASARA_PYM)"

echo
echo "Verifying under a CLEAN environment (simulating a GUI launch, no shell PYTHONPATH)..."
env -i HOME="$HOME" "$VENV/bin/python" - <<'PYEOF'
import numpy, scipy, sklearn, shapely
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import yasara
print("OK: yasara + numpy/scipy/sklearn/shapely + matplotlib tkagg all import in a clean env.")
PYEOF

echo
echo "Done. Set  yasara.ini  ->  PythonPath $VENV/bin/python  and restart YASARA."
