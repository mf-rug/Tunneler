# Tunneler

A [YASARA](http://www.yasara.org) plugin for detecting, inspecting, and measuring
**tunnels and cavities** in protein structures — the channels that connect a buried
active site to the solvent, or internal pockets that never reach it.

Tunneler finds these channels geometrically (no force field, no docking), lets you
explore them interactively in the YASARA scene, measures their diameter profiles,
and — optionally — hands the geometry off to [CAVER 3.0](https://caver.cz) for
formal tunnel-pathway calculation.

---

## How it works

Detection is a purely geometric point-cloud pipeline:

1. **Seed a point cloud** around the protein on a regular grid (spacing set by the
   *ball spacing* control — finer spacing = more points = higher resolution and cost).
2. **Carve out the protein.** Every grid point closer than a probe radius to any
   protein atom is deleted. What remains is solvent — bulk solvent *plus* the tunnels
   and cavities threading through the protein.
3. **Cluster** the remaining points with **DBSCAN**. Each spatially-connected cluster
   is one tunnel / cavity / pocket.
4. **Classify inside vs. outside** with a rolling-ball concave hull. For each point we
   compute its **critical radius R\*** — the smallest probe radius for which the point
   still falls inside the protein's solvent-excluded surface. Deep, enclosed points have
   a small R\*; shallow surface grooves have a large R\*; bulk-solvent points never fall
   inside (R\* = ∞). This is computed with morphological closings (`scipy.ndimage`
   distance transforms) swept over a range of radii, and stored per-point.
   A single **R knob** (default 5 Å) then trims everything shallower than the chosen
   probe — instantly, since R\* is precomputed.
5. **Measure geometry.** For a selected tunnel, PCA gives the principal axis; slicing
   perpendicular planes along it and fitting maximum inscribed circles yields a
   **diameter / bottleneck profile**.

Optionally, short **MD simulations** can be interleaved between steps to sample
conformational flexibility, so transient channels that only open in some conformations
are captured.

The pipeline scales with *tunnel* volume, not protein volume: interior points are
pre-filtered in NumPy before they ever reach YASARA, and point clouds are loaded in
adaptive chunks to sidestep YASARA's O(N²) per-file load cost.

---

## Features

- **Three-tab dialog** — *Create* (detection parameters), *Appearance* (display, the
  R trim knob, coloring), *Inspect* (per-tunnel selection, diameter profiles, residues).
- **One-knob surface trim** — a single probe-radius slider cleanly separates buried
  channels from surface grooves and bulk solvent; no manual point pruning.
- **Diameter / bottleneck analysis** — cross-section sweep along the tunnel axis with
  maximum-inscribed-circle radii and area profiles.
- **Multiple tunnel representations** — point balls, merged-sphere isosurfaces
  (marching cubes), and Wavefront `.obj` mesh export.
- **CAVER 3.0 integration** — export the detected geometry as seed points and run the
  GPLv3 CAVER engine (auto-downloaded on first use, or point at an existing install);
  import CAVER results back into the scene.
- **MD-aware detection** — optional short simulations between steps to catch transient
  tunnels.
- **Performant mode** — auto-recommended from machine + protein size; trades some
  visual detail for speed on large systems.
- **Save / Load scene** — full round-trip of the detection result and all dialog state.
- **License-tier safe** — works on YASARA View through Structure; heavy operations use
  wrappers that degrade gracefully across tiers.

---

## Installation

### Requirements

- **YASARA** (View, Model, Dynamics, or Structure) on macOS.
- A **real Python 3.12** with a working tkinter/Tcl-Tk (Homebrew `python@3.12` works).
  YASARA's bundled `epy` interpreter cannot `pip install`, so the plugin runs against a
  dedicated external Python.

The scientific stack (pinned in `requirements.txt`): `numpy`, `scipy`, `scikit-learn`,
`scikit-image`, `shapely`, `matplotlib`, plus `requests` / `pandas`.

### 1. Build the plugin's Python environment

A GUI-launched YASARA runs plugins in a clean environment that does **not** inherit your
shell's `PYTHONPATH`, so a plain `pip install -r requirements.txt` is not enough — the
environment also has to be told where YASARA's own `yasara` module lives. The included
`setup_venv.sh` does both: it builds a dedicated venv, installs the pinned dependencies,
and drops a `.pth` file pointing at YASARA's `pym/` directory.

```bash
./setup_venv.sh                 # builds ~/.yasara-venv by default
```

Override the defaults with environment variables if your paths differ:

```bash
BASE_PY=/opt/homebrew/opt/python@3.12/bin/python3.12 \
YASARA_PYM=/Applications/YASARA.app/Contents/yasara/pym \
./setup_venv.sh ~/.yasara-venv
```

The script finishes by verifying — under a simulated clean GUI environment — that
`yasara`, the scientific stack, and matplotlib's tkagg backend all import.

> The venv is built with `--copies` (not symlinks) on purpose: YASARA resolves the real
> path of its `PythonPath` before exec, and a symlinked interpreter would resolve back to
> the base Python, silently bypassing the venv.

### 2. Point YASARA at that Python

Edit `yasara.ini` (in YASARA's app resources) and set:

```
PythonPath /Users/<you>/.yasara-venv/bin/python
```

Then restart YASARA.

### 3. Install the plugin files

Copy — or symlink — all `Tunneler_*.py` files into YASARA's plugin directory
(`/Applications/YASARA.app/Contents/yasara/plg/`):

```
Tunneler_LoadMenu_tk_con.py     # GUI entry point (the plugin YASARA loads)
Tunneler_function_con.py        # detection pipeline
Tunneler_diameter_functions.py  # geometry / cross-section analysis
Tunneler_caver.py               # CAVER 3.0 integration (helper module)
Tunneler_env_check.py           # startup dependency checker
Tunneler_meshwrite.py           # parallel OBJ-mesh writer worker
```

A development checkout can be symlinked into `plg/` — the entry point handles the
symlink `import yasara` edge case itself.

> On first launch the plugin runs a dependency self-check (`Tunneler_env_check.py`). If
> anything is missing or broken it reports exactly what and how to fix it, rather than
> crashing with a raw `ImportError`.

### 4. Run it

In YASARA: **Analyze → Tunnels → Predict Tunnels**. Load a structure, set the ball
spacing and probe radii, and detect.

---

## Module layout

| File | Role |
|------|------|
| `Tunneler_LoadMenu_tk_con.py` | GUI entry point — the 3-tab tkinter dialog; the file YASARA loads as the plugin. |
| `Tunneler_function_con.py` | Core detection pipeline (point cloud → carve → DBSCAN → classify). |
| `Tunneler_diameter_functions.py` | Post-detection geometry: PCA axis, cross-sections, inscribed circles, A\* pathfinding. |
| `Tunneler_caver.py` | CAVER 3.0 discovery/fetch, config generation, run, and result import. Pure logic — no tkinter/yasara imports. |
| `Tunneler_env_check.py` | Startup dependency diagnosis with functional probes (stdlib-only, safe to import first). |
| `Tunneler_meshwrite.py` | Standalone subprocess worker for the parallel Wavefront-OBJ writer. |

---

## CAVER 3.0

CAVER 3.0 (https://caver.cz) is separate GPLv3 software by the Loschmidt Laboratories,
Masaryk University. Tunneler does **not** redistribute it — on first use it is fetched
from the official site (source + license included), or you can point at an existing
install. Only the GPLv3 CAVER 3.0 engine is used.

---

## License

GPL — see [www.gnu.org](https://www.gnu.org). Author: M.J.L.J. Fürst.
