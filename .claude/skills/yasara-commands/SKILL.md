---
name: yasara-commands
description: YASARA command reference — look up syntax, arguments, and usage for any YASARA Python command
user_invocable: false
---

# YASARA Command Reference

This skill provides access to ~551 YASARA commands organized by category.

## How to Find Commands

1. Check the index: `docs/commands/index.md`
2. Browse by category file in `docs/commands/`
3. Search `src/yasaramodule.py` for the Python function signature

## Command Documentation Format

Each command page shows:
- **Format**: Yanaconda syntax with argument names, types, defaults, min/max
- **Python**: Python function signature with keyword args
- **Related**: Links to related commands
- **Description**: What the command does, with examples

## Category Files

- `general.md` — Clear, Reset, RecordMacro, ShowCom
- `pdb-files.md` — LoadPDB, SavePDB, LoadCIF, SaveCIF
- `yasara-objects.md` — LoadYOb, SaveYOb
- `yasara-scenes.md` — LoadSce, SaveSce, and many more
- `simulation-snapshots.md` — LoadSim, SaveSim, LoadXTC, SaveXTC, etc.
- `split-points.md` — JoinObj, SplitObj, AddBond, DelBond, building, swapping
- `geometry-manipulation.md` — Move, Rotate, Pos, Distance, Angle, Dihedral
- `geometry-optimization.md` — Optimize, OptimizeLoop
- `surface-environments.md` — AddEnv, RemoveEnv, AddHyd, DelHyd, OptHyd
- `names-and-numbers.md` — Name, Number, List, Count commands
- `preparing-a-simulation.md` — Cell, Boundary, ForceField, FillCell, etc.
- `force-fields.md` — ForceField, Coulomb, Longrange, etc.
- `simulation-parameters.md` — Temp, Pressure, TimeStep, SpeedCap, Fix, Free, etc.
- `running-a-simulation.md` — Sim, Experiment, Wait, etc.
- `atom-position.md` — Group, Pos, RMSD, Sup (superposition), alignment
- `geometry-analysis.md` — SurfAtom, Volume, Mass, Charge, etc.
- `scene-content.md` — List, Count, Check, Compare commands
- `contact-analysis.md` — ListHBo, ListCon, etc.
- `secondary-structure-analysis.md` — SecStr analysis
- `surface-analysis.md` — SurfAtom, accessible surface area
- `cavity-analysis.md` — cavity detection and analysis
- `alternative-views.md` — Style, BallStick, Stick, Trace, Ribbon, etc.
- `ions-and-electrostatic-potentials.md` — ESP, charges, potentials
- `position-and-orientation.md` — Camera, Look, Zoom, Rotate scene
- `marks-and-zooms.md` — Mark, ShowArrow, ShowDis, labels, etc.
- `wire-frames.md` — Show/Hide specific display elements
- `coordinate-system.md` — PosOriObj, Transform, etc.
- `input-devices.md` — Mouse, Keyboard, Console
- `messages-and-buttons.md` — ShowMessage, HUD, menus
- `custom-windows.md` — ShowWin, window management

## Top 20 Commands Quick Reference

```python
# File I/O
obj = LoadPDB("1crn")[0]
SavePDB(obj, "output.pdb")
LoadSce("scene.sce")
SaveSce("scene.sce")

# Display
ColorRes("all", "Element")
BallStickAll()
StickRes("Ala")
RibbonAll()
HideRes("HOH")              # hide water

# Structure editing
AddHydAll()                  # add hydrogens
CleanAll()                   # prepare for simulation
DelRes("HOH")                # delete waters
SwapRes("Ala 42", "Gly")    # mutate residue
BuildMol("ACDEFG")           # build peptide

# Analysis
rmsd = RMSDAtom("CA Obj 1", "CA Obj 2")
d = Distance(atom1, atom2)
sup = SupAtom("CA Obj 1", "CA Obj 2")
namelist = NameAtom("all")
count = CountRes("Ala")

# Simulation
ForceField("AMBER14")
Cell("Auto", 10)
Experiment("Minimization")
Experiment("On")
Wait("ExpEnd")
```

## Supporting Reference Files

- `selection-syntax.md` — Full selection expression reference
- `structure-cmds.md` — Structure manipulation commands
- `analysis-cmds.md` — Analysis commands (RMSD, contacts, surfaces)
- `visualization-cmds.md` — Display, coloring, surfaces
- `simulation-cmds.md` — MD simulation setup and analysis
- `modeling-cmds.md` — Homology modeling, docking, building
