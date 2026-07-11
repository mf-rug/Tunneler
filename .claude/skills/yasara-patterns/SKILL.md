---
name: yasara-patterns
description: Common YASARA Python scripting patterns, idioms, and best practices
user_invocable: false
---

# YASARA Scripting Patterns

## Basic Script Structure

```python
from yasara import *

# Load structure
obj = LoadPDB("1crn")[0]

# Do work
AddHydAll()
ColorRes("all", "Element")

# Save results
SavePDB(obj, "output.pdb")
```

## Performance: Batch Operations

YASARA Python calls are IPC — each call has overhead. Minimize calls in loops.

```python
# SLOW: one call per atom
for i in range(info.atoms):
    name = NameAtom(i+1)[0]

# FAST: one call for all atoms
names = NameAtom("all")
```

Always call `Console("off")` before bulk operations to suppress screen updates, and `Console("on")` after.

## Getting Multiple Properties

```python
Console("off")
names = NameAtom("all")
positions = PosAtom("all")  # flat list: [x1,y1,z1,x2,y2,z2,...]
bfactors = BFactorAtom("all")
Console("on")

# Positions come as flat xyz triplets
for i in range(0, len(positions), 3):
    x, y, z = positions[i], positions[i+1], positions[i+2]
```

## Working with Multiple Objects

```python
objects = info.objects  # cache before loop
for i in range(1, objects + 1):
    name = NameObj(i)
    rmsd = RMSDAtom(f"CA Obj {i}", f"CA Obj 1")
```

## Error Handling

YASARA doesn't raise Python exceptions for most errors — it prints warnings. Check return values:

```python
result = LoadPDB("nonexistent")
# result will be an empty list if file not found
if not result:
    print("Failed to load")
```

## Iterating Over Residues

```python
# Get all residue numbers
resnums = ListRes("all")
resnames = NameRes("all")
for num, name in zip(resnums, resnames):
    print(f"Residue {name} {num}")
```

## Multi-Model / NMR Ensemble

```python
objlist = LoadPDB("2m67")  # NMR structure — multiple models
for obj in objlist:
    rmsd = RMSDAtom(f"CA Obj {obj}", f"CA Obj {objlist[0]}")
    print(f"Model {obj}: RMSD = {rmsd}")
```

## Writing Results to Files

```python
from yasara import *

obj = LoadPDB("1crn")[0]
Console("off")
resnums = ListRes(f"Protein Obj {obj}")
resnames = NameRes(f"Protein Obj {obj}")
bfactors = BFactorRes(f"Protein Obj {obj}")
Console("on")

with open("results.csv", "w") as f:
    f.write("ResNum,ResName,BFactor\n")
    for num, name, bf in zip(resnums, resnames, bfactors):
        f.write(f"{num},{name},{bf}\n")
```

## Running in Text Mode (No GUI)

```python
from yasara import *
# YASARA starts in text mode by default when imported as module
# info.mode == 'txt'
# All commands work the same, just no visual output
```

## Waiting for Operations

```python
Experiment("Minimization")
Experiment("On")
Wait("ExpEnd")      # blocks until experiment finishes

# Or with timeout
Wait(100)           # wait 100 simulation steps
```

## Coordinate Manipulation with NumPy

```python
import numpy as np
from yasara import *

obj = LoadPDB("1crn")[0]
pos = PosAtom("all")
coords = np.array(pos).reshape(-1, 3)

# Compute centroid
centroid = coords.mean(axis=0)

# Set new positions (flat list)
# PosAtom can also set positions — check docs
```

## Common Workflow: Compare Two Structures

```python
from yasara import *

obj1 = LoadPDB("wt.pdb")[0]
obj2 = LoadPDB("mutant.pdb")[0]

# Superpose on CA atoms
rmsd = SupAtom(f"CA Obj {obj1}", f"CA Obj {obj2}")
print(f"Overall RMSD: {rmsd}")

# Per-residue comparison
resnums = ListRes(f"Protein Obj {obj1}")
for rn in resnums:
    d = RMSDRes(f"Res {rn} Obj {obj1}", f"Res {rn} Obj {obj2}")
```
