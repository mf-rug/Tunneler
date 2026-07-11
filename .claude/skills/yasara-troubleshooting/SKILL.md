---
name: yasara-troubleshooting
description: Common YASARA Python errors, gotchas, and their solutions
user_invocable: false
---

# YASARA Troubleshooting

## Common Gotchas

### 1. LoadPDB Returns a List, Not a Single Object
```python
# WRONG
obj = LoadPDB("1crn")
ColorObj(obj, "Red")   # Error: obj is a list

# RIGHT
obj = LoadPDB("1crn")[0]
ColorObj(obj, "Red")
```
LoadPDB returns a list because NMR structures have multiple models. Always use `[0]` for single-model structures.

### 2. Selection Expressions Must Not Contain Commas
```python
# WRONG
ColorRes("Ala, Gly", "Red")

# RIGHT
ColorRes("Ala Gly", "Red")   # space-separated = implicit OR
```
Commas separate command arguments, not selection items.

### 3. Python Argument Names Are Lowercase
```python
# WRONG
AddBond(atom1, atom2, Order=2)

# RIGHT
AddBond(atom1, atom2, order=2)
```

### 4. info Properties Are Live — Cache Before Loops
```python
# SLOW (re-queries YASARA each iteration)
for i in range(info.atoms):
    ...

# FAST
natoms = info.atoms
for i in range(natoms):
    ...
```

### 5. PosAtom Returns a Flat List
```python
pos = PosAtom("all")
# pos = [x1, y1, z1, x2, y2, z2, ...]  NOT [(x1,y1,z1), ...]

# Access atom i's coordinates:
x, y, z = pos[i*3], pos[i*3+1], pos[i*3+2]
```

### 6. Atom/Residue Numbers Start at 1
YASARA uses 1-based numbering, not 0-based.

### 7. Console Spam Slows Scripts
```python
Console("off")   # before bulk operations
# ... many commands ...
Console("on")    # restore
```

### 8. run() for Commands Without Python Wrappers
Some commands (especially WHAT IF) don't have Python wrappers:
```python
run("SomeCommand arg1,arg2")
```

### 9. Empty Results Don't Raise Exceptions
Most YASARA commands print warnings but don't raise Python exceptions. Check return values explicitly.

### 10. Object Numbers Can Change
After deleting objects, remaining objects may be renumbered. Use `ListObj("all")` to get current numbers.

### 11. Selections Are Atom Flags, Not Ordered
You cannot select atoms "in order" — a selection is just a set of flags.

### 12. ForceField Must Be Set Before Simulation
```python
ForceField("AMBER14")  # must come before Experiment
Cell("Auto", extension=10)
Experiment("Minimization")
Experiment("On")
```

### 13. Wait("ExpEnd") Is Required
Without `Wait("ExpEnd")`, the script continues before the experiment finishes.

### 14. File Paths
YASARA resolves paths relative to its own working directory. Use absolute paths to be safe:
```python
import os
LoadPDB(os.path.abspath("input.pdb"))
SavePDB(1, os.path.abspath("output.pdb"))
```

## Error Messages

| Error | Cause | Fix |
|-------|-------|-----|
| "No atoms selected" | Selection expression matched nothing | Check spelling, use ListAtom/CountAtom to verify |
| "Object X not found" | Object was deleted or number changed | Use ListObj("all") to check current objects |
| "Could not open file" | Wrong path or YASARA working dir | Use absolute paths |
| Empty list returned | LoadPDB failed silently | Check file exists, try absolute path |
