# Structure Manipulation Commands

## Loading & Saving

```python
obj = LoadPDB("1crn")[0]              # Load from PDB (returns list)
obj = LoadYOb("myprotein")[0]         # Load YASARA object
LoadSce("scene.sce")                  # Load entire scene
SavePDB(obj, "output.pdb")           # Save as PDB
SaveYOb(obj, "output.yob")           # Save as YASARA object
SaveSce("scene.sce")                  # Save entire scene
LoadCIF("structure.cif")             # Load mmCIF
```

## Building

```python
BuildMol("ACDEFGHIK")                 # Build peptide from sequence
BuildSMILES("c1ccccc1")              # Build from SMILES
BuildAtom("C")                        # Single atom
BuildRes("Ala")                       # Single residue
AddRes("Ala", obj)                    # Append residue to chain
```

## Editing

```python
SwapRes("Ala 42", "Gly")             # Mutate residue
ReplaceRes("Ala 42", "Gly")          # Replace residue (different from Swap)
DelAtom("Atom H*")                    # Delete atoms
DelRes("HOH")                         # Delete water
DelObj(2)                             # Delete object
DuplicateObj(1)                       # Duplicate object
JoinObj(2, 1)                         # Merge obj 2 into obj 1
SplitObj(1)                           # Split at split points
```

## Hydrogens & Cleanup

```python
AddHydAll()                           # Add missing hydrogens
DelHydAll()                           # Remove all hydrogens
OptHydAll()                           # Optimize hydrogen network
CleanAll()                            # Prepare for simulation (fix issues)
AddCapAll()                           # Add terminal caps
AddTerAll()                           # Add C-terminal oxygens
```

## Crystallography

```python
OligomerizeObj(1)                     # Build biological unit
CrystallizeObj(1)                     # Fill unit cell
BuildSymRes("Ala 1")                  # Build symmetry mates
```

## Naming & Numbering

```python
NameObj(1, "MyProtein")               # Rename object
NumberRes("all", first=1)             # Renumber residues
```

See `docs/commands/` files for full argument details.
