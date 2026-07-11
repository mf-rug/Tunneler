# YASARA Selection Syntax Reference

## Basics

The command name includes the **final unit**: `ColorAtom`, `ColorRes`, `ColorMol`, `ColorObj`, `ColorAll`. This sets the initial selection type.

```
VerbUnit SelectionExpression, Arg2, Arg3, ...
```

## Selection by Name or Number

```python
ColorAtom("CA", "Red")         # by atom name
ColorAtom(1256, "Blue")        # by atom number
ColorRes("Ala", "Green")       # by residue name (3-letter code)
ColorRes(42, "Red")            # by residue number
ColorMol("A", "Yellow")        # by chain/molecule name
ColorObj(1, "Blue")            # by object number
ColorObj("Crambin", "Red")     # by object name
```

## Ranges

```python
ColorRes("1-50", "Red")        # residue range
ColorAtom("100-200", "Blue")   # atom range
```

## Wildcards

- `?` matches one character: `C?` matches CA, CB, CG...
- `*` matches multiple: `Lys*` matches Lys, LysH, etc.

## Switching Selection Type Mid-Expression

Use keywords `Atom`, `Res`, `Mol`, `Obj` to switch:

```python
ColorRes("Ala Mol A", "Red")      # Ala residues in chain A
ColorAtom("CA Res 1-50", "Red")   # CA atoms in residues 1-50
ColorAtom("CA Obj 1", "Red")      # CA atoms in object 1
```

## Implicit OR

Tokens of the same type are OR'd:
```python
ColorRes("Ala Gly Val", "Red")    # Ala OR Gly OR Val
```

## Spatial Selections

```python
# 'with distance < X from' — atoms within X Å
ColorRes("Arg with distance <5 from Asp", "Red")

# 'with distance > X from' — atoms farther than X Å
ColorAtom("CA with distance >10 from Atom 1", "Blue")
```

## Logical Operators

```python
# NOT: 'not' or '!'
ColorRes("not Ala", "Red")        # everything except Ala

# AND: implicit when switching type
ColorRes("Ala Mol A", "Red")      # Ala AND in Mol A
```

## Special Keywords

- `all` — select everything
- `selected` — previously selected atoms (via Select command)
- `HOH` or `WAT` — water molecules
- `Protein` — protein residues
- `NucAcid` — nucleic acid residues
- `Hetgroup` — hetero groups (ligands, ions, waters)

## Property-Based Selection

```python
ColorRes("BFactor>50", "Red")     # high B-factor residues
ColorAtom("Charge<0", "Blue")     # negatively charged atoms
```

## In Python

Selection is always the first positional argument. Use strings for expressions:
```python
ColorRes("Ala Gly Mol A", "Red")
# NOT: ColorRes(Ala, Gly, Mol, A, "Red")  — this is wrong
```

For numeric selections (single atom/residue number), you can pass an int:
```python
ColorAtom(42, "Red")
# equivalent to ColorAtom("42", "Red")
```
