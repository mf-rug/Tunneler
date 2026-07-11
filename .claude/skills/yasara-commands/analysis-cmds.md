# Analysis Commands

## RMSD & Superposition

```python
# RMSD without superposition
rmsd = RMSDAtom("CA Obj 1", "CA Obj 2")

# Superpose and get RMSD
rmsd = SupAtom("CA Obj 1", "CA Obj 2")

# Superpose with options
rmsd = SupAtom("CA Obj 1", "CA Obj 2", flip="Yes")
```

## Distances, Angles, Dihedrals

```python
d = Distance(atom1, atom2)                    # returns float in Å
a = Angle(atom1, atom2, atom3)                # returns degrees
dih = Dihedral(atom1, atom2, atom3, atom4)    # returns degrees

# Set values (moves atoms)
Distance(atom1, atom2, set=3.5)
```

## Counting & Listing

```python
n = CountAtom("all")
n = CountRes("Ala")
n = CountObj("all")

names = NameAtom("all")         # returns list of atom names
names = NameRes("all")          # returns list of residue names
nums = ListAtom("all")          # returns list of atom numbers
```

## Properties

```python
mass = MassAtom("all")          # atom masses
charge = ChargeAtom("all")      # atom charges
bfac = BFactorAtom("all")       # B-factors
pos = PosAtom("all")            # returns [x1,y1,z1,x2,y2,z2,...]
```

## Contacts & Hydrogen Bonds

```python
hbonds = ListHBoAtom("Obj 1", "Obj 2")
contacts = ListConAtom("Res Ala", "Res Gly")
```

## Surface Analysis

```python
sasa = SurfAtom("all")          # solvent-accessible surface area
```

## Secondary Structure

```python
secstr = SecStrRes("all")       # returns list: H=helix, E=sheet, C=coil, T=turn
```

## Energy

```python
energy = EnergyAll()            # total energy
energy = EnergyObj(1)           # energy of object 1
```

## Sequence Alignment

```python
AlignObj(1, 2)                  # sequence alignment
AlignObj(1, 2, method="GlobalSeq")
```
