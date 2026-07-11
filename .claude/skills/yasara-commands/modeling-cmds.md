# Modeling Commands

## Homology Modeling

```python
# Use the built-in macro approach
# or call commands directly:
AlignObj(1, 2, method="GlobalSeq")
BuildLoop("Res 42-48 Obj 1")
OptimizeLoop("Res 42-48 Obj 1")
```

See `docs/examples/hm_build.mcr` for the full homology modeling workflow.

## Docking

```python
# Receptor-ligand docking uses the Experiment framework
# See docs/examples/dock_run.mcr for full workflow
```

## Loop Modeling

```python
BuildLoop("Res 42-48 Obj 1")        # Build loop
OptimizeLoop("Res 42-48 Obj 1")     # Optimize loop
SampleLoop("Res 42-48 Obj 1")       # Sample conformations
```

## Mutations

```python
SwapRes("Ala 42 Mol A", "Gly")      # Point mutation
# Then minimize to relax:
ExperimentMinimization()
Experiment("On")
Wait("ExpEnd")
```

## Structure Prediction

```python
FoldMol("ACDEFGHIKLMNPQRSTVWY")     # AI structure prediction (AlphaFold-like)
```

## Building Small Molecules

```python
BuildSMILES("c1ccccc1")             # Benzene from SMILES
BuildAtom("Fe")                      # Single atom
```
