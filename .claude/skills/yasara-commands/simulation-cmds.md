# Simulation Commands

## Energy Minimization

```python
# Quick minimization
ExperimentMinimization()
Experiment("On")
Wait("ExpEnd")

# Or step by step
ForceField("AMBER14")
Cell("Auto", extension=10)
AddHydAll()
CleanAll()
Experiment("Minimization")
Experiment("On")
Wait("ExpEnd")
```

## MD Simulation Setup

```python
# Load structure
obj = LoadPDB("1crn")[0]

# Prepare
CleanAll()
AddHydAll()

# Force field
ForceField("AMBER14")           # or AMBER03, NOVA, etc.

# Simulation cell
Cell("Auto", extension=10)      # auto-size with 10 Å padding

# Fill with water and ions
FillCellWater()
# or
Experiment("Neutralization")
Experiment("On")
Wait("ExpEnd")

# Set parameters
Temp(298)                        # Temperature in K
Pressure("NPT")                 # Pressure coupling
TimeStep(2.5)                   # fs (with constraints)
```

## Running Simulation

```python
Sim("On")                        # Start simulation
Sim("Off")                       # Stop simulation
Sim("Pause")                     # Pause
Sim("Continue")                  # Continue

# Or use Experiment framework
Experiment("Simulation")
Experiment("On")
Wait("ExpEnd")
```

## Simulation Parameters

```python
SpeedCap("On")                   # Cap atom speeds
FixAtom("CA")                    # Fix atoms in place
FreeAtom("CA")                   # Release fixed atoms
Temp(300)                        # Set temperature
TempCtrl("Rescale")              # Temperature control: Rescale, Berendsen, etc.
Pressure("SolventProbe")         # Pressure control
Boundary("Periodic")             # Boundary conditions
Longrange("PME")                 # Long-range electrostatics
Cutoff(8.0)                      # Non-bonded cutoff in Å
```

## Saving Snapshots

```python
SaveSim("snapshot", steps=1000)  # Auto-save every 1000 steps
SaveXTC("trajectory.xtc")       # Save as Gromacs XTC
```

## Loading Trajectories

```python
LoadSim("snapshot")
LoadXTCObj(1, "trajectory.xtc")
# Navigate
Sim("Forward")
Sim("Backward")
```

## Analysis During/After Simulation

```python
# Per-residue RMSF
rmsf = RMSFAtom("CA Obj 1")

# Energies
energy = EnergyAll()
pot = EnergyPot()
kin = EnergyKin()

# Temperature
t = TempAtom("all")
```

## Pre-built Macros

The `docs/examples/` directory contains official macros:
- `md_run.mcr` — Standard MD simulation
- `md_runfast.mcr` — Fast MD
- `md_runmembrane.mcr` — Membrane MD
- `md_analyze.mcr` — Analysis
- `em_run.mcr` — Energy minimization
- `dock_run.mcr` — Docking
- `hm_build.mcr` — Homology modeling
