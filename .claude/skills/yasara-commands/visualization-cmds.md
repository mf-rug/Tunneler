# Visualization Commands

## Display Styles

```python
BallStickAll()                    # Ball and stick
StickAll()                        # Stick only
BallAll()                         # Spacefill / VDW spheres
RibbonAll()                       # Cartoon ribbon
TraceAll()                        # CA trace
WireAll()                         # Wireframe
HideAll()                         # Hide everything

# Per-level variants
StickRes("Ala")
BallStickObj(1)
RibbonMol("A")
```

## Coloring

```python
ColorAll("Red")                   # Uniform color
ColorRes("all", "Element")        # CPK element coloring
ColorObj(1, "SecStr")             # By secondary structure
ColorRes("all", "BFactor")        # By B-factor
ColorRes("all", "Chain")          # By chain

# Specific colors
ColorRes("Ala", "Red")
ColorRes("Gly", 255, 128, 0)     # RGB values
```

## Show/Hide

```python
ShowRes("Ala")                    # Show specific residues
HideRes("HOH")                   # Hide waters
ShowObj(1)                        # Show object
HideObj(2)                       # Hide object
ShowAtom("CA")                   # Show specific atoms
```

## Surfaces

```python
ShowSurfObj(1)                    # Molecular surface
ShowSurfMol("A")
SurfPar(Type="Molecular")        # Surface parameters
SurfPar(Type="SAS")              # Solvent-accessible surface
```

## Labels & Annotations

```python
LabelAtom("CA", format="RESName RESNUM")
LabelRes("all", format="Name Num")
ShowArrow(start="AtAtom", selection1=1, end="AtAtom", selection2=100)
ShowDis(atom1, atom2)             # Show distance measurement
```

## Camera & View

```python
ZoomAll()                         # Zoom to fit all
ZoomObj(1)                       # Zoom to object
ZoomRes("Ala 42")                # Zoom to residue
RotateAll("Y", 90)               # Rotate scene
```

## Screenshots

```python
SavePNG("screenshot.png")
SavePNG("screenshot.png", width=1920, height=1080)
RayTrace("screenshot.pov")       # POVRay render
```

## Fog & Depth Cue

```python
Fog(50)                           # 50% fog
DepthCue("On")
```
