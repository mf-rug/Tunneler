# YASARA PLUGIN
# TOPIC:       Tunnels
# TITLE:       LoadCaver
# AUTHOR:      M.J.L.J. Fürst
# LICENSE:     GPL (www.gnu.org)
# DESCRIPTION: This plugin loads the output of the Caver tool and visualizes the results
#
 
"""
MainMenu: Analyze
  PullDownMenu: Tunnels
    Submenu: LoadCaver
      FileSelectionWindow: Select target pdb file in Caver's /data/ dir
        MultipleSelections: No
        Filename: pdb/*.pdb
      Request: loadcaver
"""

from yasara import *
import os
Console('Off')

# radii should be list of numbers, single number or 'bfactor'
# colors should be list of numbers, single number, or 'atomcol', or 'perobj'
def show_spheres(atomselection, alpha, radius, color, small_H=True, level=2):
    sphere_atoms = " ".join([str(x) for x in atomselection])
    atomlist = ListAtom(sphere_atoms)
    elements = [x[0] for x in ListAtom(sphere_atoms, format='ATOMNAME')]
    p = PosAtom(sphere_atoms,coordsys='global')

    if color == 'atomcol':
        col = ColorAtom(sphere_atoms)
    elif len(color) == len(atomlist):
        col = color
    else:
        col = [color] * len(atomlist)

    if radius == 'bfactor':
        rad = BFactorAtom(sphere_atoms)
    elif isinstance(radius, list) and len(radius) == len(atomlist):
        rad = radius
    else:
        rad = [radius] * len(atomlist)

    if small_H:
         radius = [r * 0.66 if e[0] == 'H' else r for r, e in zip(radius, elements)] 

    for o in range(len(atomlist)):
        obj = ShowSphere(radius=rad[o], color=col[o], alpha=alpha, level=level)
        PosObj(obj, p[(o+1)*3-3], p[(o+1)*3-2], p[(o+1)*3-1])

    jobj = ListObj('sphere')[0]
    JoinObj('sphere', jobj)
    NameObj(jobj, 'caver_sphere')


pdb_file = selection[0].filename[0]
prot = LoadPDB(pdb_file, center=False, correct=False)[0]

dir_name = os.path.dirname(pdb_file)
tunnel_dir = os.path.join(dir_name, 'clusters_timeless')
tunnel_pdbs = [os.path.join(tunnel_dir, f) for f in os.listdir(tunnel_dir)]

mols = []
for i, pdb in enumerate(tunnel_pdbs):
    ShowMessage(f'Loading tunnel {i} / {len(tunnel_pdbs)}.')
    tun = LoadPDB(pdb, center=False, correct=False)[0]
    ColorObj(tun, int(360 / len(tunnel_pdbs) * i))
    molname = 't' + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]
    NameMol(f'obj {tun}', molname)
    mols.append(molname)
    JoinObj(tun, prot)
    Wait(1)

align =\
    ShowWin("Custom","Align Object?",435,180,
            "Text",         20, 48, "Do you want to align the caver target",
            "Text",         20, 73, "with an object currently loaded?",
            "Button",      170,130,"Yes",
            "Button",      260,130,"No")[0]
if align == 'Yes':
    align_target = ShowWin('ObjectSelection', 'Select alignment target')[0]
    AlignObj(prot, align_target, method='sheba')

HideObj(prot)
show_spheres(atomselection = ListAtom(f'Obj {prot} Mol {" ".join(mols)}'),
                alpha= 100,
                radius='bfactor',
                color='white',
                small_H=False)

plugin.end()