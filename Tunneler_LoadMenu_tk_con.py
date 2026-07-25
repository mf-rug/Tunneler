# YASARA PLUGIN
# TOPIC:       Tunnels
# TITLE:       Tunneler
# AUTHOR:      M.J.L.J. Fürst
# LICENSE:     GPL (www.gnu.org)
# PLATFORMS:   ExternalPython,MacOS
# DESCRIPTION: This plugin loads the menu for the tunnel inspection
#
 
"""
MainMenu: Analyze
  PullDownMenu: Tunnels
    Submenu: Predict Tunnels
      Request: loadmenu
"""

# Tunneler GUI entry point — tkinter dialog for tunnel detection and inspection.

# This file is loaded directly by YASARA as a plugin. It:
#   1. Checks and optionally installs Python dependencies (numpy, sklearn, etc.)
#   2. Imports the core pipeline from Tunneler_function_con and geometry
#      functions from Tunneler_diameter_functions
#   3. Builds a 3-tab tkinter dialog (Create / Appearance / Inspect) via
#      tunneler_dialog(), which runs as a mainloop

# The giant tunneler_dialog() closure contains ~70 nested functions that share
# state through local variables. This is intentional — it avoids global state —
# but makes the function very long. Section banners below help navigate it.


# ============================================================
#  DEPENDENCY CHECKS & IMPORTS
# ============================================================

import sys
import os

# YASARA runs plugins from the plg/ folder (it sets the working directory there)
# and provides the `yasara` module as plg/yasara.py. Normally sys.path[0] is the
# script's folder (plg/), so the import just works -- but when this file is a
# SYMLINK (e.g. a dev checkout linked into plg/), Python 3.11+ sets sys.path[0]
# to the link *target* instead, and `import yasara` fails. Make the import robust
# either way by putting the plugin folder (cwd) on the path first.
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())
from yasara import *
from Tunneler_env_check import ensure_dependencies


# ============================================================
#  TOP-LEVEL UTILITIES
#  (Some of these duplicate functions in Tunneler_function_con.py
#   because they are needed before that module is imported, or
#   because the UI file was developed independently.)
# ============================================================

def rescale_floats_to_range(float_list, min_int, max_int, min_float=None, max_float=None):
    """Linearly scale a list of floats to an integer range [min_int, max_int].

    Used to map distance values to YASARA color codes.
    """
    if min_float == None:
        min_float = min(float_list)
    if max_float == None:
        max_float = max(float_list)
    scaled_list = [int((x - min_float) / (max_float - min_float) * (max_int - min_int) + min_int) for x in float_list]
    return scaled_list

def chunks(lst, n):
    """Yield successive length-*n* slices of *lst* (used to batch coloring work)."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Verify the scientific stack is present AND functional before importing it.
# ensure_dependencies() diagnoses the running Python (rejects YASARA's bundled
# 'epy'), functionally probes each dependency (a bare `import matplotlib` misses
# the tkagg/Tcl breakage), and offers a correct install/upgrade. It must run
# BEFORE the heavy imports below, which would otherwise crash on a bad env.
_env_status = ensure_dependencies(interactive=True)
if _env_status != 'ok':
    # 'repaired-restart': packages were fixed but this process still holds the
    # old modules, so the user was asked to relaunch. 'abort': env unusable and
    # already explained. Either way, stop before the heavy imports.
    plugin.end()

import numpy as np
import re
import os
from configparser import ConfigParser
from Tunneler_function_con import *
import threading
import tkinter as tk
import tkinter.ttk as ttk
from Tunneler_diameter_functions import *
from sklearn.cluster import DBSCAN
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def tunneler_dialog():
    """Build and run the 3-tab Tunneler tkinter dialog.

    This is one giant closure: ~70 nested functions share state through local
    variables (tk Vars, widget references, etc.). The function does not return
    until the user clicks Exit.

    Sections (search for '# ---' banners):
      - State helpers (target, get_config, convert_status, switch_status)
      - Tab 1 — Create Tunnels (widgets + callbacks)
      - Tab 2 — Appearance (widgets + callbacks)
      - Tab 3 — Inspect Tunnel (widgets + callbacks)
      - Cross-section / diameter analysis
      - Dialog mainloop
    """
    Console("OFF")

    # --------------------------------------------------------
    #  STATE HELPERS — Determine current target, read config
    # --------------------------------------------------------

    def target():
        """Return the object number (as string) of the protein that has tunnels, or None."""
        Console("OFF")
        if PairObj('All', 'ball_spacing') != [] and ListObj('?Cl??????? ??Cl???????') != []:
            return(re.findall(r'^\d+', ListObj('?Cl??????? ??Cl???????', format='OBJNAME')[0])[0])
        else:
            return(None)
        
    def forget_crosssection():
        """Remove all cross-section widgets from Tab 3 (when switching to 'All')."""
        diamter_height_scale.place_forget()
        diam_up.place_forget()
        diam_down.place_forget()
        reset_ax.place_forget()
        adjust_ax.place_forget()
        dia_plot.place_forget()
        make_path.place_forget()
        cut_axis_alpha_scale.place_forget()
        cut_axis_alpha_value_label.place_forget()
        cut_axis_alpha_label.place_forget()
        cut_points_button.place_forget()
        rough_path_button.place_forget()
        axis_button.place_forget()
        expose_path_button.place_forget()
        canvas_widget.place_forget()

    def place_crosssection():
        """Show all cross-section widgets on Tab 3 (when a specific tunnel is selected)."""
        diamter_height_scale.place(anchor="nw", x=19, y=220, width=115)
        diam_up.place(anchor="nw", x=134, y=218)
        diam_down.place(anchor="nw", x=0, y=218)
        axis_button.place(anchor="nw", x=0, y=245)
        adjust_ax.place(anchor="nw", x=50, y=243)
        reset_ax.place(anchor="nw", x=103, y=243)
        cut_axis_alpha_label.place(anchor="nw", x=0, y=267)
        cut_axis_alpha_value_label.place(anchor="nw", x=128, y=267)
        cut_axis_alpha_scale.place(anchor="nw", x=40, y=268, width=85)
        cut_points_button.place(anchor="nw", x=0, y=289)
        dia_plot.place(anchor="nw", x=0, y=305)
        make_path.place(anchor="nw", x=208, y=0)
        rough_path_button.place(anchor="nw", x=250, y=-2)
        expose_path_button.place(anchor="nw", x=250, y=14)
        canvas_widget.place(anchor="nw", x=155, y=215, width=tnl_dia_canv_width, height=tnl_dia_canv_height)


    def Recluster():
        """Re-run DBSCAN clustering on the current tunnel points (Tab 2 action)."""
        Console("OFF")
        ShowMessage('Reclustering, please wait.')
        Wait(1)
        tar = target()
        # check if the user used the hide surface atom slider to hide some points
        if ListObj(f'{tar}excl_pts') != [] or CountAtom(f'obj {tar}Cl???????') > CountAtom(f'obj {tar}Cl??????? visible'):
            exclude_hidden = exclude_chk.get()
        
            if exclude_hidden:
                hidden_points = DuplicateAtom(f'obj {tar}Cl??????? !visible')
                if hidden_points != []:
                    JoinObj(" ".join(str(i) for i in hidden_points), hidden_points[0])
                    NameObj(hidden_points[0], f'{tar}excl_pts')
                    DelAtom(f'obj {tar}Cl??????? !visible')
                    ColorObj(hidden_points[0], 'white')
                    SwitchObj(f'{tar}excl_pts', 'off')
                    ShowObj(f'{tar}excl_pts')

            else:
                # renaming makes its name match downstream
                NameObj(f'{tar}excl_pts', f'{tar}C00000000')
                SegObj(f'{tar}roughsurf', '.')
                ShowObj(f'{tar}Cl???????')
            
        get_config(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Tunneler_config.ini'))
        points = np.array(PosAtom(f'Obj {target()}Cl???????', coordsys='global')).reshape(-1,3)
        # Use the LIVE slider values for the clustering thresholds so Recluster
        # reflects the current settings (min cluster volume + connect cutoff).
        # ball_spacing stays the STORED value the cloud was generated with:
        # Recluster re-thresholds the existing points, it does not regenerate them.
        cluster_tunnel_points_dbscan(tar, points, min_vol_scale_chk.get(), float(ball_spacing),
                                     connect_cut_scale_chk.get(), recluster=True)
        SwitchObj(f'{tar}Cl????????', "OFF")
        SwitchObj(ListObj(f'{tar}Cl???????')[:5], "ON")
        transf_and_fix_ss(tar)
        DelObj('???_sphere ???_shape')
        PairObj(tar, 'dist_sel', '')
        HideMessage()
        Wait(1)
        Console("hidden")



    def convert_status(value):
        """Convert a boolean/int checkbox value to YASARA 'ON'/'OFF' string."""
        Console("OFF")
        return "ON" if value == 1 else "OFF"

    def Tunnelonoff():
        """Toggle visibility of all tunnel cluster objects."""
        Console("OFF")
        SwitchObj(f'{target()}Cl???????', convert_status(tunnel_chk.get()))
        Wait(1)
        Console("hidden")

    def Targetonoff():
        """Toggle visibility of the protein target object."""
        Console("OFF")
        SwitchObj(target(), convert_status(target_chk.get()))
        Wait(1)
        Console("hidden")


    def Nonprot():
        """Toggle display of non-protein residues (ligands, cofactors, etc.)."""
        Console("OFF")
        tar = target()
        if ListObj(f'{tar}NonProt', format='OBJNUM') == []:
            new = DuplicateObj(tar)[0]
            SwitchObj(new, 'ON')
            HideObj(new)
            HideSecStrObj(new)
            ShowRes(f'Obj {new} res !protein and !hoh')
            NameObj(new, f'{tar}NonProt')
        else:
            SwitchObj(f'{tar}NonProt', convert_status(nonprot_chk.get()))       
        Wait(1)
        Console("hidden")

    def H2O():
        """Toggle display of water molecules."""
        Console("OFF")
        tar = target()
        if ListObj(f'{tar}H2O', format='OBJNUM') == []:
            new = DuplicateObj(tar)[0]
            SwitchObj(new, 'ON')
            HideObj(new)
            HideSecStrObj(new)
            ShowRes(f'Obj {new} res hoh')
            NameObj(new, f'{tar}H2O')
        else:
            SwitchObj(f'{tar}H2O', convert_status(h2o_chk.get()))       
        Wait(1)
        Console("hidden")

    def Surf(*args, mind_console=True):
        """Create or update the molecular surface visualization object."""
        if mind_console:
            Console("OFF")
        tar = target()
        if ListObj(f'{tar}Surf') == []:
            new = DuplicateObj(tar)[0]
            SwitchObj(new, 'ON')
            HideObj(new)
            HideSecStrObj(new)
            NameObj(new, f'{tar}Surf')
            on_cut()
            return
        if surf_col.get() == 'choose..':
            col = ShowWin("ColorSelection","Select tunnel residues color", "Bow","Background","100")[0]
        elif surf_col.get() == 'outcol':
            col = 'white'
        else:
            col = surf_col.get()
        if surf_incol.get() == 'choose..':
            incol = ShowWin("ColorSelection","Select tunnel residues color", "Bow","Background","100")[0]
        elif surf_incol.get() == 'as outside':
            incol = 'atomcol'
        elif surf_incol.get() == 'incol':
            incol = '000001'
        elif surf_incol.get() == 'black':
            incol = '000001'
        else:
            incol = surf_incol.get()
        ColorRes(f'obj {tar}Surf', col)
        HideSurfObj(f'{tar}Surf')
        ShowSurfRes(f'obj {tar}Surf res protein', 'molecular', outcol='atomcol', outalpha=surf_col_alpha_chk.get(), incol=incol, inalpha=surf_incol_alpha_chk.get())
        SwitchObj(f'{tar}Surf', convert_status(surf_chk.get()))   
        Wait(1)
        if mind_console:
            Console("hidden")

    def Removeinsidepoints():
        """Hide or delete tunnel points that are buried inside the point cloud (not on the surface)."""
        Console("OFF")
        import re
        tunnel_objs = ListObj(f'{target()}Cl???????')
        delete_inside = del_hide_chk.get()
        if delete_inside:
            ShowMessage(f'Deleting inside points')
        else:
            ShowMessage(f'Hiding inside points')
        Wait(22)
        for targetobj in tunnel_objs:
            RemoveEnvRes('all')
            AddEnvRes(targetobj)
            n_before = CountAtom(f'Obj {targetobj}')
            if delete_inside:
                DelAtom(f'obj {targetobj} with distance > 2.58 from accessible surface of obj {targetobj}')
            else:
                HideAtom(f'obj {targetobj} with distance > 2.58 from accessible surface of obj {targetobj}')
            n = CountAtom(f'Obj {targetobj}')
            NameObj(targetobj, re.sub('[0-9]+$', f'{n:06d}', NameObj(targetobj)[0]))
            NameObj(targetobj + 1, 
                    re.sub('[0-9]+$', f'{n:06d}', NameObj(targetobj)[0]) + 'A')
            ShowMessage(f'Removed {n_before - n} inside points from object {targetobj}')
            Wait(1)
        
        HideMessage()
        Wait(1)
        Console("hidden")

    def Exit():
        """Save scene and clean up on dialog exit."""
        Console("hidden")
        if target() != None:
            SaveSce(f'{NameObj(target())[0]}_tunnels_exit.sce')

    def Balls():
        """Switch tunnel display to ball-stick mode."""
        Console("OFF")
        on_spheres = [x for x,y in zip(NameObj('???_Sphere'), SwitchObj('???_Sphere')) if y == 'On']
        if len(on_spheres) == 0:
            on_spheres = [x for x,y in zip(NameObj('???_shape'), SwitchObj('???_shape')) if y == 'On']
        add_tunnels = [match.group() for x in on_spheres for match in [re.search('[0-9]+', x)] if match]
        SwitchObj(" ".join(add_tunnels), 'on')
        SwitchObj(" ".join(on_spheres), 'off')
        SwitchObj('???_shape', 'off')
        BallAtom(f'Obj {target()}Cl???????')
        Wait(1)
        Console("hidden")

    def Points():
        """Switch tunnel display to stick (point) mode."""
        Console("OFF")
        on_spheres = [x for x,y in zip(NameObj('???_Sphere'), SwitchObj('???_Sphere')) if y == 'On']
        if len(on_spheres) == 0:
            on_spheres = [x for x,y in zip(NameObj('???_shape'), SwitchObj('???_shape')) if y == 'On']
        add_tunnels = [match.group() for x in on_spheres for match in [re.search('[0-9]+', x)] if match]
        SwitchObj(" ".join(add_tunnels), 'on')
        SwitchObj(" ".join(on_spheres), 'off')
        SwitchObj('???_shape', 'off')
        StickAtom(f'Obj {target()}Cl???????')
        Wait(1)
        Console("hidden")

    def Spheres(new=False, progress=False):
        """Switch tunnel display to sphere mode (one YASARA sphere per tunnel point)."""
        Console("OFF")
        if ListObj('???_Sphere') != [] and not new:
            on_tunnels = [x for x,y in zip(ListObj(f'{target()}Cl???????'), SwitchObj(f'{target()}Cl???????')) if y == 'On']
            if len(on_tunnels) == 0:
                on_tunnels = [int(x[2:3]) for x,y in zip(NameObj('???_shape'), SwitchObj('???_shape')) if y == 'On']
            SwitchObj(" ".join([str(f'{x:03d}') + '_Sphere' for x in on_tunnels]), "on")
        else:
            DelObj('???_Sphere')
            tunnel_objs = ListObj(f'{target()}Cl???????')
            total_spheres = CountAtom(f'Obj {target()}Cl???????')
            if total_spheres < 3000:
                ShowMessage(f"Creating {total_spheres} Spheres.")
                sphere_level = 3
            elif total_spheres < 10000:
                ShowMessage(f"Creating {total_spheres} Spheres of rad {r}. This process can take several minutes.")
                sphere_level = 2
            elif total_spheres < 40000:
                ShowMessage(f"Creating {total_spheres:,} Spheres. This process can take very long. Click Continue or type StopPlugin in the Console")
                sphere_level = 1
                Wait('Continuebutton')
            else:
                ShowMessage(f"Creating {total_spheres:,} Spheres. This process can take extremely long. Click Continue or type StopPlugin in the Console")
                sphere_level = 0
                Wait('Continuebutton')
            Wait(1)
            done = 0
            for targetobj in tunnel_objs:
                ShowMessage(f"Creating {CountAtom(f'Obj {targetobj}'):,} Spheres of tunnel {NameObj(targetobj)[0]}")
                Wait(1)
                on_off = SwitchObj(targetobj)[0]
                SwitchObj(targetobj, 'Off')
                DelObj(f'sphere x{targetobj:03d}_sphere')
                atomlist = ListAtom(f'obj {targetobj}')
                p = PosAtom(f'obj {targetobj}',coordsys='global')
                col = ColorAtom(f'obj {targetobj}')

                modulo_value = 0
                for o in range(len(atomlist)):
                    obj = ShowSphere(radius=rad_chk.get() /100 * 2.5, color=col[o], alpha=alpha_chk.get(), level=sphere_level)
                    PosObj(obj, p[(o+1)*3-3], p[(o+1)*3-2], p[(o+1)*3-1])
                    if o == modulo_value:
                        modulo_value += 1000
                        jobj = ListObj('sphere')[0]
                        JoinObj('sphere', jobj)
                        ShowMessage(f'Created {o:,} / {len(atomlist):,} spheres of tunnel {NameObj(targetobj)[0]}.')
                        Wait(1)
                        if progress:
                            progress_var.set((o + done) / total_spheres *100)
                            percent_label.config(text=f'{(o + done) / total_spheres *100:.0f}%')
                done = done + len(atomlist)
                if progress:
                    progress_var.set(done / total_spheres *100)
                    percent_label.config(text=f'{done / total_spheres *100:.0f}%')

                jobj = ListObj('sphere')[0]
                JoinObj('sphere', jobj)
                SwitchObj(jobj, on_off)
                NameObj('sphere', f'{targetobj:03d}_sphere')

        SwitchObj(f'{target()}Cl???????', 'off')
        SwitchObj('???_shape', 'off')
        HideMessage()
        Wait(1)
        Console("hidden")

    def group_and_color(atomlist, collist, mind_console=True):
        """Batch-color atoms by grouping them by color first (much faster than per-atom calls)."""
        if mind_console:
            Console("OFF")
        from collections import defaultdict
        # Create a dictionary to store the grouped items
        grouped_items = defaultdict(list)

        # Group items from atomlist based on values in collist
        for atoms, col in zip(atomlist, collist):
            grouped_items[col].append(atoms)

        # Iterate over the grouped items and call ColorAtom
        for col, atoms in grouped_items.items():
            ColorAtom(" ".join(str(x) for x in atoms), col)

          
    def Colorbytunneldist(shapes=True):
        """Color tunnel points by their distance to a user-selected reference atom/center."""
        Console("OFF")
        tar = target()
        save_pairs = PairObj(tar)
        objs = [str(x) for x in ListObj(f'{target()}Cl???????')]

        # if the distance center is the same as before, we can reuse the color stored in SegAtom
        dist_sel = PairObj(tar, 'dist_sel')
        if dist_sel == []:
            atms = SelectDistAtom(win=True)
            dist_sel = PairObj(tar, 'dist_sel')
        else:
            atms = SelectDistAtom(win=False)
        dist_col = PairObj(tar, 'dist_col')
        if dist_col != [] and dist_sel == dist_col:
            atomlist = ListAtom(f'obj {tar}Cl???????')
            collist = [x[1:] for x in SegAtom(f'obj {tar}Cl???????')]
            group_and_color(atomlist, collist)
            HideMessage()
            Wait(1)
            if ListObj('???_shape') != []:
                atomlist = ListAtom(f'obj ???_shape')
                collist = [x[1:] for x in SegAtom(f'obj {tar}Cl???????')]
                if len(atomlist) == len(collist):
                    group_and_color(atomlist, collist)
                else:
                    ShowMessage('Problem occured: ???_shape and Cl?????? objects do not have the same number of atoms. Try deleting shapes and recreate.')
                    Wait('Continuebutton')
            Console("hidden")
            return

        # if the distance center is new, we calculate all distances
        ShowMessage('Coloring by distance')
        Wait(1)
        start_time = time.perf_counter()
        min_color = color1_entry.get()
        max_color = color2_entry.get()
        fast_mode = fast_chk.get()
        per_tunnel = pertun_chk.get()

        if len(atms) > 1:
            ShowMessage('Creating center helper object')
            Wait(1)
            c = DuplicateAtom(" ".join([str(x) for x in atms]))
            JoinObj(" ".join(str(i) for i in c), c[0])
            cx,cy,cz = PosAtom("obj " + str(c[0]), mean=True, coordsys='global')
            cen = BuildAtom("C")
            NameObj(cen, 'CenterHlp')
            PosAtom("obj " + str(cen), x = cx,y = cy, z = cz, coordsys='global')
            DelObj(c[0])
            center=str(ListAtom('Obj ' + str(cen), format='ATOMNUM')[0])
        else:
            center = " ".join(atms)

        # if calculating for all points, check min and max distance first
        if not per_tunnel:
            ShowMessage('Getting min and max distance')
            Wait(1)
            
            atm_obj = ListObj('atom ' + center, format='OBJNUM')[0]
            mind = ListAtom(f'obj {" ".join(objs)} with minimum distance from atom {center}')[0]
            x = DuplicateAtom(mind)[0]
            SwapAtom('obj ' + str(x), 'Du')
            JoinObj(x, atm_obj)
            mind = Distance(f'Obj {atm_obj} element Du', f'Obj {atm_obj} atom {center}')[0]
            DelAtom(f'Obj {atm_obj} element Du')
            maxd = ListAtom(f'obj {" ".join(objs)} with maximum distance from atom {center}')[0]
            x = DuplicateAtom(maxd)[0]
            SwapAtom('obj ' + str(x), 'Du')
            JoinObj(x, atm_obj)
            maxd = Distance(f'Obj {atm_obj} element Du', f'Obj {atm_obj} atom {center}')[0]
            DelAtom(f'Obj {atm_obj} element Du')

        dist_target = ListObj(f'atom {center}', format='OBJNUM')[0]
        atm_count = CountAtom(f"obj {target()}Cl???????")
        if atm_count < 5000:
            chunk_len = int(atm_count / 10)
        else:
            chunk_len = 5000
        for tunnel in objs:
            tname = ListObj(tunnel, format='OBJNAME')[0]

            n = DuplicateObj(tunnel)[0]
            natoms = CountAtom("obj " + str(n))

            SwapAtom(f'Obj {n}', "Du")
            JoinObj(n, dist_target)
            if natoms > 1000:
                ShowMessage(f'Obj {tunnel}: getting distances of {natoms} points')
                Wait(1)

            disto = Distance(f'obj {dist_target} element Du', center)

            new = DuplicateAtom(f'obj {dist_target} element Du')
            DelAtom(f'obj {dist_target} element Du')
            NameObj(new, 'coltunnel')
            on_off = SwitchObj(tunnel)[0]
            SwitchObj(new, on_off)
            DelObj(tunnel)
            RenumberObj(n, tunnel)
            NameObj(tunnel, tname)
            SwapAtom(f'obj {tunnel}', 'H', rename=False)

            atomlist = ListAtom(f'obj {tunnel}')

            if natoms > 1000:
                ShowMessage(f'Obj {tunnel}: Coloring {natoms:,} points')
                Wait(1)

            if per_tunnel:
                mind = min(disto)
                maxd = max(disto)
            
            all_cols = rescale_floats_to_range(disto, int(min_color), int(max_color), mind, maxd)
            
            if fast_mode:
                jmp=max(1, int(len(atomlist) / 100))
                atomlist = [x for _, x in sorted(zip(disto, atomlist))]
                all_cols = [x for _, x in sorted(zip(disto, all_cols))]
                disto = sorted(disto)
                for i, chunk in enumerate(list(chunks(atomlist, chunk_len))):
                    ShowMessage(f'Obj {tunnel}: Colored {i * chunk_len:,} / {len(atomlist):,} points.')
                    Wait(1)
                    for j in range(0, len(chunk), jmp):
                        ColorAtom(atomlist[(i * chunk_len) + j:(i * chunk_len) + j + jmp], 
                                all_cols[(i * chunk_len) + j])
                        SegAtom(atomlist[(i * chunk_len) + j:(i * chunk_len) + j + jmp], f'c{all_cols[(i * chunk_len) + j]}')
            else:
                for i in range(len(atomlist)):
                    ColorAtom(atomlist[i], int(all_cols[i]))

        if ListObj('???_shape') != []:
            atomlist = ListAtom(f'obj ???_shape')
            collist = [x[1:] for x in SegAtom(f'obj {tar}Cl???????')]
            if len(atomlist) == len(collist):
                group_and_color(atomlist, collist)
            else:
                ShowMessage('Problem occured: ???_shape and Cl?????? objects do not have the same number of atoms. Try deleting shapes and recreate.')
                Wait('Continuebutton')

        for i in range(0, len(save_pairs), 2):
            PairObj(tar, save_pairs[i], save_pairs[i + 1])

        PairObj(tar, 'dist_col', dist_sel[0])
        HideMessage()
        DelObj('CenterHlp')
        Wait(1)
        Console("hidden")


    def SecStr(*args):
        """Toggle secondary structure visualization (ribbon/cartoon/tube/trace)."""
        Console("OFF")
        tar = target()
        if ss_chk.get():
            SwitchObj(tar, 'OFF')
            DelObj(f'{tar}SS')
            new = DuplicateObj(tar)[0]
            SwitchObj(new, 'ON')
            NameObj(new, f'{tar}SS')
            HideObj(new)
            if ss_style.get() != 'Trace':
                ShowSecStrObj(new, ss_style.get())
            else:
                HideSecStrObj(new)
                ShowTrace(f'obj {new} atom CA')
                HideAtom(f'obj {new}')
                ShowAtom(f'obj {new} atom CA')
                BallStickAtom(f'obj {new} atom CA')
            ss_col_change()
        else:
            SwitchObj(f'{tar}SS', 'OFF')
  
        Wait(1)
        Console("hidden")


    # Variable to control the loop
    continue_loop = True

    def switch_status(obj):
        Console("OFF")
        if obj == None:
            return None
        stat = SwitchObj(obj, 'OnOff')
        bool_stat = [True if x == 'On' else False for x in stat]
        return any(bool_stat)

    def on_cancel():
        Console("OFF")
        nonlocal continue_loop
        Exit()
        continue_loop = False
        root.destroy()

    # --------------------------------------------------------
    #  DIALOG SETUP — Root window, notebook, config
    # --------------------------------------------------------

    root = tk.Tk()
    root.title("Tunneler Customization Menu")
    root.attributes("-topmost", True)  
    root.geometry(f"+{root.winfo_x()}+{int(root.winfo_y() +55)}")

    initializing = True

    # Create the Notebook widget
    notebook = ttk.Notebook(root)
    notebook.pack(expand=True, fill='both', padx=0)


    # get previous settings
    def get_config(config_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Tunneler_config.ini')):
        """Load tunnel parameters from the INI config file into global variables."""
        if os.path.exists(config_file):
            config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Tunneler_config.ini')
            config = ConfigParser()
            config.read(config_file)
            global ignore_surface, ball_spacing, max_ball_protein, surf_con_prev, keep_surf_points, mds, min_vol, connect_cut, build_pol, prog
            ignore_surface, ball_spacing, max_ball_protein, surf_con_prev, keep_surf_points, mds, min_vol, connect_cut, build_pol, prog = float(config['Variables']['ignore_surface']), float(config['Variables']['ball_spacing']), float(config['Variables']['max_ball_protein']), float(config['Variables']['surf_con_prev']), bool(config['Variables']['keep_surf_points']), int(config['Variables']['mds']), float(config['Variables']['min_vol']), int(config['Variables']['connect_cut']), bool(config['Variables']['build_pol']), str(config['Variables']['prog'])
        else:
            ignore_surface, ball_spacing, max_ball_protein, surf_con_prev, keep_surf_points, mds, min_vol, connect_cut, build_pol, prog = 3.8, 0.33, 2.8, 2.7, False, 0, 5, 1, True, 2
    
    get_config()
    style = ttk.Style()
    current_theme = root.tk.call("ttk::style", "theme", "use")
    style.theme_create( "MyStyle", parent=current_theme, settings={
            "TNotebook.Tab": {"configure": {"padding": [10, 10, 10, 10] },},
            "TNotebook": {"configure": {"tabmargins": [5, 0, 5, 5] } },
            })
    style.theme_use("MyStyle")

    # --------------------------------------------------------
    #  TAB 1 — CREATE TUNNELS
    #  Widgets: parameter sliders, target dropdown, residue
    #  exclusion listbox, find-tunnels button
    #  Callbacks: run_tun, show_progress_tunneler, reset
    # --------------------------------------------------------
    tab1_mktun = ttk.Frame(notebook)
    tab1_mktun.configure(height=375, width=310)  # Set dimensions as needed
    notebook.add(tab1_mktun, text='Create Tunnels', padding=0)  # Add tab1_mktun as the second tab

    def update_label(var, label, n=1):
        """Update the label with the value of the variable."""
        # The value is retrieved from the passed variable and set on the passed label.
        label.config(text=f"{var.get():.{n}f}")

 
    ign_surf_scale_chk = tk.DoubleVar(value=ignore_surface)
    ign_surf_value_label = tk.Label(tab1_mktun, text=f"{ign_surf_scale_chk.get():.1f}")
    ign_surf_scale = ttk.Scale(tab1_mktun, from_=0, to=10, orient="horizontal", variable=ign_surf_scale_chk,
                            command=lambda value, var=ign_surf_scale_chk, label=ign_surf_value_label: update_label(var, label))
    ign_surf_label = tk.Label(tab1_mktun, text=f"Ignore surface up to (\u212B)")
    update_label(ign_surf_scale_chk, ign_surf_value_label)


    surf_con_scale_chk = tk.DoubleVar(value=surf_con_prev)
    surf_con_value_label = tk.Label(tab1_mktun, text=f"{surf_con_scale_chk.get():.1f}")
    surf_con_scale = ttk.Scale(tab1_mktun, from_=0, to=10, orient="horizontal", variable=surf_con_scale_chk,
                            command=lambda value, var=surf_con_scale_chk, label=surf_con_value_label: update_label(var, label))
    surf_con_label = tk.Label(tab1_mktun, text=f"Prevent surface connect (\u212B)")
    update_label(surf_con_scale_chk, surf_con_value_label)


    prot_space_scale_chk = tk.DoubleVar(value=max_ball_protein)
    prot_space_value_label = tk.Label(tab1_mktun, text=f"{prot_space_scale_chk.get()}")
    prot_space_scale = ttk.Scale(tab1_mktun, from_=0, to=5, orient="horizontal", variable=prot_space_scale_chk,
                            command=lambda value, var=prot_space_scale_chk, label=prot_space_value_label: update_label(var, label, 1))
    prot_space_label = tk.Label(tab1_mktun, text=f"Ball-protein distance (\u212B)")
    update_label(prot_space_scale_chk, prot_space_value_label, 2)


    min_vol_scale_chk = tk.DoubleVar(value=min_vol)
    min_vol_value_label = tk.Label(tab1_mktun, text=f"{min_vol_scale_chk.get():.0f}")
    min_vol_scale = ttk.Scale(tab1_mktun, from_=0, to=200, orient="horizontal", variable=min_vol_scale_chk,
                            command=lambda value, var=min_vol_scale_chk, label=min_vol_value_label: update_label(var, label, 0))
    min_vol_label = tk.Label(tab1_mktun, text=f"Minimum cluster volume (\u212B\u00b3)")
    update_label(min_vol_scale_chk, min_vol_value_label, 0)


    num_md_scale_chk = tk.IntVar(value=mds)
    num_md_value_label = tk.Label(tab1_mktun, text=f"{num_md_scale_chk.get()}")
    num_md_scale = ttk.Scale(tab1_mktun, from_=0, to=10, orient="horizontal", variable=num_md_scale_chk,
                            command=lambda value, var=num_md_scale_chk, label=num_md_value_label: update_label(var, label, 0))
    num_md_label = tk.Label(tab1_mktun, text=f"Number of MD simulations")
    update_label(num_md_scale_chk, num_md_value_label, 0)


    ball_spacing_scale_chk = tk.DoubleVar(value=ball_spacing)
    ball_spacing_value_label = tk.Label(tab1_mktun, text=f"{ball_spacing_scale_chk.get()}")
    ball_spacing_scale = ttk.Scale(tab1_mktun, from_=0.15, to=1.7, orient="horizontal", variable=ball_spacing_scale_chk,
                            command=lambda value, var=ball_spacing_scale_chk, label=ball_spacing_value_label: update_label(var, label, 2))
    ball_spacing_label = tk.Label(tab1_mktun, text=f"Ball spacing")
    update_label(ball_spacing_scale_chk, ball_spacing_value_label, 2)
    
    def on_scale_change(var, scale, reset=False):
        if not reset:
            # Round the scale's current value to the nearest integer
            new_value = round(scale.get())
            # Update the variable and the scale's position
            var.set(new_value)
            scale.set(new_value)
        else:
            var.set(1)
            scale.set(1)


    connect_cut_scale_chk = tk.IntVar(value=connect_cut)
    connect_cut_value_label = tk.Label(tab1_mktun, text=f"{connect_cut_scale_chk.get()}")
    connect_cut_scale = ttk.Scale(tab1_mktun, from_=1, to=5, orient="horizontal", variable=connect_cut_scale_chk,
                            command=lambda value, var=connect_cut_scale_chk, label=connect_cut_value_label: update_label(var, label, 0))
    connect_cut_label = tk.Label(tab1_mktun, text=f"Connect cutoff (\u00D7 ball sp.)")
    update_label(connect_cut_scale_chk, connect_cut_value_label, 0)
    connect_cut_scale_chk.trace_add("write", lambda *args: on_scale_change(connect_cut_scale_chk, connect_cut_scale))

    # place all the sliders of tab 1
    ign_surf_label.place(anchor="nw", x=0, y=0)
    ign_surf_value_label.place(anchor="nw", x=160, y=17)
    ign_surf_scale.place(anchor="nw", x=0, y=18, width=162)
    surf_con_label.place(anchor="nw", x=0, y=46)
    surf_con_value_label.place(anchor="nw", x=160, y=63)
    surf_con_scale.place(anchor="nw", x=0, y=64, width=162)
    prot_space_label.place(anchor="nw", x=0, y=92)
    prot_space_value_label.place(anchor="nw", x=160, y=109)
    prot_space_scale.place(anchor="nw", x=0, y=110, width=162)
    min_vol_label.place(anchor="nw", x=0, y=138)
    min_vol_value_label.place(anchor="nw", x=160, y=155)
    min_vol_scale.place(anchor="nw", x=0, y=156, width=162)
    num_md_label.place(anchor="nw", x=0, y=184)
    num_md_value_label.place(anchor="nw", x=160, y=201)
    num_md_scale.place(anchor="nw", x=0, y=202, width=162)
    ball_spacing_label.place(anchor="nw", x=0, y=230)
    ball_spacing_value_label.place(anchor="nw", x=160, y=247)
    ball_spacing_scale.place(anchor="nw", x=0, y=248, width=162)
    connect_cut_label.place(anchor="nw", x=0, y=276)
    connect_cut_value_label.place(anchor="nw", x=160, y=293)
    connect_cut_scale.place(anchor="nw", x=0, y=294, width=162)


    separator1 = ttk.Separator(tab1_mktun)
    separator1.configure(orient="vertical")
    separator1.place(anchor="nw", height=300, width=2, x=193, y=0)

    target_label = tk.Label(tab1_mktun, text=f"Target object:")
    target_label.place(anchor="nw", x=200, y=0)

    target_options_list = ListObj('All', format='OBJNUM: OBJNAME')
    target_option = tk.StringVar(value=target_options_list[0])  # Setting default value to 'select'

    dropdown = ttk.OptionMenu(tab1_mktun, target_option, target_option.get(), *target_options_list)
    dropdown.place(anchor="nw", width=97, height=27, x=205, y=22)

    def get_tnl_name():
        """Return the YASARA object name of the currently selected tunnel (or wildcard for 'All')."""
        if tnl_insp_option.get() != 'All':
            targ = target()
            tnl_objnum = re.findall(r"\d+(?=:)", tnl_insp_option.get())[0]
            return NameObj(tnl_objnum)[0]
        else:
            if target() != None:
                return f'{target()}Cl???????'
            else:
                return None

    def target_changed(*args):
        Console("OFF")
        cur_target = re.findall(r"\d+(?=:)", target_option.get())[0]
        if ListRes(f'obj {cur_target} res HOH') != []:
            items = ['HOH']
        else:
            items = []
        [items.append(x) for x in ListRes(f'Obj {cur_target} res !protein and !hoh', format='RESNAME RESNUM')]
        listbox.delete(0, tk.END)
        for item in items:
            listbox.insert(tk.END, item)
        if ListRes(f'obj {cur_target} res HOH') != []:
            listbox.selection_set(0)

    # Link the function to the variable, so it gets called when the selection changes
    target_option.trace_add("write", target_changed)

    def on_sel_exclude_res(event):
        Console("OFF")
        selected_indices = listbox.curselection()
        selected_res = " ".join([listbox.get(i) for i in selected_indices])
        ShowRes(selected_res)
        UnselectAll()
        SelectRes(selected_res)
        Wait(1)
        Console("hidden")

    res_label = tk.Label(tab1_mktun, text=f"Exclude residues:")
    res_label.place(anchor="nw", x=200, y=51)


    listbox = tk.Listbox(tab1_mktun, selectmode='multiple')
    listbox.place(x=208, y=75, height=85, width=90)
    scrollbar = tk.Scrollbar(listbox, command=listbox.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    listbox.config(yscrollcommand=scrollbar.set, borderwidth=0.1)
    target_changed()

    listbox.bind('<<ListboxSelect>>', on_sel_exclude_res)

    polygon_chk = tk.BooleanVar(value=True)  # Set to True for prechecked
    polygon_button = ttk.Checkbutton(tab1_mktun)
    polygon_button.configure(text='Build polygon', variable=polygon_chk)
    polygon_button.place(anchor="nw", x=200, y=175)

    # # Radio button variable
    show_prog_var = tk.StringVar()
    show_prog_var.set('vis')

    show_prog_var_radio = ttk.Radiobutton(tab1_mktun)
    show_prog_var_radio.configure(value='vis', variable=show_prog_var, text="Visualize steps")
    show_prog_var_radio.place(anchor="nw", x=200, y=202)

    show_prog_var_radio = ttk.Radiobutton(tab1_mktun)
    show_prog_var_radio.configure(value='fast', variable=show_prog_var, text="No steps (fast)")
    show_prog_var_radio.place(anchor="nw", x=200, y=223)

    show_prog_var_radio = ttk.Radiobutton(tab1_mktun)
    show_prog_var_radio.configure(value='wait', variable=show_prog_var, text="Debug (Wait)")
    show_prog_var_radio.place(anchor="nw", x=200, y=244)


    def show_progress_tunneler():
        """Create a progress window and run the Tunneler pipeline (called in a thread)."""
        Console("OFF")
        global progress_window, progress_var, percent_label, initializing
        initializing = True
        progress_window = tk.Toplevel(root)
        progress_window.title("Creating tunnels")
        progress_window.lift()
        progress_window.attributes("-topmost", True) 
        progress_var = tk.IntVar()
        progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=200, mode="determinate", variable=progress_var, maximum=100)
        progress_bar.pack(padx=5, pady=5)
        percent_label = ttk.Label(progress_window, text="0%")  # Initial text for the label
        percent_label.pack(pady=5)

        Tunneler(target=re.findall(r"\d+(?=:)", target_option.get())[0], ignore_res=[listbox.get(i) for i in listbox.curselection()],
                 ignore_surface=ign_surf_scale_chk.get(), 
                 ball_spacing=ball_spacing_scale_chk.get(),
                 max_ball_protein=prot_space_scale_chk.get(), 
                 surf_con_prev=surf_con_scale_chk.get(), 
                 keep_surf_points=False, 
                 mds=num_md_scale_chk.get(), 
                 min_vol=min_vol_scale_chk.get(), 
                 connect_cut=connect_cut_scale_chk.get(), 
                 build_pol=polygon_chk.get(),
                 prog=show_prog_var.get(),
                 progress_var=progress_var,
                 percent_label=percent_label)
        progress_window.destroy()

        if target():
            notebook.add(tab2_appear, text = 'Appearance')
            notebook.add(tab3_inspect, text = 'Inspect Tunnel')
            notebook.select(tab2_appear)
            tnl_insp_options_list = ['All']
            for x in ListObj(f'{target()}Cl???????', format='OBJNUM: OBJNAME'):
                tnl_insp_options_list.append(x)
            update_option_menu(tab3_inspect, tnl_insp_option, tnl_insp_options_list, current_value='All')
        initializing = False
        Wait(1)
        Console("hidden")

    def run_tun(*args):
        Console("OFF")
        UnselectAll()
        threading.Thread(target=show_progress_tunneler).start()


    run_tun_button = ttk.Button(tab1_mktun)
    run_tun_button.configure(style='Toolbutton', text='    Find tunnels    ', command=run_tun)
    run_tun_button.place(anchor="nw",  width=111,height=33, x=200, y=272)

    def reset():
        Console("OFF")
        ign_surf_scale_chk.set(3.8), 
        ball_spacing_scale_chk.set(0.33),
        prot_space_scale_chk.set(2.8), 
        surf_con_scale_chk.set(2.7), 
        num_md_scale_chk.set(0), 
        min_vol_scale_chk.set(5),
        connect_cut_scale_chk.set(1), 
        polygon_chk.set(True),
        show_prog_var.set('vis')
        update_label(ign_surf_scale_chk, ign_surf_value_label, 1)
        update_label(ball_spacing_scale_chk, ball_spacing_value_label, 2)
        update_label(prot_space_scale_chk, prot_space_value_label, 2)
        update_label(surf_con_scale_chk, surf_con_value_label, 1)
        update_label(num_md_scale_chk, num_md_value_label, 0)
        update_label(min_vol_scale_chk, min_vol_value_label, 0)
        on_scale_change(connect_cut_scale_chk, connect_cut_scale, reset=True)

    reset_button = ttk.Button(tab1_mktun)
    reset_button.configure(text = 'Reset to default', command=reset)
    reset_button.place(anchor="nw", x=10, y=315)

    # --------------------------------------------------------
    #  TAB 2 — APPEARANCE
    #  Sections: Show/Hide, display mode (Points/Balls/Spheres/Shape),
    #  Color by (tunnel / distance), Actions (remove points, recluster),
    #  Surface points slider
    #  Callbacks: Tunnelonoff, Targetonoff, Balls, Points, Spheres,
    #    Shapes, Colorbytunnel, Colorbytunneldist, Removeinsidepoints,
    #    Recluster, ml_outside_points, SecStr, Surf, on_cut
    # --------------------------------------------------------
    tab2_appear = ttk.Frame(notebook)
    notebook.add(tab2_appear, text='Appearance', padding=0) 
    tab2_appear.configure(height=375, width=310)

    ## show/hide section
    separator1 = ttk.Separator(tab2_appear)
    separator1.configure(orient="horizontal")
    separator1.place(anchor="nw", height=2, width=242, x=60, y=7)

    label2 = ttk.Label(tab2_appear)
    label2.configure(
        font="TkSmallCaptionFont",
        foreground="#797979",
        text='Show/Hide')
    label2.place(anchor="nw", x=0, y=0)

    if target() != None:
        targ_switch = switch_status(f'{target()}Cl???????')
    else:
        targ_switch = False
    tunnel_chk = tk.BooleanVar(value=targ_switch)  # Set to True for prechecked
    checkbutton4 = ttk.Checkbutton(tab2_appear)
    checkbutton4.configure(text='Tunnels', variable=tunnel_chk, command=Tunnelonoff)
    checkbutton4.place(anchor="nw", x=0, y=16)

    target_chk = tk.BooleanVar(value=targ_switch)  # Set to True for prechecked
    checkbutton1 = ttk.Checkbutton(tab2_appear)
    checkbutton1.configure(text='Target', variable=target_chk, command=Targetonoff)
    checkbutton1.place(anchor="nw", x=73, y=16)

    radio_var = tk.StringVar()
    radio_var.set('points')

    radiobutton4 = ttk.Radiobutton(tab2_appear)
    radiobutton4.configure(text='Points', variable=radio_var, value="points", command=Points)
    radiobutton4.place(anchor="nw", x=3, y=65)

    radiobutton5 = ttk.Radiobutton(tab2_appear)
    radiobutton5.configure(text='Balls', variable=radio_var, value="balls", command=Balls)
    radiobutton5.place(anchor="nw", x=3, y=85)

    separator8 = ttk.Separator(tab2_appear)
    separator8.configure(orient="horizontal")
    separator8.place(anchor="nw", height=2, width=9, x=78, y=116)
 
    separator6 = ttk.Separator(tab2_appear)
    separator6.configure(orient="vertical")
    separator6.place(anchor="nw", height=33, width=2, x=85, y=95)

    def show_progress_spheres(new=True):
        Console("OFF")
        global progress_window, progress_var, percent_label
        progress_window = tk.Toplevel(root)
        progress_window.title("Creating spheres")
        progress_window.lift()
        progress_window.attributes("-topmost", True) 
        progress_var = tk.IntVar()

        progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=200, mode="determinate", variable=progress_var, maximum=100)
        progress_bar.pack(padx=5, pady=30)

        percent_label = ttk.Label(progress_window, text="0%")  # Initial text for the label
        percent_label.pack(pady=30)
        
        threading.Thread(target=on_new_sphere, args=(new, True)).start()


    def on_new_sphere(new=False, progress=False):
        Console("OFF")
        progress_var.set(1)
        radio_var.set('spheres')
        Spheres(new=new, progress=progress)
        progress_window.destroy()

    radiobutton6 = ttk.Radiobutton(tab2_appear)
    radiobutton6.configure(text='Spheres', variable=radio_var, value="spheres", command=lambda: show_progress_spheres(new=False))
    radiobutton6.place(anchor="nw", x=3, y=105)

    alpha_chk = tk.IntVar()
    scale2 = ttk.Scale(tab2_appear, from_=1, to=100)
    scale2.configure(orient="horizontal", state="normal", variable=alpha_chk)
    scale2.place(anchor="nw", x=125, y=85, height=30, width=136)
    alpha_chk.set(19)

    rad_chk = tk.IntVar()
    scale3 = ttk.Scale(tab2_appear, from_= 4, to=100)
    scale3.configure(orient="horizontal", state="normal", variable=rad_chk)
    scale3.place(anchor="nw", x=125, y=106, height=30, width=136)
    rad_chk.set(18)

    button13 = ttk.Button(tab2_appear)
    button13.configure(style="Toolbutton", text='new', command=show_progress_spheres)
    button13.place(anchor="nw", width=39, x=267, y=97)
 
    label1 = ttk.Label(tab2_appear)
    label1.configure(text = 'alpha')
    label1.place(anchor="nw", x=90, y=88)

    label4 = ttk.Label(tab2_appear)
    label4.configure(text = 'size')
    label4.place(anchor="nw", x=90, y=109)

    def Shapes(*args, new=False):
        """Switch tunnel display to surface-shape mode (molecular/VdW/accessible surface)."""
        start_time = time.perf_counter()
        Console("OFF")
        cur_surf = PairObj('???_shape', 'surf')
        if cur_surf == [] or cur_surf[0] == '' or cur_surf[0] != shape_surf_option.get():
            new = True
        if ListObj('???_shape') != [] and new == False:
            on_tunnels = [x for x,y in zip(ListObj(f'{target()}Cl???????'), SwitchObj(f'{target()}Cl???????')) if y == 'On']
            if len(on_tunnels) == 0:
                on_tunnels = [int(x[2:3]) for x,y in zip(NameObj('???_Sphere'), SwitchObj('???_Sphere')) if y == 'On']
            SwitchObj(" ".join([str(f'{x:03d}') + '_shape' for x in on_tunnels]), "on")
        else:
            shapes_on = zip(ListObj('???_shape', format='OBJNAME'), SwitchObj('???_shape'))
            DelObj('???_shape')
            tunnel_objs = ListObj(f'{target()}Cl???????')
            for targetobj in tunnel_objs:
                ShowMessage(f"Creating shape of tunnel {NameObj(targetobj)[0]}")
                Wait(1)
                new = DuplicateObj(targetobj)[0]
                HideObj(new)
                SwapAtom(f'obj {new}', 'H')
                ShowSurfObj(new, shape_surf_option.get(), outcol='atomcol', outalpha=shape_alpha_chk.get())
                HideObj(new)
                NameObj(new, f'{targetobj:03d}_shape')
                on_off = SwitchObj(targetobj)[0]
                SwitchObj(new, on_off)
                PairObj(new, 'surf', shape_surf_option.get())
            for obj, on_off in shapes_on:
                SwitchObj(obj, on_off)

            # recolor because swapatom resets color, imperfect. change if swapatom keepcol becomes avail.
            if radio_col_var.get() == 'tunnel':
                Colorbytunnel(shapes=True, mind_console=False)
            else:
                atomlist = ListAtom(f'obj ???_shape')
                collist = [x[1:] for x in SegAtom(f'obj {target()}Cl???????')]
                if len(atomlist) == len(collist):
                    group_and_color(atomlist, collist, mind_console=False)
                else:
                    ShowMessage('Problem occured: ???_shape and Cl?????? objects do not have the same number of atoms. Try deleting shapes and recreate.')
                    wc()
        
        SwitchObj(f'{target()}Cl??????? ???_sphere', 'off')
        HideMessage()
        Wait(1)
        Console("hidden")

    radiobutton7 = ttk.Radiobutton(tab2_appear)
    radiobutton7.configure(text='Shape', variable=radio_var, value="shape", command=lambda: Shapes())
    radiobutton7.place(anchor="nw", x=3, y=125)

    #  Tunnel AA backbone atom type
    shape_surf_option = tk.StringVar()
    shape_surf_option.set("molecular")
    shape_surf_dropdown = ttk.OptionMenu(tab2_appear, shape_surf_option, "VdW", "VdW", "accessible")
    shape_surf_dropdown.place(anchor="nw", width=100, height=27, x=85, y=127)

    shape_surf_option.trace_add("write", Shapes)

    shape_alpha_chk = tk.IntVar()
    shape_alpha_scale = ttk.Scale(tab2_appear, from_=1, to=100)
    shape_alpha_scale.configure(orient="horizontal", state="normal", variable=shape_alpha_chk)
    shape_alpha_scale.place(anchor="nw", x=200, y=125, height=30, width=100)
    shape_alpha_chk.set(80)


    def ml_outside_points(by=25.5):
        """Hide tunnel points near the protein surface (the 'Surface points' slider callback).

        Uses the roughsurf object to determine the accessible surface, then hides
        tunnel points closer than a distance threshold controlled by surf_pts_chk.
        """
        Console("OFF")
        tar = target()
        SupAtom(f'obj {tar}roughsurf', f'obj {tar}', match='Yes')
        SwitchObj('???_spheres ???_shape', 'OFF')
        if radio_var.get() == 'spheres':
            radio_var.set('balls')
            Balls()
        rough_surf_Du = DuplicateObj(f'{tar}roughsurf')[0]
        DelAtom(f'obj {rough_surf_Du} element !Du')

        rough_surf = ListAtom(f'obj {rough_surf_Du} element Du')
        if stagen(stage) > stagen('View'):
            surf_atom = FirstSurfAtom(rough_surf, 'accessible')[0]
        else:
            surf_atom = ListAtom(f'obj {rough_surf_Du} element Du with maximum distance from obj {tar}')
            print('Warning: this command might give unexpected results because you are using the free version of Yasara.')
 
        min_dist = PairObj(f'{tar}roughsurf', key = 'min_dist')
        max_dist = PairObj(f'{tar}roughsurf', key = 'max_dist')
        if min_dist == [] or max_dist == [] or not is_float(min_dist[0]) or not is_float(max_dist[0]) :
            disto = ListAtom(f'obj {tar}Cl??????? with minimum distance from obj {rough_surf_Du}')[0]
            TransferObj(f'{rough_surf_Du}', ListObj(f'atom {disto}'), 'fix')
            min_dist = float(Distance(disto, ListAtom(f'obj {rough_surf_Du} element Du with minimum distance from atom {disto}'))[0] + 2.2)
            ShowMessage('Getting min and max distance, this may take a while')
            Wait(1)
            i = 1
            total_atm_count = CountAtom(f'obj {tar}Cl???????')
            while True:
                count_atms = len(ListAtom(f'obj {tar}Cl??????? with distance < {i} from accessible surface touched by {surf_atom}'))
                ShowMessage(f'Checking min and max distance, now checking: {i} atom count: {count_atms}/{total_atm_count}')
                if count_atms != 0 and count_atms == total_atm_count:
                    max_dist = float(i -2)
                    break
                else:
                    i += 2
                Wait(1)

            PairObj(f'{tar}roughsurf', 'min_dist', f'{min_dist:.2f}')
            PairObj(f'{tar}roughsurf', 'max_dist', f'{max_dist:.2f}')
        else:
            min_dist = float(min_dist[0])
            max_dist = float(max_dist[0])

        cur_dist = min_dist + (max_dist - min_dist) * surf_pts_chk.get()

        ShowObj(f'obj {tar}Cl???????')
        HideAtom(f'obj {tar}Cl??????? with distance < {cur_dist:.1f} from accessible surface touched by {surf_atom}')
        DelObj(rough_surf_Du)
        Wait(1)
        Console("hidden")

    surf_chk = tk.IntVar(value=switch_status(f'{target()}Surf'))  
    checkbutton5 = ttk.Checkbutton(tab2_appear)
    checkbutton5.configure(text='Surf', variable=surf_chk, command=Surf)
    checkbutton5.place(anchor="nw", x=0, y=40)

    surf_col = tk.StringVar()
    surf_col_drop = ttk.OptionMenu(tab2_appear, surf_col, "white", "white", "element", "restype", "Bfactor", "SecStr", "Occupancy", 'choose..', command=Surf)
    surf_col.set("outcol")
    surf_col_drop.place(anchor="nw", width=65, height=27, x=50, y=37)

    surf_col_alpha_chk = tk.IntVar()
    surf_col_alpha_scale = ttk.Scale(tab2_appear, from_=1, to=100)
    surf_col_alpha_scale.configure(orient="horizontal", state="normal", variable=surf_col_alpha_chk, command=Surf)
    surf_col_alpha_scale.place(anchor="nw", x=115, y=37, height=30, width=46)
    surf_col_alpha_chk.set(80)

    surf_incol = tk.StringVar()
    surf_incol_drop = ttk.OptionMenu(tab2_appear, surf_incol, "black", "as outside", "black", "white", 'choose..', command=Surf)
    surf_incol.set("incol")
    surf_incol_drop.place(anchor="nw", width=65, height=27, x=161, y=37)

    surf_incol_alpha_chk = tk.IntVar()
    surf_incol_alpha_scale = ttk.Scale(tab2_appear, from_=1, to=100)
    surf_incol_alpha_scale.configure(orient="horizontal", state="normal", variable=surf_incol_alpha_chk, command=Surf)
    surf_incol_alpha_scale.place(anchor="nw", x=226, y=37, height=30, width=46)
    surf_incol_alpha_chk.set(80)

    cut_surf_label = ttk.Label(tab2_appear)
    cut_surf_label.configure(text = 'Cut surf:')
    cut_surf_label.place(anchor="nw", x=266, y=16)

    def surf_obj(tar):
        sobj = DuplicateObj(tar)[0]
        SwitchObj(sobj, 'on')
        NameObj(sobj, f'{tar}Surf')
        ShowSurfRes(f'obj {sobj}', 'molecular')
        HideObj(sobj)
        HideSecStrObj(sobj)
        return sobj


    def on_cut(*args):
        Console('off')
        tar = target()
        DelObj(f'{tar}CutPlane')
        DelObj(f'{tar}Surf')
        cut = int(cut_surf_times.get())
        if cut > 0:
            for i in range(cut):
                sobj = surf_obj(tar)
                cobj = CutObj(sobj)[0]
                NameObj(cobj, f'{tar}CutPlane')
                SwitchObj(cobj, 'off')
                if i == 1:
                    RotateObj(cobj, 90)
                elif i == 2:
                    RotateObj(cobj, 0, 90)
        else:
            surf_obj(tar)
        Surf(mind_console=False)
        Wait(1)
        Console("hidden")


    cut_surf_times = tk.StringVar()
    cut_surf_times_drop = ttk.OptionMenu(tab2_appear, cut_surf_times, "2", "0", "1", "2", "3", command=on_cut)
    cut_surf_times.set("2")
    cut_surf_times_drop.place(anchor="nw", width=50, height=27, x=272, y=37)
    cut_surf_times.trace_add("write", on_cut)


    h2o_chk = tk.BooleanVar(value=switch_status(f'{target()}H2O'))  # Variable to track the checkbox status
    checkbutton6 = ttk.Checkbutton(tab2_appear)
    checkbutton6.configure(text='H\u2082O', variable=h2o_chk, command=H2O)
    checkbutton6.place(anchor="nw", x=136, y=16)

    # --- Color utilities (YASARA hue ↔ RGB conversion) ---
    import colorsys
    color_names = {
        "blue": 0,
        "magenta": 60,
        "red": 120,
        "yellow": 180,
        "green": 240,
        "cyan": 300,
        "gray": None,
    }
    def col_to_num(hue, grey='g'):
        """Convert a color name or hue string to a YASARA numeric hue value."""
        Console("OFF")
        if isinstance(hue, str):
            hue_lower = hue.lower()
            if hue_lower in color_names:
                hue = color_names[hue_lower]
                if hue is None:  # Special handling for grey
                    return grey  # return special
            else:
                try:
                    # Attempt to convert string to a number
                    hue = int(hue)
                except ValueError:
                    plugin.end()
            return hue
        elif isinstance(hue, int) or isinstance(hue, float):
            return int(hue)
        else:
            plugin.end()
        

    def get_contrasting_text_color(hex_color):
        """Return black or white hex color for readable text on the given background."""
        Console("OFF")
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return '#000000' if luminance > 0.5 else '#FFFFFF'
    
    def hue_to_rgb(hue, grey='w'):
        """Convert a YASARA hue value (0-360+) to a hex RGB color string (#rrggbb)."""
        if isinstance(hue, str):
            hue_lower = hue.lower()
            if hue_lower in color_names:
                hue = color_names[hue_lower]
                if hue is None:  # Special handling for grey
                    return "#FFFFFF"  # white
            else:
                try:
                    # Attempt to convert string to a number
                    hue = float(hue)
                except ValueError:
                    ShowMessage(f'this is a bug, tried to use this hue: {hue}')
                    wc()
                    plugin.end()
 
        adjusted_hue = (int(hue) + 240) % 360
        r, g, b = colorsys.hsv_to_rgb(adjusted_hue / 360, 1, 1)
        hex_color = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        return hex_color


    ss_chk = tk.BooleanVar(value=switch_status(f'{target()}SS'))  # Variable to track the checkbox status
    checkbutton7 = ttk.Checkbutton(tab2_appear)
    checkbutton7.configure(text='SecStr', variable=ss_chk, command=SecStr)
    checkbutton7.place(anchor="nw", x=70, y=65)
    

    ss_col = tk.StringVar()
    ss_col.set("element")
    ss_col_drop = ttk.OptionMenu(tab2_appear, ss_col, "white", "white", "element", "restype", "Bfactor", "SecStr", "Occupancy", 'choose..')
    ss_col_drop.place(anchor="nw", width=85, height=27, x=140, y=61)

    ss_style = tk.StringVar()
    ss_style.set("ribbon")
    ss_style_drop = ttk.OptionMenu(tab2_appear, ss_style, "Ribbon", "Ribbon", "Cartoon", "Tube", "Trace", command=SecStr)
    ss_style_drop.place(anchor="nw", width=85, height=27, x=230, y=61)

    # Function to handle the selection change
    def ss_col_change(*args):
        Console("OFF")
        if ss_col.get() == 'choose..':
            col = ShowWin("ColorSelection","Select tunnel residues color", "Bow","Background","100")[0]
        else:
            col = ss_col.get()
        ColorObj(f'{target()}SS', col)
        Wait(1)
        Console("hidden")

    # Link the function to the variable, so it gets called when the selection changes
    ss_col.trace_add("write", ss_col_change)

    nonprot_chk = tk.BooleanVar(value=switch_status(f'{target()}NonProt'))  # Variable to track the checkbox status
    checkbutton8 = ttk.Checkbutton(tab2_appear)
    checkbutton8.configure(text='NonProt', variable=nonprot_chk, command=Nonprot)
    checkbutton8.place(anchor="nw", x=185, y=16)

    # # Radio button variable
    radio_col_var = tk.StringVar()
    radio_col_var.set('tunnel')

    def Colorbytunnel(shapes=True, mind_console=True):
        """Color each tunnel cluster a different hue (stepped by the 'step' entry value)."""
        if mind_console:
            Console("OFF")
        objs = ListObj(f'{target()}Cl???????')
        try:
            step = int(step_entry.get())
        except:
            ShowMessage('Invalid step size, you must use a number between 0 and 360. Defaulting to 25')
            Wait(30)
            step = 25
            step_entry.delete(0, tk.END)
            step_entry.insert(0, 25)
        if not shapes:
            for objnum in objs:
                ColorObj(objnum, (objnum +1) * step)            
        else:
            for objnum in objs:
                ColorObj(objnum, (objnum +1) * step)
                ColorObj(str(f'{objnum:03d}') + '_shape', (objnum +1) * step)
        HideMessage()
        Wait(1)
        if mind_console:
            Console("hidden")

    ### color by section
    label3 = ttk.Label(tab2_appear)
    label3.configure(
        font="TkSmallCaptionFont",
        foreground="#797979",
        text='Color by')
    label3.place(anchor="nw", x=0, y=145)

    separator4 = ttk.Separator(tab2_appear)
    separator4.configure(orient="horizontal")
    separator4.place(anchor="nw", height=2, width=250, x=50, y=151)
  
    separator2 = ttk.Separator(tab2_appear)
    separator2.configure(orient="vertical")
    separator2.place(anchor="nw", height=55, width=2, x=12, y=168)
 
    radiobutton14 = ttk.Radiobutton(tab2_appear)
    radiobutton14.configure(text='Tunnel', variable=radio_col_var, value="tunnel", command=Colorbytunnel)
    radiobutton14.place(anchor="nw", x=5, y=158)

    label14 = ttk.Label(tab2_appear)
    label14.configure(text='step')
    label14.place(anchor="nw", x=20, y=180)

    step_entry = ttk.Entry(tab2_appear)
    step_entry.place(anchor="nw", x=20, y=198, width=40)
    step_entry.insert(0, '25')

    def on_colbydist():
        SelectDistAtom(win=True)
        Colorbytunneldist()
        radio_col_var.set('distance')

    def SelectDistAtom(win=False):
        """Prompt user to select reference atom(s) for distance-based coloring."""
        Console("OFF")
        selection = PairObj(target(), 'dist_sel')
        if win or selection == []:
            selection = ShowWin('AtomSelection', 'Select atom from which to calculate distance')[0]
            selection = " ".join([str(x) for x in ListAtom(selection)])
            PairObj(target(), 'dist_sel', selection)

        atms = [str(x) for x in ListAtom(selection)]

        if len(atms) > 1:
            atm_names = ListAtom(' '.join(str(x) for x in atms), format='ATOMNAME')
            if len(set(atm_names)) == 1:
                show_txt = f'{"".join(set(atm_names))}, {len(atms)} atom\'s center'         
            else:
                show_txt = str(len(atms)) + ' atom\'s center'
        elif len(atms) == 0:
            return
        else:
            show_txt = " ".join(atms)
        button18.configure(style="Toolbutton", text=show_txt, command=on_colbydist)
        Wait(1)
        return atms

  
    separator3 = ttk.Separator(tab2_appear)
    separator3.configure(orient="vertical")
    separator3.place(anchor="nw", height=55, width=2, x=79, y=168)
 
    radiobutton13 = ttk.Radiobutton(tab2_appear)
    radiobutton13.configure(text='Distance to', variable=radio_col_var, value="distance", command=lambda: Colorbytunneldist())
    radiobutton13.place(anchor="nw", x=72, y=158)

    button18 = ttk.Button(tab2_appear)
    button18.configure(text='select', command=on_colbydist)
    button18.place(anchor="nw", x=165, y=155)

    fast_chk = tk.BooleanVar(value=True)  # Variable to track the checkbox status
    checkbutton7 = ttk.Checkbutton(tab2_appear)
    checkbutton7.configure(text='fast mode', variable=fast_chk)
    checkbutton7.place(anchor="nw", x=90, y=182)

    pertun_chk = tk.BooleanVar()  # Variable to track the checkbox status
    checkbutton7 = ttk.Checkbutton(tab2_appear)
    checkbutton7.configure(text='calc per tunnel', variable=pertun_chk)
    checkbutton7.place(anchor="nw", x=183, y=182)

    def choose_color():
        Console("OFF")
        color_code = ShowWin("ColorSelection","Select color for close atoms", "Bow","Background","100")
        if color_code:
            col = hue_to_rgb(color_code[0])
            color1_entry.delete(0, tk.END)
            color1_entry.insert(0, col_to_num(color_code[0]))
            style.configure(style_name1, foreground=get_contrasting_text_color(col), background=col)

    def choose_color2():
        Console("OFF")
        color_code = ShowWin("ColorSelection","Select color for distant atoms", "Bow","Background","100")
        if color_code:
            col = hue_to_rgb(color_code[0])
            color2_entry.delete(0, tk.END)
            color2_entry.insert(0, col_to_num(color_code[0]))
            style.configure(style_name2, foreground=get_contrasting_text_color(col), background=col)


    # Use different style names for each entry
    style = ttk.Style()
    style_name1 = "Colored1.TEntry"
    style.configure(style_name1, foreground="white", background=hue_to_rgb(100))

    style_name2 = "Colored2.TEntry"
    style.configure(style_name2, foreground="white", background=hue_to_rgb(380))

    color1_entry = ttk.Entry(tab2_appear, style=style_name1)
    color1_entry.place(anchor="nw", x=145, y=203, width=45)
    color1_entry.insert(0, '100')

    color2_entry = ttk.Entry(tab2_appear, style=style_name2)
    color2_entry.place(anchor="nw", x=258, y=203, width=45)
    color2_entry.insert(0, '380')
           
    button19 = ttk.Button(tab2_appear)
    button19.configure(style='Toolbutton', text='Min col', command=choose_color)
    button19.place(anchor="nw", x=90, y=203)

    button20 = ttk.Button(tab2_appear)
    button20.configure(style='Toolbutton', text='Max col', command=choose_color2)
    button20.place(anchor="nw", x=200, y=203)  

    ### actions section
    label13 = ttk.Label(tab2_appear)
    label13.configure(
        font="TkSmallCaptionFont",
        foreground="#797979",
        text='Actions')
    label13.place(anchor="nw", x=0, y=266)

    separator9 = ttk.Separator(tab2_appear)
    separator9.configure(orient="horizontal")
    separator9.place(anchor="nw", height=2, width=255, x=45, y=274)

    button1 = ttk.Button(tab2_appear)
    button1.configure(style="Toolbutton", text='Remove invisible points', command=Removeinsidepoints)
    button1.place(anchor="nw", x=0, y=284)


    def create_tooltip(widget, text, delay=775):
        """Attach a hover tooltip to a tkinter widget."""
        tooltip_window = None
        after_id = None

        def show_tooltip(event):
            nonlocal tooltip_window, after_id
            def create_tooltip_window():
                nonlocal tooltip_window
                if tooltip_window:
                    return
                x, y, width, height = widget.bbox("insert")
                x = x + widget.winfo_rootx() + 20
                y = y + height + widget.winfo_rooty() + 20
                tooltip_window = tk.Toplevel(widget)
                tooltip_window.wm_overrideredirect(True)
                tooltip_window.wm_geometry(f"+{x}+{y}")
                label = tk.Label(tooltip_window, text=text, justify='left',
                                background='lightyellow', relief='solid', borderwidth=0.2,
                                font=("tahoma", "8", "normal"))
                label.pack(ipadx=1, ipady=26)
                tooltip_window.lift()
                tooltip_window.transient(widget.winfo_toplevel())

            # Schedule the tooltip to appear after a delay
            after_id = widget.after(delay, create_tooltip_window)

        def hide_tooltip(event):
            nonlocal tooltip_window, after_id
            if after_id:
                widget.after_cancel(after_id)
                after_id = None
            if tooltip_window:
                tooltip_window.destroy()
                tooltip_window = None

        widget.bind('<Enter>', show_tooltip)
        widget.bind('<Leave>', hide_tooltip)


    create_tooltip(button1, "This action removes tunnel points on the inside of the point cloud.\nWhile they are normally invisible, tunnels will appear 'empty'\nif some tunnel points are removed, or upon close zoom.")

    del_hide_chk = tk.BooleanVar()  # Variable to track the checkbox status
    checkbutton9 = ttk.Checkbutton(tab2_appear)
    checkbutton9.configure(text='permanently', variable=del_hide_chk)
    checkbutton9.place(anchor="nw", x=157, y=286)

    button2 = ttk.Button(tab2_appear)
    button2.configure(style="Toolbutton", text='Recluster', command=Recluster)
    button2.place(anchor="nw", x=0, y=312)
    
    exclude_chk = tk.BooleanVar()  # Variable to track the checkbox status
    checkbutton7 = ttk.Checkbutton(tab2_appear)
    checkbutton7.configure(text='only visible points', variable=exclude_chk)
    checkbutton7.place(anchor="nw", x=70, y=315)

    def on_rotsurf():
        Console("OFF")
        GrabObj(f'{target()}CutPlane')
        ShowMessage('Rotate surface cutplanes now. Press Continue to accept new rotation')
        Wait('continuebutton')
        GrabAll()
        HideMessage()
        Console("hidden")

    button5 = ttk.Button(tab2_appear)
    button5.configure(style="Toolbutton", text='Rotate surf. cut', command=on_rotsurf)
    button5.place(anchor="nw", x=206, y=312)

    ## surface points section
    label10 = ttk.Label(tab2_appear)
    label10.configure(
        font="TkSmallCaptionFont",
        foreground="#797979",
        text='Surface points')
    label10.place(anchor="nw", x=0, y=228)

    separator5 = ttk.Separator(tab2_appear)
    separator5.configure(orient="horizontal")
    separator5.place(anchor="nw", height=2, width=220, x=80, y=235)

    label11 = ttk.Label(tab2_appear)
    label11.configure(
        font="TkSmallCaptionFont",
        foreground="#919191",
        text='All')
    label11.place(anchor="nw", x=10, y=247)

    label12 = ttk.Label(tab2_appear)
    label12.configure(
        font="TkSmallCaptionFont",
        foreground="#919191",
        text='None')
    label12.place(anchor="nw", x=274, y=247)
  
    surf_pts_chk = tk.DoubleVar()  # Variable to track the checkbox status
    scale1 = ttk.Scale(tab2_appear)
    scale1.configure(orient="horizontal", state="normal", variable=surf_pts_chk, command=ml_outside_points)
    scale1.place(
        anchor="nw",
        relwidth=0.5,
        relx=0.0,
        width=80,
        x=30,
        y=245)

    # Function to toggle the window always on top
    def toggle_always_on_top():
        Console("OFF")
        if always_on_top_var.get():
            root.attributes("-topmost", True)
        else:
            root.attributes("-topmost", False)

    # Variable for the always on top checkbox
    always_on_top_var = tk.BooleanVar(value=True)
    checkbutton21 = ttk.Checkbutton(root)
    checkbutton21.configure(text='Keep dialog on top', variable=always_on_top_var, command=toggle_always_on_top)
    checkbutton21.place(anchor="nw", x=5, y=405)

    button4 = ttk.Button(root)
    button4.configure(text='Exit', command=on_cancel)
    button4.place(anchor="nw", x=256, y=401)

    separator7 = ttk.Separator(root)
    separator7.configure(orient="horizontal")
    separator7.place(anchor="nw", height=2, width=310, x=5, y=400)


    # --------------------------------------------------------
    #  TAB 3 — INSPECT TUNNEL
    #  Sections: tunnel selector, amino acid display options,
    #  surface display, cross-section / diameter analysis,
    #  pathfinding
    #  Callbacks: inspect_changed, on_tnl_aas, on_tnl_aa_bb,
    #    on_tnl_aa_lab, on_tnl_aa_surf, on_diameter,
    #    draw_diameter_plot, on_make_path
    # --------------------------------------------------------
    tab3_inspect = ttk.Frame(notebook)
    tab3_inspect.configure(height=375, width=310)  # Set dimensions as needed
    notebook.add(tab3_inspect, text='Inspect Tunnel', padding=0)  # Add tab1_mktun as the second tab

    tnl_insp_label = tk.Label(tab3_inspect, text=f"Select:")
    tnl_insp_label.place(anchor="nw", x=2, y=1)

    zoomsteps = 10
    def inspect_changed(*args):
        """Callback when the tunnel selector dropdown changes. Zooms to the selected tunnel."""
        if 'initializing' in globals():
            global initializing
        elif 'initializing' not in locals():
            initializing = False
        if initializing:
            return
        Console("OFF") 
        if target() != None:
            targ = target()
            tnl_name = get_tnl_name()
            ShowObj(tnl_name)
            DelObj(f'???_slice ???_axis')
            if tnl_insp_option.get() != 'All':
                tnl_objnum = re.findall(r"\d+(?=:)", tnl_insp_option.get())[0]
                SwitchObj(f'{targ}Cl???????? ???_sphere ???_shape', 'OFF')
                SwitchObj(tnl_objnum, 'ON')
                ZoomAtom(f'Obj {tnl_name}?', zoomsteps)
                Wait(zoomsteps)
                CellAuto(1, 'cuboid', f'obj {tnl_objnum}')
                SwitchObj('SimCell', 'off')
                NameObj('SimCell', 'CntrOfRot')
                on_tnl_aas()
                place_crosssection()
            else:
                forget_crosssection()
                ZoomAtom('all', zoomsteps)
                DelObj('CntrOfRot')
                MarkAtom('none')
                SwitchObj(f'{targ}Cl???????? {targ}excluded {targ}Close2Surf {targ}Close2Prot', 'OFF')
                SwitchObj(ListObj(f'{targ}Cl???????')[:5], "ON")
            HideSurfObj(f'{targ}tnlAAsurf')
            Wait(1)
        Console("hidden")

    tnl_insp_options_list = ['All']
    if target() != None:
        for x in ListObj(f'{target()}Cl???????', format='OBJNUM: OBJNAME'):
            tnl_insp_options_list.append(x)

    tnl_insp_option = tk.StringVar(value='All')  # Set default value

    dropdown_insp = ttk.OptionMenu(tab3_inspect, tnl_insp_option, tnl_insp_option.get(), *tnl_insp_options_list)
    dropdown_insp.place(anchor="nw", width=150, height=27, x=52, y=0)
    tnl_insp_option.trace_add("write", inspect_changed)

    def update_option_menu(parent, variable, options, current_value=None):
        if hasattr(parent, 'dropdown_insp'):
            parent.dropdown_insp.destroy()

        parent.dropdown_insp = ttk.OptionMenu(parent, variable, current_value if current_value else options[0], *options)
        parent.dropdown_insp.place(anchor="nw", width=150, height=27, x=52, y=0)



    # row 2, separater show/hide
    label2 = ttk.Label(tab3_inspect)
    label2.configure(
        font="TkSmallCaptionFont",
        foreground="#797979",
        text='Show/Hide amino acids\' ...')
    label2.place(anchor="nw", x=0, y=26)

    separator1 = ttk.Separator(tab3_inspect)
    separator1.configure(orient="horizontal")
    separator1.place(anchor="nw", height=2, width=170, x=141, y=35)



    # row 3, AAs and color label
    def on_tnl_aas():
        """Toggle display of tunnel-lining amino acid sidechains."""
        Console("OFF")
        tnl_name = get_tnl_name()
        if tnl_name != None:
            SwitchObj(f'{tnl_name}A', convert_status(tnl_aas.get()))
            Wait(1)
            Console("hidden")


    tnl_aas = tk.BooleanVar(value=True)
    chkbox_tnl_aas = ttk.Checkbutton(tab3_inspect)
    chkbox_tnl_aas.configure(text='sidechains', variable=tnl_aas, command=on_tnl_aas)
    chkbox_tnl_aas.place(anchor="nw", x=0, y=45)


    #  Tunnel AA backbone atom type
    tnl_res_bbatmtyp_option = tk.StringVar()
    tnl_res_bbatmtyp_option.set("BallSticks")
    tnl_res_bbatmtyp_dropdown = ttk.OptionMenu(tab3_inspect, tnl_res_bbatmtyp_option, "BallSticks", "Sticks", "BallSticks", "Balls")
    tnl_res_bbatmtyp_dropdown.place(anchor="nw", width=85, height=27, x=100, y=68)

    def tnl_res_bbatmtyp_changed(*args):
        if initializing:
            return
        Console("OFF")
        tnl_name = get_tnl_name()
        if tnl_name != None:
            if tnl_res_bbatmtyp_option.get() == 'Sticks':
                StickAtom(f'obj {tnl_name}A atom backbone')
            elif tnl_res_bbatmtyp_option.get() == 'BallSticks':
                BallStickAtom(f'obj {tnl_name}A atom backbone')
            else:
                BallAtom(f'obj {tnl_name}A atom backbone')
        Console("hidden")

    tnl_res_bbatmtyp_option.trace_add("write", tnl_res_bbatmtyp_changed)

    #  Tunnel AAs atom type
    def tnl_res_atmtyp_changed(*args):
        if initializing:
            return
        tnl_name = get_tnl_name()
        if tnl_name != None:
            Console("OFF")
            tnl_aas_name = f'{tnl_name}A'
            if tnl_res_atmtyp_option.get() == 'Sticks':
                StickObj(tnl_aas_name)
            elif tnl_res_atmtyp_option.get() == 'BallSticks':
                BallStickObj(tnl_aas_name)
            else:
                BallObj(tnl_aas_name)
            tnl_res_bbatmtyp_changed()
            Console("hidden")

    tnl_res_atmtyp_option = tk.StringVar()
    tnl_res_atmtyp_option.set("element")
    tnl_res_atmtyp_dropdown = ttk.OptionMenu(tab3_inspect, tnl_res_atmtyp_option, "Sticks", "Sticks", "BallSticks", "Balls")
    tnl_res_atmtyp_dropdown.place(anchor="nw", width=85, height=27, x=100, y=43)
    tnl_res_atmtyp_option.trace_add("write", tnl_res_atmtyp_changed)

    tnl_res_col_label = tk.Label(tab3_inspect, text=f"Color atoms:")
    tnl_res_col_label.place(anchor="nw", x=200, y=46)

    tnl_res_atmtyp_changed()


    # row 4, AA bbs and color dropdown

    def on_tnl_aa_bb():
        """Toggle display of backbone atoms in tunnel-lining residues."""
        Console("OFF")
        tnl_name = get_tnl_name()
        if tnl_name != None:
            if tnl_aa_bb.get():
                ShowAtom(f'obj {tnl_name}A atom C N O')
            else:
                HideAtom(f'obj {tnl_name}A atom C O or res !pro atom N')
            Wait(1)
        Console("hidden")

    tnl_aa_bb = tk.BooleanVar(value=False)
    chkbox_tnl_aa_bb = ttk.Checkbutton(tab3_inspect)
    chkbox_tnl_aa_bb.configure(text='backbones', variable=tnl_aa_bb, command=on_tnl_aa_bb)
    chkbox_tnl_aa_bb.place(anchor="nw", x=0, y=70)


    tnl_res_bbatmtyp_changed()

    tnl_res_col_option = tk.StringVar()
    tnl_res_col_option.set("element")
    tnl_res_col_dropdown = ttk.OptionMenu(tab3_inspect, tnl_res_col_option, "element", "element", "restype", "Bfactor", "SecStr", "Occupancy", "Distance to tunnel", "choose..")
    tnl_res_col_dropdown.place(anchor="nw", width=115, height=27, x=200, y=63)

    # Tunnel aa color selection
    def col_by_dist_to_tun(tunnel):
        Console("OFF")
        if tnl_aa_surf.get():
            SwitchObj(f'{target()}tnlAAsurf', 'OFF')
        tnl_aa_obj = ListObj(NameObj(tunnel)[0] + 'A')[0]
        tnl_aa_atms = ListAtom(f'obj {tnl_aa_obj}')
        tnl_points = ListObj(tunnel)[0]

        TransferObj(tnl_aa_obj, tnl_points, 'fix')

        min_color = 100
        max_color = 360

        disto = [round(Distance(x, ListAtom(f'obj {tnl_points} with minimum distance from {x}')[0])[0],2) for x in tnl_aa_atms]
        all_cols = rescale_floats_to_range(disto, int(min_color), int(max_color))

        for i in range(len(tnl_aa_atms)):
            ColorAtom(tnl_aa_atms[i], int(all_cols[i]))

        if tnl_surf_col_option.get() == 'atomcol':
            for i in range(len(tnl_aa_atms)):
                ColorAtom(f'obj {target()}tnlAAsurf with distance < 0.1 from {tnl_aa_atms[i]}', int(all_cols[i]))
        if tnl_aa_surf.get():
            SwitchObj(f'{target()}tnlAAsurf', 'On')
        Wait(1)
        Console("hidden")

    # Function to handle the selection change
    def tnl_res_col_changed(*args):
        Console("OFF")
        tnl_name = get_tnl_name()
        if tnl_name != None:
            tnl_aas_name = f'{tnl_name}A'
            if tnl_res_col_option.get() != "Distance to tunnel" and tnl_res_col_option.get() != "choose..":
                ColorObj(tnl_aas_name, tnl_res_col_option.get())
                if tnl_surf_col_option.get() == 'atomcol':
                    ColorObj(f'{target()}tnlAAsurf', tnl_res_col_option.get())
            else:
                if tnl_res_col_option.get() == "Distance to tunnel":
                    for obj in ListObj(tnl_name, format='OBJNAME'):
                        col_by_dist_to_tun(obj)
                else:
                    col = ShowWin("ColorSelection","Select tunnel residues color", "Bow","Background","100")[0]
                    ColorObj(tnl_aas_name, col)
                    if tnl_surf_col_option.get() == 'atomcol':
                        ColorObj(f'{target()}tnlAAsurf', col)
            Wait(1)
        Console("hidden")

    # Link the function to the variable, so it gets called when the selection changes
    tnl_res_col_option.trace_add("write", tnl_res_col_changed)



    # row 5, AA label, size, col
    def on_tnl_aa_lab(*args):
        if initializing:
            return
        Console("OFF")
        tnl_name = get_tnl_name()
        if tnl_name != None:
            UnlabelAtom(f'obj {tnl_name}A')
            if tnl_aa_lab.get():
                if tnl_res_lab_col_option.get() == 'auto':
                    ca_list = ListAtom(f'obj {tnl_name}A atom CA')
                    for ca in ca_list:
                        ca_col = ColorAtom(ca)[0]
                        LabelAtom(ca, 'RESNAME1RESNUM', tnl_aa_lab_size.get(), get_contrasting_text_color(hue_to_rgb(str(ca_col)))[1:])
                else:
                    LabelAtom(f'obj {tnl_name}A atom CA', 'RESNAME1RESNUM', tnl_aa_lab_size.get(), tnl_res_lab_col_option.get())
            Wait(1)
        Console("hidden")

    tnl_aa_lab = tk.BooleanVar(value=False)
    chkbox_tnl_aa_lab = ttk.Checkbutton(tab3_inspect)
    chkbox_tnl_aa_lab.configure(text='label', variable=tnl_aa_lab, command=on_tnl_aa_lab)
    chkbox_tnl_aa_lab.place(anchor="nw", x=0, y=95)

    tnl_aa_lab_col_label = tk.Label(tab3_inspect, text=f"Color:")
    tnl_aa_lab_col_label.place(anchor="nw", x=200, y=94)

    tnl_res_lab_col_option = tk.StringVar()
    tnl_res_lab_col_option.set("black")
    tnl_res_lab_col_dropdown = ttk.OptionMenu(tab3_inspect, tnl_res_lab_col_option, "black", "black", "white", "auto")
    tnl_res_lab_col_dropdown.place(anchor="nw", width=70, height=27, x=245, y=93)

    tnl_res_lab_col_option.trace_add("write", on_tnl_aa_lab)


    def new_tnl_aa_lab_size(var, label, n=2):
        if initializing:
            return
        tnl_name = get_tnl_name()
        if tnl_name != None:
            Console('off')        
            UnlabelAtom(f'obj {tnl_name}A')
            on_tnl_aa_lab()
            label.config(text=f"{var.get():.{n}f}")
            Console("hidden")

    tnl_aa_lab_size = tk.DoubleVar(value=0.22)
    tnl_aa_lab_size_value_label = tk.Label(tab3_inspect, text=f"{tnl_aa_lab_size.get():.12}")
    tnl_aa_lab_size_value_label.place(anchor="nw", x=150, y=94)
    tnl_aa_lab_size_scale = ttk.Scale(tab3_inspect, from_=0.2, to=0.8, orient="horizontal", variable=tnl_aa_lab_size,
                            command=lambda value, var=tnl_aa_lab_size, label=tnl_aa_lab_size_value_label: new_tnl_aa_lab_size(var, label))
    tnl_aa_lab_size_scale.place(anchor="nw", x=53, y=95, width=95)
    new_tnl_aa_lab_size(tnl_aa_lab_size, tnl_aa_lab_size_value_label)


    #  Tunnel AA ss style
    def tnl_res_ssstyle_changed(*args):
        Console("OFF")
        tnl_name = get_tnl_name() + "A"
        if tnl_name != None:
            if tnl_aa_ss.get():
                if tnl_res_ssstyle_option.get() != 'Trace':
                    ShowSecStrObj(tnl_name, tnl_res_ssstyle_option.get())
                    HideTrace(f'obj {tnl_name} atom CA')
                    tnl_res_atmtyp_changed()
                else:
                    HideSecStrObj(tnl_name)
                    ShowTrace(f'obj {tnl_name} atom CA')
                    BallStickAtom(f'obj {tnl_name} atom CA')
            else:
                HideSecStrObj(tnl_name)
                HideTrace(f'obj {tnl_name} atom CA')
        Console("hidden")



    # row 6, ss
    tnl_aa_ss = tk.BooleanVar(value=True)
    chkbox_tnl_aa_ss = ttk.Checkbutton(tab3_inspect)
    chkbox_tnl_aa_ss.configure(text='SecStr', variable=tnl_aa_ss, command=tnl_res_ssstyle_changed)
    chkbox_tnl_aa_ss.place(anchor="nw", x=0, y=120)

    tnl_res_ssstyle_option = tk.StringVar()
    tnl_res_ssstyle_option.set("Ribbon")
    tnl_res_ssstyle_dropdown = ttk.OptionMenu(tab3_inspect, tnl_res_ssstyle_option, "Ribbon", "Ribbon", "Cartoon", "Tube", "Trace")
    tnl_res_ssstyle_dropdown.place(anchor="nw", width=100, height=27, x=65, y=117)
    tnl_res_ssstyle_option.trace_add("write", tnl_res_ssstyle_changed)



    # row 7, surf
    tnl_aa_surf = tk.BooleanVar(value=False)

    def on_tnl_aa_surf(*args, dist=None):
        """Toggle molecular surface display around tunnel-lining residues."""
        if initializing:
            return
        tar = target()
        if tar != None:
            Console("OFF")
            tnl_name = get_tnl_name()
            if tnl_name != None and tnl_aa_surf.get():
                if ListObj(f'{tar}tnlAAsurf') == []:
                    new = DuplicateObj(tar)[0]
                    HideObj(new)
                    HideSecStrObj(new)
                    MoveObj(new, x=0.01)
                    NameObj(new, f'{tar}tnlAAsurf')
                if dist == None:
                    max1 = ListAtom(f'obj {tnl_name}A with maximum distance from obj {tnl_name}')[0]
                    max2 = ListAtom(f'obj {tnl_name} with minimum distance from {max1}')[0]

                    # an imperfect solution for the fact that try - except doesn't work for yasara commands.
                    if not all([True if x == 'True' else False for x in PairObj(f'{tar}Cl???????A', 'fix')]):
                        ShowMessage('Aligning coordinate systems, please wait.')
                        Wait(1)
                        transf_and_fix_ss(tar)
                        HideMessage()
                    maxd = Distance(max1, max2)[0]

                    try:
                        mind = float(PairObj(tar, 'max_ball_protein')[0])
                    except IndexError:
                        mind = float(PairObj('All', 'max_ball_protein')[0])

                    dist = round(mind + (tnl_aa_surf_dist.get() / 100.0) * (maxd - mind), 2)

                AddEnvRes(f'obj {tar}tnlAAsurf res protein')
                HideSurfObj(f'{tar}tnlAAsurf')
                surf_atms = " ".join(str(x) for x in ListAtom(f'obj {tar}tnlAAsurf res protein with distance < 0.1 from obj {tnl_name}A'))

                ShowSurfAtom(f'{surf_atms} with distance < {dist} from obj {tnl_name}', tnl_surfstyle_option.get(), outcol='atomcol', outalpha=tnl_aa_surf_alpha.get())
            else:
                HideSurfObj(f'obj {tar}tnlAAsurf')

            Wait(1)
            Console("hidden")
 
    def new_tnl_aa_surf_alpha(var, label, n=0):
        on_tnl_aa_surf()
        label.config(text=f"{var.get():.{n}f}")

    chkbox_tnl_aa_surf = ttk.Checkbutton(tab3_inspect)
    chkbox_tnl_aa_surf.configure(text='surface', variable=tnl_aa_surf, command=on_tnl_aa_surf)
    chkbox_tnl_aa_surf.place(anchor="nw", x=0, y=145)

    tnl_surfstyle_option = tk.StringVar()
    tnl_surfstyle_option.set("molecular")
    tnl_surfstyle_dropdown = ttk.OptionMenu(tab3_inspect, tnl_surfstyle_option, "molecular", "molecular", "VdW", "accessible")
    tnl_surfstyle_dropdown.place(anchor="nw", width=110, height=27, x=75, y=142)
    tnl_surfstyle_option.trace_add("write", on_tnl_aa_surf)

    tnl_surf_col_label = tk.Label(tab3_inspect, text=f"Color:")
    tnl_surf_col_label.place(anchor="nw", x=187, y=143)

    def on_tnl_surf_col(*args):
        if tnl_surf_col_option.get() == 'atomcol':
            col = tnl_res_col_option.get()
        elif tnl_surf_col_option.get() == 'element':
            col = 'element'
        else:
            col = ShowWin("ColorSelection","Select tunnel surface color", "Bow","Background","100")[0]
        if col == 'choose..':
            col = ColorAtom(f'obj {get_tnl_name()}A')[0]
        elif col == 'Distance to tunnel':
            obj = get_tnl_name()
            col = ColorAtom(f'obj {obj}A with minimum distance from obj {obj}')[0]
        ColorAtom(f'obj {target()}tnlAAsurf', col)
        on_tnl_aa_surf()
        Wait(1)

    tnl_surf_col_option = tk.StringVar()
    tnl_surf_col_option.set("black")
    tnl_surf_col_dropdown = ttk.OptionMenu(tab3_inspect, tnl_surf_col_option, "atomcol", "atomcol", 'element', "choose...")
    tnl_surf_col_dropdown.place(anchor="nw", width=83, height=27, x=232, y=142)

    tnl_surf_col_option.trace_add("write", on_tnl_surf_col)


    tnl_aa_surf_alpha_label = tk.Label(tab3_inspect, text=f"alpha:")
    tnl_aa_surf_alpha_label.place(anchor="nw", x=0, y=165)

    tnl_aa_surf_alpha = tk.DoubleVar(value=75)
    tnl_aa_surf_alpha_value_label = tk.Label(tab3_inspect, text=f"{tnl_aa_surf_alpha.get():.0f}")
    tnl_aa_surf_alpha_value_label.place(anchor="nw", x=288, y=166)
    tnl_aa_surf_alpha_scale = ttk.Scale(tab3_inspect, from_=0, to=100, orient="horizontal", variable=tnl_aa_surf_alpha,
                            command=lambda value, var=tnl_aa_surf_alpha, label=tnl_aa_surf_alpha_value_label: new_tnl_aa_surf_alpha(var, label))
    tnl_aa_surf_alpha_scale.place(anchor="nw", x=45, y=167, width=240)
    new_tnl_aa_surf_alpha(tnl_aa_surf_alpha, tnl_aa_surf_alpha_value_label)

    tnl_aa_surf_dist_label = tk.Label(tab3_inspect, text=f"dist:")
    tnl_aa_surf_dist_label.place(anchor="nw", x=0, y=184)


    def new_tnl_aa_surf_dist(var, label, n=1):
        on_tnl_aa_surf()
        label.config(text=f"{var.get():.{n}f}")

    tnl_aa_surf_dist = tk.DoubleVar(value=0.5)
    tnl_aa_surf_dist_value_label = tk.Label(tab3_inspect, text=f"{tnl_aa_surf_dist.get():.0f}")
    tnl_aa_surf_dist_value_label.place(anchor="nw", x=288, y=184)
    tnl_aa_surf_dist_scale = ttk.Scale(tab3_inspect, from_=0, to=100, orient="horizontal", variable=tnl_aa_surf_dist,
                            command=lambda value, var=tnl_aa_surf_dist, label=tnl_aa_surf_dist_value_label: new_tnl_aa_surf_dist(var, label))
    tnl_aa_surf_dist_scale.place(anchor="nw", x=30, y=186, width=255)
    new_tnl_aa_surf_dist(tnl_aa_surf_dist, tnl_aa_surf_dist_value_label)


    ## Tunnel diameter section
    separator1 = ttk.Separator(tab3_inspect)
    separator1.configure(orient="horizontal")
    separator1.place(anchor="nw", height=2, width=242, x=70, y=212)

    label2 = ttk.Label(tab3_inspect)
    label2.configure(
        font="TkSmallCaptionFont",
        foreground="#797979",
        text='Tunnel crosssection')
    label2.place(anchor="nw", x=0, y=205)

    dpi = 52  
    tnl_dia_canv_width = 155
    tnl_dia_canv_height = 130
    figsize_inches = (tnl_dia_canv_width / dpi, tnl_dia_canv_height / dpi)  # Convert pixel dimensions to inches

    fig = Figure(dpi=dpi)
    ax = fig.add_subplot(111)

    # Update the figure size
    fig.set_size_inches(figsize_inches[0], figsize_inches[1], forward=True)

    # Now calculate plot_width and plot_height based on figure size and dpi
    plot_width, plot_height = figsize_inches[0] * dpi, figsize_inches[1] * dpi


    def make_axis(tnl_name):
        """Create a PCA-based principal axis for a tunnel and load it as a YASARA object.

        Saves the tunnel surface as a .obj file, computes PCA to find the principal
        axis, identifies the two extreme points (inner/outer), and creates an arrow
        and axis object in YASARA.

        Returns (ext_center_atom, ext_outer_atom) — atom numbers of the two endpoints.
        """
        DelObj('???_axis ???_slice')
        # create the surface of the tunnel points as static object
        stat_surf = ShowSurfObj(tnl_name, 'vdw', 'static')

        # save the surface as obj
        surf_obj_file = os.path.join(PWD(), f'{NameObj(target())[0]}_{tnl_name}.obj')
        SaveWOb(stat_surf, surf_obj_file)
        Wait('continuebutton')
        
        DelObj(stat_surf)

        # use load_obj to create vertices and faces
        my_vertices, my_faces = load_obj(surf_obj_file)

        # write vertices as pdb file, with extreme points marked in residue column as EXT
        vert_pdb = os.path.join(PWD(), f'{NameObj(target())[0]}_{tnl_name}_vertRot.pdb')
        axis = find_principal_axis(my_vertices)
        extremes = find_extreme_points(my_vertices, axis)
        # for some reason, the points  need to be rotated 180° around the y-axis for transfering to work
        write_vertices_to_pdb(my_vertices, extremes, vert_pdb, rotate=True)

        # load pdb and transfer to the same coordinate system as the original tunnel points
        n = LoadPDB(vert_pdb, center=True)[0]
        Wait('continuebutton')
        TransferObj(n, tnl_name, 'keep')
        Wait('continuebutton')
        
        # create helper atom at center of target to determine what is inside and what is outside
        cx,cy,cz = PosAtom(f"obj {target()}", mean=True, coordsys='global')
        cen = BuildAtom("C")
        PosAtom(f"obj {cen}", x = cx,y = cy, z = cz, coordsys='global')
        DelRes(f'obj {n} res UNL')
        ext_center = ListAtom(f'obj {n} with minimum distance from obj {cen}')[0]
        ext_outer = ListAtom(f'obj {n} with maximum distance from obj {cen}')[0]
        NameAtom(ext_center, 'In')
        NameAtom(ext_outer, 'Out')

        # show arrow between extremes
        ShowArrow('atatom', ext_outer, 'atatom',  ext_center, color='black')
        StickObj(n)
        ColorObj(n, 'black')
        NameObj(n, f'{ListObj(tnl_name)[0]:03d}_axis')
        DelObj(f'{cen}')
        return ext_center, ext_outer

    def draw_diameter_plot(tnl_name, slice_obj, fig, ax, on_canvas=True, only_area=True):
        """Draw the cross-section plot for a tunnel at the current slice position.

        Projects tunnel points near the cutting plane into 2D, clusters them,
        calculates areas (and optionally inscribed circles), and renders the
        plot either on the embedded canvas or in a separate matplotlib window.
        """
        if on_canvas:
            ax.clear()
        else:
            fig, ax = plt.subplots()

        plane_points_pos = np.array(PosAtom(f'obj {slice_obj}', coordsys='global')).reshape(-1,3)
        tnl_points_pos = np.array(PosAtom(f'Obj {tnl_name}', coordsys='global')).reshape(-1,3)
        ball_spacing = float(PairObj(target(), 'ball_spacing')[0])
        threshold = ball_spacing / 800
        near_plane, near_indices, not_near_plane, not_near_indices = find_points_near_plane(tnl_points_pos, plane_points_pos, distance_threshold=threshold)
        if near_indices.size > 0:
            tnl_points = np.array(ListAtom(f'Obj {tnl_name}'))
            SegAtom(tnl_points, '.')
            SegAtom(tnl_points[near_indices], 'cutp')
            if cut_points_chk.get():
                ShowAtom(tnl_points)
                HideAtom(tnl_points[not_near_indices])

            near_plane_points_projected, original_indices, plane_origin, u, v = project_points_onto_plane(near_plane, plane_points_pos)

            if len(near_plane_points_projected) > 0:
                # Apply DBSCAN clustering to the near plane points
                eps = 1.4
                min_samples = 1  # Minimum samples for a core point, this could be adjusted based on your point density
                cluster_labels = cluster_points_with_dbscan(near_plane_points_projected, eps, min_samples)

                # You can now separate the points by clusters based on the labels
                unique_labels = set(cluster_labels)
                clusters = {label: near_plane_points_projected[cluster_labels == label] for label in unique_labels if label != -1}
                # Initialize variables to track the min and max bounds of all clusters
                all_data_x_min = float('inf')
                all_data_x_max = float('-inf')
                all_data_y_min = float('inf')
                all_data_y_max = float('-inf')

                cluster_shapes = {}
                total_area = 0

                if not only_area:
                    StickAtom(tnl_points)

                # Loop over clusters to calculate bounds
                for label, cluster_points in clusters.items():
                    area, merged_shape = calculate_area_of_points(cluster_points, ball_spacing * 2, radius=0.75)
                    # print(f'cluster {label} has area {area}')
                    x, y = merged_shape.exterior.xy
                    if not only_area:
                        try:
                            max_circle_center, max_circle_radius = find_maximum_inscribed_circle(merged_shape)
                            if max_circle_center is not None:
                                # Find the index of the closest original 3D point
                                closest_point_index = find_closest_point_index(near_plane_points_projected, max_circle_center)
                                original_point_index = original_indices[closest_point_index]
                                correct_index = near_indices[original_point_index]
                                descriptor = int(tnl_points[correct_index])
                                BallAtom(descriptor)
                        except AttributeError:
                            continue
                    total_area += area
                    cluster_shapes[label] = (x, y, area)
                    # Update the bounds for all clusters
                    all_data_x_min = min(all_data_x_min, min(x))
                    all_data_x_max = max(all_data_x_max, max(x))
                    all_data_y_min = min(all_data_y_min, min(y))
                    all_data_y_max = max(all_data_y_max, max(y))

                # Set the axis limits after determining the bounds for all clusters
                all_data_width = all_data_x_max - all_data_x_min
                all_data_height = all_data_y_max - all_data_y_min
                max_data_extent = max(all_data_width, all_data_height)
                fig_aspect_ratio = plot_width / plot_height

                if np.isnan(max_data_extent) or not np.isfinite(max_data_extent):
                    return

                if all_data_width > all_data_height:
                    new_y_half_extent = max_data_extent / fig_aspect_ratio / 2
                    ax.set_ylim([all_data_y_min - new_y_half_extent, all_data_y_max + new_y_half_extent])
                    ax.set_xlim([all_data_x_min, all_data_x_max])
                else:
                    new_x_half_extent = max_data_extent * fig_aspect_ratio / 2
                    ax.set_xlim([all_data_x_min - new_x_half_extent, all_data_x_max + new_x_half_extent])
                    ax.set_ylim([all_data_y_min, all_data_y_max])

                # Plot the clusters now with the updated axis limits
                for label, (x, y, area) in cluster_shapes.items():
                    ax.fill(x, y, color='black', label=f'Cluster {label}')
                    ax.plot(x, y, color='cyan', linewidth=0.75, label=f'Cluster {label}')
                    # Add labels only in the separate window
                    if not on_canvas:
                        # Calculate the centroid of the cluster
                        centroid_x = sum(x) / len(x)
                        centroid_y = sum(y) / len(y)

                        # Place the text annotation near the centroid
                        ax.text(centroid_x, centroid_y, f"{area:.2f} \u212B\u00b2", 
                                ha='center', va='center', color='red', fontsize=8.5, fontweight='bold')


                # Set the aspect ratio and other plot properties
                ax.set_aspect('equal', adjustable='datalim')
                ax.set_title(f"total area ({diamter_height.get():.1f}): {total_area:.2f} \u212B\u00b2")
                ax.grid(True, linewidth=0.3, color='gray')


                # Move ticks to the bottom and left spines
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')

                # Redraw the canvas
                if on_canvas:
                    canvas.draw()
                else:
                    plt.show()

            else:
                ShowMessage('Unexpected state: near_indices and projected points mismatch.')
                wc()
        else:
            ax.clear()
            ax.set_title(f"tunnel crosssection area: 0 \u212B\u00b2")
            ax.axis('off')
            canvas.draw()


    def on_diameter(*args):
        """Callback for cross-section height slider. Builds cutting plane and draws the plot."""
        if initializing:
            return
        Console('OFF')
        if tnl_insp_option.get() != 'All':
            Wait(1)
            tnl_name = get_tnl_name()

            # if not existant, make axis:
            if ListObj(f'{ListObj(tnl_name)[0]:03d}_axis') == []:
                ext_center, ext_outer = make_axis(tnl_name)
            else:
                DelObj
                ext_center = ListAtom('obj ' + f'{ListObj(tnl_name)[0]:03d}_axis ' + 'atom In')[0]
                ext_outer = ListAtom('obj ' + f'{ListObj(tnl_name)[0]:03d}_axis ' + 'atom Out')[0]

            DelObj(f'{ListObj(tnl_name)[0]:03d}_slice')
            
            # Create square cutting through tunnel along axis
            point1 = PosAtom(ext_outer, coordsys='global')
            point2 = PosAtom(ext_center, coordsys='global')
            
            side_length = CUTTING_PLANE_SIDE_LENGTH

            position = diamter_height.get() / 100
            vertices = square_vertices(point1, point2, side_length, position)

            # display a black square to indicate slice
            for i in range(4):
                n= BuildAtom('du')
                PosAtom(f'Obj {n}', *vertices[i], coordsys='global')
                if i == 0: 
                    slice_obj = n
                else:
                    JoinObj(n, slice_obj)
            NameObj(slice_obj, f'{ListObj(tnl_name)[0]:03d}_slice')
            HideObj(slice_obj)
            ShowPolygonAtoms('black', cut_axis_alpha.get(), 4, *ListAtom(f'obj {slice_obj}'))

            draw_diameter_plot(tnl_name, slice_obj, fig, ax)

        else:
            ax.clear()
            ShowMessage('Select a tunnel first')
            Wait(25)
            HideMessage()            
        Wait(1)
        Console("hidden")


    canvas = FigureCanvasTkAgg(fig, master=tab3_inspect)  
    canvas_widget = canvas.get_tk_widget()

    diamter_height = tk.DoubleVar(value=0.5)
    diamter_height_scale = ttk.Scale(tab3_inspect, from_=0, to=100, orient="horizontal", variable=diamter_height, command=on_diameter)

    def on_axis(*args):
        Console('off')
        SwitchObj(f'???_axis', convert_status(axis_chk.get())) 
        Wait(1)   
        Console("hidden")  

    axis_chk = tk.BooleanVar(value=True)  
    axis_button = ttk.Checkbutton(tab3_inspect)
    axis_button.configure(text='axis', variable=axis_chk, command=on_axis)


    def on_cut_points(*args):
        Console('off')
        if not cut_points_chk.get():
            ShowObj(get_tnl_name())
        else:
            if ListAtom(f'obj {get_tnl_name()} segment cutp') != []:
                HideObj(get_tnl_name())
                ShowAtom(f'obj {get_tnl_name()} segment cutp')
            else:
                on_diameter()
        Console("hidden")  

    cut_points_chk = tk.BooleanVar(value=True)  
    cut_points_button = ttk.Checkbutton(tab3_inspect)
    cut_points_button.configure(text='only cut pts', variable=cut_points_chk, command=on_cut_points)


    def new_cut_axis_alpha(var, label, n=0):
        on_diameter()
        label.config(text=f"{var.get():.{n}f}")

    cut_axis_alpha_label = tk.Label(tab3_inspect, text=f"alpha")
    cut_axis_alpha = tk.IntVar(value=90)
    cut_axis_alpha_value_label = tk.Label(tab3_inspect, text=f"{cut_axis_alpha.get():.0f}")
    cut_axis_alpha_scale = ttk.Scale(tab3_inspect, from_=1, to=100, orient="horizontal", variable=cut_axis_alpha,
                            command=lambda value, var=cut_axis_alpha, label=cut_axis_alpha_value_label: new_cut_axis_alpha(var, label))
    new_cut_axis_alpha(cut_axis_alpha, cut_axis_alpha_value_label)

    def on_cut_detail():
        Console('off')
        if tnl_insp_option.get() != 'All' and len(ListObj('???_slice')) > 0:
            draw_diameter_plot(get_tnl_name(), ListObj('???_slice')[0], fig, ax, on_canvas=False)
        else:
            ax.clear()
            ShowMessage('Select and slice a tunnel first')
            Wait(25)
            HideMessage()            
        Wait(1)
        Console("hidden")

    dia_plot = ttk.Button(tab3_inspect)
    dia_plot.configure(text='Detailed Plot', command=on_cut_detail)
    


    
    def plot_radius_vs_height(circles, only_area=True):
        """
        Plot the radius of the maximum inscribed circles and area as a function of the height.

        :param circles: List of tuples containing (height, descriptor, max_circle_center, max_circle_radius, area).
        :param only_area: Boolean flag to indicate if only the area should be plotted.
        """
        height_to_radii = {}
        height_to_areas = {}

        for height, descriptor, center, radius, area in circles:
            if height not in height_to_radii:
                height_to_radii[height] = []
                height_to_areas[height] = []
            height_to_radii[height].append(radius)
            height_to_areas[height].append(area)

        heights = []
        radii = []
        areas = []

        for height in sorted(height_to_radii.keys()):
            for radius in height_to_radii[height]:
                heights.append(float(height))
                radii.append(radius)
            for area in height_to_areas[height]:
                areas.append(area)

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Set the x-ticks every 5 steps using numpy.arange
        ax1.set_xticks(np.arange(0, 100, 5))

        # Enable gridlines
        ax1.grid(True, which='both')

        # If not only_area, add second axis
        if not only_area:
            color = 'tab:blue'
            ax1.set_xlabel('Height')
            ax1.set_ylabel('Maximum Radius of Inscribed Circle', color=color)
            ax1.plot(heights, radii, 'o', linestyle='None', color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:red'
            ax2.set_ylabel('Area', color=color)  # we already handled the x-label with ax1
            ax2.plot(heights, areas, 'o', linestyle='None', color=color)
            ax2.tick_params(axis='y', labelcolor=color)
        else:
            color = 'tab:green'
            ax1.set_xlabel('Height')
            ax1.set_ylabel('Area', color=color)
            ax1.plot(heights, areas, 'o', linestyle='None', color=color)
            ax1.tick_params(axis='y', labelcolor=color)

        # Adjust the x-tick labels using numpy.arange
        plt.xticks(np.arange(0, 100, 5))

        # Display the plot
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.title('Maximum Radius of Inscribed Circle and Area vs. Height' if not only_area else 'Area vs. Height')
        plt.grid(True)
        plt.show()



    def on_make_path():
        """Find and display the shortest A* path through the selected tunnel."""
        Console('Off')
        ShowObj(get_tnl_name())

        tnl_name = get_tnl_name()  # Replace with the actual tunnel name
        if ListObj(f'{ListObj(tnl_name)[0]:03d}_axis') == []:
            ext_center, ext_outer = make_axis(tnl_name)
        else:
            ext_center = ListAtom('obj ' + f'{ListObj(tnl_name)[0]:03d}_axis ' + 'atom In')[0]
            ext_outer = ListAtom('obj ' + f'{ListObj(tnl_name)[0]:03d}_axis ' + 'atom Out')[0]

        ext_center = ListAtom(f'Obj {tnl_name} with minimum distance from atom {ext_center}')[0]
        ext_outer = ListAtom(f'Obj {tnl_name} with minimum distance from atom {ext_outer}')[0]
        
        HideArrowAtom(f'obj {tnl_name}')
        NameAtom(f'obj {tnl_name}', 'UNL')
        shortest_path_points = find_shortest_path(tnl_name, ball_spacing * 1.02, ball_spacing * connect_cut * 1.021, ext_center, ext_outer, coarse_only=rough_path_chk.get())
        
        if shortest_path_points:
            NameAtom(shortest_path_points, '_SP')
            for i in range(len(shortest_path_points)):
                if i > 0:
                    ShowArrow('atatom', int(shortest_path_points[i]), 'atatom', int(shortest_path_points[i-1]), 0.2, 0, ColorAtom(int(shortest_path_points[i]))[0])
            expose_path()
        else:
            ShowMessage('No path found for this tunnel.')
            Wait('continuebutton')
            HideMessage()
            
        Console('Hidden')


    make_path = ttk.Button(tab3_inspect)
    make_path.configure(text='Path', style='Toolbutton', command=on_make_path)
    
    def expose_path():
        Console('off')
        tnl_name = get_tnl_name()
        SwitchObj(tnl_name, 'ON')
        ShowObj(tnl_name)
        if expose_path_chk.get():
            HideAtom(f'obj {tnl_name} atom UNL')

        Console("hidden")  
        
    rough_path_chk = tk.BooleanVar(value=True)  
    rough_path_button = ttk.Checkbutton(tab3_inspect)
    rough_path_button.configure(text='coarse', variable=rough_path_chk)

    expose_path_chk = tk.BooleanVar(value=True)  
    expose_path_button = ttk.Checkbutton(tab3_inspect)
    expose_path_button.configure(text='expose', variable=expose_path_chk, command = expose_path)


    def on_adjust():
        Console("OFF")
        GrabObj(f'???_axis ???_slice')
        CenterObj(f'???_axis')
        ShowMessage('Rotate axis now. Press Continue to accept new rotation')
        Wait('continuebutton')
        GrabAll()
        HideMessage()
        on_diameter()
        Console("hidden")        

    adjust_ax = ttk.Button(tab3_inspect)
    adjust_ax.configure(text='Adjust', style='Toolbutton', command=on_adjust)

    def on_reset():
        Console('off')
        DelObj('???_axis')
        if tnl_insp_option.get() != 'All':
            make_axis(get_tnl_name())
            on_diameter()
        else:
            ShowMessage('Select a tunnel first.')
            Wait(20)
            HideMessage()
        Console("hidden")

    reset_ax = ttk.Button(tab3_inspect)
    reset_ax.configure(text='Reset', style='Toolbutton', command=on_reset)

    def on_diam_up():
        diamter_height.set(diamter_height.get() + 0.5)
        on_diameter()

    def on_diam_down():
        diamter_height.set(diamter_height.get() - 0.5)
        on_diameter()

    diam_down = ttk.Button(tab3_inspect)
    diam_down.configure(text='<', style='Toolbutton', command=on_diam_down)

    diam_up = ttk.Button(tab3_inspect)
    diam_up.configure(text='>', style='Toolbutton', command=on_diam_up)

    # --------------------------------------------------------
    #  DIALOG MAINLOOP
    # --------------------------------------------------------

    # Hide Tab 2 and Tab 3 if no tunnels exist yet
    if target() == None:
        notebook.tab(1, state='hidden')
        notebook.tab(2, state='hidden')

    Console("hidden")
    initializing = False

    root.mainloop()


# Keep showing the dialog until the user clicks "Cancel"
while tunneler_dialog():
    pass

