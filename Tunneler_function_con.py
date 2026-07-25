"""
Tunneler core pipeline — tunnel detection, geometry helpers, and YASARA wrappers.

This module contains the main Tunneler() entry point and all supporting functions
for detecting tunnels/cavities in protein structures loaded in YASARA. The pipeline:

  1. Build a rough surface point cloud around the protein (point_clouder)
  2. Load those points into YASARA as dummy atoms (load_points_yasara)
  3. Remove points too close to protein atoms → remaining = tunnel points (generate_tunnel_points)
  4. Cluster tunnel points with DBSCAN → individual tunnels (cluster_tunnel_points_dbscan)
  5. Optionally run short MD simulations between steps to sample conformational flexibility

The module also provides:
  - Stage-aware wrappers for DuplicateObj/Res/Atom that work in all YASARA license tiers
  - Geometry helpers for convex hulls, point clouds, and CIF file I/O
  - Visualization helpers for polygon display and surface-point filtering

Imported by Tunneler_LoadMenu_tk_con.py (the GUI entry point).
"""

from yasara import *
from yasara import DuplicateObj as y_DuplicateObj
from yasara import DuplicateRes as y_DuplicateRes
from yasara import DuplicateAtom as y_DuplicateAtom

import itertools
from configparser import ConfigParser
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from sklearn.cluster import DBSCAN

# ============================================================
#  CONSTANTS
# ============================================================

ROUGH_SURF_SPACING = 1.5        # Angstroms — grid spacing for the initial rough surface point cloud
OBJECT_Z_OFFSET = -50           # Angstroms — Z offset to hide helper objects off-screen
REFINED_SURF_SPACING = 0.6      # Angstroms — grid spacing for the refined surface representation
SURFACE_REFINE_DISTANCE = 2.5   # Angstroms — max distance from accessible surface for refined points
SURFACE_CONNECT_DISTANCE = 2.0  # Angstroms — flood-fill connectivity distance for surface shell
SURFACE_BUFFER = 2.2            # Angstroms — buffer added when initializing surface distance threshold
NEARBY_RESIDUE_DISTANCE = 4     # Angstroms — how close a residue must be to a tunnel to be included
DBSCAN_EPS_FACTOR = 1.01        # Small factor to slightly enlarge DBSCAN epsilon

# ============================================================
#  UTILITIES — Small helpers used throughout the plugin
# ============================================================

def int2let(number):
    """
    Convert an integer to a single letter code (0-Z), cycling through if the number is higher than 35.
    :param number: The integer to convert.
    :return: The single letter code as a string.
    """
    digits = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # Cycle through the digits
    single_digit = number % 36
    return digits[single_digit]


def wc(msg=''):
    """Show a message in YASARA and wait for the user to click Continue."""
    ShowMessage(msg)
    Wait('continuebutton')


def short_ss_to_long(x):
    """Convert a one-letter secondary-structure code to the full YASARA name.

    Mapping: C→Coil, T→Turn, H/I/G→Helix, E*→Sheet.
    """
    if x == 'C':
        return 'Coil'
    elif x == 'T':
        return 'Turn'
    elif x == 'H' or x == 'I' or x == 'G':
        return 'Helix'
    elif x[0] == 'E':
        return 'Sheet'


def transf_and_fix_ss(target):
    """Transfer coordinates and restore secondary structure in the 'A' (amino-acid) companion objects.

    Each tunnel cluster object (e.g. 3Cl...) has an 'A' companion holding nearby residues.
    After coordinate changes, the companion's position and sec-str assignments must be synced
    back from the original target object.
    """
     # transfer and fix ss in 'A' objs
    Console('off')
    for obj in ListObj(f'{target}Cl???????', format='OBJNAME'):
        TransferObj(f'{obj}A', obj, 'fix')
        PairObj(f'{obj}A', 'fix', True)
        reslist = ListRes(f'obj {obj}A', format='Mol MOLNAME res RESNAME RESNUM')
        sslist = []
        for x in reslist:
            ss = SecStrRes(f'Obj {target} {x}')[0]
            sslist.append(ss)
        for j in range(len(reslist)):
            SecStrRes(f'obj {obj}A {reslist[j]}', short_ss_to_long(sslist[j]))


def stagen(stage):
    """Convert a YASARA stage name to a numeric tier (View=1, Model=2, Dynamics=3, Structure=4).

    Used to gate features that require a certain license level.
    """
    return(['View', 'Model', 'Dynamics', 'Structure'].index(stage) +1)

# ============================================================
#  YASARA STAGE-AWARE WRAPPERS — DuplicateObj / Res / Atom
#
#  YASARA's built-in Duplicate commands only work in View stage.
#  These wrappers achieve the same result in higher stages by
#  saving to a temp .yob file, reloading, and transferring coords.
# ============================================================

def DuplicateObj(obj_sel):
    """Duplicate one or more YASARA objects into new object(s), working in any stage.

    Delegates to the built-in DuplicateObj. On every tier tested (View and
    Structure, YASARA 26.4) it creates isolated duplicate object(s) with global
    coordinates and names preserved — verified identical to the old manual
    routine (max coord diff 0.0 even under an applied translation+rotation), so
    the previous per-object SaveYOb/LoadYOb + TransferObj round-trip was
    unnecessary. This mirrors the DuplicateAtom fix (perf item B2); the round-trip
    here was per-object rather than full-scene, so a lighter cost and smaller
    crash-surface, but still redundant disk I/O.

    Returns a list with the new object number(s). (Old manual implementation is
    in git history if an edge case ever needs it.)
    """
    if obj_sel is None:
        ShowMessage('bugggg, obj_sel is None')
        wc()
        return([''])
    return y_DuplicateObj(obj_sel)

def DuplicateRes(res):
    """Duplicate residues, stage-aware. Delegates to DuplicateAtom in non-View stages."""
    if stagen(stage) == stagen('View'):
        new = y_DuplicateRes(res)
        return(new)
    else:
        return(DuplicateAtom(ListRes(res)))


def DuplicateAtom(atm):
    """Duplicate atoms into new object(s) — one new object per source object.

    Delegates to YASARA's built-in DuplicateAtom. On every tier tested (View and
    Structure, YASARA 26.4) the built-in already produces exactly what the old
    manual routine did: isolated new object(s) — one per source object — with
    global coordinates and the source object's name preserved, returned as a list
    of new object numbers.

    The previous non-View branch instead saved the ENTIRE scene to disk, deleted
    all but the selected atoms, saved the survivors per-object, reloaded the whole
    scene, then reloaded the pieces — a full SaveSce/LoadSce round-trip on EVERY
    call (once per tunnel cluster + once per residue companion). On large scenes
    (e.g. 5m10, ~8 round-trips/run over thousands of point-cloud atoms) the
    repeated full-scene serialization intermittently corrupted YASARA's heap
    (fatal error 39, "mem_free: Pointer to free not found") and was the single
    biggest per-run cost (perf item B2). The built-in avoids the round-trip
    entirely. Equivalence verified: 1mbn/1cv2/5m10 cloud fingerprints unchanged.

    (Old manual implementation preserved in git history if an edge case ever
    needs it.)
    """
    return y_DuplicateAtom(atm)

def stop_plugin(message):
    """Show an error message and terminate the plugin."""
    Console("OFF")
    ShowMessage(message)
    Wait('Continuebutton')
    HideMessage()
    Console("ON")
    plugin.end()


def is_float(value):
    """Return True if *value* can be converted to a float."""
    try:
        float(value)
        return True
    except ValueError:
        return False


# ============================================================
#  VISUALIZATION — Polygon display, surface-point filtering,
#                  object renumbering
# ============================================================

def ml_outside_points(target, mode, by=0.5):
    """Hide or show tunnel points near the protein surface (the 'surface points' slider).

    Adjusts a distance threshold stored in the roughsurf object's Seg field,
    then hides tunnel cluster points closer than that threshold to the accessible surface.

    Args:
        target: Object number of the protein target.
        mode: 'less' to show fewer surface points, anything else to show more.
        by: Step size in Angstroms for each adjustment.
    """
    Console("OFF")
    cur_dist = SegObj(f'{target}roughsurf')[0]
    if not is_float(cur_dist):
        disto = ListAtom(f'obj {target}Cl??????? with minimum distance from obj {target}roughsurf')[0]
        TransferObj(f'{target}roughsurf', ListObj(f'atom {disto}'))
        cur_dist = Distance(disto, ListAtom(f'obj {target}roughsurf with minimum distance from atom {disto}'))[0] + SURFACE_BUFFER
    else:
        if mode == 'less':
            cur_dist = float(cur_dist) + by
        else:
            cur_dist = float(cur_dist) - by
    cur_dist = f'{cur_dist:.1f}'
    SegObj(f'{target}roughsurf', cur_dist)
    ShowObj(f'{target}Cl???????')
    rough_surf = ListAtom(f'obj {target}roughsurf')
    if stagen(stage) > stagen('View'):
        surf_atom = FirstSurfAtom(rough_surf, 'accessible')[0]
    else:
        surf_atom = ListAtom(f'obj {target}roughsurf with maximum distance from obj {target}')
        print('Warning: this command might give unexpected results because you are using the free version of Yasara.')

    HideAtom(f'obj {target}Cl??????? with distance < {cur_dist} from accessible surface touched by {surf_atom}')
    Console("ON")


def sort_objs(target):
    """Renumber tunnel-related YASARA objects into a tidy consecutive sequence.

    After clustering, objects can have scattered numbers. This renumbers them so
    cluster objects and their 'A' companions alternate (target+1, target+2, ...),
    followed by helper objects (polygons, surfaces, etc.).
    """
    objs = ListObj(f'{target}Cl???????? {target}TPolygon? {target}Surf {target}CutPlane {target}roughsurf', format='OBJNUM')
    objs_str = " ".join([str(x) for x in objs])
    free_from = list(reversed((ListObj('All', format='OBJNUM'))))[0]
    RenumberObj(objs_str, free_from + 1)

    objs = ListObj(f'{target}Cl???????', format='OBJNUM')
    objs_str = " ".join([str(x) for x in objs])
    for i, obj in enumerate(objs):
        RenumberObj(obj, target + (i * 2) + 1)

    objs = ListObj(f'{target}Cl???????A', format='OBJNUM')
    objs_str = " ".join([str(x) for x in objs])
    for i, obj in enumerate(objs):
        RenumberObj(obj, target + (i * 2) + 2)

    objs = ListObj(f'{target}TPolygon? {target}Surf {target}CutPlane {target}roughsurf', format='OBJNUM')
    objs_str = " ".join([str(x) for x in objs])

    RenumberObj(objs_str, list(reversed(ListObj(f'{target}Cl???????A', format='OBJNUM')))[0] + 1)


def w(message=''):
    """Display a progress message in YASARA, respecting the current progress mode.

    Behavior depends on the global `prog` variable:
      - 'vis':  show message + briefly render dummy sticks (visual feedback)
      - 'wait': show message + wait for Continue button (debug mode)
      - 'fast': do nothing (silent)
    """
    global prog
    if 'prog' not in globals():
        prog = 'vis'
    if prog in ['vis', 'wait']:
        ShowMessage(message)
    if prog == 'vis':
        StickAtom("element Du")
        Wait(1)
    elif prog == 'wait':
        Wait('Continuebutton')


# ============================================================
#  GEOMETRY & POINT CLOUDS — Convex hull, grid filling, CIF I/O
# ============================================================

def write_cif_file(points, output_file, ori='right'):
    """Write a set of 3D points as dummy atoms in a minimal CIF file.

    YASARA can load CIF files, so this is used as a transport format to get
    numpy point clouds into the YASARA scene as atoms.

    Args:
        points: (N, 3) numpy array of coordinates.
        output_file: Path for the .cif file.
        ori: 'right' negates the X coordinate (YASARA uses a right-handed
             coordinate system that needs mirroring for some operations).
    """
    # YASARA maps _atom_site_label -> the atom name, which is capped at 4
    # characters. Labels like 'ATOM1' trip error 101 ("atom name is longer
    # than four characters") on newer YASARA CIF parsers, while older ones
    # accepted them. Encode the index in base-36 so labels stay <=4 chars
    # (unique up to 36**4 points; harmless duplicates beyond -- the label is
    # never referenced downstream, atoms are selected by object/element/pos).
    atom_labels = [np.base_repr(i % 36**4, 36) for i in range(len(points))]
    atom_symbols = ['X' for _ in range(len(points))]
    with open(output_file, 'w') as cif_file:
        cif_file.write('data_\n_cell_length_a   1.0\n_cell_length_b   1.0\n_cell_length_c   1.0\n_cell_angle_alpha   90.0\n_cell_angle_beta    90.0\n_cell_angle_gamma   90.0\nloop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n')
        if ori == 'right':
            for i in range(len(points)):
                cif_file.write('{} {} {:8.3f} {:8.3f} {:8.3f}\n'.format(atom_labels[i], atom_symbols[i], -points[i, 0], points[i, 1], points[i, 2]))
        else:
            for i in range(len(points)):
                cif_file.write('{} {} {:8.3f} {:8.3f} {:8.3f}\n'.format(atom_labels[i], atom_symbols[i], points[i, 0], points[i, 1], points[i, 2]))


def show_polygon(target, hull_vertices, hull_simplices, color='green', name='TPolygon'):
    """Render a convex hull as YASARA polygon objects and join them into one."""
    for i in range(len(hull_vertices[hull_simplices])):
        o = ShowPolygonPoints(color, 25, 3,
                            hull_vertices[hull_simplices][i][0][0],hull_vertices[hull_simplices][i][0][1],hull_vertices[hull_simplices][i][0][2],
                            hull_vertices[hull_simplices][i][1][0],hull_vertices[hull_simplices][i][1][1],hull_vertices[hull_simplices][i][1][2],
                            hull_vertices[hull_simplices][i][2][0],hull_vertices[hull_simplices][i][2][1],hull_vertices[hull_simplices][i][2][2])
        NameObj(o, str(target) + name)

    pobj = ListObj(str(target) + name)
    JoinObj(str(target) + name,pobj[0])
    Wait(1)

def get_hull(points):
    """Compute the convex hull of a set of 3D points. Returns (vertices, simplices)."""
    points = np.array(points).reshape(-1, 3)
    hull = ConvexHull(points)
    hull_vertices = hull.points
    hull_simplices = hull.simplices
    return(hull_vertices, hull_simplices)

def get_cube_points(hull_vertices, ball_spacing):
    """Generate a regular 3D grid of points covering the bounding box of the hull."""
    # Compute the bounding box around the Convex Hull
    bbox_min = np.min(hull_vertices, axis=0) - ball_spacing
    bbox_max = np.max(hull_vertices, axis=0) + ball_spacing

    # Generate points inside the bounding box at the fixed distance
    x_coords = np.arange(bbox_min[0], bbox_max[0], ball_spacing)
    y_coords = np.arange(bbox_min[1], bbox_max[1], ball_spacing)
    z_coords = np.arange(bbox_min[2], bbox_max[2], ball_spacing)
    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords)
    cube_points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    return(cube_points)


def get_shape_points(cube_points, hull_vertices):
    """Filter grid points to keep only those inside the convex hull (Delaunay test)."""
   # Delete points outside the first shape
    tri = Delaunay(hull_vertices)
    inside_mask = tri.find_simplex(cube_points) >= 0
    shape_points = cube_points[inside_mask]
    return(shape_points)

def create_surrounding_points(points, spacing, remove_edge=True):
    """Expand each point into a 3x3x3 neighborhood cube, then deduplicate.

    Used to build the rough surface point cloud around protein atoms.
    When remove_edge=True, only keeps neighbors that share at least one
    coordinate with the original point (i.e. face/edge neighbors, not pure corners).
    """
    all_coords = []
    for point in points:
        cube_coords = []
        for i in range(len(point)):
            grid_points = [point[i], point[i] - spacing, point[i] + spacing]
            cube_coords.append(grid_points)
        all_coords.append(cube_coords)
    all_permuts = []
    for i in range(len(all_coords)):
        coord = all_coords[i]
        point = points[i]
        coord_permuts = list(itertools.product(coord[0],coord[1],coord[2],))
        if remove_edge:
            coord_permuts = [tup for tup in coord_permuts if any(item in point for item in tup)]
        all_permuts.append(coord_permuts)
    ret_coords = np.array([point for sublist in all_permuts for point in sublist])
    return(np.unique(ret_coords, axis=0))


# ============================================================
#  TUNNEL ANALYSIS PIPELINE
#
#  point_clouder  →  load_points_yasara  →  generate_tunnel_points
#     →  cluster_tunnel_points_dbscan  →  (called from Tunneler)
# ============================================================

def point_clouder(target, ball_spacing, ignore_surface, keep_surf_points, surf_con_prev, build_polygon=False):
    """Build a point cloud that fills the interior of the protein.

    Steps:
      1. Get protein atom positions and create a rough surrounding grid (spacing 1.5 A).
      2. Load that grid as a CIF into YASARA (the 'roughsurf' object, shifted Z-50 to hide it).
      3. Get interior atoms (those buried deeper than ignore_surface from the accessible surface).
      4. Compute convex hull of interior atoms; fill it with a fine grid (ball_spacing).
      5. Keep only grid points inside the hull.

    If keep_surf_points is True, also computes an outer shell of points between two
    surface cutoffs — these are later used to prevent tunnels from connecting through
    the protein surface.

    Returns [outer_points, shape_points] where outer_points may be None.
    """
    Console("OFF")
    pdb_points = PosAtom(f'obj {target}', coordsys='global')
    pdb_points = np.array(pdb_points).reshape(-1, 3)
    ddh_points = create_surrounding_points(pdb_points, ROUGH_SURF_SPACING)
    ddh_points = np.unique(np.round(ddh_points, 0), axis=0)

    write_cif_file(ddh_points, PWD() + os.path.sep + f'{target}roughsurf.cif', ori = 'right')
    ddh =  LoadCIF(PWD() + os.path.sep + f'{target}roughsurf.cif', center=False, correct=True)
    MoveObj(ddh, z=OBJECT_Z_OFFSET)  # Shift off-screen so it doesn't interfere with the main view
    StickObj(ddh)
    ColorObj(ddh, 'white')
    SwitchObj(ddh, 'off')

    if keep_surf_points:
        p1 = PosAtom(f'obj {target} with distance > {ignore_surface} from accessible surface of obj {target}', coordsys='global')
        if build_polygon:
            p2 = PosAtom(f'obj {target} with distance > {ignore_surface + surf_con_prev} from accessible surface of obj {target}', coordsys='global')
    else:
        p1 = PosAtom(f'obj {target} with distance > {ignore_surface + surf_con_prev} from accessible surface of obj {target}', coordsys='global')
        if build_polygon:
            p2 = PosAtom(f'obj {target} with distance > {ignore_surface} from accessible surface of obj {target}', coordsys='global')

    if len(p1) == 0:
        ShowMessage('No points that fit these parameters. You probably should reduce the surface cutoff.')
        Wait('Continuebutton')
        HideMessage()
        plugin.end()
    hull_vertices, hull_simplices = get_hull(p1)

    if build_polygon:
        show_polygon(f'{target}', hull_vertices, hull_simplices)
        hull_vertices2, hull_simplices2 = get_hull(p2)
        show_polygon(f'{target}', hull_vertices2, hull_simplices2, 'yellow', 'Tpolygon2')

    cube_points = get_cube_points(hull_vertices, ball_spacing)
    shape_points = get_shape_points(cube_points, hull_vertices)

    if keep_surf_points:
        shape_points2 = get_shape_points(cube_points, hull_vertices2)

        nrows, ncols = shape_points.shape
        dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
                'formats': ncols * [shape_points.dtype]}

        non_common_rows = np.setdiff1d(shape_points.view(dtype), shape_points2.view(dtype))
        
        # Convert structured array to a regular array
        outer_points = non_common_rows.view(shape_points.dtype).reshape(-1, ncols)
        shape_points = shape_points2

    else:
        outer_points = None

    w('|   Calculated convex hull and filled it with points. Loading points in Yasara.')
    return([outer_points, shape_points])


def load_points_yasara(target, point_cloud, keep_exclusion):
    """Load the numpy point cloud into YASARA as CIF dummy-atom objects.

    Creates '{target}inside' (tunnel candidate points) and optionally
    '{target}excluded' (outer shell points). Both are shifted Z-50 to keep
    them out of the main viewport.
    """
    Console("OFF")
    # keep excluded points at the surface as separate object
    if keep_exclusion:
        write_cif_file(point_cloud[0], PWD() + os.path.sep + f'{target}outside.cif')
        outside_points = LoadCIF(f'{PWD()}{os.path.sep}{target}outside.cif', correct=True, center=False)[0]
        MoveObj(outside_points,z=OBJECT_Z_OFFSET)
        NameObj(outside_points, f'{target}excluded')

    write_cif_file(point_cloud[1], os.path.join(PWD(), f'{target}inside.cif'))
    inside_points = LoadCIF(f'{PWD()}{os.path.sep}{target}inside.cif', correct=True, center=False)[0]
    os.remove(os.path.join(PWD(), f'{target}inside.cif'))
    MoveObj(inside_points,z=OBJECT_Z_OFFSET)

    StickObj(f'{target}inside {target}outside')
    w(f'|   Loaded {CountAtom(f"obj {target}inside {target}outside"):,} points successfully. Determining tunnel points.')


def generate_tunnel_points(target, point_protein_distance, ignore_res, mds):
    """Remove grid points that are too close to protein atoms — the survivors are tunnel points.

    Also merges with any existing 'TunnelPoints' object (for iterative MD analysis)
    and renames the result to 'TunnelPoints'.

    Returns the tunnel points as an (N, 3) numpy array ready for DBSCAN clustering.
    """
    Console("OFF")
    # Delete points close to target from the point cloud
    if ignore_res == [0] or ignore_res == []:
        ignore_res = ''
    else:
        ignore_res = ListRes(f'obj {target} res !' + " and !".join(ignore_res))
        ignore_res = f'obj {target} ' + " ".join(ignore_res)

    DelAtom(f'Obj {target}inside {target}excluded with distance < {point_protein_distance} from Obj {target} {ignore_res}')

    # Combine existing tunnel points (if we are doing iterative analysis)
    if ListObj('TunnelPoints') != []:
        DelAtom(f'Obj {target}inside with distance < 0.01 from obj TunnelPoints')
        if ListObj(f'{target}inside') != []:
            JoinObj(f'{target}inside', 'TunnelPoints')
    NameObj(f'{target}inside', 'TunnelPoints')

    # save the remaining (=tunnel) points
    points_to_cluster = np.array(PosAtom(f'Obj TunnelPoints')).reshape(-1, 3)

    if mds > 0:
        w('Proceeding with MD.')
    else:
        w('|   Clustering points.')
    return(points_to_cluster)


def cluster_tunnel_points_dbscan(target, points, min_vol, ball_spacing, connect_cut, recluster=False):
    """Cluster tunnel points using DBSCAN and create per-cluster YASARA objects.

    For each cluster above the minimum volume threshold:
      - Creates a new object named '{target}Cl{letter}{count:06d}'
      - Colors it with a unique hue
      - Marks buried interior points with segment 'bur'
      - Creates an 'A' companion object with nearby protein residues

    Args:
        target: Object number of the protein.
        points: (N, 3) numpy array of tunnel point coordinates.
        min_vol: Minimum cluster volume (in A^3) to keep.
        ball_spacing: Grid spacing used during point cloud generation.
        connect_cut: Multiplier for DBSCAN epsilon (eps = ball_spacing * connect_cut * 1.01).
        recluster: If True, merges existing cluster objects back before re-clustering.
    """
    Console("OFF")

    # Perform DBSCAN clustering
    eps = (ball_spacing * connect_cut) * DBSCAN_EPS_FACTOR
    clustering = DBSCAN(eps=eps, min_samples=2).fit(points)
    labels = clustering.labels_

    # Set the minimum number of points a cluster must have.
    # The cloud is a cubic grid at spacing `ball_spacing`, so it holds ~1 point
    # per ball_spacing**3 of volume. Converting the minimum tunnel VOLUME
    # (min_vol, in A^3) to a point count therefore divides by ball_spacing**3.
    # (Previously this divided by ball_spacing, so min_vol was not actually a
    # volume and the "A^3" label was wrong; this also makes the threshold a true
    # physical volume, independent of the chosen grid spacing.)
    min_points = min_vol / ball_spacing ** 3

    # Initialize an empty list for each cluster
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    clusters = [[] for _ in range(num_clusters)]

    # Populate the clusters with point indices
    for index, label in enumerate(labels):
        if label != -1:
            clusters[label].append(index)

    # Filter out small clusters
    filtered_clusters = [cluster for cluster in clusters if len(cluster) >= min_points]
    sorted_cluster_indices = sorted(filtered_clusters, key=len, reverse=True)
    if recluster:
        tnl_obj = ListObj(f'{target}Cl???????', format='OBJNUM')[0]
        JoinObj(f'{target}Cl???????', tnl_obj)
        NameObj(tnl_obj, 'TunnelPoints')

    points_names = np.array(ListAtom(f'Obj TunnelPoints', format='ATOMNUM'))

    DelObj(f'{target}Cl???????A')

    for i, indices in enumerate(sorted_cluster_indices):
        c = DuplicateAtom(" ".join(str(i) for i in points_names[indices]))[0]
        NameObj(c, f"{target}Cl{int2let(c)}{len(indices):06d}")
        ColorObj (c, (i +1) * 25)
        # mark points burried inside with the segment field
        SegAtom(f'obj {c} with distance >1 from vdw surface of obj {c}', 'bur')
        new_res = ListRes(f'obj {target} res protein with distance < {NEARBY_RESIDUE_DISTANCE} from obj {c}', format='RESNUM')
        new = DuplicateRes(f'obj {target} res {" ".join([str(x) for x in new_res])}')
        aa_obj = NameObj(c)[0] + 'A'
        NameObj(new, aa_obj)
        ShowObj(new)

    sort_objs(int(target))
    CenterAtom('All')
    BallStickAll()
    DelObj(f'TunnelPoints')
    w('|   Finished clustering.')


# ============================================================
#  MAIN ENTRY POINT
# ============================================================

def Tunneler(target, ignore_res, ignore_surface=3.8, ball_spacing=0.33, max_ball_protein=2.8,
             surf_con_prev=2.7, keep_surf_points=False, mds=0, min_vol=5, connect_cut=1, build_pol=True, prog='vis', progress_var=None,percent_label=None):
    """Run the full tunnel detection pipeline on a YASARA protein object.

    This is the main entry point called by the GUI. It orchestrates the entire
    pipeline: point cloud → tunnel points → DBSCAN clustering → visualization.

    Optionally runs short MD simulations between iterations to sample flexibility.

    After completion, creates a refined surface representation (roughsurf) and
    saves the scene as '{name}_tunnels.sce'.

    Args:
        target: YASARA object number of the protein to analyze.
        ignore_res: List of residue identifiers to exclude from the analysis.
        ignore_surface: Distance (A) from accessible surface to ignore (surface cutoff).
        ball_spacing: Grid spacing (A) for the point cloud.
        max_ball_protein: Maximum allowed distance (A) between a grid point and the protein.
        surf_con_prev: Extra distance (A) to prevent tunnel-surface connections.
        keep_surf_points: If True, keep an outer shell of points at the surface.
        mds: Number of MD simulation iterations (0 = no MD).
        min_vol: Minimum cluster volume (A^3) to report as a tunnel.
        connect_cut: DBSCAN connectivity multiplier (eps = ball_spacing * connect_cut).
        build_pol: If True, build convex hull polygon visualization.
        prog: Progress display mode ('vis', 'fast', or 'wait').
        progress_var: Optional tkinter IntVar for progress bar updates.
        percent_label: Optional tkinter Label for percentage display.
    """
    # Publish the chosen progress mode to the module global that w() reads.
    # (Can't use `global prog` here: `prog` is also a parameter name, so we set
    # the module global explicitly. Without this the GUI's fast/vis/wait choice
    # was ignored and w() always fell back to its 'vis' default.)
    globals()['prog'] = prog
    Console("OFF")
    Print('------------------------------------------------------------------------------------')
    Print('|                                -----------.                                ')
    Print('|                                | TUNNELER |                                ')
    Print('|                                -----------*                                ')
    Print('|                                           ')

    # write config
    variables = {
        'ignore_surface':ignore_surface,
        'ball_spacing':ball_spacing,
        'max_ball_protein':max_ball_protein,
        'surf_con_prev':surf_con_prev,
        'keep_surf_points':keep_surf_points,
        'mds':mds,
        'min_vol':min_vol,
        'connect_cut':connect_cut,
        'build_pol':build_pol,
        'prog':prog
    }
    config = ConfigParser()

    # Add the variables to the ConfigParser object
    config['Variables'] = {}
    for name, value in variables.items():
        config['Variables'][name] = str(value)
        PairObj(target, name, value)

    # Save the variables to an INI file
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Tunneler_config.ini')
    with open(config_file, 'w') as configfile:
        config.write(configfile)

    # check Yasara stage requirements
    if mds > 0 and stagen(stage) < stagen('Dynamics'):
        stop_plugin('Yasara Dynamics or higher is required for tunnel analysis with MD simulations.')

    if build_pol and stagen(stage) < stagen('Model'):
        stop_plugin('Yasara Model or higher is required for building the polygon.')

    # parse settings
    target_name = NameObj(target)[0]

    start_time = time.perf_counter()
    if ListImage('All') != []:
        if stagen(stage) > stagen('View'):
            PrintImage(1)
            FillRect(color='None')
        else:
            print('Warning: Possibly unexpected visualization. An image was detected that cannot be deleted because you are using the free version of Yasara.')
        DelImage(1)

    DelObj(f'{target}excluded {target}TPolygon? {target}Cl????????? {target}roughsurf {target}Close2Surf {target}tnlAAsurf {target}Close2Prot {target}Surf {target}SS {target}NonProt {target}H2O CenterHlp Du {target}excl_pts {target}CutPlane ???_shape ???_Sphere CntrOfRot')

    # Create dummy objects to fill gaps in the object number list.
    # YASARA's RenumberObj needs consecutive slots; dummies are deleted at the end.
    all_objs = ListObj('all')
    empty_objs = list(set(range(1,max(all_objs))) - set(all_objs))
    [BuildAtom('Du') for _ in empty_objs]

    w('|   Starting tunnel analysis, please wait.')

    if progress_var != None and percent_label != None:
        progress_var.set(3)
        percent_label.config(text=f'3%')

    point_cloud = point_clouder(target, ball_spacing=ball_spacing, ignore_surface=ignore_surface, keep_surf_points=keep_surf_points, surf_con_prev=surf_con_prev, build_polygon=build_pol)

    if progress_var != None and percent_label != None:
        progress_var.set(10)
        percent_label.config(text=f'10%')

    load_points_yasara(target, point_cloud, keep_exclusion = keep_surf_points)

    if progress_var != None and percent_label != None:
        progress_var.set(20)
        percent_label.config(text=f'20%')

    points_to_cluster = generate_tunnel_points(target, point_protein_distance = max_ball_protein, ignore_res=ignore_res, mds=mds)

    if progress_var != None and percent_label != None:
        progress_var.set(30)
        percent_label.config(text=f'30%')

    if mds > 0:
        # MD
        ShowMessage('Preparing MD')
        md_obj = DuplicateObj(target)[0]
        NameObj(md_obj, f'{target_name}_neutr')
        RemoveObj(f'!{md_obj}')
        CleanObj(md_obj)
        OptHydObj(md_obj, 'YASARA')
        CellAuto(2, 'cuboid')
        Boundary('periodic')
        ForceField('Amber14', setpar=False)
        FillCellWater()
        neut_result = ExperimentNeutralization()
        Experiment('On')
        Wait('ExpEnd')
        HideObj('Water')
        NameObj(md_obj, f'{target_name}_min')
        min_result = ExperimentMinimization()
        Experiment('On')
        Wait('ExpEnd')
        HideObj('Water')
        NameObj(md_obj, f'{target_name}_md')
        for i in range(0, mds):
            if i > 0:
                RemoveObj('TunnelPoints')
                AddObj('Water')
                AddObj(md_obj)
                RemoveObj(target)
            ShowMessage('starting MD')
            ShowMessage(f'Running 1000 fs of MD {i + 1} / {mds}')
            Sim('On')
            Wait(1000, unit='femtoseconds')
            Sim('Off')
            AddObj('TunnelPoints')
            RemoveObj('Water')
            analysis_obj = DuplicateObj(md_obj)[0]
            RemoveObj(md_obj)
            DelRes(f'obj {analysis_obj} res hoh')
            DelAtom(f'obj {analysis_obj} element h')
            if i == 0:
                NameObj(target, f'org_{target_name}'[:12])
            NameObj(analysis_obj, f'MD{i + 1}_{target_name}'[:12])
            RenumberObj(analysis_obj, target)
            ShowMessage('starting point analysis')

            # in-MD tunnel analysis
            load_points_yasara(target, point_cloud, keep_exclusion = keep_surf_points)
            points_to_cluster = generate_tunnel_points(target, point_protein_distance = max_ball_protein, ignore_res=ignore_res, mds=mds)
            SaveSce(f'{target}_MD{i}.sce')
            if i == mds -1:
                RenumberObj(f'org_{target_name}'[:12], target)
                NameObj(f'org_{target_name}'[:12], target_name)
                DelObj('Water SimCELL md_obj')
                AddObj('All')
                SwitchObj(f'MD?_{target_name}', 'OFF')

    if progress_var != None and percent_label != None:
        progress_var.set(80)
        percent_label.config(text=f'80%')

    cluster_tunnel_points_dbscan(target, points_to_cluster, min_vol, ball_spacing, connect_cut)

    if progress_var != None and percent_label != None:  
        progress_var.set(90)
        percent_label.config(text=f'90%')

    DelObj("Du")
    SwitchObj(f'{target}Cl???????? {target}excluded {target}Close2Surf {target}Close2Prot', 'OFF')
    SwitchObj(ListObj(f'{target}Cl???????')[:5], "ON")
    HideMessage()
    SwitchObj(f'{str(target)}TPolygon?', 'off')

    if progress_var != None and percent_label != None:  
        progress_var.set(95)
        percent_label.config(text=f'95%')

    # --- Refine the rough surface representation ---
    # Build a finer point cloud (0.6 A spacing) near the protein surface, then
    # trim it to only keep points close to the accessible surface. This replaces
    # the initial coarse roughsurf with a smoother version used for visualization.
    tpoints = PosAtom(f'obj {target} with distance < {NEARBY_RESIDUE_DISTANCE} from accessible surface of obj {target}roughsurf', coordsys='global')
    tpoints_hull_vertices, hull_simplices = get_hull(tpoints)
    tpoints_cube_points = get_cube_points(tpoints_hull_vertices, REFINED_SURF_SPACING)
    tpoints_shape_points = get_shape_points(tpoints_cube_points, tpoints_hull_vertices)
    write_cif_file(tpoints_shape_points, PWD() + os.path.sep + f'{target}tpoints_shape_points.cif')
    tpoints_outside_points = LoadCIF(f'{PWD()}{os.path.sep}{target}tpoints_shape_points.cif', correct=True, center=False)[0]
    MoveObj(tpoints_outside_points,z=OBJECT_Z_OFFSET)
    # Keep only points within SURFACE_REFINE_DISTANCE of the protein's accessible surface
    DelAtom(f'obj {tpoints_outside_points} with distance > {SURFACE_REFINE_DISTANCE} from accessible surface of obj {target}')
    # Keep only the connected surface shell (flood-fill from one surface atom)
    a = FirstSurfAtom(f'obj {tpoints_outside_points}', 'accessible')[0]
    DelAtom(f'obj {tpoints_outside_points} with distance > {SURFACE_CONNECT_DISTANCE} from accessible surface touched by {a}')
    # Replace the old roughsurf with this refined version
    n = ListObj(f'{target}roughsurf', format='OBJNUM')[0]
    HideObj(tpoints_outside_points)
    DelObj(f'{target}roughsurf')
    NameObj(f'{tpoints_outside_points}', f'{target}roughsurf')
    RenumberObj(f'{target}roughsurf', n)
    # Add backbone atoms to roughsurf for better surface rendering
    tar = DuplicateObj(target)[0]
    DelAtom(f'obj {tar} atom !backbone')
    JoinObj(tar, f'{target}roughsurf')

    if progress_var != None and percent_label != None:  
        progress_var.set(99)
        percent_label.config(text=f'99%')


    ShowSurfRes(f'obj {target}roughsurf element Du', 'accessible',outcol='blue', outalpha=50,incol='red',inalpha=50)
    HideObj(f'{target}roughsurf')
    SwitchObj(f'{target}roughsurf', 'off')
    SwitchObj('cutplane','off')

    # transfer and fix ss in 'A' objs
    transf_and_fix_ss(target)
    PairObj(target, 'dist_col','')
    PairObj(target, 'dist_sel','')

    SaveSce(f'{NameObj(target)[0]}_tunnels.sce')
    PrintCon()
    Print(f'|   Ran tunnels plugin in {"{:.2f}".format(time.perf_counter()  - start_time)} seconds on object {target} with parameters: \n|      Exclude surface atoms up to                     {ignore_surface}\n|      Ball spacing                                    {ball_spacing}\n|      Connect cutoff                                  {connect_cut}\n|      Min volume                                      {min_vol}\n|      Maximum allowed ball distance to protein        {max_ball_protein}\n|      Prevent tunnel surface connection with cutoff   {surf_con_prev}\n------------------------------------------------------------------------------------')
