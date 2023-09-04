# YASARA PLUGIN
# TOPIC:       Protein Tunnels
# TITLE:       Yasara Tunneler
# AUTHOR:      M.J.L.J. Fürst
# LICENSE:     GPL (www.gnu.org)
# DESCRIPTION: This plugin identifies tunnels in protein structures
#              Dependencies:
#              Python:
#              $ pip3 install numpy scipy
#              PCL: I recommend installing via a conda environment
#              $ conda create pcl_env
#              $ conda activate pcl_env
#              (pcl_env) $ conda install -c sirokujira pcl --channel conda-forge
#              (pcl_env) $ conda install -c sirokujira python-pcl --channel conda-forge
#              
#              Parameters: I recommend testing default parameters first. Then, play with the surface 

"""
MainMenu: Analyze
  PullDownMenu: Tunnels
    Submenu: Find Tunnels
      ObjectSelectionWindow: Step 1: Select target objects to find tunnels for
      Request: Tunneler
"""

# check if dependencies are available
from yasara import *
import importlib
import sys
import subprocess
import re
from configparser import ConfigParser

def check_and_install_module(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False

def w(message=''):
    if prog > 1:
        ShowMessage(message)
    if prog == 2:
        StickAtom("element Du")
        Wait(1)
    elif prog == 3:
        Wait('Continuebutton')

# Yield successive n-sized chunks from lst.
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

  
def write_cif_file(points, output_file, ori='right'):
    atom_labels = ['ATOM{}'.format(i) for i in range(1, len(points) + 1)]
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
    points = np.array(points).reshape(-1, 3)
    hull = ConvexHull(points)
    hull_vertices = hull.points
    hull_simplices = hull.simplices
    return(hull_vertices, hull_simplices)

def get_cube_points(hull_vertices, ball_spacing):
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
   # Delete points outside the first shape
    tri = Delaunay(hull_vertices)
    inside_mask = tri.find_simplex(cube_points) >= 0
    shape_points = cube_points[inside_mask]
    return(shape_points)

def point_cloud(target, ignore_surface, keep_surf_points, surf_con_prev, build_polygon=False):
    if keep_surf_points:
        p1 = PosAtom(f'obj {target} with distance > {ignore_surface} from accessible surface of obj {target}', coordsys='global')
        if build_polygon:
            p2 = PosAtom(f'obj {target} with distance > {ignore_surface + surf_con_prev} from accessible surface of obj {target}', coordsys='global')
    else:
        p1 = PosAtom(f'obj {target} with distance > {ignore_surface + surf_con_prev} from accessible surface of obj {target}', coordsys='global')
        if build_polygon:
            p2 = PosAtom(f'obj {target} with distance > {ignore_surface} from accessible surface of obj {target}', coordsys='global')
        
    hull_vertices, hull_simplices = get_hull(p1)

    if build_polygon:
        show_polygon(f'{target}', hull_vertices, hull_simplices)
        hull_vertices2, hull_simplices2 = get_hull(p2)
        show_polygon(f'{target}', hull_vertices2, hull_simplices2, 'yellow', 'Tpolygon2')

    cube_points = get_cube_points(hull_vertices, ball_spacing)
    shape_points = get_shape_points(cube_points, hull_vertices)

    if keep_surf_points:
        # # Calculate the plane normals
        # triangle_normals = np.cross(
        #     hull_vertices[hull_simplices[:, 1]] - hull_vertices[hull_simplices[:, 0]],
        #     hull_vertices[hull_simplices[:, 2]] - hull_vertices[hull_simplices[:, 0]]
        # )
        # triangle_normals /= np.linalg.norm(triangle_normals, axis=1)[:, np.newaxis]

        # # Calculate the distances from each point to the planes defined by the triangles
        # distances = np.empty((len(shape_points), len(hull_simplices)))
        # for i, triangle in enumerate(hull_simplices):
        #     triangle_vertices = hull_vertices[triangle]
        #     distances[:, i] = np.abs(
        #         np.dot(shape_points - triangle_vertices[0], triangle_normals[i])
        #     )

        # # Check if any distance in each row is less than or equal to the threshold
        # threshold_mask = np.any(distances <= surf_con_prev, axis=1)
        # outer_points = shape_points[threshold_mask]
        # shape_points = shape_points[~threshold_mask]


        shape_points2 = get_shape_points(cube_points, hull_vertices2)

        # # Combine the arrays and sort
        # combined_array = np.vstack((shape_points, shape_points2))
        # sorted_indices = np.lexsort(combined_array.T)
        # sorted_combined = combined_array[sorted_indices]

        # # Calculate the differences between consecutive rows and get 0s (duplos)
        # differences = np.diff(sorted_combined, axis=0)
        # duplicate_mask = np.allclose(differences, 0, atol=1e-6)
        # duplicate_indices = np.where(duplicate_mask)[0]

        # # Separate  duplicate rows from  combined array & find unique rows in shape_points
        # duplicate_rows = sorted_combined[duplicate_indices]
        # outer_points = np.setdiff2d(shape_points, duplicate_rows, assume_unique=True, equal_nan=True)
        # shape_points = shape_points2

        nrows, ncols = shape_points.shape
        dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
                'formats': ncols * [shape_points.dtype]}

        non_common_rows = np.setdiff1d(shape_points.view(dtype), shape_points2.view(dtype))
        
        # Convert structured array to a regular array
        outer_points = non_common_rows.view(shape_points.dtype).reshape(-1, ncols)
        shape_points = shape_points2


    else:
        outer_points = None

    w('Calculated convex hull and filled it with points. Loading points in Yasara.')
    return([outer_points, shape_points])


def load_points_yasara(target, point_cloud, keep_exclusion, keep_close):
    # keep excluded points at the surface as separate object
    if keep_exclusion:
        write_cif_file(point_cloud[0], PWD() + os.path.sep + f'{target}outside.cif')
        outside_points = LoadCIF(f'{PWD()}{os.path.sep}{target}outside.cif', correct=False, center=False)[0]
        MoveObj(outside_points,z=-50)
        NameObj(outside_points, f'{target}excluded')

    write_cif_file(point_cloud[1], PWD() + os.path.sep + f'{target}inside.cif')
    inside_points = LoadCIF(f'{PWD()}{os.path.sep}{target}inside.cif', correct=False, center=False)[0]
    MoveObj(inside_points,z=-50)

    # keep points close to the protein as separate object
    if keep_close:
        keep = DuplicateAtom(f'obj {target}outside {target}inside element Du with distance < {max_ball_protein} from Obj {target} res protein')
        NameObj(keep, f'{target}Close2Prot')
        JoinObj(f'{target}Close2Prot', ListObj(f'{target}Close2Prot', format='OBJNUM')[0])
        ColorObj(f'{target}Close2Prot', 'white')
    StickObj(f'{target}inside {target}outside')
    w(f'Loaded {CountAtom(f"obj {target}inside {target}outside"):,} points successfully. Determining tunnel points.')


def generate_tunnel_points(target, point_protein_distance, pclfile, ignore_res):
    # Delete points close to target from the point cloud
    if ignore_res == [0]:
        ignore_res = ''
    else:
        ignore_res = ListRes(f'obj {target} res !' + " and !".join(ignore_res[1:]))
        ignore_res = f'obj {target} ' + " ".join(ignore_res)

    DelAtom(f'Obj {target}inside {target}excluded with distance < {point_protein_distance} from Obj {target} {ignore_res}')

    # remove points outside the accessible surface with large water probe
    # Deprecated feature 
    # if remove_surface_points:
    #     SurfPar(probe=3)
    #     prot = DuplicateAtom(f'obj {target} res protein')[0]
    #     tun = DuplicateObj(f'{target}inside')[0]
    #     JoinObj(tun, prot)
    #     keep = DuplicateAtom(f'obj {target}inside with distance < 7.5 from accessible surface of obj {prot}')
    #     NameObj(keep, f'{target}Close2Surf')
    #     DelAtom(f'obj {target}inside with distance < 7.5 from accessible surface of obj {prot}')
    #     DelObj(prot)
    #     RenumberObj(keep, prot)
    #     SurfPar(probe=1.4)

    # Combine existing tunnel points (if we are doing iterative analysis)
    if ListObj('TunnelPoints') != []:
        DelAtom(f'Obj {target}inside with distance < 0.01 from obj TunnelPoints')
        if ListObj(f'{target}inside') != []:
            JoinObj(f'{target}inside', 'TunnelPoints')
    NameObj(f'{target}inside', 'TunnelPoints')

    # save the remaining (=tunnel) points
    points_to_cluster = np.array(PosAtom(f'Obj TunnelPoints')).reshape(-1, 3)
    np.save(f'{PWD()}{os.path.sep}{pclfile}', points_to_cluster)
    if mds > 0:
        w('Proceeding with MD.')
    else:
        w('Clustering points.')


def cluster_tunnel_points(target, pclfile):
    print('Using pcl clustering method')
    # run pcl to cluster the points
    noerr = subprocess.run(f"conda run -n pcl_env python -c \"\
import json\n\
import pcl\n\
from numpy import load, float32\n\
points_to_cluster = load(\"{PWD()}{os.path.sep}{pclfile}\")\n\
cloud = pcl.PointCloud()\n\
cloud.from_array(points_to_cluster.astype(float32))\n\
ec = cloud.make_EuclideanClusterExtraction()\n\
ec.set_ClusterTolerance(1.01 * {ball_spacing})\n\
ec.set_MinClusterSize({min_vol / ball_spacing})\n\
with open(\"{PWD()}{os.path.sep}point_clusters.json\", \"w\") as file:\n\
    json.dump(ec.Extract(), file)\"", 
    shell=True, capture_output=True, text=True)
    with open(f'{PWD()}{os.path.sep}point_clusters.json', 'r') as file:
        sorted_cluster_indices = json.load(file)
 
    points_names = np.array(ListAtom(f'Obj TunnelPoints', format='ATOMNUM'))
    for i, indices in enumerate(sorted_cluster_indices):
        c = DuplicateAtom(" ".join(str(i) for i in points_names[indices]))[0]
        NameObj(c, f"{target}Clu{len(indices):06d}")
        ColorObj (c, (i +1) * 25)
        new = DuplicateRes(f'obj {target} res protein with distance < 4 from obj {c}')
        NameObj(new, NameObj(c)[0] + 'A')

    RenumberObj(f'{target} {target}excluded {target}Close2Prot {target}Close2Surf', target)
    RenumberObj(f'{target} {target}excluded {target}Close2Prot {target}Close2Surf {target}Clu???????', target)
    DelObj(f'TunnelPoints')
    w('Finished clustering.')

def cluster_tunnel_points_nopcl(target, ball_spacing, pc_object='TunnelPoints'):
    print('Using Yasara clustering method')
    counter = 1
    while True:
        if counter > 1:
            chk = ListObj(pc_object)
            if chk == []:
                break
        try:
            cur = ListAtom('obj ' + str(pc_object))
        except RuntimeError:
            ShowMessage('No tunnels found')
            plugin.end()
        SelectAtom(cur[0])
        org = 0
        while True:
            new_atms = ListAtom(f'obj {pc_object} atom !selected with distance < {(ball_spacing / 2) + ball_spacing} from obj {pc_object} atom selected', format="ATOMNUM")
            SelectAtom(" ".join([str(x) for x in new_atms]), mode="add")
            org = org + len(new_atms)
            if len(new_atms) == 0:
                c = DuplicateAtom('Selected')[0]
                NameObj(c, f"{target}Clu{CountAtom(f'obj {c}'):06d}")
                counter += 1
                DelAtom('obj ' + str(pc_object) + ' atom selected')
                UnselectAll()
                break

    org_obj_nums = ListObj(f'{target}Clu??????', format='OBJNUM')
    free_obj_num = max(ListObj('All', format='OBJNUM')) + 1
    RenumberObj(f'{target}Clu??????', free_obj_num)

    obj_names = ListObj(f'{target}Clu??????', format='OBJNAME')
    obj_nums = ListObj(f'{target}Clu??????', format='OBJNUM')
    obj_nums_sorted = [x for _, x in reversed(sorted(zip(obj_names, obj_nums)))]

    for i, obj in enumerate(obj_nums_sorted):
        ColorObj (obj, (i +1) * 25)
        RenumberObj(obj, org_obj_nums[i])
    
    for obj in ListObj(f'{target}Clu??????', format='OBJNUM'):
        new = DuplicateRes(f'obj {target} res protein with distance < 4 from obj {obj}')
        NameObj(new, NameObj(obj)[0] + 'A')

    RenumberObj(f'{target} {target}excluded {target}Close2Prot {target}Close2Surf', target)
    RenumberObj(f'{target} {target}excluded {target}Close2Prot {target}Close2Surf {target}Clu???????', target)
    DelObj(f'TunnelPoints')
    w('Finished clustering.')

Console("off")

if request == 'Tunneler':
    required_modules = { 'numpy': 'numpy', # Import name before colon, pip installation name after colon
                        'scipy': 'scipy' } 
    missing_modules = [module for module in required_modules if not check_and_install_module(module)]
    if missing_modules:
        print('Python interpreter:', sys.executable)
        install_choice =\
            ShowWin("Custom","Missing software",600,180,
                    "Text",         20, 48,"The following python modules weren't found on your system:",
                    "Text",         20, 73, ",".join(missing_modules),
                    "Text",         20, 98,"Do you want to install these automatically?",
                    "Button",      250,130,"No",
                    "Button",      350,130,"Yes")[0]
        
        if install_choice == 'Yes':
            for module in missing_modules:
                install_name = required_modules[module]
                subprocess.call([sys.executable, '-m', 'pip', 'install', install_name])
        else:
            ShowMessage("Some python modules are missing, try installing manually. Exiting.")
            plugin.end()

    # check previously saved parameters
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Tunneler.ini')):
        print('found prev. settings')
        settings_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Tunneler.ini')
    else:
        print('no sets')
    # Ask user input for parameters
    ignore_surface, ball_spacing, max_ball_protein, keep_prot_points, surf_con_prev, keep_surf_points, mds, min_vol, use_all_res, build_pol, prog =\
        ShowWin("Custom","Step 2: Tunnel analysis parameters",600,380,
                "NumberInput",         20, 50, "_I_gnore surface up to (A)", 3.8, 0, 10,
                "NumberInput",         315, 50, "Ball _s_pacing (A)", 0.33, 0, 2,
                "NumberInput",         315, 125, "Max ball-protein _d_istance (A)", 2.8, 2, 4,
                "Text",                477, 157, "Keep points",
                "CheckBox",            440, 148, "", False,
                "NumberInput",         20, 125, "Surface connection prevention (A)", 2.7, 0, 10,
                "Text",                192, 157, "Keep points",
                "CheckBox",            155, 148, "", False,
                "NumberInput",         20, 200, "Number of MD iterations", 0, 0, 10,
                "NumberInput",         315, 200, "Min. cluster volume (A^3)", 50, 1, 1000,
                "RadioButtons",  2, 2, 250, 260,  "Use all residues in selected target",
                                    250, 300,  "Select residues to ignore",
                "CheckBox",            250, 340, "Build Polygon", True,
                "RadioButtons",  3, 2, 20, 260,  "No progress (fast)",
                                    20, 300,  "Show progress",
                                    20, 340,  "Debug",
                "Button",              540, 340, "_O_K")
    


    # Example variables
    variables = {
        'ignore_surface':ignore_surface,
        'ball_spacing':ball_spacing,
        'max_ball_protein':max_ball_protein,
        'keep_prot_points':keep_prot_points,
        'surf_con_prev':surf_con_prev,
        'keep_surf_points':keep_surf_points,
        'mds':mds,
        'min_vol':min_vol,
        'use_all_res':use_all_res,
        'build_pol':build_pol,
        'prog':prog
    }
    config = ConfigParser()

    # Add the variables to the ConfigParser object
    config['Variables'] = {}
    for name, value in variables.items():
        config['Variables'][name] = str(value)

    # Save the variables to an INI file
    with open('config.ini', 'w') as configfile:
        config.write(configfile)

    print("Variables saved to config.ini")

    target = [selection[0].object[j].number.inyas for j in range(selection[0].objects)][0]
    target_name = NameObj(target)[0]

    if use_all_res == 2:
        uniq_res = set(ListRes(f'Obj {target} res !protein', format='RESNAME'))
        ignore_res = ShowWin('Custom', "Step 3: Select ignore part",300,340,
                             "Text",        20, 50, "Tunnel analysis will act as if the",
                             "Text",        20, 70, "selected residues weren't there.",
                             "Text",        20, 90, "Usually at least HOH is selected.",
                             "List",        50,130,f"Residues in object {target}",195,128,"Yes", len(uniq_res), sorted([item for item in uniq_res]),
                             "Button",     105,290, "OK")
        
    w('Software dependency check. ')
    PrintCon()
    # old pcl check
    # s_start_time = time.perf_counter()
    # conda_test = subprocess.run(f"conda run -n pcl_env python -c 'import pcl'",shell=True, capture_output=True, text=True)
    # if conda_test.stderr != '':
    #     if 'command not found' in conda_test.stderr:
    #         ShowMessage('Error: `conda` wasn\'t found on the system.')
    #     elif 'EnvironmentLocationNotFound' in conda_test.stderr:
    #         ShowMessage('Error: `conda` found, but environment pcl_env doesn\t exist.')
    #     elif 'No module named' in conda_test.stderr:
    #         ShowMessage('Error: `conda` works and pcl_env exists, but pcl and/or python-pcl wasn\'t installed.')
    #     print('To install pcl, install (Ana/mini)conda and do:\n$ conda create pcl_env\n$ conda activate pcl_env\n(pcl_env) $ conda install -c sirokujira pcl --channel conda-forge\n(pcl_env) $ conda install -c sirokujira python-pcl --channel conda-forge')
    #     plugin.end()
   
    # Print(f'software check took {"{:.6f}".format(time.perf_counter()  - s_start_time)} seconds')

    s_start_time = time.perf_counter()
    conda_test = subprocess.run(f"conda run -n pcl_env python -c \"import importlib.util; check = importlib.util.find_spec(\"pcl\"); print(check)\"",shell=True, capture_output=True, text=True)
    use_pcl = True

    if conda_test.stderr != '':
        if 'command not found' in conda_test.stderr:
            w('Warning: `conda` wasn\'t found on the system. Using (slow) Yasara clustering')
        elif 'EnvironmentLocationNotFound' in conda_test.stderr:
            w('Warning: `conda` found, but environment pcl_env doesn\t exist. Using (slow) Yasara clustering')
        else:
            w('Warning: the plugin was unsuccessful in running conda. Check the console. Using (slow) Yasara clustering')
            print(conda_test.stderr)
        print('To install pcl, install (Ana/mini)conda and do:\n$ conda create pcl_env\n$ conda activate pcl_env\n(pcl_env) $ conda install -c sirokujira pcl --channel conda-forge\n(pcl_env) $ conda install -c sirokujira python-pcl --channel conda-forge')
        use_pcl = False
    else:
        if conda_test.stdout == 'None\n\n':
            w('Warning: `conda` works and pcl_env exists, but pcl and/or python-pcl wasn\'t installed. Using (slow) Yasara clustering')
            Wait(20)
            print('To install pcl, install (Ana/mini)conda and do:\n$ conda create pcl_env\n$ conda activate pcl_env\n(pcl_env) $ conda install -c sirokujira pcl --channel conda-forge\n(pcl_env) $ conda install -c sirokujira python-pcl --channel conda-forge')
            # plugin.end()
            use_pcl = False
 
    Print(f'software check v2 took {"{:.6f}".format(time.perf_counter()  - s_start_time)} seconds')


    # import required libraries
    import numpy as np
    from scipy.spatial import ConvexHull, Delaunay
    import json

    start_time = time.perf_counter()

    DelObj(f'{target}excluded {target}TPolygon? {target}Clu???????? {target}Close2Surf {target}Close2Prot {target}Surf CenterHlp Du {target}CutPlane ???_Sphere')

    # create dummy objects to fill object list gaps
    all_objs = ListObj('all')
    empty_objs = list(set(range(1,max(all_objs))) - set(all_objs))
    [BuildAtom('Du') for _ in empty_objs]

    w('Starting tunnel analysis, please wait.')
    point_cloud = point_cloud(target, ignore_surface=ignore_surface, keep_surf_points=keep_surf_points, surf_con_prev=surf_con_prev, build_polygon=build_pol)
    load_points_yasara(target, point_cloud, keep_exclusion = keep_surf_points, keep_close = keep_prot_points)
    generate_tunnel_points(target, point_protein_distance = max_ball_protein, pclfile='points_to_cluster.npy', ignore_res=ignore_res)

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
            load_points_yasara(target, point_cloud, keep_exclusion = keep_surf_points, keep_close = keep_prot_points)
            generate_tunnel_points(target, point_protein_distance = max_ball_protein, pclfile='points_to_cluster.npy', ignore_res=ignore_res)
            SaveSce(f'{target}_MD{i}.sce')
            if i == mds -1:
                RenumberObj(f'org_{target_name}'[:12], target)
                NameObj(f'org_{target_name}'[:12], target_name)
                DelObj('Water SimCELL md_obj')
                AddObj('All')
                SwitchObj(f'MD?_{target_name}', 'OFF')

    if use_pcl:
        cluster_tunnel_points(target, pclfile='points_to_cluster.npy')
    else:
        cluster_tunnel_points_nopcl(target, ball_spacing, pc_object='TunnelPoints')

    CenterAtom('All')
    BallStickAll()
    DelObj("Du")
    SwitchObj(f'{target}Clu??????? {target}excluded {target}Close2Surf {target}Close2Prot', 'OFF')
    SwitchObj(ListObj(f'{target}Clu??????')[:5], "ON")
    HideMessage()

    surf1 = DuplicateObj(target)[0]
    surf2 = DuplicateObj(target)[0]
    co1 = CutObj(surf1)[0]
    co2 = CutObj(surf2)[0]
    HideObj(f'{surf1} {surf2}')
    HideSecStrObj(f'{surf1} {surf2}')
    NameObj(f'{co1} {co2}', f'{target}CutPlane')
    ShowSurfRes(f'Obj {surf1} {surf2} res protein', 'molecular', outcol=1232, outalpha=85, incol='3f3f3f', inalpha=50)
    NameObj(f'Obj {surf1} {surf2}', f'{target}Surf'[:12])
    RotateObj(co2, x=90)
    SwitchObj(f'{co1} {co2}', 'OFF')
    SwitchObj(f'{str(target)}TPolygon?', 'off')
    SaveSce(f'org_{NameObj(target)[0]}_tunnels.sce')
    PrintCon()
    Print(f'Ran tunnels plugin in {"{:.2f}".format(time.perf_counter()  - start_time)} seconds on object {selection[0].objects} with parameters: \n   Exclude surface atoms up to                     {ignore_surface}\n   Ball spacing                                    {ball_spacing}\n   Maximum allowed ball distance to protein        {max_ball_protein}\n   Prevent tunnel surface connection with cutoff   {surf_con_prev}')

    img = MakeImage("Buttons",topcol="None",bottomcol="None")
    ShowImage(img,alpha=66,priority=0)
    PrintImage(img)

    def ShowButtons():
        Font("Arial",height=14,color="black")

        ShowButton("Tunnel on/off",x='12%', y='55%',color="White", height=40)
        ShowButton("Target on/off",x='12%', y='60%',color="White", height=40)
        ShowButton("Surf",x='3%', y='65%',color="White", height=40)
        ShowButton("SecStr",x='9%', y='65%',color="White", height=40)
        ShowButton("Nonprot",x='17%', y='65%',color="White", height=40)
        ShowButton("Color by tunnel/dist",x='12%', y='70%',color="White", height=40)
        ShowButton("Spheres",x='5%', y='75%',color="White", height=40)
        ShowButton("Balls",x='12%', y='75%',color="White", height=40)
        ShowButton("Points",x='18%', y='75%',color="White", height=40)
        ShowButton("Remove inside points",x='12%', y='80%',color="White", height=40)
        ShowButton("Exit",x='12%', y='85%',color="Red", height=40)


    ShowButtons()

elif request == 'Tunnelonoff':
    current = SwitchObj('?Clu??????')[0]
    if current == 'Off':
        SwitchObj('?Clu??????', 'ON')
    else:
        SwitchObj('?Clu??????', 'OFF')

elif request == 'Targetonoff':
    target = ListObj('?Surf', format='OBJNAME')[0][0]
    current = SwitchObj(target)[0]
    if current == 'Off':
        SwitchObj(target, 'ON')
    else:
        SwitchObj(target, 'OFF')

elif request == 'Surf':
    current = SwitchObj('1Surf')[0]
    if current == 'Off':
        SwitchObj('1Surf', 'ON')
    else:
        SwitchObj('1Surf', 'OFF')

elif request == 'Removeinsidepoints':
    import re
    tunnel_objs = ListObj('?Clu??????')
    delete_inside =\
        ShowWin("Custom","Remove non-visible points",435,180,
                "Text",         20, 48, "Do you want to delete or just hide the points?",
                "Text",         20, 73, "Hide for a medium gain in performance",
                "Text",         20, 98, "Delete for max. performance improvement",
                "Button",      170,130,"Hide",
                "Button",      260,130,"Delete")[0]
    ShowMessage(f'Removing inside points.')
    for targetobj in tunnel_objs:
        RemoveEnvRes('all')
        AddEnvRes(targetobj)
        n_before = CountAtom(f'Obj {targetobj}')
        if delete_inside == 'Delete':
            DelAtom(f'obj {targetobj} with distance > 2.55 from accessible surface of obj {targetobj}')
        else:
            HideAtom(f'obj {targetobj} with distance > 2.55 from accessible surface of obj {targetobj}')
        n = CountAtom(f'Obj {targetobj}')
        NameObj(targetobj, re.sub('[0-9]+$', f'{n:06d}', NameObj(targetobj)[0]))
        NameObj(targetobj + 1, 
                re.sub('[0-9]+$', f'{n:06d}', NameObj(targetobj)[0]) + 'A')
        ShowMessage(f'Removed {n_before - n} inside points from object {targetobj}')
        Wait(1)
    
    HideMessage()

elif request == 'Exit':
    # yasara bug? Doesn't remove button
    # FillRect(color='None')
    SaveSce('ExitTunnels.sce')
    Clear()
    LoadSce('ExitTunnels.sce')

elif request == 'Balls':
    on_tunnels = [x for x,y in zip(ListObj('?Clu??????'), SwitchObj('?Clu??????')) if y == 'On']
    on_spheres = [x for x,y in zip(NameObj('???_Sphere'), SwitchObj('???_Sphere')) if y == 'On']
    add_tunnels = [match.group() for x in on_spheres for match in [re.search('[0-9]+', x)] if match]
    SwitchObj(" ".join(add_tunnels), 'on')
    SwitchObj(" ".join(on_spheres), 'off')
    BallAtom('Obj ?Clu??????')

elif request == 'Points':
    on_tunnels = [x for x,y in zip(ListObj('?Clu??????'), SwitchObj('?Clu??????')) if y == 'On']
    on_spheres = [x for x,y in zip(NameObj('???_Sphere'), SwitchObj('???_Sphere')) if y == 'On']
    add_tunnels = [match.group() for x in on_spheres for match in [re.search('[0-9]+', x)] if match]
    SwitchObj(" ".join(add_tunnels), 'on')
    SwitchObj(" ".join(on_spheres), 'off')
    StickAtom('Obj ?Clu??????')

elif request == 'Spheres':
    if ListObj('???_Sphere') != []:
        on_tunnels = [x for x,y in zip(ListObj('?Clu??????'), SwitchObj('?Clu??????')) if y == 'On']
        SwitchObj(" ".join([str(f'{x:03d}') + '_Sphere' for x in on_tunnels]), "on")
        SwitchObj('?Clu??????', 'off')
        # SwitchObj(" ".join([x for x in NameObj('All') if re.search('^[0-9]+_sphere$', x)]), 'ON')
    else:
        a,r = ShowWin("Custom","Parameters for spheres",300,170,
                      "NumberInput",  20, 48,"Alpha",19,1,100,
                      "NumberInput", 175, 48,"Radius",0.45,0,1,
                      "Button",      150,120,"OK")
        tunnel_objs = " ".join([str(x) for x in ListObj('?Clu??????')])
        n = CountAtom(f'Obj {tunnel_objs}')
        if n < 3000:
            ShowMessage(f"Creating {CountAtom(f'Obj {tunnel_objs}')} Spheres.")
            sphere_level = 3
        elif n < 10000:
            ShowMessage(f"Creating {CountAtom(f'Obj {tunnel_objs}')} Spheres. This process can take several minutes.")
            sphere_level = 2
        elif n < 40000:
            ShowMessage(f"Creating {CountAtom(f'Obj {tunnel_objs}'):,} Spheres. This process can take very long. Click Continue or type StopPlugin in the Console")
            sphere_level = 1
            Wait('Continuebutton')
        else:
            ShowMessage(f"Creating {CountAtom(f'Obj {tunnel_objs}'):,} Spheres. This process can take extremely long. Click Continue or type StopPlugin in the Console")
            sphere_level = 0
            Wait('Continuebutton')
        Wait(1)
        tunnel_objs = ListObj(tunnel_objs)


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
                obj = ShowSphere(radius=r, color=col[o], alpha=a, level=sphere_level)
                PosObj(obj, p[(o+1)*3-3], p[(o+1)*3-2], p[(o+1)*3-1])
                if o == modulo_value:
                    modulo_value += 5000
                    jobj = ListObj('sphere')[0]
                    JoinObj('sphere', jobj)
                    ShowMessage(f'Created {o:,} / {len(atomlist):,} spheres of tunnel {NameObj(targetobj)[0]}.')
                    Wait(1)

            jobj = ListObj('sphere')[0]
            JoinObj('sphere', jobj)
            SwitchObj(jobj, on_off)
            NameObj('sphere', f'{targetobj:03d}_sphere')
    HideMessage()


elif request == 'Colorbytunneldist':
    from yasara import *

    def InsertObj(old_objnum, new_objnum):
        current_objs = ListObj('All', format='OBJNUM')
        index_start = [int(new_objnum) >= i for i in current_objs].index(False) -1
        shift_objs = current_objs[index_start:]
        [RenumberObj(i, i + 1) for i in list(reversed(shift_objs)) if i != old_objnum]
        RenumberObj(old_objnum, new_objnum)

    chunk_len=5000

    Console('Off')

    start_time = time.perf_counter()

    objs = [str(x) for x in ListObj('?Clu??????')]
    atms = [str(x) for x in ListAtom(ShowWin('AtomSelection', 'Select atom from which to calculate distance')[0])]


    min_color, max_color, per_tunnel, fast_mode = ShowWin("Custom","Parameters for tunnel colors",310,320,
                                                        "NumberInput",  20, 48,"Color of closest points",100,0,720,
                                                        "NumberInput",  20, 123,"Color of most distant points",380,0,720,
                                                        "CheckBox",     20,185,"Calculate distance per tunnel", False,
                                                        "CheckBox",     20,235,"Fast mode", True,
                                                        "Button",       150,280,"_O_K") 

    PrintCon()

    if len(atms) > 1:
        ShowMessage(f'Warning: {len(atms)} atoms match the selection for the distance calculations; using geometric mean of all selected atoms')
        Wait(35)
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
        print(f"Selected single atom {center} as center.")

    # if calculating for all points, check min and max distance first
    if not per_tunnel:
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


    for tunnel in objs:
        # object=selection[0].object[j]
        # tunnel = object.number.inyas

        tname = ListObj(tunnel, format='OBJNAME')[0]
        target = ListObj(f'atom {center}', format='OBJNUM')[0]

        ShowMessage(f'Coloring tunnel {tname}')
        Wait(1)

        n = DuplicateObj(tunnel)[0]
        # Wait('continuebutton')
        natoms = CountAtom("obj " + str(n))

        SwapAtom(f'Obj {n}', "Du")
        JoinObj(n, target)
        ShowMessage(f'Obj {tunnel}: getting distances of {natoms} points')
        Wait(1)

        disto = Distance(f'obj {target} element Du', center)

        new = DuplicateAtom(f'obj {target} element Du')
        DelAtom(f'obj {target} element Du')
        NameObj(new, 'coltunnel')
        on_off = SwitchObj(tunnel)[0]
        SwitchObj(new, on_off)
        DelObj(tunnel)
        RenumberObj(n, tunnel)
        NameObj(tunnel, tname)
        SwapAtom(f'obj {tunnel}', 'H', rename=False)

        atomlist = ListAtom(f'obj {tunnel}')

        ShowMessage(f'Obj {tunnel}: Coloring {natoms:,} points')
        Wait(1)

        def rescale_floats_to_range(float_list, min_int, max_int, min_float=None, max_float=None):
            print(f'rescaling between {min_int} and {max_int}')
            if min_float == None:
                min_float = min(float_list)
            if max_float == None:
                max_float = max(float_list)
            scaled_list = [int((x - min_float) / (max_float - min_float) * (max_int - min_int) + min_int) for x in float_list]
            return scaled_list
        
        if per_tunnel:
            mind = min(disto)
            maxd = max(disto)
        
        all_cols = rescale_floats_to_range(disto, min_color, max_color, mind, maxd)

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
                            #   int(100 + ((disto[(i * chunk_len) + j] - mind) / (maxd - mind)) * (280)))
        else:
            for i, chunk in enumerate(list(chunks(atomlist, chunk_len))):
                ShowMessage(f'Obj {tunnel}: Colored {i * chunk_len:,} / {len(atomlist):,} points.')
                Wait(1)
                for j in range(len(chunk)):
                    ColorAtom(atomlist[(i * chunk_len) + j], int(100 + ((disto[(i * chunk_len) + j] - mind) / (maxd - mind)) * (280)))
            
        
        print(f"Done in {'{:.6f}'.format(time.perf_counter()  - start_time)} seconds")

        # StickObj(tunnel)
        HideMessage()
    DelObj('CenterHlp')


plugin.end()


import timeit

