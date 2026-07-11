"""
Tunneler geometry module — cross-section analysis, diameter measurement, and A* pathfinding.

This module provides functions for analyzing tunnel geometry after detection:

  - PCA-based principal axis determination
  - Cutting-plane construction (square perpendicular to the axis)
  - 2D cross-section computation (project points onto a plane, cluster, measure area)
  - Maximum inscribed circle calculation (Voronoi-based)
  - Diameter analysis pipeline (sweep slices along the tunnel axis)
  - A* pathfinding through tunnel point clouds
  - File I/O helpers for Wavefront .obj and PDB formats

Imported by Tunneler_LoadMenu_tk_con.py (the GUI).
"""

import numpy as np
from sklearn.decomposition import PCA
from yasara import *
from sklearn.cluster import DBSCAN
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from scipy.spatial import Voronoi

# ============================================================
#  CONSTANTS
# ============================================================

CUTTING_PLANE_SIDE_LENGTH = 30  # Angstroms — side length of the square cutting plane for cross-sections

# ============================================================
#  PCA & AXIS ANALYSIS
# ============================================================

def find_principal_axis(vertices):
    """
    Find the principal axis of a set of vertices using PCA.
    
    :param vertices: Vertices of the shape.
    :return: The principal axis (first principal component).
    """
    pca = PCA(n_components=1)
    pca.fit(vertices)
    return pca.components_[0]

def calculate_height_along_axis(vertices, axis):
    """
    Calculate the height of the shape along a given axis.
    
    :param vertices: Vertices of the shape.
    :param axis: The axis along which to calculate the height.
    :return: Height of the shape along the axis.
    """
    # Project vertices onto the axis
    projections = np.dot(vertices, axis)
    
    # Calculate the height as the difference between the max and min projections
    height =  np.max(projections) - np.min(projections)
    return(height)


def find_extreme_points(vertices, axis):
    """
    Find the two extreme points along a given axis.

    :param vertices: Vertices of the shape.
    :param axis: The axis along which to find the extreme points.
    :return: Two points (numpy arrays) representing the extremes.
    """
    projections = np.dot(vertices, axis)
    min_index = np.argmin(projections)
    max_index = np.argmax(projections)

    return vertices[min_index], vertices[max_index]



# ============================================================
#  2D POLYGON OPERATIONS
# ============================================================

def order_polygon_vertices_2D(points):
    """
    Order vertices of a 2D polygon based on their angle relative to the centroid.
    
    :param points: A list of tuples representing the points.
    :return: Ordered list of points.
    """
    centroid = np.mean(points, axis=0)
    def sort_key(point):
        return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])
    return sorted(points, key=sort_key)


def shoelace_area(vertices):
    """
    Calculate the area of a polygon using the Shoelace formula.
    
    :param vertices: Ordered vertices of the polygon.
    :return: Area of the polygon.
    """
    n = len(vertices)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += vertices[i][0] * vertices[j][1]
        area -= vertices[j][0] * vertices[i][1]
    return abs(area) / 2

# ============================================================
#  3D SLICING / CROSS-SECTIONING
# ============================================================

def intersect_plane_with_triangle(plane_normal, plane_point, triangle, vertices):
    """
    Calculate intersection points of a plane with a triangle.
    
    :param plane_normal: Normal vector of the plane.
    :param plane_point: A point on the plane.
    :param triangle: Vertices of the triangle.
    :return: Intersection points as a list of tuples.
    """
    intersection_points = []
    for i in range(3):
        p1, p2 = vertices[triangle[i]], vertices[triangle[(i + 1) % 3]]
        edge_vec = p2 - p1
        if edge_vec.ndim > 1:
            edge_vec = edge_vec.reshape(-1)  # Ensure edge_vec is 1D
        if plane_normal.ndim > 1:
            plane_normal = plane_normal.reshape(-1)  # Ensure plane_normal is 1D

        dot_product = np.dot(edge_vec, plane_normal)

        if np.isclose(dot_product, 0):
            continue

        t = np.dot(plane_point - p1, plane_normal) / dot_product
        if 0 <= t <= 1:
            intersection_point = p1 + t * edge_vec
            intersection_points.append(tuple(intersection_point))
    return intersection_points


def calculate_shape_cross_section(vertices, faces, num_slices=None, height=None):
    """
    Calculate cross-section areas of a shape along its height.
    
    :param vertices: Vertices of the shape.
    :param height: Height of the shape.
    :param num_slices: Number of slices.
    :return: List of areas of each cross-section.
    """

    # Principal axis (tip to base center)
    principal_axis = find_principal_axis(vertices)

    if height == None and num_slices > 0:
        height = calculate_height_along_axis(vertices, principal_axis)
        start = 0
    
    elif num_slices == None and height > 0:
        num_slices = 0
        start = 1
    
    else:
        raise ValueError('You have to specify either num_slices or a specific height.')

    tip, base_center = find_extreme_points(vertices, principal_axis)

    cross_sections = []
    all_intersection_points = []

    for i in range(start, num_slices + 1):  # Include 0 and n in the loop
        slice_height = i * height / num_slices
        plane_point = tip + (slice_height / height) * (base_center - tip)
        intersection_points = set()

        for face in faces:
            for point in intersect_plane_with_triangle(principal_axis, plane_point, face, vertices):
                intersection_points.add(point)

        all_intersection_points.append(intersection_points)

        if len(intersection_points) == 0:
            cross_sections.append(0)  # No intersection or invalid shape
        else:
            intersection_points_2D = [(p[0], p[1]) for p in intersection_points]  # Assuming X and Y coordinates
            ordered_points_2D = order_polygon_vertices_2D(intersection_points_2D)
            area = shoelace_area(ordered_points_2D)
            cross_sections.append(area)

    return all_intersection_points, cross_sections, principal_axis, [tip, base_center]


# ============================================================
#  FILE I/O — Wavefront .obj and PDB formats
# ============================================================

def load_obj(filename):
    """
    Load a Wavefront .obj file and extract vertices and faces.

    :param filename: Path to the .obj file.
    :return: Tuple of numpy arrays (vertices, faces)
    """
    vertices = []
    faces = []

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                # Vertex definition
                _, x, y, z = line.split()
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith('f '):
                # Face definition
                face = [int(idx.split('/')[0]) - 1 for idx in line.split()[1:]]  # OBJ files are 1-indexed
                faces.append(face)

    return np.array(vertices), np.array(faces)


def write_vertices_to_pdb(vertices, extremes, output_file, rotate=False):
    """
    Write vertices to a PDB file, treating each vertex as an atom.
    Optionally rotate the vertices 180 degrees around the Y-axis.
    Special residue names are assigned to extreme points.

    :param vertices: Numpy array of vertices.
    :param extremes: Array of extreme points.
    :param output_file: Path to the output PDB file.
    :param rotate: Boolean, if True, rotate vertices 180 degrees around the Y-axis.
    """
    with open(output_file, 'w') as file:
        for i, vertex in enumerate(vertices, start=1):
            # Check if the vertex is one of the extremes
            residue_name = "EXT" if any(np.array_equal(vertex, ext) for ext in extremes) else "UNL"
            if rotate:
                # Rotate 180 degrees around the Y-axis
                vertex = [-vertex[0], vertex[1], -vertex[2]]

            file.write(
                f"ATOM  {i:5d}  Du  {residue_name} A{i:4d}    {vertex[0]:8.3f}{vertex[1]:8.3f}{vertex[2]:8.3f}  1.00 20.00\n"
            )



# ============================================================
#  VECTOR / PLANE UTILITIES
# ============================================================

def midpoint(p1, p2):
    """Return the midpoint between two points."""
    return (p1 + p2) / 2

def normalize(v):
    """Normalize a vector to unit length. Returns zero vector if norm is 0."""
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def square_vertices(point1, point2, side_length, fraction):
    """Compute the 4 vertices of a square cutting plane perpendicular to a line.

    The square is centered at the point that is *fraction* of the way from
    point1 to point2, oriented perpendicular to the line between them.

    Args:
        point1, point2: Endpoints of the axis line.
        side_length: Side length of the square (in Angstroms).
        fraction: Position along the line (0.0 = at point1, 1.0 = at point2).

    Returns:
        (4, 3) numpy array of vertex coordinates.
    """
    # Convert points to numpy arrays
    p1 = np.array(point1)
    p2 = np.array(point2)

    # Compute the line vector 
    line_vec = p2 - p1

    # Find the center of the square on the line
    center = p1 + fraction * line_vec

    # Step 2: Compute the normal vector
    normal_vector = normalize(p2 - p1)

    # Step 3: Find two perpendicular vectors in the plane
    # Here we find one arbitrary perpendicular vector and then use the cross product to find another.
    if (normal_vector == np.array([1, 0, 0])).all():
        # Special case to handle collinearity
        perp_vector1 = np.array([0, 1, 0])
    else:
        perp_vector1 = normalize(np.cross(normal_vector, np.array([1, 0, 0])))
    perp_vector2 = np.cross(normal_vector, perp_vector1)

    # Step 4: Scale the perpendicular vectors to the half side length
    half_side = side_length / 2
    perp_vector1 *= half_side
    perp_vector2 *= half_side

    # Step 5: Calculate the vertices
    vertices = np.array([
        center + perp_vector1 + perp_vector2,
        center - perp_vector1 + perp_vector2,
        center - perp_vector1 - perp_vector2,
        center + perp_vector1 - perp_vector2
    ])

    return vertices


# ============================================================
#  PLANE PROJECTION & CLUSTERING
# ============================================================

def find_points_near_plane(points, plane_points, distance_threshold=0.1):
    """Identify points that are close to the plane."""
    # Compute the plane normal (assuming points are coplanar)
    v1 = plane_points[1] - plane_points[0]
    v2 = plane_points[2] - plane_points[0]
    plane_normal = np.cross(v1, v2)
    
    norm = np.linalg.norm(plane_normal)
    if norm == 0 or np.isclose(norm, 0):
        # Handle the case where the plane normal cannot be computed
        raise ValueError("Unable to compute a valid plane normal.")

    plane_normal = plane_normal.astype(float) / norm
    
    # Compute the distance of each point from the plane
    distances = np.abs(np.dot((points - plane_points[0]), plane_normal))
    near_mask = np.abs(distances) <= distance_threshold
    

    near_indices = np.where(near_mask)[0]
    not_near_indices = np.where(~near_mask)[0]
    # Use boolean mask to filter points
    near_plane = points[near_mask]
    not_near_plane = points[~near_mask]

    return near_plane, near_indices, not_near_plane, not_near_indices


def cluster_points_with_dbscan(points, eps, min_samples):
    """
    Cluster points using DBSCAN and return the cluster labels.
    
    :param points: The input points to cluster.
    :param eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    :param min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    :return: Cluster labels for each point.
    """
    # Use only X and Y for clustering since the points are coplanar
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, :2])
    return db.labels_

def project_points_onto_plane(points, plane_points):
    """Project 3D points orthogonally onto a plane and return 2D coordinates.

    The plane is defined by the first 3 points in plane_points. Returns the
    projected points in a local 2D coordinate system (u, v) on the plane.

    Returns:
        (points_2d, indices, plane_origin, u_basis, v_basis)
    """
    # Calculate the plane's normal vector
    v1 = plane_points[1] - plane_points[0]
    v2 = plane_points[2] - plane_points[0]
    plane_normal = np.cross(v1, v2)
    plane_normal = plane_normal.astype(float)  # Ensure the vector is of float type
    plane_normal /= np.linalg.norm(plane_normal)

    # Compute D for the plane equation Ax + By + Cz + D = 0
    D = -np.dot(plane_normal, plane_points[0])

    # Project points orthogonally onto the plane
    projected_points = points - np.outer(np.dot(points, plane_normal) + D, plane_normal)

    # Define new basis vectors for the plane
    u = v1 / np.linalg.norm(v1)
    v = np.cross(plane_normal, u)

    # Transform the projected points to the 2D plane coordinates
    points_2d = np.array([[np.dot(p, u), np.dot(p, v)] for p in projected_points])

    return points_2d, np.arange(len(points)), plane_points[0], u, v


# ============================================================
#  AREA & INSCRIBED CIRCLE CALCULATIONS
# ============================================================

def calculate_area_of_points(points, grid_spacing, radius=False):
    """
    Calculate the area covered by points, assuming each point represents a circle.
    Also return the merged shape.

    :param points: Array of points.
    :param grid_spacing: The spacing of the grid, used to determine the radius of circles.
    :return: Tuple containing the total area covered by the points and the merged shape.
    """
    if not radius:
        radius = grid_spacing / 2
    circles = [Point(p).buffer(radius) for p in points]  # Create circles around each point
    merged_shape = unary_union(circles)  # Merge all circles into a single shape
    return merged_shape.area, merged_shape


def find_maximum_inscribed_circle(polygon):
    """
    Find the maximum inscribed circle within a given polygon.
    
    :param polygon: Shapely Polygon object.
    :return: Tuple containing the center and radius of the largest inscribed circle.
    """
    points = np.array(polygon.exterior.coords)
    vor = Voronoi(points)
    max_circle_center = None
    max_circle_radius = 0
    
    for region in vor.regions:
        if not -1 in region and len(region) > 0:
            polygon_region = Polygon([vor.vertices[i] for i in region])
            if polygon_region.intersects(polygon):
                intersected_region = polygon_region.intersection(polygon)
                if not intersected_region.is_empty and intersected_region.area > 0:
                    interior_points = np.array(intersected_region.exterior.coords)
                    for point in interior_points:
                        distance = polygon.exterior.distance(Point(point))
                        if distance > max_circle_radius:
                            max_circle_radius = distance
                            max_circle_center = point
                            
    return max_circle_center, max_circle_radius


def find_closest_point_index(points, target_point):
    """
    Find the index of the point in 'points' that is closest to 'target_point'.
    
    :param points: Array of points.
    :param target_point: The target point to find the closest point to.
    :return: Index of the closest point.
    """
    distances = np.linalg.norm(points - target_point, axis=1)
    return np.argmin(distances)



# ============================================================
#  DIAMETER ANALYSIS PIPELINE
# ============================================================

def calculate_diameter_analysis(tnl_name, slice_heights, ball_spacing, ext_center, ext_outer, only_area=False):
    """
    Calculate the maximum inscribed circles and areas for a given tunnel and multiple slice heights.

    :param tnl_name: Name of the tunnel.
    :param slice_heights: List of slice heights to analyze.
    :param ball_spacing: Spacing for the balls in the analysis.
    :param ext_center: External center from make_axis.
    :param ext_outer: External outer from make_axis.
    :return: List of tuples containing (height, descriptor, max_circle_center, max_circle_radius, area).
    """
    all_max_circles = []

    try:
        # Get tunnel points
        tnl_points_pos = np.array(PosAtom(f'Obj {tnl_name}', coordsys='global')).reshape(-1, 3)
        tnl_points = np.array(ListAtom(f'Obj {tnl_name}'))
        NameAtom(tnl_points, 'UNL')
        SegAtom(tnl_points, '.')
        StickAtom(tnl_points)

        threshold = ball_spacing / 800
        point1 = PosAtom(ext_outer, coordsys='global')
        point2 = PosAtom(ext_center, coordsys='global')
        side_length = CUTTING_PLANE_SIDE_LENGTH
        max_label = 0
        init = True
        for i, height in enumerate(slice_heights):
            try:
                # Adjust slice position based on height
                position = height / 100.0
                vertices = square_vertices(point1, point2, side_length, position)
                
                plane_points_pos = np.array(vertices).reshape(-1, 3)
                
                near_plane, near_indices, not_near_plane, not_near_indices = find_points_near_plane(tnl_points_pos, plane_points_pos, distance_threshold=threshold)
                if near_indices.size > 0:
                    
                    near_plane_points_projected, original_indices, plane_origin, u, v = project_points_onto_plane(near_plane, plane_points_pos)

                    if len(near_plane_points_projected) > 0:
                        eps = 1.4
                        min_samples = 1

                        cluster_labels = cluster_points_with_dbscan(near_plane_points_projected, eps, min_samples)

                        old_max_label = max_label
                        max_label = max(np.append(cluster_labels,max_label))
                        unique_labels = set(cluster_labels)
                        
                        clusters = {}
                        cluster_indices = {}
                        cluster_new_label = {}

                        SegAtom(f'atom {" ".join([str(x) for x in tnl_points[near_indices]])} and atom !X?', 'cutp')
                        
                        for label in unique_labels:
                            if label != -1:
                                cluster_mask = (cluster_labels == label)
                                label = tuple(label) if isinstance(label, np.ndarray) else label
                                clusters[label] = near_plane_points_projected[cluster_mask]
                                cluster_indices[label] = near_indices[original_indices[cluster_mask]] 
                                cluster_new_label[label] = label  + 1 + old_max_label

                        clusters_revdict = {v: k for k, v in clusters.items()}

                        if init == True:
                            for org_label, cluster_points in clusters.items():
                                SegAtom(tnl_points[cluster_indices[org_label]], org_label)
                                NameAtom(tnl_points[cluster_indices[org_label]], f'X{i}')
                            init = False
                        else:
                            for org_label, cluster_points in clusters.items():
                                closest_seg = SegAtom(f'X? with minimum distance from atom {" ".join([str(x) for x in tnl_points[cluster_indices[org_label]]])}')[0]
                            StopPlugin()

                        for org_label, cluster_points in clusters.items():
                            area, merged_shape = calculate_area_of_points(cluster_points, ball_spacing * 2, radius=0.75)

                            if not only_area:
                                try:
                                    max_circle_center, max_circle_radius = find_maximum_inscribed_circle(merged_shape)

                                    if max_circle_center is not None:
                                        closest_point_index = find_closest_point_index(near_plane_points_projected, max_circle_center)
                                        original_point_index = original_indices[closest_point_index]
                                        correct_index = near_indices[original_point_index]
                                        descriptor = int(tnl_points[correct_index])

                                        all_max_circles.append((height, descriptor, max_circle_center, max_circle_radius, area))

                                except AttributeError as e:
                                    print(f'Warning: cluster {org_label} gave AttributeError. {e}')
                                    continue
                            else:
                                all_max_circles.append((height, None, None, None, area))
                        
                    else:
                        all_max_circles.append((height, None, None, 0, 0))
                else:
                    all_max_circles.append((height, None, None, 0, 0))

            except Exception as e:
                print(f"Error processing slice at height {height}: {e}")
                StopPlugin()

    except Exception as e:
        print(f"Error in calculate_diameter_analysis: {e}")
    
    return all_max_circles, only_area


# ============================================================
#  PATHFINDING (A*)
# ============================================================

from heapq import heappop, heappush

def get_neighbors(point, point_index_map, ball_spacing, tolerance=1e-6):
    """Find all points within ball_spacing distance of *point* (brute-force search)."""
    neighbors = []
    for key in point_index_map:
        if np.linalg.norm(np.array(point) - np.array(key)) <= ball_spacing + tolerance:
            neighbors.append(key)
    return neighbors



def heuristic(a, b):
    """Euclidean distance heuristic for A*."""
    return np.linalg.norm(np.array(a) - np.array(b))


def astar(start, end, points, point_index_map, spacing):
    """A* pathfinding through a 3D point cloud.

    Finds the shortest path from *start* to *end* by traversing neighboring
    points (within *spacing* distance). Uses Euclidean distance as the heuristic.

    Returns a list of point tuples forming the path, or None if no path exists.
    """
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {point: float('inf') for point in points}
    g_score[start] = 0
    f_score = {point: float('inf') for point in points}
    f_score[start] = heuristic(start, end)

    while open_set:
        current_f_score, current = heappop(open_set)

        if current == end:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        neighbors = get_neighbors(current, point_index_map, spacing)

        for neighbor in neighbors:
            tentative_g_score = g_score[current] + 1
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, end)
                heappush(open_set, (f_score[neighbor], neighbor))
    
    return None  # No path found
    


def find_shortest_path(tnl_name, min_spacing, max_spacing, start_name, end_name, coarse_only):
    """Find the shortest path through a tunnel's point cloud between two named atoms.

    First tries with min_spacing (fine grid). If no path is found and coarse_only
    is False, retries with max_spacing (coarser, more connected grid).

    Returns a list of atom numbers along the path, or None.
    """
    tnl_points_pos = np.array(PosAtom(f'Obj {tnl_name}', coordsys='global')).reshape(-1, 3)
    tnl_points = np.array(ListAtom(f'Obj {tnl_name}'))

    # Create a point to index map for fast lookups
    point_index_map = {tuple(point): idx for idx, point in enumerate(tnl_points_pos)}
    
    # Create a name to index map
    name_index_map = {name: idx for idx, name in enumerate(tnl_points)}

    # Get the indices of the start and end points based on their names
    try:
        start_idx = name_index_map[start_name]
        end_idx = name_index_map[end_name]
    except KeyError as e:
        print(f"Error: Name '{e.args[0]}' not found in tnl_points")
        return None

    start_point = tuple(tnl_points_pos[start_idx])
    end_point = tuple(tnl_points_pos[end_idx])

    # Ensure BallAtom receives a list
    BallAtom([tnl_points[start_idx]])
    BallAtom([tnl_points[end_idx]])

    # Find the shortest path using A*
    if coarse_only:
        path = astar(start_point, end_point, point_index_map.keys(), point_index_map, max_spacing)
        if not path:
            ShowMessage(f'No path found despite coarse ball_spacing ({max_spacing})!')
            Wait('continuebutton')
    else:
        path = astar(start_point, end_point, point_index_map.keys(), point_index_map, min_spacing)
        if not path:
            ShowMessage('No path found with ball_spacing, extending grid search with connect cutoff (=coarse)')
            Wait(1)
            path = astar(start_point, end_point, point_index_map.keys(), point_index_map, max_spacing)

    if path:
        # Convert path of points back to names using indices
        path_names = [tnl_points[point_index_map[tuple(point)]] for point in path]
        return path_names
    
    HideMessage()
    return None

