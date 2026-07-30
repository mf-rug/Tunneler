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
#  VECTOR / PLANE UTILITIES
# ============================================================

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

    # Step 3: Find two perpendicular vectors in the plane.
    # Cross the normal with whichever cardinal axis it is *least* aligned with,
    # so the cross product is never near-zero. The old code only guarded the
    # +X case, so an axis pointing along -X (or ±Y/±Z) produced a zero-length
    # perp_vector1 and a degenerate, zero-area cutting square.
    ref = np.eye(3)[np.argmin(np.abs(normal_vector))]
    perp_vector1 = normalize(np.cross(normal_vector, ref))
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

    # Transform the projected points to the 2D plane coordinates (B6: exact
    # vectorization of the old per-point [[p·u, p·v] for p ...] list comprehension).
    points_2d = np.column_stack((projected_points @ u, projected_points @ v))

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

    Accepts either a Shapely Polygon or a MultiPolygon. A merged cross-section
    (``unary_union`` of point-buffers) is a MultiPolygon whenever it splits into
    disconnected lobes; the old code assumed a single Polygon and crashed on
    ``.exterior`` for those slices. For a MultiPolygon we evaluate every
    component and return the largest inscribed circle found across all of them.
    The single-Polygon result is unchanged.

    :param polygon: Shapely Polygon or MultiPolygon object.
    :return: Tuple containing the center and radius of the largest inscribed circle.
    """
    if isinstance(polygon, MultiPolygon):
        parts = [g for g in polygon.geoms if not g.is_empty and g.area > 0]
    else:
        parts = [polygon]

    best_center = None
    best_radius = 0
    for part in parts:
        center, radius = _max_inscribed_circle_single(part)
        if radius > best_radius:
            best_radius = radius
            best_center = center
    return best_center, best_radius


def _max_inscribed_circle_single(polygon):
    """Maximum inscribed circle of a single Shapely Polygon (Voronoi method).

    This is the original ``find_maximum_inscribed_circle`` body, unchanged, so
    the numeric result for a plain Polygon is identical to before.
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
#  PATHFINDING (A*)
# ============================================================

from heapq import heappop, heappush
from scipy.spatial import cKDTree


def heuristic(a, b):
    """Euclidean distance heuristic for A*."""
    return np.linalg.norm(np.array(a) - np.array(b))


def astar(start, end, points, point_index_map, spacing):
    """A* pathfinding through a 3D point cloud.

    Finds the shortest path from *start* to *end* by traversing neighboring
    points (within *spacing* distance). Uses Euclidean distance as the heuristic.

    Neighbor lookup uses a KD-tree (scipy.cKDTree) rather than an O(N) brute-force
    scan per node, turning the search from O(N^2) into ~O(N log N) on large clouds.
    The result is identical to the brute-force version: the same neighbor set per
    node (radius = spacing + 1e-6, matching the old tolerance), the same hop-count
    step cost, and the same (f_score, point) heap ordering fully determine the path
    regardless of the order neighbors are visited.

    Returns a list of point tuples forming the path, or None if no path exists.
    """
    pts = list(points)
    coords = np.asarray(pts, dtype=float)
    tree = cKDTree(coords)
    radius = spacing + 1e-6   # matches the old brute-force tolerance

    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {point: float('inf') for point in pts}
    g_score[start] = 0
    f_score = {point: float('inf') for point in pts}
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

        for nb in tree.query_ball_point(np.asarray(current, dtype=float), radius):
            neighbor = pts[nb]
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

