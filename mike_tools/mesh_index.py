import numpy as np
import matplotlib.tri as mtri
from scipy.spatial import cKDTree
from functools import lru_cache
from collections import defaultdict

MIKE_NULL = 1.00000001800251e-35

# Simple cache for mesh helpers keyed by id(mesh_dict)
_mesh_cache = {}

def _build_mesh_helpers(mesh_dict):
    """
    Build and cache triangulation, trifinder, kdtree, and node->tris adjacency.
    """
    key = id(mesh_dict)
    if key in _mesh_cache:
        return _mesh_cache[key]

    xMesh = np.asarray(mesh_dict["xMesh"], dtype=float).ravel()
    yMesh = np.asarray(mesh_dict["yMesh"], dtype=float).ravel()
    faces = np.asarray(mesh_dict["meshIndices"], dtype=int)

    # If meshIndices appear MATLAB 1-based, convert to 0-based internally
    if faces.min() >= 1:
        faces = faces - 1

    # Clip to valid nodes (defensive)
    Nv = len(xMesh)
    faces = np.clip(faces, 0, Nv - 1)

    triang = mtri.Triangulation(xMesh, yMesh, faces)
    trifinder = triang.get_trifinder()
    vertices = np.column_stack([xMesh, yMesh])
    tree = cKDTree(vertices)

    # Build node -> triangle adjacency (list of triangle indices per node)
    node_to_tris = defaultdict(list)
    for tri_idx, verts in enumerate(faces):
        for v in verts:
            node_to_tris[int(v)].append(int(tri_idx))

    # Convert defaultdict lists to numpy arrays for faster indexing
    node_to_tris_arr = {k: np.array(v, dtype=int) for k, v in node_to_tris.items()}

    helpers = {
        "xMesh": xMesh,
        "yMesh": yMesh,
        "faces": faces,
        "triang": triang,
        "trifinder": trifinder,
        "tree": tree,
        "node_to_tris": node_to_tris_arr
    }
    _mesh_cache[key] = helpers
    return helpers


def _barycentric_coords_vector(px, py, xverts, yverts):
    """
    Vectorized barycentric coords.
    xverts, yverts are shape (n_tri, 3)
    px, py are shape (n_pts,)
    Returns l1,l2,l3 arrays of shape (n_tri, n_pts) OR if called with single triangle,
    returns shape (n_pts,).
    """
    # Ensure shapes:
    x1 = xverts[:, 0][:, None]
    x2 = xverts[:, 1][:, None]
    x3 = xverts[:, 2][:, None]
    y1 = yverts[:, 0][:, None]
    y2 = yverts[:, 1][:, None]
    y3 = yverts[:, 2][:, None]

    detT = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)  # shape (n_tri,1)

    # px/py -> (1, n_pts)
    px_row = px[None, :]
    py_row = py[None, :]

    l1 = ((y2 - y3) * (px_row - x3) + (x3 - x2) * (py_row - y3)) / detT
    l2 = ((y3 - y1) * (px_row - x3) + (x1 - x3) * (py_row - y3)) / detT
    l3 = 1.0 - l1 - l2
    return l1, l2, l3  # each shape (n_tri, n_pts)


def mesh_index(
    xp, yp, mesh_dict,
    tol=1e-12, fallback=True, matlab_indexing=True,
    trifinder=None, tree=None,
    trifinder_chunk=200_000
):
    """
    Fast mesh index lookup.

    Parameters
    ----------
    xp, yp : array-like
        Coordinates. Can be any shape; result matches shape.
    mesh_dict : dict
        Must contain 'xMesh', 'yMesh', 'meshIndices'.
    tol : float
        tolerance for barycentric tests.
    fallback : bool
        whether to run KD-tree fallback for outside points.
    matlab_indexing : bool
        if True, return 1-based triangle indices.
    trifinder : optional
        prebuilt trifinder to use (overrides cached).
    tree : optional
        prebuilt KD-tree to use (overrides cached).
    trifinder_chunk : int
        chunk size for batch trifinder calls to limit memory/overhead.

    Returns
    -------
    elem_idx : ndarray
        Triangle indices (NaN outside mesh). Shape matches xp.
    """

    # --- Prepare input arrays and remember original shape ---
    xp_arr = np.asarray(xp, dtype=float)
    yp_arr = np.asarray(yp, dtype=float)
    orig_shape = xp_arr.shape
    xp_flat = xp_arr.ravel()
    yp_flat = yp_arr.ravel()
    Np = xp_flat.size

    # --- Replace NaNs/Infs with MIKE_NULL for trifinder (so KD-tree query won't crash) ---
    finite_mask = np.isfinite(xp_flat) & np.isfinite(yp_flat)
    xp_fixed = xp_flat.copy()
    yp_fixed = yp_flat.copy()
    if not np.all(finite_mask):
        xp_fixed[~finite_mask] = MIKE_NULL
        yp_fixed[~finite_mask] = MIKE_NULL

    # --- Build or fetch mesh helpers ---
    helpers = _build_mesh_helpers(mesh_dict)
    xMesh = helpers["xMesh"]
    yMesh = helpers["yMesh"]
    faces = helpers["faces"]
    triang = helpers["triang"]
    trifinder_cached = helpers["trifinder"]
    tree_cached = helpers["tree"]
    node_to_tris = helpers["node_to_tris"]

    # If user provided trifinder/tree, prefer them
    trifinder_use = trifinder if trifinder is not None else trifinder_cached
    tree_use = tree if tree is not None else tree_cached

    # --- Fast trifinder lookup in chunks ---
    elem_idx = np.full(Np, np.nan, dtype=float)
    start = 0
    while start < Np:
        stop = min(start + trifinder_chunk, Np)
        tri_res = trifinder_use(xp_fixed[start:stop], yp_fixed[start:stop]).astype(float)
        tri_res[tri_res < 0] = np.nan
        elem_idx[start:stop] = tri_res
        start = stop

    # --- If fallback enabled, handle points where tri_res is NaN but the original coords were finite ---
    if fallback:
        need_fix = np.isnan(elem_idx) & finite_mask
        if np.any(need_fix):
            # Query nearest nodes for those points
            pts = np.column_stack([xp_fixed[need_fix], yp_fixed[need_fix]])
            _, nearest_nodes = tree_use.query(pts, k=1)

            # Map from node -> indices in elem_idx that correspond to that node
            idx_positions = np.where(need_fix)[0]
            # group indices by nearest node for vectorised processing
            node_to_point_indices = {}
            for pos_idx, node in zip(idx_positions, nearest_nodes):
                node_to_point_indices.setdefault(int(node), []).append(pos_idx)

            # For each node, do candidate triangle tests vectorised over all points assigned to that node
            for node, point_positions in node_to_point_indices.items():
                cand_tris = node_to_tris.get(node)
                if cand_tris is None or cand_tris.size == 0:
                    continue
                # candidate triangles (n_tri_cand,)
                cand_tris = np.asarray(cand_tris, dtype=int)
                # build xverts,yverts arrays (n_tri_cand, 3)
                tri_verts = faces[cand_tris]  # shape (n_tri_cand,3)
                xverts = xMesh[tri_verts]     # shape (n_tri_cand,3)
                yverts = yMesh[tri_verts]     # shape (n_tri_cand,3)

                # Points assigned to this node: gather px,py arrays
                pts_idx_local = np.array(point_positions, dtype=int)
                px = xp_fixed[pts_idx_local]
                py = yp_fixed[pts_idx_local]
                # compute barycentric coords: returns (n_tri_cand, n_pts_local)
                l1, l2, l3 = _barycentric_coords_vector(px, py, xverts, yverts)

                # For each point (column), find a triangle where all li >= -tol
                # l1/l2/l3 shape: (n_tri_cand, n_pts_local)
                inside_mask = (l1 >= -tol) & (l2 >= -tol) & (l3 >= -tol)
                # For points, find first triangle that contains them
                # idx_of_first is index into cand_tris array
                # If no triangle found, argmax will return 0; we need to check mask
                first_idx = np.argmax(inside_mask, axis=0)  # shape (n_pts_local,)
                # check if any triangle actually matched per point
                any_match = inside_mask[first_idx, np.arange(inside_mask.shape[1])]
                # Set elem_idx accordingly (tri index is cand_tris[first_idx])
                matched_points = np.where(any_match)[0]
                if matched_points.size > 0:
                    pts_global_positions = pts_idx_local[matched_points]
                    tri_selected = cand_tris[first_idx[matched_points]]
                    elem_idx[pts_global_positions] = tri_selected.astype(float)
                # points that didn't match remain NaN

    # --- Restore NaN for original non-finite inputs (they were given MIKE_NULL for searching) ---
    elem_idx[~finite_mask] = np.nan

    # --- Apply MATLAB indexing if requested ---
    if matlab_indexing:
        mask = np.isfinite(elem_idx)
        elem_idx[mask] = elem_idx[mask] + 1.0

    return elem_idx.reshape(orig_shape)
