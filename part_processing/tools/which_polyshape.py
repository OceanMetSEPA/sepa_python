import numpy as np
from shapely.geometry import Point, Polygon
from shapely.errors import ShapelyError

def which_polyshape(x, y, polyStructs, closest=False):
    """
    Identify which polygon(s) contain or touch a point (x, y),
    or return the closest polygon if requested.

    Notes
    -----
    - `polyStructs` should contain polygons with fields:
        'Longitude' and 'Latitude' (lat/lon coordinates)
      NOT 'X' and 'Y' from shapefiles or projected coordinates.
    - Polygons may contain NaNs separating disjoint parts (islands/holes).
    - If a point lies exactly on the boundary, a small tolerance buffer is applied.

    Parameters
    ----------
    x, y : float or array-like
        Point(s) to test.
    polyStructs : list or NumPy array of dict/struct
        Each polygon should have 'Longitude' and 'Latitude' arrays.
    closest : bool
        If True, return the index of the nearest polygon when
        the point is not inside/touching any polygon.

    Returns
    -------
    list of int/float
        Indices of polygons containing/touching each point, or [np.nan] if not found.
    """

    # --- Flatten MATLAB-style struct arrays into list of dicts ---
    def _flatten_structs(ps):
        out = []
        if isinstance(ps, np.ndarray):
            for el in ps.flat:
                if isinstance(el, dict):
                    out.append(el)
                elif isinstance(el, np.void):  # structured array
                    out.append({k: el[k] for k in el.dtype.names})
                elif isinstance(el, np.ndarray) and el.dtype.names:  # nested struct array
                    out.append({k: el[k][0] for k in el.dtype.names})
        elif isinstance(ps, list):
            out.extend(ps)
        elif isinstance(ps, dict):
            out.append(ps)
        return out

    polys_list = _flatten_structs(polyStructs)

    x = np.atleast_1d(np.asarray(x, float))
    y = np.atleast_1d(np.asarray(y, float))
    pts = [Point(xi, yi) for xi, yi in zip(x, y)]

    polygons = []
#    print("=== Building polygons (lat/lon, tolerant boundary) ===")

    for idx, ps in enumerate(polys_list):
        try:
            xv = np.ravel(np.asarray(ps.get("Longitude", []), dtype=float))
            yv = np.ravel(np.asarray(ps.get("Latitude", []), dtype=float))
            if xv.size == 0 or yv.size == 0:
                continue

            # Mask for finite coordinates
            mask = np.isfinite(xv) & np.isfinite(yv)
            if not np.any(mask):
                continue

            # Split by NaNs into contiguous segments (islands, holes)
            splits = np.where(~mask)[0]
            segments = []
            start = 0
            for s in np.append(splits, len(xv)):
                if np.any(mask[start:s]):
                    xi = xv[start:s][np.isfinite(xv[start:s])]
                    yi = yv[start:s][np.isfinite(yv[start:s])]
                    if len(xi) > 2:
                        segments.append(np.column_stack((xi, yi)))
                start = s + 1
            if not segments:
                continue

            # Keep the largest segment
            largest = max(segments, key=len)
            # Ensure polygon is explicitly closed
            if not np.allclose(largest[0], largest[-1], equal_nan=True):
                largest = np.vstack([largest, largest[0]])

            poly = Polygon(largest)

            if poly.is_valid and not poly.is_empty:
                polygons.append((idx, poly))

        except ShapelyError as e:
            print(f"⚠️ Skipping poly {idx}: {e}")
        except Exception as e:
            print(f"⚠️ Unexpected error building poly {idx}: {e}")

#    print(f"Built {len(polygons)} polygons")

    # --- Evaluate each point ---
    results = []
    buffer_size = 1e-5  # small tolerance in degrees (~1m)
    for pt in pts:
        found = None
        for idx, poly in polygons:
            try:
                if poly.contains(pt) or poly.touches(pt) or pt.buffer(buffer_size).intersects(poly):
                    found = idx
                    break
            except Exception:
                continue

        # Fallback to closest polygon
        if found is None and closest and polygons:
            dists = np.array([poly.distance(pt) for _, poly in polygons])
            nearest_idx = int(np.argmin(dists))
            found = polygons[nearest_idx][0]
#            print(f"Closest polygon {found} at distance {dists[nearest_idx]:.6f}")

        results.append(found if found is not None else np.nan)

    return results
