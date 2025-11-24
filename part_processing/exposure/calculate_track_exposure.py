import numpy as np
from scipy import sparse

def calculate_track_exposure(conc_data, fish_legs, offset=0, verbose=False):
    """
    Compute exposure per fish leg and offset.

    Supports conc_data as:
      - a single dense or sparse array, OR
      - a dict of arrays (returns dict of exposures)

    Parameters
    ----------
    conc_data : np.ndarray, scipy.sparse matrix, or dict of arrays
        Concentration data [Ncells x NtimeSteps].
    fish_legs : dict
        Must contain keys 'CellIndex', 'timeStep', 'legDuration'.
    offset : int or array-like, optional
        MATLAB-style offset(s). Relative shift applied to timeStep.
    verbose : bool, optional
        If True, prints one line per leg per offset.

    Returns
    -------
    exposure_arr : np.ndarray or dict of arrays
        Exposure per leg and offset.
    """

    # --- Handle dictionary input ---
    if isinstance(conc_data, dict):
        result = {}
        nSites = len(conc_data)
        for i, (site, data) in enumerate(conc_data.items(), start=1):
            print(f"Processing {site} ({i}/{nSites})...")
            exposure = calculate_track_exposure(data, fish_legs, offset=offset, verbose=verbose)
            result[site] = exposure
        return result

    # --- Flatten inputs ---
    cell_array = np.atleast_1d(fish_legs["CellIndex"]).flatten().astype(int)
    time_array = np.atleast_1d(fish_legs["timeStep"]).flatten().astype(int)
    legDur_array = np.atleast_1d(fish_legs["legDuration"]).flatten().astype(float)

    offsets = np.atleast_1d(offset).astype(int)
    nOffsets = len(offsets)
    nLegs = len(cell_array)

    if len(time_array) != nLegs or len(legDur_array) != nLegs:
        raise ValueError("fish_legs fields must have the same number of elements.")

    nRows, nCols = conc_data.shape

    i_idx = np.tile(cell_array - 1, (nOffsets, 1)).T
    j_idx = np.tile(time_array - 1, (nOffsets, 1)).T + offsets

    i_idx = np.clip(i_idx, 0, nRows - 1)
    j_idx = np.clip(j_idx, 0, nCols - 1)

    conc_mat = np.zeros((nLegs, nOffsets), dtype=float)

    # --- Sparse matrix lookup ---
    if sparse.issparse(conc_data):
        coo = conc_data.tocoo()
        key_vals = coo.row.astype(np.int64) * np.int64(nCols) + coo.col.astype(np.int64)
        order = np.argsort(key_vals)
        key_sorted = key_vals[order]
        vals_sorted = coo.data[order]

        req_keys = (i_idx.astype(np.int64) * np.int64(nCols) + j_idx.astype(np.int64)).ravel()
        pos = np.searchsorted(key_sorted, req_keys)

        # Initialize result
        found = np.zeros(req_keys.shape, dtype=float)

        # Only consider valid positions within bounds
        in_bounds = pos < key_sorted.size
        if np.any(in_bounds):
            pos_valid = pos[in_bounds]
            req_valid = req_keys[in_bounds]
            match = key_sorted[pos_valid] == req_valid
            found[in_bounds] = match * vals_sorted[pos_valid]  # 0 where no match

        conc_mat = found.reshape(i_idx.shape)

    else:
        # Dense array lookup
        i_flat = i_idx.ravel()
        j_flat = j_idx.ravel()
        conc_flat = conc_data[i_flat, j_flat]
        conc_mat = np.array(conc_flat).reshape(i_idx.shape)

    # --- Exposure computation ---
    legDur_col = legDur_array.reshape(-1, 1)
    if legDur_col.shape[0] != conc_mat.shape[0]:
        legDur_col = np.broadcast_to(legDur_col, conc_mat.shape)

    spd = 24.0 * 60.0 * 60.0
    exposure_arr = conc_mat * (legDur_col / spd)

    # --- Verbose printing ---
    if verbose:
        header = ("Idx | CellIndex | timeStep(MATLAB) | timeStep+offset(MATLAB) | "
                  "Python_col | Concentration | legDuration(s) | Exposure")
        print("------ DEBUG: Exposure per Fish Leg ------")
        print(header)
        for r in range(nLegs):
            for c in range(nOffsets):
                cell_m = int(cell_array[r])
                time_m = int(time_array[r])
                off = int(offsets[c])
                col_m = time_m + off
                py_col = int(j_idx[r, c])
                conc_val = conc_mat[r, c]
                dur = legDur_array[r]
                exp_val = exposure_arr[r, c]
                print(f"{r:3d} | {cell_m:8d} | {time_m:17d} | {col_m:21d} | "
                      f"{py_col:10d} | {conc_val:13.6g} | {dur:12.3f} | {exp_val:12.6g}")
        print("-----------------------------------------")

    # --- Output shape ---
    if nOffsets == 1:
        return exposure_arr.ravel()
    return exposure_arr
