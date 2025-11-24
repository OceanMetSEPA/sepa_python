import numpy as np

def calculate_concentration(mapped_dict, mesh_dict, zCutoff=np.inf, surface=False, scaleFactor=1e9):
    """
    Calculate surface or volumetric concentration from mapped particle dict.

    Parameters
    ----------
    mapped_dict : dict
        Particle tracking dictionary containing at least:
        'meshIndex', 'ParticleActive', 'dateTime', optionally 'depthBelowSurface'
    mesh_dict : dict
        Mesh dictionary containing:
        'meshIndices' (nCells x 3), 'cellArea' (nCells,), 'zMesh' (Nv,)
    zCutoff : float
        Maximum depth to include in surface concentration or volumetric fraction
    surface : bool
        If True, compute surface concentration (particles above zCutoff)
        If False, compute volumetric concentration
    scaleFactor : float
        Multiplicative factor to convert units (default 1e9: kg → µg)

    Returns
    -------
    conc : ndarray
        Concentration array [nCells x nTimeSteps]
    """

    nCells = mesh_dict['meshIndices'].shape[0]
    nTime = mapped_dict['dateTime'].shape[0]

    conc = np.zeros((nCells, nTime))

    # Handle particle indices
    meshIndex_raw = mapped_dict['meshIndex']
    valid_mask = ~np.isnan(meshIndex_raw)
    meshIndex = np.full_like(meshIndex_raw, -1, dtype=int)
    meshIndex[valid_mask] = meshIndex_raw[valid_mask].astype(int) - 1  # MATLAB 1-based → Python 0-based

    pa = mapped_dict['ParticleActive']

    if surface:
        # Require depthBelowSurface
        if 'depthBelowSurface' not in mapped_dict:
            raise ValueError("depthBelowSurface field required for surface=True")
        depth = mapped_dict['depthBelowSurface']

    else:
        # compute cell depths if volumetric
        if 'z' in mesh_dict:
            z_cell = mesh_dict['z'].ravel()
        else:
            tri = mesh_dict['meshIndices'] - 1  # Python 0-based
            z_mesh = mesh_dict['zMesh'].ravel()
            z_tri = z_mesh[tri]                  # (nCells x 3)
            z_cell = np.mean(z_tri, axis=1)     # (nCells,)
        z_cell_safe = z_cell.copy()
        z_cell_safe[z_cell_safe == 0] = np.nan

    cellArea = mesh_dict['cellArea'].ravel()

    for t_idx in range(nTime):
        if t_idx % 100 == 0:
            print(f"Timestep {t_idx+1}/{nTime}")

        mi_t = meshIndex[:, t_idx]
        pa_t = pa[:, t_idx]

        if surface:
            depth_t = depth[:, t_idx]
            valid = (mi_t >= 0) & (mi_t < nCells) & (pa_t > 0) & (depth_t <= zCutoff)
        else:
            valid = (mi_t >= 0) & (mi_t < nCells) & (pa_t > 0)

        mi_valid = mi_t[valid]
        pa_valid = pa_t[valid]

        for m in np.unique(mi_valid):
            inds = mi_valid == m
            conc[m, t_idx] = np.sum(pa_valid[inds])

    # Scale and divide
    if surface:
        conc /= cellArea[:, None]
    else:
        conc /= (cellArea * z_cell_safe)[:, None]

    conc *= scaleFactor

    return conc
