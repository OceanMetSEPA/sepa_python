import numpy as np

def calculate_surface_concentration(mapped_dict, cellArea, zCutoff=2,scaleFactor=1e9):
    """
    Calculate surface concentration (µg/m²) from particle tracking struct.

    Parameters
    ----------
    mapped_dict : dict
        Dictionary containing particle tracking data
    cellArea : array_like
        Area of each mesh cell in m²
    zCutoff : float
        Maximum depth (m) to include in surface concentration
    scaleFactor : float
        Factor to multiply resulting concentration (default 1e9: kg → µg)

    Returns
    -------
    concPerM2 : ndarray
        Surface concentration matrix [numCells x numTimeSteps] in µg/m²
    """

    numCells = len(cellArea)
    numTimeSteps = mapped_dict['dateTime'].shape[0]
    concPerM2 = np.zeros((numCells, numTimeSteps))

    # Mask invalid indices before converting to int to avoid warnings
    meshIndex_raw = mapped_dict['meshIndex']
    valid_mask = ~np.isnan(meshIndex_raw)
    meshIndex = np.full_like(meshIndex_raw, -1, dtype=int)  # default invalid index
    meshIndex[valid_mask] = meshIndex_raw[valid_mask].astype(int) - 1  # MATLAB 1-based -> Python 0-based

    particleActive = mapped_dict['ParticleActive']
    depthBelowSurface = mapped_dict['depthBelowSurface']

    for t_idx in range(numTimeSteps):
        if t_idx % 100 == 0:
            print(f"Processing timestep {t_idx+1}/{numTimeSteps}")

        mesh_t = meshIndex[:, t_idx]
        active_t = particleActive[:, t_idx]
        depth_t = depthBelowSurface[:, t_idx]

        # Valid particles for this timestep
        valid = (mesh_t >= 0) & (mesh_t < numCells) & (~np.isnan(active_t)) & (depth_t <= zCutoff)
        mesh_valid = mesh_t[valid]
        active_valid = active_t[valid]

        # Accumulate per cell
        for m in np.unique(mesh_valid):
            inds = mesh_valid == m
            concPerM2[m, t_idx] = active_valid[inds].sum() / cellArea[m]

    # Apply scale factor
    concPerM2 *= scaleFactor

    return concPerM2
