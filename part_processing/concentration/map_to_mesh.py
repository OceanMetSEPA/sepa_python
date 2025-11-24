import numpy as np
from concurrent.futures import ThreadPoolExecutor
from mike_tools.mesh_index import mesh_index


def map_particle_tracks_to_mesh(particle_dict, mesh_dict, matlab_indexing=True):
    """
    Map particle tracks to unstructured mesh & compute depthBelowSurface and waterSurface.

    Output dict = input dict + new fields:
        meshIndex
        waterSurface  (if zCoordinate available)
        depthBelowSurface (if z available)
    """

    x = np.asarray(particle_dict["x"], float)
    y = np.asarray(particle_dict["y"], float)

    has_z = "z" in particle_dict
    z = np.asarray(particle_dict["z"], float) if has_z else None

    xMesh = np.asarray(mesh_dict["xMesh"], float).ravel()
    yMesh = np.asarray(mesh_dict["yMesh"], float).ravel()
    meshIndices = np.asarray(mesh_dict["meshIndices"], int)

    has_zcoord = "zCoordinate" in mesh_dict
    zMesh = np.asarray(mesh_dict["zCoordinate"], float) if has_zcoord else None

    Np, Nt = x.shape

    # Prepare output arrays
    meshIndex = np.full((Np, Nt), np.nan, float)
    waterSurface = np.full((Np, Nt), np.nan, float) if has_zcoord else None
    depthBelowSurface = np.full((Np, Nt), np.nan, float) if (has_zcoord and has_z) else None

    do_depth = has_zcoord and has_z
    print(f"Mapping {Np} particles over {Nt} timesteps…")
    if do_depth:
        print("Mesh has zCoordinate → computing depth and waterSurface.")
    elif has_zcoord:
        print("Mesh has zCoordinate but particle_dict has no z → only meshIndex computed.")
    else:
        print("No zCoordinate in mesh_dict → only meshIndex computed.")

    # -------------------------------
    # Process ONE timestep
    # -------------------------------
    def process_timestep(t):
        xp = x[:, t]
        yp = y[:, t]
        zp = z[:, t] if has_z else None

        valid = np.isfinite(xp) & np.isfinite(yp)
        if has_z:
            valid &= np.isfinite(zp)

        idx_valid = np.where(valid)[0]
        mi_t = meshIndex[:, t].copy()

        # --- Compute mesh element indices ---
        if idx_valid.size > 0:
            mi_val = mesh_index(
                xp[idx_valid], yp[idx_valid],
                mesh_dict,
                matlab_indexing=False,   # always compute 0-based, convert later
                fallback=True
            )
            mi_t[idx_valid] = mi_val

        # --- Depth + water surface ---
        if do_depth:
            ws_t = waterSurface[:, t].copy()
            dbs_t = depthBelowSurface[:, t].copy()

            for pi in idx_valid:
                if np.isnan(mi_t[pi]):
                    continue

                tri_idx = int(mi_t[pi])
                verts = meshIndices[tri_idx]

                xv = xMesh[verts]
                yv = yMesh[verts]
                zv = zMesh[verts, t]

                # barycentric weights
                x1, x2, x3 = xv
                y1, y2, y3 = yv

                detT = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
                if abs(detT) < 1e-14:
                    continue

                px, py = xp[pi], yp[pi]
                l1 = ((y2-y3)*(px-x3) + (x3-x2)*(py-y3)) / detT
                l2 = ((y3-y1)*(px-x3) + (x1-x3)*(py-y3)) / detT
                l3 = 1.0 - l1 - l2

                top = float(l1*zv[0] + l2*zv[1] + l3*zv[2])
                ws_t[pi] = top
                dbs_t[pi] = top - float(zp[pi])

        # ------------------------
        # PRINT (only once per timestep)
        # ------------------------
        if t % 50 == 0:
            print(f"Timestep {t+1}/{Nt}")

        if do_depth:
            return t, mi_t, ws_t, dbs_t
        else:
            return t, mi_t, None, None

    # -------------------------------
    # Parallel execution
    # -------------------------------
    with ThreadPoolExecutor() as ex:
        results = list(ex.map(process_timestep, range(Nt)))

    # -------------------------------
    # Store into output arrays
    # -------------------------------
    for t, mi, ws, dbs in results:
        meshIndex[:, t] = mi
        if do_depth:
            waterSurface[:, t] = ws
            depthBelowSurface[:, t] = dbs

    # convert to MATLAB 1-based indexing
    if matlab_indexing:
        valid = np.isfinite(meshIndex)
        meshIndex[valid] += 1

    # -------------------------------
    # Build output dictionary
    # -------------------------------
    out = dict(particle_dict)  # copy original fields
    out["meshIndex"] = meshIndex

    if has_zcoord:
        out["waterSurface"] = waterSurface
    if do_depth:
        out["depthBelowSurface"] = depthBelowSurface

    return out
