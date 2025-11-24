from pathlib import Path
import numpy as np

from sepa_tools.load_mat_file import load_mat_file
from part_processing.tracks.track_coordinates import track_coordinates
from part_processing.tools.which_polyshape import which_polyshape
from part_processing.tools.flatten_struct_array import flatten_struct_array

def wspz_track_description(ip, geostuff, version=1, path=None, loadIfFile=False):
    """
    Describe a fish track in terms of River, PZ, WB.

    Parameters
    ----------
    ip : str/Path (MAT-file), dict with 'x','y', or (x,y) tuple
        If filename, coordinates are loaded if loadIfFile=True.
    geostuff : dict
        Must contain 'RiverMouths', 'WaterBodies', 'WSPZ_Individual'
    version : int, optional
        Version number for the track code and file_name (default 1)
    path : str or Path, optional
        Base path to prepend to generated file_name
    loadIfFile : bool, optional
        If True and ip is filename, load coordinates from file.
        If False, generate output fields from filename string.
    """

    file_path = None
    x = y = None

    # ------------------------
    # 1. Handle input type
    # ------------------------
    if isinstance(ip, (str, Path, np.str_)):
        ip = str(ip)
        file_path = ip
        if loadIfFile:
            tf = load_mat_file(ip)
            x, y = track_coordinates(tf)
    elif isinstance(ip, dict):
        if 'x' in ip and 'y' in ip:
            x = np.ravel(ip['x'])
            y = np.ravel(ip['y'])
        else:
            raise TypeError("dict input must have 'x' and 'y' keys")
    elif isinstance(ip, tuple) and len(ip) == 2:
        x, y = np.ravel(np.asarray(ip[0])), np.ravel(np.asarray(ip[1]))
    else:
        raise TypeError("ip must be filename (str/Path), dict with x,y, or (x,y) tuple")

    # ------------------------
    # 2. Initialize outputs
    # ------------------------
    riverId = WBID = None
    pzIds = []
    riverName = WBName = None
    pzNames = []
    Version = version

    # ------------------------
    # 3. Process coordinates if available
    # ------------------------
    if x is not None and y is not None and len(x) > 0:
        riverMouthsAll = flatten_struct_array(geostuff["RiverMouths"])
        waterBodiesAll = flatten_struct_array(geostuff["WaterBodies"])
        pzStructs = flatten_struct_array(geostuff["WSPZ_Individual"])

        # River: first track point
        start_point = np.array([x[0], y[0]])
        riverMouths = [rm for rm in riverMouthsAll if int(np.ravel(rm.get('ModelDomain', 0))[0]) > 0]
        river_coords = np.array([[np.ravel(rm['Longitude'])[0], np.ravel(rm['Latitude'])[0]] for rm in riverMouths])
        dists = np.sum((river_coords - start_point)**2, axis=1)
        riverIndex = int(np.argmin(dists))
        river = riverMouths[riverIndex]
        riverId = int(np.ravel(river['River_ID'])[0])
        riverName = str(np.ravel(river['River_Name'])[0])

        # PZs: full track
        pzIndices_all = which_polyshape(x, y, pzStructs, closest=False)
        valid_indices = [int(i) for i in pzIndices_all if i is not None and not np.isnan(i)]
        seen = set()
        valid_indices = [i for i in valid_indices if not (i in seen or seen.add(i))]
        if valid_indices:
            pzIds = [int(np.ravel(pzStructs[i]['WSPZ_ID'])[0]) for i in valid_indices]
            pzNames = [str(np.ravel(pzStructs[i]['WSPZ_NAME'])[0]) for i in valid_indices]

        # WB: last track point
        wbIndices = which_polyshape(x[-1], y[-1], waterBodiesAll, closest=True)
        if wbIndices and wbIndices[0] is not None and not np.isnan(wbIndices[0]):
            wbIndex = int(wbIndices[0])
            WBID = int(np.ravel(waterBodiesAll[wbIndex]['WB_ID'])[0])
            WBName = str(np.ravel(waterBodiesAll[wbIndex]['WATER_BODY'])[0])

    # ------------------------
    # 4. Handle filename-only parsing
    # ------------------------
    if x is None or len(x) == 0:
        if file_path:
            file_stem = Path(file_path).stem
            # Parse version
            if '_v' in file_stem:
                base, ver_str = file_stem.rsplit('_v', 1)
                try:
                    Version = int(ver_str)
                except Exception:
                    Version = version
            else:
                base = file_stem
                Version = version

            parts = base.split('_')
            if len(parts) >= 3:
                try:
                    riverId = int(parts[0])
                    WBID = int(parts[-1])
                    pzIds = [int(p) for p in parts[1:-1]]
                except Exception:
                    riverId = WBID = None
                    pzIds = []

            # Lookup names from geostuff
            if riverId is not None:
                riverMouthsAll = flatten_struct_array(geostuff["RiverMouths"])
                rmatch = [rm for rm in riverMouthsAll if int(np.ravel(rm['River_ID'])[0]) == riverId]
                riverName = str(np.ravel(rmatch[0]['River_Name'])[0]) if rmatch else None

            if WBID is not None:
                waterBodiesAll = flatten_struct_array(geostuff["WaterBodies"])
                wmatch = [wb for wb in waterBodiesAll if int(np.ravel(wb['WB_ID'])[0]) == WBID]
                WBName = str(np.ravel(wmatch[0]['WATER_BODY'])[0]) if wmatch else None

            if pzIds:
                pzStructs = flatten_struct_array(geostuff["WSPZ_Individual"])
                pzNames = []
                for pid in pzIds:
                    pmatch = [pz for pz in pzStructs if int(np.ravel(pz['WSPZ_ID'])[0]) == pid]
                    pzNames.append(str(np.ravel(pmatch[0]['WSPZ_NAME'])[0]) if pmatch else None)

    # ------------------------
    # 5. Generate Code and file_name (include path if provided)
    # ------------------------
    if riverId is not None:
        pzPart = '_'.join(map(str, pzIds)) if pzIds else '0'
        WBPart = WBID if WBID is not None else 0
        Code = f"{riverId}_{pzPart}_{WBPart}"
    else:
        Code = 'Unknown'
            
    file_name = f"{Code}_v{Version}.mat"

    # If the input was a string or Path, include its directory
    if file_path:
        file_name = str(Path(file_path).parent / file_name)
    elif path is not None:
        file_name = str(Path(path) / file_name)


    return {
        'Code': Code,
        'RiverID': riverId,
        'RiverName': riverName,
        'PZID': pzIds,
        'PZName': pzNames,
        'WBID': WBID,
        'WBName': WBName,
        'Version': Version,
        'fileName': file_name
    }