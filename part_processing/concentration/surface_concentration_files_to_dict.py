from part_processing.tools import site_name_from_string
from pathlib import Path
from sepa_tools.load_mat_file import load_mat_file
from sepa_tools.string_tools import file_finder
import numpy as np

import scipy.sparse as sp

def surface_concentration_files_to_dict(concFiles):
    """
    Generate a dictionary where each key is a site name and each value is the
    contents of a surface concentration .mat file (2D sparse or dense array).

    If the array has more than 2000 columns, it will be downsampled by taking
    every 4th column (e.g. convert 15-min to hourly).
    """
    # Handle case where user passes a directory instead of list
    if isinstance(concFiles, (str, Path)):
        concFiles = file_finder(concFiles, 'surfaceConc.mat', subdir=True)

    surface_conc_dict = {}
    total_files = len(concFiles)

    for site_index, fi in enumerate(concFiles, start=1):
        site_name = site_name_from_string(fi.name)
        print(f"Loading {site_name} ({site_index} of {total_files})")

        conc_data = load_mat_file(fi)

        # Unwrap singleton lists or dicts if needed
        if isinstance(conc_data, (list, tuple)) and len(conc_data) == 1:
            conc_data = conc_data[0]

        # Ensure it's numpy array or sparse
        if sp.issparse(conc_data):
            n_cols = conc_data.shape[1]
        else:
            conc_data = np.asarray(conc_data)
            n_cols = conc_data.shape[1] if conc_data.ndim == 2 else 0

        # ✅ Downsample if necessary (every 4th column)
        if n_cols > 2000:
            print(f"  -> {site_name}: {n_cols} columns detected, downsampling by 4 (approx hourly)")
            idx = np.arange(0, n_cols, 4)
            if sp.issparse(conc_data):
                conc_data = conc_data.tocsc()[:, idx]
            else:
                conc_data = conc_data[:, idx]
        else:
            print(f"  -> {site_name}: {n_cols} columns detected, keeping all")

        surface_conc_dict[site_name] = conc_data

    # Sort alphabetically by site name
    surface_conc_dict = {k: surface_conc_dict[k] for k in sorted(surface_conc_dict)}

    return surface_conc_dict
