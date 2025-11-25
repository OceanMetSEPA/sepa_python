import numpy as np
from typing import Dict, Union
import pandas as pd
from copy import deepcopy
from scipy.sparse import issparse, csr_matrix

# custom SEPA imports
#from part_processing.tools import source_term_version_data
from sepa_tools.maths_tools import summary_statistics


def scale_conc_dict(
    s: Dict[str, np.ndarray],
    scale_factors: Union[float, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Memory-efficient sparse-safe version.

    Supports:
    - scalar
    - vector [Nf]
    - time-varying [Nf, Nt]

    Returns dict of same types (sparse stays sparse).
    """

    fn = list(s.keys())
    Nf = len(fn)

    # Check shapes
    first_shape = s[fn[0]].shape
    if not all(s[f].shape == first_shape for f in fn[1:]):
        raise ValueError("Mismatch in field sizes")

    Np, Nt = first_shape

    # ---- Prepare scale_factors into shape (Nf, 1) or (Nf, Nt) ----
    if np.isscalar(scale_factors):
        scale_factors = np.full((Nf, 1), scale_factors, dtype=float)

    else:
        scale_factors = np.atleast_2d(np.asarray(scale_factors, float))

        if scale_factors.shape[0] != Nf:
            if scale_factors.shape[1] == Nf:
                scale_factors = scale_factors.T  # swap orientation
            else:
                raise ValueError("Scale factors should have one per field")

    # Validate shape
    Nsf_cols = scale_factors.shape[1]
    if Nsf_cols != 1 and Nsf_cols != Nt:
        raise ValueError("Scale factors must have 1 or Nt columns")

    # ---- Apply scaling ----
    scaled_struct = {}

    for i, f in enumerate(fn):

        vals = s[f]
        sf = scale_factors[i]

        # Convert to CSR if not already (efficient for column ops)
        if issparse(vals) and not isinstance(vals, csr_matrix):
            vals = vals.tocsr()

        # Case 1: scalar per farm
        if sf.size == 1:
            scaled_struct[f] = vals * sf[0]
            continue

        # Case 2: time-varying per timestep (sf has length Nt)
        # Scale each column independently, in-place, sparse-safe
        sf_t = sf  # shape: (Nt,)

        # Efficient column-scaling for CSR:
        # Multiply each column's data in-place
        vals = vals.tocsc()  # CSC is efficient for column operations

        for t in range(Nt):
            start = vals.indptr[t]
            end = vals.indptr[t+1]
            if start != end:
                vals.data[start:end] *= sf_t[t]

        scaled_struct[f] = vals.tocsr()

    return scaled_struct


def biomass_to_lice(
    biomass: Union[float, np.ndarray],
    overfill: float = 1.75,
    mass_per_fish: float = 5.0,
    lice_per_fish: Union[float, np.ndarray] = 0.4,
    new_lice_per_day: float = 30.0,
    particle_time_spacing: float = 5.0,  # minutes
    mass_of_particle: float = 1e9,       # µg
    model_duration: float = 74.0,        # days
) -> np.ndarray:
    """
    Calculate scale factor for particle tracking from µg to lice.

    Parameters
    ----------
    biomass : float or ndarray
        Biomass in tonnes.
    overfill : float, default 1.75
        Factor to account for operators keeping biomass near maximum.
    mass_per_fish : float, default 5
        Average mass per fish in kg.
    lice_per_fish : float or ndarray, default 0.4
        Mean number of adult ovigerous female lice per fish.
    new_lice_per_day : float, default 30
        Number of new lice released per adult female per day.
    particle_time_spacing : float, default 5
        Time between particle releases in minutes.
    mass_of_particle : float, default 1e9
        Mass of particle in µg.
    model_duration : float, default 74
        Length of model run in days.

    Returns
    -------
    scale_factor : ndarray
        Scale factor for converting concentration/exposure values.
    """

    biomass = np.asarray(biomass)

    # Handle lice_per_fish input if structured (MATLAB struct)
    lice_per_fish = np.asarray(lice_per_fish)

    # Step 1: convert biomass to number of fish
    number_of_fish = biomass * 1000 * overfill / mass_per_fish

    # Step 2: calculate lice per farm
    number_of_lice = number_of_fish * lice_per_fish

    # Step 3: calculate new lice per farm per day
    new_lice_per_day_total = number_of_lice * new_lice_per_day

    # Step 4: calculate lice per particle
    particles_per_day = 24 * 60 / particle_time_spacing
    lice_per_particle = new_lice_per_day_total / particles_per_day

    # Final scale factor
    scale_factor = lice_per_particle / mass_of_particle

    return scale_factor



def get_default_farm_lice_count(siteData):
    """
    Return default lice-per-fish values:
      - 0.4 for all farms
      - 0.1 for farms whose Notes contain 'closed' (case-insensitive)
    siteData: pandas DataFrame with columns ['SiteName', 'Notes']
    """

    # Start with all farms = 0.4
    lice = np.full(len(siteData), 0.4)

    # Find closed containment farms
    if "Notes" in siteData.columns:
        is_closed = siteData["Notes"].str.contains("closed", case=False, na=False)
        lice[is_closed] = 0.1

    # Return DataFrame similar to MATLAB output
    return pd.DataFrame({
        "SiteName": siteData["SiteName"].values,
        "LiceCount": lice
    })

def scale_exposure_dict(exposure_data, version_data):
    # Normalize input
    single_input = False
    if isinstance(exposure_data, dict):
        exposure_list = [exposure_data]
        single_input = True
    else:
        exposure_list = exposure_data

    # Ensure biomass numeric
    version_data["Biomass"] = pd.to_numeric(version_data["Biomass"], errors="coerce")

    # Lice count
    lice_count = get_default_farm_lice_count(version_data)["LiceCount"].values

    # Scale factors
    scale_factors = biomass_to_lice(
        biomass=version_data["Biomass"].values,
        lice_per_fish=lice_count
    )

    out = []

    for s in exposure_list:
        s = deepcopy(s)

        farm_names = list(s["total_exposure_per_farm"].keys())

        # Scale exposures per farm
        for i, farm in enumerate(farm_names):
            s["total_exposure_per_farm"][farm] *= scale_factors[i]

        # Stack along new axis (farm axis) and sum
        farm_arrays = [s["total_exposure_per_farm"][f] for f in farm_names]
        stacked = np.stack(farm_arrays, axis=1)  # shape: (N_time, N_farms)
        s["total_exposure"] = np.sum(stacked, axis=1)  # sum over farms

        # Summary statistics
        stats = summary_statistics(s["total_exposure"])
        s.update(stats)


        out.append(s)

    return out[0] if single_input else out
