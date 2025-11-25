#%% Process multiple concentration files 
#
# *) find _surfaceConc.mat files
# *) load them into a python dict, where each entry [siteName] contains an arry of concentration values µg/m2
# *) calculate total concentration (sum of individual contributions)
# *) scale, converting from µg/m2 to lice/m2
# *) calculate time-average

# NB make sure we're in folder above part_processing
#cd C:\Python\
# (or add to path - sort this later!)

# python imports
from scipy.io import savemat
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# custom SEPA imports
from part_processing.concentration import surface_concentration_files_to_dict
from part_processing.tools import source_term_version_data
from part_processing.tools import scale_conc_dict
from part_processing.tools.scale_stuff import biomass_to_lice

from sepa_tools.string_tools import file_finder
from sepa_tools import load_mat_file
from sepa_tools.dict_tools import dict_filter, sum_sparse_dict

from part_processing.exposure import calculate_track_exposure,exposure_file_to_dict


#%% Define base folder with conc files
base_path=Path(r'C:\SeaLice')
model_domain='ECLH'
model_path = base_path/model_domain

#%% Find surfaceConc files
conc_path=model_path / 'XMLTracksAnddfs0'              
conc_files=file_finder(conc_path,'surfaceConc.mat',subdir=True) 
print(conc_files)
len(conc_files)

#%% Load all concentration files for model domain
conc_per_farm_raw=surface_concentration_files_to_dict(conc_files)
# NB some of these were originally created at 15 minute intervals- function above loads all as hourly values
# (makes combining them easier)

#%% Load timesteps from original modelling
# These are 15 minute intervals from mid-March to end of May
# conc_per_farm_raw contains hourly values; want to filer these to April/May
# -> filter datenums to hourly values, then to months 4/5

# Path to .mat file
f = model_path / 'modelDatenums.mat'

# Load the .mat file
t = load_mat_file(f)  # shape (7105,1)
t = t.ravel()          # flatten to (7105,)

# Convert MATLAB datenum to Python datetime
py_dates = np.array([datetime.fromordinal(int(d)) + timedelta(days=d % 1) - timedelta(days=366) for d in t])

# Round to nearest minute
py_dates = np.array([dt.replace(second=0, microsecond=0) + (timedelta(minutes=1) if dt.microsecond >= 500_000 else timedelta()) 
                     for dt in py_dates])

# Keep only full hours
mask_hourly = np.array([dt.minute == 0 for dt in py_dates])

# Keep only April or May
mask_month = np.isin([dt.month for dt in py_dates], [4,5])

# Combined mask on original t
time_mask_orig = mask_hourly & mask_month

# Map this mask to hourly timesteps in concPerFarmRaw
# Assume concPerFarmRaw uses every 4th timestep (15min → 1h) starting from t[0]
hourly_indices = np.where(mask_hourly)[0]  # indices of full hours
mask_hourly_apr_may = np.isin(np.arange(len(hourly_indices)), np.where(mask_month[hourly_indices])[0])

# This is the mask to apply to concPerFarmRaw (length = number of hourly timesteps)
conc_mask = mask_hourly_apr_may

# Filter conc dict by mask (no axis keyword)
conc_per_farm = dict_filter(conc_per_farm_raw, conc_mask)


#%% Save 
conc_mat_file=model_path / 'concsByPython.mat'
savemat(conc_mat_file,conc_per_farm)

#%% Scale concs
fn=conc_per_farm.keys()
farm_info=source_term_version_data(fn,0)
# lice scale factor -
sf=biomass_to_lice(farm_info.Biomass) # (118,)
# scale concs -
conc_per_farm_scaled=scale_conc_dict(conc_per_farm,sf)

#%% Get total conc
total_conc=sum_sparse_dict(conc_per_farm_scaled) # ~30s

#%% time-average
time_average_conc=np.mean(total_conc,axis=1)
time_average_conc.shape

#%% Plot time-average

mesh_file=file_finder(model_path,'meshStruct.mat')[0]
mesh_dict=load_mat_file(mesh_file)
meshIndices=mesh_dict['meshIndices']
xMesh=mesh_dict['xMesh']
yMesh=mesh_dict['yMesh']
zMesh=mesh_dict['zMesh']
# Example: ensure meshIndices is zero-indexed for matplotlib
triangles = meshIndices - 1 if meshIndices.min() == 1 else meshIndices

# Flatten the x, y, z arrays if needed
x = xMesh.flatten()
y = yMesh.flatten()
z = zMesh.flatten()

# Create Triangulation object
triang = tri.Triangulation(x, y, triangles)

# --- Fix triangle indices (0-indexed) ---
triangles = meshIndices - 1 if meshIndices.min() == 1 else meshIndices

# --- Fix concentration (1 value per triangle) ---
concentration = np.asarray(time_average_conc).squeeze()

# Cap concentration at 1
conc_clipped = np.minimum(concentration, 1.0)

# --- Bathymetry base layer (deep = dark) ---
triang = tri.Triangulation(x, y, triangles)

fig, ax = plt.subplots(figsize=(11, 9))

tpc_bathy = ax.tripcolor(
    triang,
    z,
    cmap='gray',      #  grayscale: deep = dark
    edgecolors='none',  # no edges
    linewidth=0
)

cb1 = plt.colorbar(tpc_bathy, ax=ax)
cb1.set_label("Bathymetry (m)")

# --- Concentration overlay (non-zero only) ---
mask = conc_clipped > 0
nonzero_tri = triangles[mask]
conc_vals = conc_clipped[mask]

if nonzero_tri.size > 0:
    triang_conc = tri.Triangulation(x, y, nonzero_tri)

    tpc_conc = ax.tripcolor(
        triang_conc,
        conc_vals,
        cmap='autumn_r',   # yellow → red, similar to flipud(autumn)
        edgecolors='none',
        alpha=0.6
    )

    cb2 = plt.colorbar(tpc_conc, ax=ax)
    cb2.set_label("Concentration (capped at 1)")

ax.set_aspect('equal')
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Bathymetry (deep = dark) + Concentration Overlay (yellow→red)")

plt.show()

#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Generate exposure from conc_per_farm above
###############################################################################
#%% Load matfile with WSPZ, river mouths etc
geostuff_file=base_path / 'geostuff.mat'
geostuff=load_mat_file(geostuff_file)

#%% setup paths
track_path=base_path / 'FishTracks'
swim_string='0.125mps'
model_domain='ECLH'

# Folder containing fishLegs
fish_legs_path=track_path / 'FishLegs' / swim_string / model_domain
# Folder containing tracks-
xy_path=track_path / 'xy'
# Folder containing exposure-
exposure_parent_path=track_path / 'Exposure'
exposure_path=exposure_parent_path / swim_string / model_domain

#%% Calculate exposure values for single track
track_name='91_67_200203_v1'
fish_legs_file=file_finder(fish_legs_path,track_name)[0]
fish_legs=load_mat_file(fish_legs_file)
offsets=np.arange(1321)
exposure=calculate_track_exposure(conc_per_farm,fish_legs,offset=offsets)

#%% Generate exposure dict (exposure data + track description)
f=file_finder(exposure_path,track_name)[0]
exposure_dict=exposure_file_to_dict(f)
savemat('exposurePython.mat',exposure)
