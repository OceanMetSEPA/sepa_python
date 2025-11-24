#%% This module does initial processing of particle tracking output
#
# 1) Convert .xml to trackStruct.mat ~25 minutes
# 2) Map trackStruct to mesh (find grid cells of every point at every timestep) ~ 10 minutes
# 3) Calculate surface concentration (calculate load in each cell and divide by area)

# There is also code for calculating volume concentration (not required for sea lice stuff)

###############################################################
# NB make sure we're in folder above part_processing

#cd C:\Python
# (or add to path - sort this later!)
###############################################################

# python imports
from scipy.io import savemat
from pathlib import Path
import time

# custom SEPA imports
import part_processing as pp
from  part_processing.concentration import calculate_concentration
from  part_processing.concentration import map_particle_tracks_to_mesh

from sepa_tools import load_mat_file
import sepa_tools.string_tools as string_tools
from sepa_tools.string_tools import file_finder
from sepa_tools.dict_tools import print_dict_shapes

#%% Aside - show all files in folder and subfolders
string_tools.utils.show_package_tree('C:\Python\sepa_python')

#%% Specify model domain / site to process
model_domain='ECLH'
site_name='GravirWest'
#base_path=Path(r'\\asb-fp-mod01\AMMU\AquacultureModelling\SeaLice')
base_path=Path(r'\\asb-fp-mod01\AMMU\AquacultureModelling\SeaLice\ECLH1993Results\pythontest20251124')
model_path=file_finder(base_path,model_domain,dirs=True)[0]

site_path=file_finder(model_path,site_name,dirs=True,subdir=True)
site_path=site_path[0] 
print(site_path)

#%% Mesh
# NB meshStruct.mat generated in matlab from .mesh file using Mike.mesh2MeshStruct
# This stores (in addition to mesh coordinates, triangulation) the area of each cell in m2.
# (required for determining surface concentration)

# For sea-lice stuff, meshStruct.mat stored in model_path. If yours is elsewhere, code below will have to be adapted!

mesh_file=file_finder(model_path,'meshStruct.mat')[0]
mesh_dict=load_mat_file(mesh_file)
print_dict_shapes(mesh_dict)

#%% Convert xml to mat (raw track info)
# Pass function below either an xml file or folder containing xml. 

##  %%time # This works but Spyder display cross in red circle so presumably doesn't like it!
start = time.perf_counter() #time how long this takes (this preferable to %%time apparently)

pp.xml_tools.process_xml_folder(model_path)

end = time.perf_counter()
print(f"Elapsed: {end - start:.3f} seconds")
# 25 minutes for conversion

#%% Process / load dfsu dict
# We might not need this going forward? Previously we did, in order to determine water level and thus particle depth relative to surface.
# But we can get depth below surface as parameter in xml file (ask Alan how!)
#
# 
# NB 1: this function designed for equally spaced sigma layers- rather than extracting every layer,
# the surface layer and (uniform) spacing is stored
# NB 2: saving to matfile (done wthin dfsu_to_surface_dict function) only works if variable <2GB.

dfsu_file=model_path / 'dfsu' / 'VolumePT.dfsu'
dfsu_dict=pp.hd.dfsu_to_dict(dfsu_file, dt=4,matlab_index=False)
# ~5 mins to generate from .dfsu file; 20s to load from dfsu.mat
print_dict_shapes(dfsu_dict)


#%% Load raw track data (file created by process_xml_folder above)
raw_track_file=file_finder(site_path,'trackStruct')[0]
raw_particle_dict=load_mat_file(raw_track_file) # ~ 1 minute or less
print_dict_shapes(raw_particle_dict)

#%% Map to grid
# NB if we want to calculate depth and it isn't in raw_particle_dict (because it wasn't included in the original xml),
# then we need to pass dfsu_dict which contains zCoordinate, the time-varying water surface
# If we're not interested in depth or it's already included, we can pass mesh_dict or dfsu_dict - we just calculate meshIndex,
# the grid cells containing the points

mapped_particle_dict=map_particle_tracks_to_mesh(raw_particle_dict,dfsu_dict, matlab_indexing=True) # takes a few minutes

# Save to file
dfsu_track_file=str(raw_track_file).replace('.mat','_dfsu.mat');
savemat(dfsu_track_file,mapped_particle_dict)

#%% surface conc - generate sparse array in units µg/m2
# Size of array is [Nn,Nt] where
# Nn = number of grid cells
# Nt = number of time steps
#
# Note - we could calculate volume concentration by setting zCutoff=inf (including particles at any depth)
# then dividing by the depth of the cell (average of the zMesh values for given cell)

surface_conc=calculate_concentration(mapped_particle_dict,mesh_dict,zCutoff=2,surface=True)
# takes about a minute

# Save to matfile
surface_conc_file=str(raw_track_file).replace('trackStruct','surfaceConc')
savemat(surface_conc_file,{'concPerM2':surface_conc})

#%% volume conc - not needed for sea-lice but might be useful elsewhere

# NB this divides the surface conc by depth specified in mesh i.e. ignores time-varying surface elevation. 
# If that's an issue, will need to adapt using dfsu_dict
volume_conc=calculate_concentration(mapped_particle_dict,mesh_dict)
