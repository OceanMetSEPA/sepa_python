# python imports
from pathlib import Path
import numpy as np

# custom SEPA imports
from sepa_tools.string_tools import file_finder
from sepa_tools.load_mat_file import load_mat_file
from part_processing.tracks import wspz_track_description
from part_processing.tools import model_domain_from_string

# Create dict containing exposure data and information about tracks
def exposure_file_to_dict(exposure_file,geostuff=None):
    exposure_file=Path(exposure_file)
    exposure_name=exposure_file.name # path removed

    # Calculate track path from file name (assumes consistent filing!)
    idx = exposure_file.parts.index("FishTracks")

    # Rebuild the path up to that part (inclusive)
    base_path = Path(*exposure_file.parts[:idx])
    
    if geostuff is None:
        geostuff=load_mat_file(base_path / 'geostuff.mat')
    
    # find associated track file
    track_path=base_path / 'FishTracks'

    # Folder containing tracks-
    xy_track_path=track_path / 'xy'

    # Find track file-
    track_file=file_finder(xy_track_path,exposure_name,subdir=True)[0]

    # Load track data-
    xy=load_mat_file(track_file) 
    # Track descripton-
    track_desc=wspz_track_description(track_file,geostuff)
    track_desc.pop('fileName')  # remove duplicate

    # Load exposure data-
    exposure_data=load_mat_file(exposure_file)
    # Total exposure for each farm (dict entry)-
    total_per_farm={k: np.array(v.sum(axis=0)).ravel() for k, v in exposure_data.items()}
    # Sum of all farms-
    total_exposure = sum(total_per_farm.values())


    # Prepare dict with key info
    s={'fileName':str(exposure_file)}
    s['fishTrackName']=exposure_file.name
    s['modelDomain']=model_domain_from_string(str(exposure_file))
    s['Code']=exposure_file.name.replace('.mat','')

    s=s|track_desc|xy|{'total_exposure_per_farm':total_per_farm,'total_exposure':total_exposure}
    return s