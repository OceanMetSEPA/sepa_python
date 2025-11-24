import numpy as np
from typing import Union, List, Tuple

def track_coordinates(ip: Union[dict, List[dict]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get x, y coordinates from fishLegs or track dict(s).

    Parameters
    ----------
    ip : dict or list of dicts
        Each dict can have either:
        - 'x0', 'y0', 'x1', 'y1' for fishLegs
        - 'x', 'y' for track

    Returns
    -------
    x, y : np.ndarray
        Concatenated coordinates. If input is a list, NaNs separate tracks.
    """

    def join_by(arrays: List[np.ndarray], sep=np.nan) -> np.ndarray:
        """Concatenate arrays with a separator (NaN by default)."""
        if not arrays:
            return np.array([])
        result = arrays[0]
        for arr in arrays[1:]:
            result = np.concatenate([result, [sep], arr])
        return result

    # Handle single dict input by converting to list
    if isinstance(ip, dict):
        ip_list = [ip]
    elif isinstance(ip, list):
        ip_list = ip
    else:
        raise TypeError("ip must be a dict or list of dicts")

    if len(ip_list) == 0:
        return np.array([np.nan]), np.array([np.nan])
    elif len(ip_list) == 1:
        track = ip_list[0]
        if all(k in track for k in ['x0', 'y0', 'x1', 'y1']):
            x = np.concatenate([np.atleast_1d(track['x0']), np.atleast_1d(track['x1'])[-1:]])
            y = np.concatenate([np.atleast_1d(track['y0']), np.atleast_1d(track['y1'])[-1:]])
        elif all(k in track for k in ['x', 'y']):
            x = np.asarray(track['x'])
            y = np.asarray(track['y'])
        else:
            raise ValueError("Unknown struct format: missing required fields")
    else:
        # Multiple tracks: recursively concatenate with NaNs in between
        x_list = []
        y_list = []
        for track in ip_list:
            xi, yi = track_coordinates(track)
            x_list.append(xi)
            y_list.append(yi)
        x = join_by(x_list, sep=np.nan)
        y = join_by(y_list, sep=np.nan)

    # Flatten and ensure arrays are 1D
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    return x, y
