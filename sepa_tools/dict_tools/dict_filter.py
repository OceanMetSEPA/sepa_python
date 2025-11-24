import numpy as np
import scipy.sparse as sp

def dict_filter(data_dict, mask):
    """
    Filter each array (dense or sparse) in a dictionary using a boolean mask.

    Automatically detects whether to filter along rows or columns, based on
    which dimension matches the mask length.

    Args:
        data_dict (dict): Keys -> arrays (NumPy or SciPy sparse)
        mask (array-like of bool): Boolean mask (1D)

    Returns:
        dict: Dictionary with filtered arrays.
    """
    mask = np.asarray(mask)
    if mask.dtype != bool:
        raise TypeError("Mask must be a boolean array (True/False).")

    filtered = {}

    for key, arr in data_dict.items():
        n_rows, n_cols = arr.shape

        if len(mask) == n_rows:
            axis = 0  # filter rows
        elif len(mask) == n_cols:
            axis = 1  # filter columns
        else:
            raise ValueError(
                f"Mask length ({len(mask)}) does not match "
                f"array dimensions {arr.shape} for key '{key}'."
            )

        if sp.issparse(arr):
            # convert for efficient slicing
            if axis == 0:
                arr = arr.tocsr()
                filtered[key] = arr[mask, :]
            else:
                arr = arr.tocsc()
                filtered[key] = arr[:, mask]
        else:
            filtered[key] = arr[mask, :] if axis == 0 else arr[:, mask]

    return filtered
