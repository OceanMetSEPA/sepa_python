import numpy as np
# -------------------------------
# Helper 1 — Robust struct flattener
# -------------------------------
def flatten_struct_array(arr):
    """
    Flatten MATLAB-style nested struct arrays into a list of plain Python dicts.
    Handles (1,n) and (n,1) shaped numpy arrays from loadmat.
    """
    result = []
    if isinstance(arr, np.ndarray):
        # Iterate through the numpy array elements
        for item in arr.flat:
            if isinstance(item, dict):
                clean_item = {}
                for k, v in item.items():
                    # Extract scalar values
                    if isinstance(v, np.ndarray):
                        if v.dtype == object:
                            v = np.array([vv for vv in v.flat])
                        if v.size == 1:
                            v = v.item()
                    clean_item[k] = v
                result.append(clean_item)
            elif isinstance(item, np.ndarray):
                # Nested array — try one more level
                result.extend(flatten_struct_array(item))
    elif isinstance(arr, dict):
        result.append(arr)
    return result