import h5py
import scipy.io
import numpy as np
from scipy.io.matlab import mat_struct

def load_mat_file(filepath, lazy_h5=False):
    # ---------- Fix MATLAB structs ----------
    def _fix_struct(obj):
        if isinstance(obj, mat_struct):
            return {name: _fix_struct(getattr(obj, name)) for name in obj._fieldnames}

        if isinstance(obj, dict):
            return {k: _fix_struct(v) for k, v in obj.items()}

        if isinstance(obj, np.ndarray):
            return _fix_array(obj)

        return obj

    # ---------- Fix numeric arrays ----------
    def _fix_array(arr):
        # unwrap 1×1 object array
        if arr.dtype == object and arr.size == 1:
            return _fix_struct(arr.flat[0])

        # recurse inside object arrays
        if arr.dtype == object:
            return np.vectorize(_fix_struct, otypes=[object])(arr)

        # reshape 1D array to column (MATLAB)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)

        return arr

    # ---------- Load HDF5 first ----------
    try:
        with h5py.File(filepath, "r") as f:
            out = {}
            for k, v in f.items():
                if k.startswith("__"):
                    continue
                out[k] = v if lazy_h5 else v[()]
            return out
    except Exception:
        pass

    # ---------- Fallback: load pre-v7.3 ----------
    mat = scipy.io.loadmat(filepath, struct_as_record=False, squeeze_me=False)
    out = {k: _fix_struct(v) for k, v in mat.items() if not k.startswith("__")}

    # unwrap top-level 1×1 object array
    if len(out) == 1:
        v = next(iter(out.values()))
        if isinstance(v, np.ndarray) and v.dtype == object and v.size == 1:
            return _fix_struct(v.flat[0])
        return v

    return out
