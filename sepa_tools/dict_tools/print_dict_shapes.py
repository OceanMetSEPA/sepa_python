def print_dict_shapes(d, parent=""):
    """
    Recursively print the name and shape of each item in a dict.

    Parameters
    ----------
    d : dict
        Dictionary to inspect.
    parent : str
        Prefix for nested keys (used in recursion).
    """
    for k, v in d.items():
        name = f"{parent}.{k}" if parent else k
        if isinstance(v, dict):
            print_dict_shapes(v, parent=name)
        elif isinstance(v, (list, tuple)):
            print(f"{name}: list/tuple of length {len(v)}")
        elif hasattr(v, "shape"):
            print(f"{name}: shape {v.shape}")
        else:
            print(f"{name}: type {type(v)}")
