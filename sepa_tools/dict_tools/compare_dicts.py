import numpy as np

def compare_dicts(dict1, dict2, rtol=1e-12, atol=1e-14, verbose=True):
    """
    Compare two dictionaries that may contain numeric and non-numeric fields.

    Numeric arrays are compared element-wise, NaNs treated as equal.
    Non-numeric fields are compared for exact equality (==).
    
    Returns a dict with per-key status, match percentage, and max discrepancy.
    """
    report = {}
    all_keys = set(dict1.keys()).union(dict2.keys())

    for k in all_keys:
        if k not in dict1:
            report[k] = {'status': 'missing_in_dict1'}
            if verbose:
                print(f"Key '{k}' missing in dict1")
            continue
        if k not in dict2:
            report[k] = {'status': 'missing_in_dict2'}
            if verbose:
                print(f"Key '{k}' missing in dict2")
            continue

        v1 = dict1[k]
        v2 = dict2[k]

        # If numeric types or arrays
        try:
            a1 = np.asarray(v1)
            a2 = np.asarray(v2)
            if np.issubdtype(a1.dtype, np.number) and np.issubdtype(a2.dtype, np.number):
                if a1.shape != a2.shape:
                    report[k] = {'status': 'shape_mismatch',
                                 'shape1': a1.shape,
                                 'shape2': a2.shape}
                    if verbose:
                        print(f"Key '{k}' shape mismatch: {a1.shape} vs {a2.shape}")
                    continue

                nan_mask = np.isnan(a1) & np.isnan(a2)
                close_mask = np.isclose(a1, a2, rtol=rtol, atol=atol)
                match_mask = close_mask | nan_mask

                match_percentage = 100 * np.sum(match_mask) / match_mask.size
                max_discrepancy = np.nanmax(np.abs(a1 - a2))

                status = 'ok' if match_percentage == 100 else 'mismatch'
                report[k] = {
                    'status': status,
                    'match_percentage': match_percentage,
                    'max_discrepancy': max_discrepancy,
                    'shape_match': True
                }

                if verbose and status != 'ok':
                    print(f"Key '{k}': {match_percentage:.2f}% matching, max discrepancy {max_discrepancy}")
                continue
        except Exception:
            pass  # Not numeric

        # Non-numeric fields
        if type(v1) != type(v2):
            report[k] = {'status': 'type_mismatch',
                         'type1': type(v1),
                         'type2': type(v2)}
            if verbose:
                print(f"Key '{k}' type mismatch: {type(v1)} vs {type(v2)}")
        else:
            # Attempt equality
            try:
                equal = v1 == v2
            except Exception:
                equal = False
            status = 'ok' if equal else 'mismatch'
            report[k] = {'status': status,
                         'match_percentage': 100.0 if equal else 0.0,
                         'max_discrepancy': None,
                         'shape_match': True}
            if verbose and status != 'ok':
                print(f"Key '{k}': non-numeric mismatch")

    return report
