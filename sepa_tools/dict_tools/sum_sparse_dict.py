from typing import Dict
from scipy.sparse import issparse, csr_matrix

def sum_sparse_dict(d: Dict[str, object]):
    """
    Sum all values in a dictionary where each entry must be
    a 2-D SciPy sparse matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse matrix equal to the sum of all matrices.

    Raises
    ------
    TypeError
        If any value is not a sparse matrix or not 2-D.
    ValueError
        If shapes do not match.
    """

    if not d:
        raise ValueError("Dictionary is empty.")

    # Extract keys and first value
    keys = list(d.keys())
    first = d[keys[0]]

    # Check first value
    if not issparse(first):
        raise TypeError(f"Value for key '{keys[0]}' is not a sparse matrix.")
    if first.ndim != 2:
        raise TypeError(f"Value for key '{keys[0]}' is not 2-D.")
    
    shape = first.shape

    # Start accumulation in CSR (fast for addition)
    total = first.tocsr()

    # Loop through remaining
    for k in keys[1:]:
        mat = d[k]

        if not issparse(mat):
            raise TypeError(f"Value for key '{k}' is not a sparse matrix.")
        if mat.ndim != 2:
            raise TypeError(f"Value for key '{k}' is not 2-D.")
        if mat.shape != shape:
            raise ValueError(
                f"Shape mismatch for key '{k}': {mat.shape} vs expected {shape}"
            )

        total += mat  # sparse-safe add

    return total
