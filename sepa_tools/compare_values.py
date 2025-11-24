from typing import Iterable, Tuple, Optional

def compare_values(x: Iterable, y: Iterable, details: bool = False) -> Optional[Tuple[set, set]]:
    """
    Compare two sets of values (similar to MATLAB compareValues).

    Prints summary of differences between x and y, and optionally returns them.

    Parameters
    ----------
    x, y : iterable
        The two sequences or sets to compare.
    details : bool, default False
        If True, print all differing values.

    Returns
    -------
    (in_x_not_y, in_y_not_x) : tuple of sets
        Only returned if used as a function (not if just printed).
    """

    # Convert to sets (automatically removes duplicates)
    x_set = set(x)
    y_set = set(y)

    if x_set == y_set:
        print("EQUAL!")
        return (set(), set())

    in_x_not_y = x_set - y_set
    in_y_not_x = y_set - x_set

    if in_x_not_y:
        print(f"{len(in_x_not_y)} values are in x but not in y")
        if details:
            print(sorted(in_x_not_y))
    else:
        print("All values in x are also in y")

    if in_y_not_x:
        print(f"{len(in_y_not_x)} values are in y but not in x")
        if details:
            print(sorted(in_y_not_x))
    else:
        print("All values in y are also in x")

    return in_x_not_y, in_y_not_x
