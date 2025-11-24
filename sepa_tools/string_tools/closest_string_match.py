def closest_string_match(candidates, query):
    """
    Replicates MATLAB's closestStringMatch behavior:

    - If there’s an exact match (case-insensitive), return it.
    - Otherwise, return all candidates that START with the query (case-insensitive).
    - Otherwise, return an empty list.

    Args:
        candidates (list-like of str or pd.Series): Available site names.
        query (str): Query string to match.

    Returns:
        list[str]: Matching site names.
    """
    # Convert pandas Series to list
    if hasattr(candidates, "tolist"):
        candidates = candidates.tolist()

    if not candidates or not query:
        return []

    # Normalize inputs
    query_lower = str(query).lower()
    candidates_lower = [str(c).lower() for c in candidates]

    # 1️⃣ Exact match first
    exact_matches = [c for c, cl in zip(candidates, candidates_lower) if cl == query_lower]
    if exact_matches:
        return exact_matches

    # 2️⃣ Prefix (startswith) match
    prefix_matches = [c for c, cl in zip(candidates, candidates_lower) if cl.startswith(query_lower)]
    if prefix_matches:
        return prefix_matches

    # 3️⃣ No match
    return []
