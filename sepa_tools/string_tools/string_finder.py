from typing import Iterable, List, Union

def string_finder(
    items: Union[str, Iterable[str]],
    patterns: Union[str, Iterable[str]],
    *,
    output: str = "char",
    operation: str = "and",
    case: bool = False,
    where: str = "any",
    exclude: Union[str, Iterable[str], None] = None,
) -> Union[List[str], List[bool], List[int]]:
    """
    Filter a list of strings by one or more patterns with flexible matching.

    Parameters
    ----------
    items : str or iterable of str
        Strings to filter. If a single string is provided, it will be treated as a one-element list.
    patterns : str or iterable of str
        Pattern(s) to match against each item. Can be substrings or exact strings.
    output : {'char', 'bool', 'index'}, default 'char'
        Determines the output type:
        - 'char' returns the matching strings
        - 'bool' returns a list of booleans of the same length as items
        - 'index' returns the indices of matching items
    operation : {'and', 'or'}, default 'and'
        How multiple patterns are combined:
        - 'and' requires all patterns to match
        - 'or' requires any pattern to match
    case : bool, default False
        If True, matching is case-sensitive; otherwise, case-insensitive.
    where : {'any', 'start', 'end', 'exact'}, default 'any'
        Where the pattern should match within the string:
        - 'any': anywhere in the string
        - 'start': string starts with the pattern
        - 'end': string ends with the pattern
        - 'exact': string exactly matches the pattern
    exclude : str or iterable of str, optional
        Pattern(s) to exclude from the results. Applied after the main matching.

    Returns
    -------
    List[str] or List[bool] or List[int]
        The filtered results, depending on `output`.

    Examples
    --------
    >>> string_finder(["apple", "banana", "grape"], "a")
    ['apple', 'banana', 'grape']

    >>> string_finder(["apple", "banana", "grape"], "a", where="start")
    ['apple']

    >>> string_finder(["apple", "banana", "grape"], "a", output="index")
    [0, 1, 2]

    >>> string_finder(["apple", "banana", "grape"], "a", exclude="gr")
    ['apple', 'banana']
    """
    if isinstance(items, str):
        items = [items]
    if isinstance(patterns, str):
        patterns = [patterns]
    if exclude and isinstance(exclude, str):
        exclude = [exclude]

    keyfunc = (lambda s: s) if case else (lambda s: s.casefold())
    items_norm = [keyfunc(s) for s in items]
    patterns_norm = [keyfunc(s) for s in patterns]
    exclude_norm = [keyfunc(s) for s in exclude] if exclude else []

    def match_one(item: str, pat: str) -> bool:
        if where == "any":
            return pat in item
        elif where == "start":
            return item.startswith(pat)
        elif where == "end":
            return item.endswith(pat)
        elif where == "exact":
            return item == pat
        else:
            raise ValueError(f"Unknown `where`: {where}")

    mask = []
    for it in items_norm:
        results = [match_one(it, pat) for pat in patterns_norm]
        keep = any(results) if operation == "or" else all(results)
        if any(ex in it for ex in exclude_norm):
            keep = False
        mask.append(keep)

    if output == "char":
        return [s for s, keep in zip(items, mask) if keep]
    elif output == "bool":
        return mask
    elif output == "index":
        return [i for i, keep in enumerate(mask) if keep]
    else:
        raise ValueError(f"Unknown output {output}")
