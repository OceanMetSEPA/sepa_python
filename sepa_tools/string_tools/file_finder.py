from pathlib import Path
from typing import List, Union
import fnmatch

def file_finder(folder: Union[str, Path],
                string: str = '',
                end: str = None,
                exclude: Union[str, List[str]] = None,
                subdir: bool = False,
                files: bool = None,
                dirs: bool = None,
                operation: str = 'or') -> List[Path]:
    """
    Search for files and/or directories with clear, predictable rules:
    
    - Default (files=None, dirs=None): return BOTH files and directories.
    - files=True  → return ONLY files.
    - files=False → return ONLY directories.
    - dirs=True   → return ONLY directories.
    - dirs=False  → return ONLY files.
    - If both files and dirs specified:
        - True/True → both
        - False/False → return empty list
    """

    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"{folder} is not a directory!")

    #
    # ---- Resolve selection mode ----
    #
    # Case 1: default behaviour
    if files is None and dirs is None:
        include_files = True
        include_dirs = True

    # Case 2: both explicitly provided
    elif files is not None and dirs is not None:
        # both False → nothing
        if files is False and dirs is False:
            return []
        include_files = bool(files)
        include_dirs = bool(dirs)

    # Case 3: only files is specified
    elif files is not None and dirs is None:
        if files:      # files=True
            include_files, include_dirs = True, False
        else:          # files=False → dirs only
            include_files, include_dirs = False, True

    # Case 4: only dirs is specified
    elif dirs is not None and files is None:
        if dirs:       # dirs=True
            include_files, include_dirs = False, True
        else:          # dirs=False → files only
            include_files, include_dirs = True, False

    #
    # ---- Collect items ----
    #
    candidates = folder.rglob('*') if subdir else folder.iterdir()

    items = []
    for f in candidates:
        if f.is_file() and include_files:
            items.append(f)
        elif f.is_dir() and include_dirs:
            items.append(f)

    #
    # ---- Apply name pattern ----
    #
    if string:
        if '*' not in string and '?' not in string:
            string = f'*{string}*'
        items = [f for f in items if fnmatch.fnmatch(f.name, string)]

    #
    # ---- Apply "end" filter ----
    #
    if end:
        items = [f for f in items if f.name.endswith(end)]

    #
    # ---- Exclude filter ----
    #
    if exclude:
        if isinstance(exclude, str):
            exclude = [exclude]
        items = [f for f in items if not any(exc in f.name for exc in exclude)]

    return items
