import os

def show_package_tree(package_path, exclude=None, prefix="", files=True):
    """
    Recursively print the package/folder tree.

    Parameters
    ----------
    package_path : str
        Root path of the package.
    exclude : list of str, optional
        Folder names to skip. Defaults to ['__pycache__', '.git'].
    prefix : str
        Internal use for recursion (do not set manually).
    files : bool, optional
        If False, only display directories. Defaults to True.
    """
    if exclude is None:
        exclude = ['__pycache__', '.git']

    basename = os.path.basename(package_path.rstrip(os.sep))
    print(prefix + basename)

    try:
        entries = sorted(os.listdir(package_path))
    except PermissionError:
        return

    for entry in entries:
        full_path = os.path.join(package_path, entry)

        # Handle directories
        if os.path.isdir(full_path):
            if entry in exclude:
                continue
            show_package_tree(full_path, exclude=exclude,
                              prefix=prefix + "├── ", files=files)

        # Handle files only if files=True
        else:
            if files:
                print(prefix + "├── " + entry)
