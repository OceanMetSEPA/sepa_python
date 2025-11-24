import pandas as pd
import numpy as np
import os
import pandas as pd
import numpy as np
from collections.abc import Iterable
from sepa_tools.string_tools import closest_string_match

def tdisp(x):
    """Format numbers like MATLAB's tdisp — integer if possible, else one decimal place."""
    if pd.isna(x):
        return "NaN"
    try:
        x = float(x)
        if x.is_integer():
            return str(int(x))
        else:
            return f"{x:.1f}"
    except Exception:
        return str(x)

def generate_source_term_table(file_path=None, force_reload=False):
    """
    Generate a table of source-term data across all versions (MATLAB-compatible version).

    Args:
        file_path (str, optional): Full path to the Excel spreadsheet. If not given, uses default.
        force_reload (bool): If True, reload spreadsheet instead of using cached data.

    Returns:
        pd.DataFrame: Table of source-term data for all farms and versions.
    """
    # Default file path (as in MATLAB)
    if file_path is None:
        file_path = r"C:\SeaLice\SourceTermSpreadsheets\Current\250812_SeaLiceScreening_Salmon_SourceTerms_V4.xlsx"

    # Global cache (persistent equivalent)
    global _source_data_cache
    if " _source_data_cache" not in globals():
        _source_data_cache = None

    if _source_data_cache is not None and not force_reload:
        print("Using cached sourceData")
        return _source_data_cache

    print("Loading sourceData from spreadsheet (this may take ~10s)...")

    # Get all sheet names (skip first)
    xls = pd.ExcelFile(file_path)
    sheet_names = [s for s in xls.sheet_names if s.lower() != "changelog"]
    if sheet_names and sheet_names[0].lower().startswith("basescreening"):
        sheet_names = sheet_names[1:]

    # Load all sheets into list of DataFrames
    all_sources = []
    for version_index, sheet in enumerate(sheet_names, start=1):
        df = pd.read_excel(xls, sheet_name=sheet)
        df["Version"] = version_index  # MATLAB-style indexing starts at 1
        all_sources.append(df)

    # Combine
    source_data_orig = pd.concat(all_sources, ignore_index=True)

    # Compute when each farm added
    farms_modelled = sorted(source_data_orig["SiteName"].dropna().unique())
    N_farms = len(farms_modelled)
    N_versions = len(sheet_names)

    # Find first version each farm appears in
    version_added = []
    for _, row in source_data_orig.iterrows():
        farm_name = row["SiteName"]
        first_ver = int(source_data_orig.loc[source_data_orig["SiteName"] == farm_name, "Version"].min())
        version_added.append(first_ver)
    source_data_orig["FarmAdded"] = version_added

    # Build full table: one row per farm per version
    records = []

    for version_index in range(1, N_versions + 1):  # MATLAB-style
        for farm_name in farms_modelled:
            k_farm = source_data_orig["SiteName"] == farm_name
            k_version = source_data_orig["Version"] == version_index
            farm_data_for_this_version = source_data_orig[k_farm & k_version]

            if farm_data_for_this_version.empty:
                # Missing farm/version -> fill logic
                first_occ = source_data_orig[k_farm].iloc[0]
                version_when_added = int(first_occ["FarmAdded"])
                biomass_for_comment = tdisp(first_occ["Biomass"])

                if version_when_added > version_index:
                    # Added later
                    biomass_this_version = 0
                    version_comment = (
                        f"No entry in spreadsheet -> biomass=0; "
                        f"(set to {biomass_for_comment} tonnes in version {version_when_added})"
                    )
                else:
                    # Possibly removed or not updated
                    zero_versions = source_data_orig.loc[k_farm & (source_data_orig["Biomass"] == 0), "Version"]
                    later_zero_versions = zero_versions[zero_versions > version_when_added]

                    if not later_zero_versions.empty:
                        biomass_this_version = 0
                        removed_in_version = int(later_zero_versions.min())
                        version_comment = (
                            f"No entry in spreadsheet; was {biomass_for_comment} tonnes, "
                            f"removed in version {removed_in_version}"
                        )
                    else:
                        biomass_this_version = first_occ["Biomass"]
                        version_comment = (
                            f"No entry in spreadsheet; using value {biomass_for_comment} tonnes "
                            f"from version {version_when_added}"
                        )

                # Fill record
                record = first_occ.copy()
                record["Biomass"] = biomass_this_version
                record["Version"] = version_index
                record["Notes"] = ""
                record["Comment"] = version_comment
            else:
                record = farm_data_for_this_version.iloc[0].copy()
                record["Comment"] = ""

            records.append(record)

    source_data = pd.DataFrame(records)

    # Cache result
    _source_data_cache = source_data.copy()

    print(f"Generated sourceData with {len(source_data)} rows "
          f"({N_farms} farms × {N_versions} versions expected = {N_farms * N_versions})")

    return source_data

# module-level cache (like MATLAB persistent)
_source_data_cache = None

def clear_source_term_cache():
    """Clear the cached source table (like `clear` in MATLAB)."""
    global _source_data_cache
    _source_data_cache = None

def source_term_version_data(*args, verbose: bool = False, force_reload: bool = False):
    """
    Retrieve or compare source-term data.

    Args:
        *args: strings (model/site names or iterables of them), numeric versions,
               and optionally the string 'compare'.
        verbose: if True, print debug info
        force_reload: if True, regenerate source table (ignore cache)

    Returns:
        pd.DataFrame: filtered table (or comparison result)
    """
    global _source_data_cache

    # -------------------------
    # Load / cache source table
    # -------------------------
    if force_reload or _source_data_cache is None:
        if verbose:
            print("Loading sourceData from spreadsheet (this may take ~10s)...")
        _source_data_cache = generate_source_term_table(force_reload=force_reload)
        if verbose:
            n = len(_source_data_cache)
            n_versions = int(_source_data_cache["Version"].max())
            n_sites = _source_data_cache["SiteName"].nunique()
            print(f"Generated sourceData with {n} rows "
                  f"({n_sites} farms × {n_versions} versions expected = {n_sites*n_versions})")
    else:
        if verbose:
            print("Using cached sourceData")

    source_data = _source_data_cache.copy()
    N_versions = int(source_data["Version"].max())

    # -------------------------
    # Parse args
    # -------------------------
    args_list = list(args)

    # Pull out 'compare' marker if present
    compare = False
    flat_strings = []
    numeric_args = []

    for a in args_list:
        # 'compare' can be a string anywhere
        if isinstance(a, str) and a.lower() == "compare":
            compare = True
            continue

        # numeric (int/float)
        if isinstance(a, (int, float, np.integer, np.floating)):
            numeric_args.append(a)
            continue

        # iterables of strings (list, dict_keys, tuple, etc.)
        if isinstance(a, Iterable) and not isinstance(a, (str, bytes, bytearray)):
            for item in a:
                if isinstance(item, str):
                    flat_strings.append(item)
            continue

        # string scalar
        if isinstance(a, str):
            flat_strings.append(a)

    # numeric -> versions
    if numeric_args:
        version_inputs = numeric_args.copy()
    else:
        version_inputs = [1]  # default

    # handle inf / 0 / negative
    resolved_versions = []
    for v in version_inputs:
        if np.isinf(v):
            # all versions 1..N_versions
            resolved_versions.extend(list(range(1, N_versions + 1)))
        else:
            iv = int(v)
            if iv == 0:
                resolved_versions.append(N_versions)
            elif iv < 0:
                # -1 => second last => N_versions + (-1) = N_versions-1
                resolved_versions.append(N_versions + iv)
            else:
                resolved_versions.append(iv)
    # unique + keep order
    seen = set()
    version_index = []
    for vv in resolved_versions:
        if vv not in seen and 1 <= vv <= N_versions:
            seen.add(vv)
            version_index.append(vv)

    if verbose:
        print("Requested versions (raw):", numeric_args)
        print("Normalized versions (MATLAB-style):", version_index)
        print("String inputs (flattened):", flat_strings)
        print("Compare flag:", compare)

    # -------------------------
    # Determine site/model filter
    # -------------------------
    # If no string inputs, don't filter by site/model
    if not flat_strings:
        k_source = pd.Series(True, index=source_data.index)
        if verbose:
            print("No model/site filters provided — selecting all sites.")
    else:
        # First attempt: exact model match (case-insensitive)
        unique_models = source_data["Model"].dropna().unique()
        matched_models = []
        for q in flat_strings:
            for m in unique_models:
                try:
                    if str(q).strip().lower() == str(m).strip().lower():
                        matched_models.append(m)
                except Exception:
                    pass

        matched_models = list(dict.fromkeys(matched_models))  # unique preserve order
        if matched_models:
            # Filter by Model
            k_source = source_data["Model"].isin(matched_models)
            if verbose:
                print(f"Matched model(s): {matched_models} -> {k_source.sum()} rows")
        else:
            # No model matches -> fallback to site-name matching via closest_string_match
            candidates = source_data["SiteName"].astype(str).tolist()
            # closest_string_match expects candidates and a query; support passing many queries
            matches = []
            for q in flat_strings:
                found = closest_string_match(candidates, q)
                # If the helper returns empty but q exactly equals a candidate (unlikely) include it:
                if (not found) and (q in candidates):
                    found = [q]
                matches.extend(found)
            matches = list(dict.fromkeys(matches))  # unique preserve order
            if not matches:
                if verbose:
                    print(f"⚠️ Warning: no matches found for {flat_strings}")
                return pd.DataFrame()  # nothing matched
            k_source = source_data["SiteName"].isin(matches)
            if verbose:
                print(f"Matched site(s): {len(matches)} -> {matches[:6]}{'...' if len(matches)>6 else ''}")

    # -------------------------
    # Build output across requested versions
    # -------------------------
    frames = []
    for vv in version_index:
        frames.append(source_data[k_source & (source_data["Version"] == vv)].copy())
        if verbose:
            print(f"Version {vv}: selected {len(frames[-1])} rows")
    if frames:
        op = pd.concat(frames, ignore_index=True)
    else:
        op = pd.DataFrame(columns=source_data.columns)

    # -------------------------
    # Comparison mode
    # -------------------------
    if compare:
        if len(version_index) != 2:
            raise ValueError("Comparison requires exactly two numeric version arguments (e.g., 1, 0, 'compare').")
        v1, v2 = version_index

        s1 = op[op["Version"] == v1].copy()
        s2 = op[op["Version"] == v2].copy()

        # If user filtered to a subset of sites/models, s1/s2 reflect that subset.

        # Columns to ignore in comparisons
        ignore_cols = {"Version", "Notes", "Comment"}

        # columns to compare (preserve order)
        compare_cols = [c for c in s1.columns if c not in ignore_cols]

        # merge by SiteName (we assume SiteName is unique per farm)
        merged = pd.merge(s1, s2, on="SiteName", how="left", suffixes=("_v1", "_v2"), indicator=True)

        # Identify SiteNames where:
        # - row exists only in v1 (left_only) -> include
        # - or exists in both but any compare column differs -> include
        # Build mask:
        left_only_mask = merged["_merge"] == "left_only"

        # For rows present in both, check if any compare column differs.
        both_mask = merged["_merge"] == "both"
        diff_mask_both = pd.Series(False, index=merged.index)
        for col in compare_cols:
            col_v1 = f"{col}_v1"
            col_v2 = f"{col}_v2"
            if col_v1 in merged.columns and col_v2 in merged.columns:
                a = merged[col_v1].fillna("__NaN__")
                b = merged[col_v2].fillna("__NaN__")
                diff_mask_both |= (a != b)

        # final mask:
        final_mask = left_only_mask | (both_mask & diff_mask_both)

        # return the rows from s1 corresponding to those SiteNames (preserve s1 columns)
        site_names_to_return = merged.loc[final_mask, "SiteName"].tolist()
        result = s1[s1["SiteName"].isin(site_names_to_return)].copy()

        # Deduplicate if necessary (keep first)
        result = result.drop_duplicates(subset=["SiteName"], keep="first").reset_index(drop=True)

        if verbose:
            print(f"Comparison: returning {len(result)} differing rows (ignores Version/Notes/Comment)")

        return result[compare_cols]

    return op.reset_index(drop=True)
