import numpy as np

def summary_statistics(x, stat_functions=None):
    """
    Python version of MATLAB summaryStatistics.m (simplified).
    Uses numpy.quantile for q95 (which matches quantileSEPA behaviour).
    """

    # 1. Flatten and remove NaNs
    x = np.asarray(x).reshape(-1)
    x = x[~np.isnan(x)]

    # 2. Default stat functions
    if stat_functions is None:
        stat_functions = ['length', 'mean', 'median', 'std', 'min', 'max']

    stats = {}

    # 3. Compute basic statistics
    for fn in stat_functions:
        if fn == 'length':
            val = len(x)
        elif fn == 'mean':
            val = np.nanmean(x) if x.size > 0 else np.nan
        elif fn == 'median':
            val = np.nanmedian(x) if x.size > 0 else np.nan
        elif fn == 'std':
            val = np.nanstd(x) if x.size > 0 else np.nan
        elif fn == 'min':
            val = np.nanmin(x) if x.size > 0 else np.nan
        elif fn == 'max':
            val = np.nanmax(x) if x.size > 0 else np.nan
        else:
            raise ValueError(f"Unknown statistic '{fn}'")

        stats[fn] = val

    # 4. q95 using numpy.quantile
    try:
        stats['q95'] = np.quantile(x, 0.95) if x.size > 0 else np.nan
    except Exception:
        stats['q95'] = np.nan

    # 5. Replace NaN outputs with 0 (your MATLAB behaviour)
    for key in stats:
        if isinstance(stats[key], float) and np.isnan(stats[key]):
            stats[key] = 0.0

    return stats
