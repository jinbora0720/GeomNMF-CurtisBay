import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import seaborn as sns
from itertools import permutations
from scipy.stats import gaussian_kde
import math
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.dates import AutoDateLocator, AutoDateFormatter
import statsmodels.formula.api as smf
from matplotlib.colors import Normalize
from itertools import cycle
from matplotlib import colors as mcolors
from typing import Dict, Any, Tuple

########
# Plot #
########
def plot_timeseries_facets(
    df,
    cols,
    time_col,                       # REQUIRED: datetime column in df
    ncols=3,
    figsize_per_panel=(6, 2.6),
    sharex=True,
    sharey=False,
    lw=1.5,
    alpha=1.0,
    xrotation=0,
    title=None,
    panel_title=None,               # dict/list/callable/format string or None
    ylabel="Value",
    # --- grouping (multi-line or faceting) ---
    covariate=None,                 # str column in df or 1-D array-like (len = len(df))
    legend_title="Group",
    label_map=None,                 # {raw_value: "Nice label"}
    group_order=None,               # iterable to control order
    facet_by_group=False,           # NEW: if True, make panels per (col, group)
    # --- highlight overlay ---
    highlight=None,                 # (start,end) | [(s,e), ...] | boolean mask (len = len(df))
    highlight_color="red",          # str OR list[str] with same length as intervals
    highlight_lw=1.5,
    highlight_alpha=1.0,
    savepath=None,
):
    """
    Faceted time-series with optional grouping and highlight overlay.

    Modes
    -----
    - covariate is None: single line per panel (one panel per 'col').
    - covariate provided, facet_by_group=False (default): one panel per 'col', multiple lines by group.
    - covariate provided, facet_by_group=True: one panel per (col, group).

    highlight:
      - (start,end) or [(s,e), ...] for time intervals
      - boolean mask (len = len(df)) to emphasize samples
    """
    import numpy as np, pandas as pd, math
    import matplotlib.pyplot as plt
    from matplotlib.dates import AutoDateLocator, AutoDateFormatter

    # normalize columns and checks
    if isinstance(cols, str):
        cols = [cols]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in df: {missing}")
    if time_col not in df.columns:
        raise KeyError(f"time_col '{time_col}' not in df")

    # --- PRESERVE TZ: build index from the Series (not .values) ---
    t = pd.DatetimeIndex(pd.to_datetime(df[time_col]), copy=False)
    data = df.set_index(t)[cols]

    # facet title getter (for col name only)
    def get_title_for_col(col):
        ft = panel_title
        if ft is None: return col
        if callable(ft): return str(ft(col))
        if isinstance(ft, dict): return str(ft.get(col, col))
        if isinstance(ft, (list, tuple)):
            if len(ft) != len(cols):
                raise ValueError("panel_title list/tuple must match len(cols).")
            return str(ft[cols.index(col)])
        if isinstance(ft, str):
            try:    return ft.format(col=col)
            except: return ft
        return col

    # prepare grouping vector (if any)
    groups = None
    if covariate is not None:
        if isinstance(covariate, str):
            if covariate not in df.columns:
                raise KeyError(f"covariate '{covariate}' not found in df")
            groups = pd.Series(df[covariate].values, index=data.index, name="group")
        else:
            groups = pd.Series(np.asarray(covariate), index=data.index, name="group")
        if len(groups) != len(data):
            raise ValueError("covariate length must match number of rows in df")

        uniq = pd.unique(groups.dropna())
        if group_order is None:
            group_order = list(uniq)
        else:
            group_order = [g for g in group_order if g in set(uniq)]

        if label_map is None:
            label_map = {g: str(g) for g in group_order}
        else:
            label_map = {g: label_map.get(g, str(g)) for g in group_order}

    # prepare highlight mode
    highlight_intervals = None
    highlight_mask = None
    if highlight is not None:
        if isinstance(highlight, tuple) and len(highlight) == 2:
            highlight_intervals = [highlight]
        elif isinstance(highlight, list) and all(isinstance(x, tuple) and len(x) == 2 for x in highlight):
            highlight_intervals = highlight
        else:
            highlight_mask = np.asarray(highlight, dtype=bool)
            if highlight_mask.shape[0] != len(data):
                raise ValueError("Boolean highlight mask must match number of rows in df.")
                
    # normalize highlight_color -> list for intervals; keep scalar for mask
    if highlight_intervals is not None:
        if isinstance(highlight_color, (list, tuple)):
            if len(highlight_color) != len(highlight_intervals):
                raise ValueError(
                    "When highlight is a list of intervals, highlight_color must be a single "
                    "color or a list of the same length."
                )
            highlight_colors = list(highlight_color)
        else:
            highlight_colors = [highlight_color] * len(highlight_intervals)
    else:
        highlight_colors = [highlight_color]  # used only for boolean mask case
        
    # --- Build panel specs ----------------------------------------------------
    # Each panel is described by dict: {"col": <col>, "group": <g or None>}
    panel_specs = []
    if groups is None or not facet_by_group:
        # one panel per col
        for col in cols:
            panel_specs.append({"col": col, "group": None})
    else:
        # NEW: one panel per (col, group)
        for col in cols:
            for g in group_order:
                panel_specs.append({"col": col, "group": g})

    # layout from panel count
    nplots = len(panel_specs)
    nrows = math.ceil(nplots / ncols)
    fig_w = figsize_per_panel[0] * ncols
    fig_h = figsize_per_panel[1] * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h),
                             sharex=sharex, sharey=sharey)
    axes = np.array(axes, ndmin=1).reshape(nrows, ncols)

    locator = AutoDateLocator()
    formatter = AutoDateFormatter(locator)

    def _to_local_naive(ts, tz="America/New_York"):
        # ts can be Timestamp or scalar-like; convert robustly
        ts = pd.to_datetime(ts)
        if ts.tz is None:
            return ts  # naive, assumed local
        return ts.tz_convert(tz).tz_localize(None)

    disp_tz = "America/New_York"

    for k, spec in enumerate(panel_specs):
        col = spec["col"]
        g_for_panel = spec["group"]  # None or actual group value

        r, c = divmod(k, ncols)
        ax = axes[r, c]
        y = data[col]

        # --- plotting index in local-naive time ---
        if y.index.tz is None:
            idx4plot = y.index
        else:
            idx4plot = y.index.tz_convert(disp_tz).tz_localize(None)

        # --- intervals to local-naive ---
        intervals_local = None
        if highlight_intervals is not None:
            intervals_local = []
            for (s, e) in highlight_intervals:
                s_local = _to_local_naive(s, disp_tz)
                e_local = _to_local_naive(e, disp_tz)
                intervals_local.append((s_local, e_local))

        yvals = y.to_numpy()

        if groups is None:
            # no grouping at all
            ax.plot(idx4plot, yvals, lw=lw, alpha=alpha, color="gray")

            # highlight overlay
            if intervals_local is not None:
                for (i, (s, e)) in enumerate(intervals_local):
                    m = (idx4plot >= s) & (idx4plot <= e)
                    if m.any():
                        ax.plot(idx4plot[m], yvals[m],
                                lw=highlight_lw, alpha=highlight_alpha, color=highlight_colors[i])
            elif highlight_mask is not None:
                m = np.asarray(highlight_mask, dtype=bool)
                if m.any():
                    ax.plot(idx4plot[m], yvals[m],
                            lw=highlight_lw, alpha=highlight_alpha, color=highlight_colors[0])

        else:
            if facet_by_group:
                # NEW: one line per panel (filtered to a single group)
                m_g = (groups == g_for_panel).to_numpy()
                if np.any(m_g):
                    y_plot = np.where(m_g, yvals, np.nan)
                    ax.plot(idx4plot, y_plot, lw=lw, alpha=alpha, color="gray")

                    # highlight for this group's samples
                    if intervals_local is not None:
                        for (i, (s, e)) in enumerate(intervals_local):
                            m_se = m_g & (idx4plot >= s) & (idx4plot <= e)
                            if m_se.any():
                                ax.plot(idx4plot[m_se], yvals[m_se],
                                        lw=highlight_lw, alpha=highlight_alpha, color=highlight_colors[i])
                    elif highlight_mask is not None:
                        m_se = m_g & np.asarray(highlight_mask, dtype=bool)
                        if m_se.any():
                            ax.plot(idx4plot[m_se], yvals[m_se],
                                    lw=highlight_lw, alpha=highlight_alpha, color=highlight_colors[0])
                # no legend in facet-by-group mode (panel title names the group)

            else:
                # original: multi-line by group in the same panel
                for gval in group_order:
                    m_g = (groups == gval).to_numpy()
                    if not np.any(m_g):
                        continue
                    y_plot = np.where(m_g, yvals, np.nan)
                    ax.plot(idx4plot, y_plot, lw=lw, alpha=alpha,
                            label=label_map.get(gval, str(gval)))

                    # highlight for this group
                    if intervals_local is not None:
                        for (i, (s, e)) in enumerate(intervals_local):
                            m_se = m_g & (idx4plot >= s) & (idx4plot <= e)
                            if m_se.any():
                                ax.plot(idx4plot[m_se], yvals[m_se],
                                        lw=highlight_lw, alpha=highlight_alpha, color=highlight_colors[i])
                    elif highlight_mask is not None:
                        m_se = m_g & np.asarray(highlight_mask, dtype=bool)
                        if m_se.any():
                            ax.plot(idx4plot[m_se], yvals[m_se],
                                    lw=highlight_lw, alpha=highlight_alpha, color=highlight_colors[0])

                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(title=legend_title, frameon=False, fontsize=8)

        # --- titles & cosmetics
        base_title = get_title_for_col(col)
        if g_for_panel is None:
            ax.set_title(base_title, fontsize=14)
        else:
            ax.set_title(f"{base_title}: {label_map.get(g_for_panel, str(g_for_panel))}", fontsize=14)

        ax.grid(True, linestyle=":", alpha=0.4)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        for lab in ax.get_xticklabels():
            lab.set_rotation(xrotation)
            lab.set_fontsize(11)
        if c == 0:
            ax.set_ylabel(ylabel, fontsize=13)
        ax.tick_params(axis='y', labelsize=11)

    # hide unused panels
    for k in range(nplots, nrows * ncols):
        r, c = divmod(k, ncols)
        axes[r, c].axis("off")

    if title:
        fig.suptitle(title, y=0.995, fontsize=14)

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=200)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

    
def plot_diurnal_W(
    W, 
    timestamps, 
    columnidx=1,
    covariate=None,                  # array/Series of group labels (len n) or None
    agg="median", 
    quantiles=(0.25, 0.75),
    figsize=(7, 4), 
    title=None, 
    ylabel=None, 
    legend_title="Group", 
    label_map=None,        # {raw_value: "Nice label"}
    color_map=None,        # {raw_value: "color"}
    linestyle_map=None,    # {raw_value: "linestyle"}
    point_map=None,        # {raw_value: "marker"}
    savepath=None,
):
    """
    Plot diurnal cycle of one column of W with optional grouping and quantile shading.

    W : ndarray (n, K)  -- rows are observations, columns are sources
    timestamps : pd.Series or array-like of datetimes, length n
    columnidx : int   -- which column of W to plot (0-based)
    covariate : array-like | Series | None
        Optional grouping variable (same length as n). If None, a single overall line is shown.
    agg : {'mean','median'}
    quantiles : tuple[float,float] or None -- e.g., (0.25, 0.75) for IQR shading; None = no band
    """
    import numpy as np, pandas as pd, matplotlib.pyplot as plt

    W = np.asarray(W)
    n, K = W.shape
    if not (0 <= columnidx < K):
        raise ValueError("column idx out of range for W")

    # timestamps -> DatetimeIndex
    t = pd.Series(pd.to_datetime(timestamps))
    if len(t) != n:
        raise ValueError("timestamps length must match number of rows in W")

    hour = t.dt.hour

    if covariate is None:
        dfh = pd.DataFrame({
            "value": W[:, columnidx],
            "hour":  hour,
            "group": "All"
        })
    else:
        dfh = pd.DataFrame({
            "value": W[:, columnidx],
            "hour":  hour,
            "group": np.asarray(covariate)
        })

    # aggregate
    g = dfh.groupby(["group", "hour"])["value"]
    center = g.mean() if agg == "mean" else g.median()

    q_lo = q_hi = None
    if quantiles is not None:
        q_lo = g.quantile(quantiles[0])
        q_hi = g.quantile(quantiles[1])

    # reshape to wide format
    hours = pd.Index(range(24), name="hour")
    center_wide = center.unstack("group").reindex(hours)

    if q_lo is not None:
        qlo_wide = q_lo.unstack("group").reindex(hours)
        qhi_wide = q_hi.unstack("group").reindex(hours)
    else:
        qlo_wide = qhi_wide = None

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(24)

    for gval in center_wide.columns:
        label = label_map.get(gval, str(gval)) if label_map is not None else str(gval)
        y = center_wide[gval].values

        # aesthetics
        linestyle = linestyle_map.get(gval, "-") if linestyle_map is not None else "-"
        marker = point_map.get(gval, "o") if point_map is not None else "o"

        line, = ax.plot(x, y, marker=marker, label=label,
                        linestyle=linestyle,
                        color=color_map.get(gval) if color_map is not None and gval in color_map else None)
        line_color = line.get_color()  

        # quantile shading
        if qlo_wide is not None and gval in qlo_wide.columns:
            ax.fill_between(
                x, qlo_wide[gval].values, qhi_wide[gval].values,
                alpha=0.2, linewidth=0, color=line_color
            )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h:02d}" for h in x])
    ax.set_xlabel("Hour of day")
    ax.set_ylabel(ylabel or f"W[:, {columnidx}]")
    ax.grid(True, linestyle=":", alpha=0.4)
    if covariate is not None: 
        ax.legend(title=legend_title, frameon=False)
    ax.set_title(title or f"Diurnal pattern of W[:, {columnidx}]")
    plt.tight_layout()

    if savepath: 
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()

    
def scatter_matrix_W_with_corr(
    W,
    covariate=None,
    colnames=None,
    figsize=(8, 8),
    s=10,
    alpha=0.7,
    diag_top=3,
    hue_labels=("0", "1"),
    tol=1e-8,
    legend=True,
    legend_title="Group",
):
    """
    Scatter-matrix for W (n x K):
      - lower triangle: scatter
      - diagonal: top-k correlations of each variable with the others (boxed text)
      - upper triangle: blank, EXCEPT (1,K) cell (top-right) hosts the legend
    A numeric covariate with values ~0/~1 (within tol) is treated as categorical {0,1}.
    """

    W = np.asarray(W)
    if W.ndim != 2:
        raise ValueError("W must be a 2D array (n x K).")
    n, K = W.shape
    if K < 2:
        raise ValueError("Need at least 2 columns (K >= 2).")

    # Column names
    if colnames is None:
        colnames = [f"Source {i}" for i in range(1, K + 1)]
    if len(colnames) != K:
        raise ValueError("len(colnames) must equal number of columns in W.")

    # Prepare categorical hue (0/1)
    cat = None
    if covariate is not None:
        cov = np.asarray(covariate).reshape(-1)
        if cov.shape[0] != n:
            raise ValueError("covariate length must match number of rows in W.")
        if np.issubdtype(cov.dtype, np.number):
            is0 = np.isfinite(cov) & (np.abs(cov - 0.0) <= tol)
            is1 = np.isfinite(cov) & (np.abs(cov - 1.0) <= tol)
            cat = np.full(n, np.nan, dtype=float)
            cat[is0] = 0.0
            cat[is1] = 1.0
        else:
            # treat non-numeric labels as strings and map to {0,1} by hue_labels
            m = {str(hue_labels[0]): 0.0, str(hue_labels[1]): 1.0}
            cat = np.array([m.get(str(x), np.nan) for x in cov], dtype=float)

    # colors + labels for categories
    colors = {0.0: "#1f77b4", 1.0: "#ff7f0e"}  # C0, C1
    labels = {0.0: str(hue_labels[0]), 1.0: str(hue_labels[1])}

    # Pairwise correlations (pairwise complete)
    def pairwise_corr(X):
        P = X.shape[1]
        C = np.full((P, P), np.nan, dtype=float)
        for i in range(P):
            xi = X[:, i]
            for j in range(P):
                xj = X[:, j]
                m = np.isfinite(xi) & np.isfinite(xj)
                if m.sum() >= 2:
                    C[i, j] = np.corrcoef(xi[m], xj[m])[0, 1]
        return C

    corr = pairwise_corr(W)

    fig, axes = plt.subplots(K, K, figsize=figsize)

    for i in range(K):
        for j in range(K):
            ax = axes[i, j]
            x = W[:, j]
            y = W[:, i]

            if i > j:
                # LOWER TRIANGLE: scatter
                if covariate is None:
                    m = np.isfinite(x) & np.isfinite(y)
                    ax.scatter(x[m], y[m], s=s, alpha=alpha)
                else:
                    # plot each category separately; drop NaN
                    for c in (0.0, 1.0):
                        m = np.isfinite(x) & np.isfinite(y) & (cat == c)
                        if np.any(m):
                            ax.scatter(x[m], y[m], s=s, alpha=alpha, c=colors[c], label=labels[c])
            elif i == j:
                # DIAGONAL: top-k correlations (boxed panel, no ticks)
                r_vec = corr[i, :].copy()
                r_vec[i] = np.nan  # drop self
                order = np.argsort(np.nan_to_num(np.abs(r_vec), nan=-np.inf))[::-1]
                order = [idx for idx in order if np.isfinite(r_vec[idx])]
                if diag_top is not None:
                    order = order[:diag_top]

                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
                for sp in ax.spines.values():
                    sp.set_visible(True)

                ax.text(0.5, 0.85, colnames[i], ha='center', va='center',
                        fontsize=10, fontweight='bold', transform=ax.transAxes)

                if len(order) == 0:
                    ax.text(0.5, 0.5, "no corr", ha='center', va='center',
                            fontsize=9, transform=ax.transAxes)
                else:
                    y0 = 0.70
                    dy = 0.12 if len(order) <= 3 else 0.08
                    for k2, j2 in enumerate(order):
                        ax.text(0.5, y0 - k2*dy,
                                f"{colnames[j2]}: r = {r_vec[j2]:.2f}",
                                ha='center', va='center', fontsize=9, transform=ax.transAxes)
            else:
                # UPPER TRIANGLE: blank by default
                ax.axis('off')

            # Ticks/labels: only bottom row / left column
            if i < K - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(colnames[j])
            if j > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(colnames[i])

    # Put legend on the (1, K) cell (top-right), boxed
    if legend and (covariate is not None):
        ax_leg = axes[0, K-1]
        # make it a boxed, empty panel
        ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, 1)
        ax_leg.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for sp in ax_leg.spines.values():
            sp.set_visible(True)

        # build proxy handles for present categories only
        present = []
        if cat is not None:
            for c in (0.0, 1.0):
                if np.any(cat == c):
                    present.append(c)
        # fallback: show both
        if not present:
            present = [0.0, 1.0]

        handles = [Line2D([], [], marker='o', linestyle='', color=colors[c],
                          markersize=6, alpha=alpha) for c in present]
        lablist = [labels[c] for c in present]
        ax_leg.legend(handles, lablist, title=legend_title, loc='center', frameon=False)

    plt.tight_layout()
    plt.show()

# functions for plotting 
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def plot_full_scatter_matrix(
    Y,
    H=None,
    H_hat=None,
    scatter_color="grey",
    alpha=0.5,
    title='Scatter Matrix',
    cvals=None,
    cmap='viridis',
    colorbar=False,
    vmin=None,
    vmax=None,
    point_size=30,
    labels=None,
    quantile=0.85,
    draw_diagonal=True,
    savepath=None,
    convexhull=True,
    # --- NEW ---
    hhat_groups=None,          # length-K labels for rows of H_hat
    shape_map=None,            # dict: group -> marker (e.g., '^','s','D','P','X','*','v','o')
    color_map=None,            # dict: group -> color (any matplotlib color)
    hhat_size=110,
    show_legend=True,
    draw_upper_corr=False,      # draw correlation in upper-diagonal cells
    corr_fmt="{:.2f}",         # format for the correlation text
    corr_fontsize=12,          # font size for the correlation text
):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial import ConvexHull

    to_arr = lambda A: (A.to_numpy() if hasattr(A, "to_numpy") else np.asarray(A)) if A is not None else None

    Y_arr    = to_arr(Y)
    H_arr    = to_arr(H)
    Hhat_arr = to_arr(H_hat)

    if Y_arr.ndim != 2:
        raise ValueError("Y must be two dimensional")
    n, J = Y_arr.shape

    Y_arr = Y_arr.astype(float, copy=False)
    Y_arr[~np.isfinite(Y_arr)] = np.nan
    if H_arr is not None:
        H_arr = H_arr.astype(float, copy=False)
        H_arr[~np.isfinite(H_arr)] = np.nan
    if Hhat_arr is not None:
        Hhat_arr = Hhat_arr.astype(float, copy=False)
        Hhat_arr[~np.isfinite(Hhat_arr)] = np.nan
        K_hat = Hhat_arr.shape[0]
        # default groups: 0..K_hat-1
        if hhat_groups is None:
            hhat_groups = np.arange(K_hat)
        else:
            hhat_groups = np.asarray(hhat_groups)
            if hhat_groups.shape[0] != K_hat:
                raise ValueError("hhat_groups must have length equal to number of rows in H_hat")

    # labels
    if labels is None:
        labels = [str(c) for c in (Y.columns if hasattr(Y, "columns") else range(1, J+1))]
    elif len(labels) != J:
        raise ValueError("labels length must equal number of columns in Y")

    # color values
    mappable = None
    numeric_cvals = False
    if cvals is not None:
        cvals = np.asarray(cvals)
        if cvals.ndim != 1 or len(cvals) != n:
            raise ValueError("cvals must be one dimensional with length equal to rows of Y")
        numeric_cvals = np.issubdtype(cvals.dtype, np.number)
        if numeric_cvals:
            cvals = cvals.astype(float, copy=False)
            cvals[~np.isfinite(cvals)] = np.nan
            if np.unique(cvals[~np.isnan(cvals)]).size == 2:
                cmap = 'bwr'
                vmin = np.nanmin(cvals); vmax = np.nanmax(cvals)

    fig, axes = plt.subplots(J, J, figsize=(3.5 * J, 3.5 * J))
    fig.suptitle(title, fontsize=20)

    qvals = None
    if quantile is not None:
        qvals = np.nanquantile(Y_arr, quantile, axis=0)

    # --- defaults for maps (used only if keys missing) ---
    default_shapes = ['^','s','D','P','X','*','v','o','<','>']
    default_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['C0','C1','C2','C3','C4','C5','C6','C7'])
    shape_map = {} if shape_map is None else dict(shape_map)
    color_map = {} if color_map is None else dict(color_map)

    # legend handles (collect once per pair so we don't duplicate)
    legend_added = set()

    for i in range(J):
        for j in range(J):
            ax = axes[i, j]

            if i < j:
                if not draw_upper_corr:
                    ax.axis('off')
                    continue
            
                # Upper triangle: show Pearson r centered
                x = Y_arr[:, j]; y = Y_arr[:, i]
                m = np.isfinite(x) & np.isfinite(y)
            
                # Default look for this cell
                ax.set_xlim(0,1)
                ax.set_ylim(0,1)
                ax.set_xlabel(labels[j], fontsize=18)
                ax.set_ylabel(labels[i], fontsize=18)
                ax.set_frame_on(True)
                ax.set_xticks([]); ax.set_yticks([])

                if m.sum() >= 3 and np.nanstd(x[m]) > 0 and np.nanstd(y[m]) > 0:
                    r = np.corrcoef(x[m], y[m])[0, 1]
                    s = corr_fmt.format(r) if isinstance(corr_fmt, str) else f"{r:{corr_fmt}}"
                else:
                    s = "NA"
            
                kw = dict(ha='center', va='center', fontsize=corr_fontsize, fontweight='bold')
                ax.text(0.5, 0.5, s, transform=ax.transAxes, **kw)
                continue


            if i == j:
                if not draw_diagonal:
                    ax.axis('off'); continue
                col = Y_arr[:, j]
                col = col[np.isfinite(col)]
                if col.size:
                    ax.hist(col, bins=30, color='grey', alpha=0.7, density=True)
                # ax.set_xlabel(labels[j], fontsize=18)
                if j == J-1:
                    ax.set_xlabel(labels[j], fontsize=18)
                ax.set_ylabel("Density", fontsize=18)
                ax.tick_params(axis='both', labelsize=17)
                if qvals is not None and np.isfinite(qvals[j]):
                    ax.axvline(qvals[j], color='red', linestyle='--', linewidth=1.5)
                continue

            # off-diagonal scatter
            x = Y_arr[:, j]; y = Y_arr[:, i]
            mask = np.isfinite(x) & np.isfinite(y)

            if cvals is None:
                ax.scatter(x[mask], y[mask], s=point_size, color=scatter_color, alpha=alpha, zorder=1)
            else:
                if numeric_cvals:
                    cm = cvals[mask]
                    sc = ax.scatter(x[mask], y[mask], s=point_size, c=cm, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax, zorder=1)
                    if mappable is None: mappable = sc
                else:
                    ax.scatter(x[mask], y[mask], s=point_size, color=np.asarray(cvals)[mask], alpha=alpha, zorder=1)

            # convex hull
            if convexhull:
                try:
                    pts = np.ascontiguousarray(np.c_[x[mask], y[mask]])
                    if pts.shape[0] >= 3:
                        hull = ConvexHull(pts)
                        for simplex in hull.simplices:
                            ax.plot(pts[simplex, 0], pts[simplex, 1], color='black', linewidth=1, zorder=2)
                except Exception:
                    pass

            # overlay H
            if H_arr is not None:
                ax.scatter(H_arr[:, j], H_arr[:, i], s=90, color='red', marker='o', label='_H', zorder=3)  # label hidden

            # overlay H_hat with per-group shape/color
            if Hhat_arr is not None:
                uniq = np.unique(hhat_groups)
                for idx_g, g in enumerate(uniq):
                    sel = (hhat_groups == g)
                    if not np.any(sel):
                        continue
                    # choose shape/color with fallbacks
                    marker = shape_map.get(g, default_shapes[idx_g % len(default_shapes)])
                    color  = color_map.get(g,  default_colors[idx_g % len(default_colors)])
                    sc = ax.scatter(
                        Hhat_arr[sel, j], Hhat_arr[sel, i],
                        s=hhat_size, marker=marker,
                        facecolors=color, edgecolors="black", linewidths=1.0,
                        alpha=1.0, label=f"Ĥ: {g}", zorder=4
                    )
                    # add one legend entry per group overall (avoid duplicates)
                    legend_key = (g, marker, color)
                    if show_legend and legend_key not in legend_added:
                        legend_added.add(legend_key)

            # labels & guides
            # ax.set_xlabel(labels[j], fontsize=18)
            # ax.set_ylabel(labels[i], fontsize=18)
            ax.tick_params(axis='both', labelsize=17)

            if i == J - 1:
                ax.set_xlabel(labels[j], fontsize=18)
            else:
                ax.set_xlabel("")
            
            if j == 0:
                ax.set_ylabel(labels[i], fontsize=18)
            else:
                ax.set_ylabel("")

            if qvals is not None:
                if np.isfinite(qvals[j]): ax.axvline(qvals[j], color='red', linestyle='--', linewidth=1)
                if np.isfinite(qvals[i]): ax.axhline(qvals[i], color='red', linestyle='--', linewidth=1)

    if colorbar and (mappable is not None):
        fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)

    # build a single legend from the first subplot with entries we added
    if show_legend and Hhat_arr is not None and len(legend_added) > 0:
        # make proxy artists for unique groups
        from matplotlib.lines import Line2D
        proxies = []
        labels_ = []
        for g, marker, color in sorted(legend_added, key=lambda t: str(t[0])):
            proxies.append(Line2D([0],[0], marker=marker, color='none',
                                  markerfacecolor=color, markeredgecolor="black",
                                  markersize=np.sqrt(hhat_size/np.pi), linestyle=''))
            labels_.append(f"Ĥ: {g}")
        axes[0,0].legend(proxies, labels_, loc='upper right', frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=200)
    plt.show()


# def plot_full_scatter_matrix(
#     Y,
#     H=None,
#     H_hat=None,
#     alpha=0.5,
#     title='Scatter Matrix',
#     cvals=None,
#     cmap='viridis',
#     colorbar=False,
#     vmin=None,
#     vmax=None,
#     point_size=30,
#     labels=None,
#     quantile=0.85,
#     draw_diagonal=True,    
#     savepath=None,
#     convexhull=True,
# ):
#     # --- helpers ---
#     to_arr = lambda A: (A.to_numpy() if hasattr(A, "to_numpy") else np.asarray(A)) if A is not None else None

#     # convert & sanitize inputs (inf -> nan)
#     Y_arr   = to_arr(Y)
#     H_arr   = to_arr(H)
#     Hhat_arr= to_arr(H_hat)

#     if Y_arr.ndim != 2:
#         raise ValueError("Y must be two dimensional")
#     n, J = Y_arr.shape

#     # Replace inf with nan to avoid warnings and bad plotting
#     Y_arr = Y_arr.astype(float, copy=False)
#     Y_arr[~np.isfinite(Y_arr)] = np.nan
#     if H_arr is not None:
#         H_arr = H_arr.astype(float, copy=False)
#         H_arr[~np.isfinite(H_arr)] = np.nan
#     if Hhat_arr is not None:
#         Hhat_arr = Hhat_arr.astype(float, copy=False)
#         Hhat_arr[~np.isfinite(Hhat_arr)] = np.nan

#     # labels
#     if labels is None:
#         if hasattr(Y, "columns"):
#             labels = [str(c) for c in Y.columns]
#         else:
#             labels = [f"Y{j+1}" for j in range(J)]
#     elif len(labels) != J:
#         raise ValueError("labels length must equal number of columns in Y")

#     # color values
#     mappable = None
#     numeric_cvals = False
#     if cvals is not None:
#         cvals = np.asarray(cvals)
#         if cvals.ndim != 1 or len(cvals) != n:
#             raise ValueError("cvals must be one dimensional with length equal to rows of Y")
#         numeric_cvals = np.issubdtype(cvals.dtype, np.number)
#         # sanitize cvals as well
#         if numeric_cvals:
#             cvals = cvals.astype(float, copy=False)
#             cvals[~np.isfinite(cvals)] = np.nan
#             # special binary coloring convenience
#             if np.unique(cvals[~np.isnan(cvals)]).size == 2:
#                 cmap = 'bwr'
#                 vmin = np.nanmin(cvals)
#                 vmax = np.nanmax(cvals)

#     fig, axes = plt.subplots(J, J, figsize=(3.5 * J, 3.5 * J))
#     fig.suptitle(title, fontsize=20)

#     # compute quantiles once (nan-aware)
#     qvals = None
#     if quantile is not None:
#         qvals = np.nanquantile(Y_arr, quantile, axis=0)

#     for i in range(J):
#         for j in range(J):
#             ax = axes[i, j]

#             # upper triangle off
#             if i < j:
#                 ax.axis('off')
#                 continue

#             # diagonal
#             if i == j:
#                 if not draw_diagonal:
#                     ax.axis('off')
#                     continue
#                 # Matplotlib histogram (density) to avoid seaborn + FutureWarning
#                 col = Y_arr[:, j]
#                 col = col[np.isfinite(col)]
#                 if col.size:
#                     ax.hist(col, bins=30, color='grey', alpha=0.7, density=True)
#                 ax.set_xlabel(labels[j], fontsize=12)
#                 ax.set_ylabel("Density", fontsize=12)
#                 ax.tick_params(axis='both', labelsize=10)
#                 if qvals is not None and np.isfinite(qvals[j]):
#                     ax.axvline(qvals[j], color='red', linestyle='--', linewidth=1.5)
#                 continue

#             # off-diagonal scatter
#             x = Y_arr[:, j]
#             y = Y_arr[:, i]
#             mask = np.isfinite(x) & np.isfinite(y)

#             if cvals is None:
#                 ax.scatter(x[mask], y[mask], s=point_size, color='grey', alpha=alpha)
#             else:
#                 if numeric_cvals:
#                     cm = cvals[mask]
#                     sc = ax.scatter(
#                         x[mask], y[mask],
#                         s=point_size, c=cm, cmap=cmap,
#                         alpha=alpha, vmin=vmin, vmax=vmax
#                     )
#                     if mappable is None:
#                         mappable = sc
#                 else:
#                     # categorical colors passed directly
#                     ax.scatter(
#                         x[mask], y[mask],
#                         s=point_size, color=np.asarray(cvals)[mask],
#                         alpha=alpha
#                     )

#             # convex hull edges
#             if convexhull: 
#                 try:
#                     pts = np.ascontiguousarray(np.c_[x[mask], y[mask]])
#                     if pts.shape[0] >= 3:
#                         hull = ConvexHull(pts)
#                         for simplex in hull.simplices:
#                             ax.plot(pts[simplex, 0], pts[simplex, 1], color='black', linewidth=1)
#                 except Exception:
#                     pass

#             # overlay H and H_hat
#             if H_arr is not None:
#                 ax.scatter(H_arr[:, j], H_arr[:, i], s=90, color='red', marker='o')
#             if Hhat_arr is not None:
#                 ax.scatter(Hhat_arr[:, j], Hhat_arr[:, i], s=100, facecolors='blue', edgecolors='blue', marker="^")

#             # axis labels
#             ax.set_xlabel(labels[j], fontsize=15)
#             ax.set_ylabel(labels[i], fontsize=15)
#             ax.tick_params(axis='both', labelsize=10)

#             # quantile guidelines
#             if qvals is not None:
#                 if np.isfinite(qvals[j]):
#                     ax.axvline(qvals[j], color='red', linestyle='--', linewidth=1)
#                 if np.isfinite(qvals[i]):
#                     ax.axhline(qvals[i], color='red', linestyle='--', linewidth=1)

#     if colorbar and (mappable is not None):
#         fig.colorbar(mappable, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)

#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     if savepath:
#         fig.savefig(savepath, bbox_inches="tight", dpi=200)
#     plt.show()
    
def plot_lower_triangular_scatter(Y, H=None, H_hat=None, alpha=0.3, title='Lower Triangular Scatter Matrix'):
    J = Y.shape[1]
    fig, axes = plt.subplots(J, J, figsize=(3 * J, 3 * J))
    fig.suptitle(title, fontsize=16)

    for i in range(J):
        for j in range(J):
            ax = axes[i, j]
            if i <= j:
                ax.axis('off')
                continue

            # Main scatter
            ax.scatter(Y[:, j], Y[:, i], s=2, color='blue', alpha=alpha)

            # Convex hull projection
            try:
                pts = Y[:, [j, i]]
                hull = ConvexHull(pts)
                for simplex in hull.simplices:
                    ax.plot(pts[simplex, 0], pts[simplex, 1], color='black', linewidth=1)
            except:
                pass

            # Plot true H (red circles)
            if H is not None:
                ax.scatter(H[:, j], H[:, i], s=30, color='red', marker='o')

            # Plot estimated H_hat (green x)
            if H_hat is not None:
                ax.scatter(H_hat[:, j], H_hat[:, i], s=40, color='#39FF14', marker='o')

            # Show tick labels only on bottom row and left column
            if i == J - 1:
                ax.set_xlabel(f'$Y_{{{j+1}}}$', fontsize=12)
            else:
                ax.set_xticks([])

            if j == 0:
                ax.set_ylabel(f'$Y_{{{i+1}}}$', fontsize=12)
            else:
                ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def plot_H_comparison(
    H_true, H_hat,
    left_title=r"$H$ (matched & permuted)", 
    right_title=r"$\widehat{H}$ (matched & permuted)",
    title=None,                 # figure-level title (suptitle)
    xlabel="Features",
    ylabel="Sources",
    cmap="viridis",             # fill color (colormap)
    vmin=None, vmax=None,       # shared color scale; if None, computed from both
    cbar_label="Value",
    cbar_location="right",      # 'right', 'bottom', 'left', 'top'
    figsize=(9, 3),
    constrained_layout=True,
    show_ticks=True, 
    savepath=None,
):
    """
    Plot side-by-side comparison of true H and H_hat with a shared colorbar.

    Parameters
    ----------
    H_true : ndarray (K x J)
        True endmember matrix.
    H_hat : ndarray (K x J)
        Estimated endmember matrix, permuted/matched to H_true.
    left_title, right_title : str
        subfigure title used for left or right panel.
    title : str or None
        Figure-level suptitle.
    xlabel, ylabel : str
        Axis labels used for both panels.
    cmap : str
        Matplotlib colormap name (fill color).
    vmin, vmax : float or None
        Common color scale limits. If None, computed from both matrices.
    cbar_label : str
        Label shown on the shared colorbar.
    cbar_location : str
        Where to place the shared colorbar: 'right', 'bottom', 'left', or 'top'.
    figsize : tuple
        Figure size (width, height).
    constrained_layout : bool
        Passed to plt.subplots for better spacing.
    show_ticks : bool
        Show integer ticks/labels for both axes.
    """

    K, J = H_true.shape
    if H_hat.shape != (K, J):
        raise ValueError(f"H_hat must have shape {(K, J)}, got {H_hat.shape}.")

    # Determine shared color scale if not provided
    if vmin is None or vmax is None:
        data_min = np.nanmin([H_true.min(), H_hat.min()])
        data_max = np.nanmax([H_true.max(), H_hat.max()])
        vmin = data_min if vmin is None else vmin
        vmax = data_max if vmax is None else vmax
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
            vmin, vmax = 0.0, 1.0  # safe fallback

    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=constrained_layout)

    # Left: True H
    im0 = axes[0].imshow(H_true, aspect='auto', cmap=cmap, norm=norm)
    axes[0].set_title(left_title)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)

    # Right: Estimated H_hat
    im1 = axes[1].imshow(H_hat, aspect='auto', cmap=cmap, norm=norm)
    axes[1].set_title(right_title)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)

    # Ticks / labels
    if show_ticks:
        for ax in axes:
            ax.set_xticks(np.arange(J))
            ax.set_yticks(np.arange(K))
            ax.set_xticklabels(np.arange(1, J+1))
            ax.set_yticklabels(np.arange(1, K+1))
    else:
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

    # One shared colorbar (common legend)
    cbar = fig.colorbar(im0, ax=axes.ravel().tolist(),
                        location=cbar_location, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label)

    if title:
        fig.suptitle(title)

    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=200)
    plt.show()

# def plot_H_comparison(H_true, H_hat):
#     """
#     Plot side-by-side comparison of true H and H_hat.

#     Parameters
#     ----------
#     H_star : ndarray (K x J)
#         True endmember matrix.
#     H_star_hat_perm : ndarray (K x J)
#         Estimated endmember matrix, permuted/matched to H_star.
#     """

#     K, J = H_true.shape

#     fig, axes = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)

#     # True H
#     im0 = axes[0].imshow(H_true, aspect='auto', cmap='viridis')
#     axes[0].set_title(r"$H$ (matched & permuted)")
#     axes[0].set_xlabel("Features")
#     axes[0].set_ylabel("Sources")
#     axes[0].set_xticks(np.arange(J))
#     axes[0].set_yticks(np.arange(K))
#     axes[0].set_xticklabels(np.arange(1, J+1))
#     axes[0].set_yticklabels(np.arange(1, K+1))
#     fig.colorbar(im0, ax=axes[0])

#     # Estimated H_hat
#     im1 = axes[1].imshow(H_hat, aspect='auto', cmap='viridis')
#     axes[1].set_title(r"$\widehat{H}$ (matched & permuted)")
#     axes[1].set_xlabel("Features")
#     axes[1].set_ylabel("Sources")
#     axes[1].set_xticks(np.arange(J))
#     axes[1].set_yticks(np.arange(K))
#     axes[1].set_xticklabels(np.arange(1, J+1))
#     axes[1].set_yticklabels(np.arange(1, K+1))
#     fig.colorbar(im1, ax=axes[1])

#     plt.show()

def facet_boxplots(
    df,
    desired_algos=None,
    showfliers=False,
    figsize_per_panel=(6, 4),
    free_y=True,
    metric_label=None,
    savepath=None,
    title=True,   # True/False/str
):
    """
    Faceted boxplots of value by n, one panel per algorithm.

    Parameters
    ----------
    df : pd.DataFrame
        Columns: ["n", "rep", "algorithm", "value"]
    desired_algos : sequence[str] | str | None
        Algorithms to include (in this order). Missing ones are skipped.
        If str, treated as a single algorithm. If None, include all present.
    showfliers : bool
        Show outliers in boxplots.
    figsize_per_panel : (w, h)
        Size of each panel; total fig size scales with #panels.
    free_y : bool
        If True, panels have independent y scales.
    metric_label : str | None
        y-axis label; defaults to "value" if None.
    savepath : str | Path | None
        If provided, save the figure here.
    title : bool | str
        True  -> per-panel titles set to algorithm names.
        False -> no titles.
        str   -> figure-level title set to this string (no per-panel titles).

    Returns
    -------
    (fig, axes)
    """

    # Checks
    required_cols = {"n", "rep", "algorithm", "value"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    # Normalize desired_algos input
    alg_series = df["algorithm"].dropna()
    present_set = set(alg_series.unique())
    if desired_algos is None:
        present_algos = list(alg_series.unique())
    else:
        if isinstance(desired_algos, str):
            desired_algos = [desired_algos]
        else:
            desired_algos = list(desired_algos)
        present_algos = [a for a in desired_algos if a in present_set]

    if len(present_algos) == 0:
        raise ValueError(
            "None of the desired_algos are present in df.\n"
            f"desired_algos={desired_algos}\n"
            f"present algos in df={sorted(present_set)}"
        )

    # Filter to algorithms we will plot
    df_plot = df[df["algorithm"].isin(present_algos)].copy()
    ns_all = sorted(df_plot["n"].unique())

    # Figure/axes
    n_panels = len(present_algos)
    fig_w = figsize_per_panel[0] * n_panels
    fig_h = figsize_per_panel[1]
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(fig_w, fig_h),
        sharey=not free_y
    )
    if n_panels == 1:
        axes = [axes]

    # Plot each panel
    for ax, algo in zip(axes, present_algos):
        sub = df_plot[df_plot["algorithm"] == algo]

        # Per-n arrays
        data_per_n = [sub[sub["n"] == n]["value"].values for n in ns_all]
        ns_present = [n for n, vals in zip(ns_all, data_per_n) if len(vals) > 0]
        data_present = [vals for vals in data_per_n if len(vals) > 0]

        if len(data_present) == 0:
            if title is True:
                ax.set_title(f"{algo} (no data)")
            ax.axis("off")
            continue

        ax.boxplot(
            data_present,
            showfliers=showfliers,
            widths=0.6,
            manage_ticks=False,
            patch_artist=True,
            medianprops=dict(color="red", linewidth=1.5),
            boxprops=dict(facecolor="lightgray", edgecolor="0.35", linewidth=1.2)
            
        )
        if title is True:
            ax.set_title(algo)

        ax.set_xticks(range(1, len(ns_present) + 1))
        ax.set_xticklabels([str(n) for n in ns_present])
        ax.set_xlabel("n")
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    # y label
    if metric_label is None:
        metric_label = "value"
    if len(axes) > 0:
        axes[0].set_ylabel(metric_label)

    # Figure-level title if a string is provided (no per-panel titles in this case)
    if isinstance(title, str):
        fig.suptitle(title)

    plt.tight_layout()
    if isinstance(title, str):
        # prevent suptitle from getting clipped by tight_layout
        plt.subplots_adjust(top=0.88)

    if savepath is not None:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")

    plt.show()

def scatter_faceted(
    dct,
    ncols=3,
    s=12,
    alpha=0.6,
    xmax_pad=0.02,
    savepath=None,
    xlabel="Truth",
    ylabel="Estimates",
    show_legend="once",          # True | False | "once"
    desired_algos=None,          # list/tuple of display names to keep; None => all present
    alg_rename=None,             # dict like {"nfindr":"N-FINDR","sourcexray":"SourceXray"}
    marker_map=None,             # dict of display_name -> marker; else auto-cycle
):
    """
    Faceted scatter of truth (x) vs estimates (y), one panel per n.
    - Supports arbitrary/multiple algorithms (keys like 'hat_*' in each dct[n] bundle)
    - Consistent marker/color per algorithm across all facets
    - Legend control: True/False/'once'
    - Optional algorithm renaming and filtering

    dct[n] = {
        "true": 1D or 2D arraylike,
        "hat_<anything>": arraylike estimates,
        "hat_<anythingElse>": ...
    }
    """
    # --- helpers ---
    def key_to_algname(k: str) -> str:
        """From 'hat_fooBar' -> display name, with optional rename map (case-insensitive)."""
        if not k.startswith("hat_"):
            return k
        raw = k[4:]
        key = raw.lower()
        if alg_rename:
            # normalize rename keys to lowercase
            for src, dst in alg_rename.items():
                if key == str(src).lower():
                    return str(dst)
        # fallback: keep original suffix, lightly prettified
        return raw.replace("_", "-")

    # which n to draw
    all_ns = sorted(k for k in dct.keys() if k in dct)
    if not all_ns:
        raise ValueError("No matching n found in results for plotting.")

    # discover all algorithms across bundles to build consistent styling
    algs_seen = []
    for n in all_ns:
        bundle = dct[n]
        for k in bundle.keys():
            if k.startswith("hat_"):
                nm = key_to_algname(k)
                if nm not in algs_seen:
                    algs_seen.append(nm)

    # filter if requested
    if desired_algos is None:
        kept_algs = algs_seen
    else:
        kept_set = {a for a in desired_algos}
        kept_algs = [a for a in algs_seen if a in kept_set]
    if not kept_algs:
        raise ValueError("After filtering, no algorithms remain to plot.")

    # markers: user map or auto cycle
    default_markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>', 'h', '*', '8', 'p']
    marker_cycle = cycle(default_markers)
    style_marker = {}
    if isinstance(marker_map, dict):
        # start with user-specified, then fill in missing
        style_marker.update(marker_map)
    for a in kept_algs:
        style_marker.setdefault(a, next(marker_cycle))

    # colors: consistent across facets using Matplotlib's default color cycle
    default_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)
    if not default_colors:
        default_colors = [f"C{i}" for i in range(10)]
    color_cycle = cycle(default_colors)
    style_color = {}
    for a in kept_algs:
        style_color.setdefault(a, next(color_cycle))

    # layout
    n_panels = len(all_ns)
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2*ncols, 4.2*nrows), squeeze=False)
    axes_flat = axes.ravel()

    # hide extras
    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    legend_done = False

    for ax, n in zip(axes_flat, all_ns):
        b = dct[n]
        true = np.asarray(b["true"]).ravel()

        # collect available (and kept) algorithms for this panel
        panel_algs = []
        for k, v in b.items():
            if not k.startswith("hat_") or v is None:
                continue
            nm = key_to_algname(k)
            if nm in kept_algs:
                panel_algs.append((nm, np.asarray(v).ravel()))

        if not panel_algs:
            ax.set_title(f"n = {n} (no estimates found)")
            ax.axis("off")
            continue

        all_vals = [true]
        for alg_name, hat in panel_algs:
            m = min(true.size, hat.size)
            x = true[:m]
            y = hat[:m]
            ax.scatter(x, y, s=s, alpha=alpha, marker=style_marker[alg_name], label=alg_name, 
                       color=("gray" if len(kept_algs) == 1 else style_color[alg_name]))
            all_vals.append(y)

        # axis limits + 45-degree line
        all_vals = np.concatenate(all_vals)
        vmin = max(0.0, np.nanmin(all_vals))
        vmax = np.nanmax(all_vals)
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = 0.0, 1.0
        pad = (vmax - vmin) * float(xmax_pad)
        a_min, a_max = vmin - pad, vmax + pad

        ax.plot([a_min, a_max], [a_min, a_max], color="red", linewidth=1.5)
        ax.set_xlim(a_min, a_max)
        ax.set_ylim(a_min, a_max)
        ax.set_aspect("equal", adjustable="box")

        ax.set_title(f"n = {n}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.4)

        # legend behavior
        if show_legend is True or (show_legend == "once" and not legend_done):
            ax.legend(title="Algorithm", loc="best", frameon=True)
            if show_legend == "once":
                legend_done = True

    plt.tight_layout()
    if savepath:
        fig.savefig(savepath, bbox_inches="tight", dpi=200)
    plt.show()
    
def plot_C_matrix_grouped_by_k(
    C_or_boot,
    title="Ĉ (grouped by source k): bootstrap mean ± std",
    col_label=None,
    show_err=True,              # if 3D input, show error bars (std)
    capsize=3, 
    savepath=None,
):
    """
    Plot grouped bars by source (k) with one bar per feature (j).

    Parameters
    ----------
    C_or_boot : array-like
        Either:
          - (J, K): single C matrix
          - (n_reps, J, K): bootstrap stack of C-hat matrices
    title : str
        Figure title.
    col_label : list[str] | None
        Feature labels of length J (row labels). If None -> "Feature 1..J".
    show_err : bool
        If 3D input, draw error bars (std across reps).
    capsize : float
        Error bar cap size (points).
    """
    A = np.asarray(C_or_boot)
    if A.ndim == 2:
        # single matrix
        J, K = A.shape
        C_mean = A
        C_std = None
    elif A.ndim == 3:
        # bootstrap stack
        n_reps, J, K = A.shape
        C_mean = np.nanmean(A, axis=0)                         # (J, K)
        C_std  = np.nanstd(A,  axis=0, ddof=1 if n_reps > 1 else 0) if show_err else None
    else:
        raise ValueError(f"Expected 2D (J,K) or 3D (n_reps,J,K); got shape {A.shape}")

    # labels
    if col_label is None:
        labels = [f"Feature {j+1}" for j in range(J)]
    else:
        if len(col_label) != J:
            raise ValueError(f"col_label must have length {J} (got {len(col_label)})")
        labels = list(col_label)

    x = np.arange(K)
    width = 0.8 / max(J, 1)

    fig, ax = plt.subplots(figsize=(10, 5))

    for j in range(J):
        # build kwargs only if we have usable stds
        kwargs = {}
        if C_std is not None:
            yerr = C_std[j, :]
            if np.any(np.isfinite(yerr)):              # avoid all-NaN / empty cases
                kwargs["yerr"] = yerr
                kwargs["error_kw"] = {"elinewidth": 1.2, "capsize": capsize}

        ax.bar(
            x + j * width,
            C_mean[j, :],
            width,
            label=labels[j],
            **kwargs,                                   # <- only present when needed
        )


    ax.set_xticks(x + (J - 1) * width / 2)
    ax.set_xticklabels([f"Source {k+1}" for k in range(K)])
    ax.set_ylabel("Expected fraction of pollutants\nattributable to each source")
    ax.set_title(title)
    ax.set_ylim(0, 1)

    # Legend outside on the right
    ax.legend(title="", loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()

# plot_W_by_covariate
def _safe_kde(x, grid):
    x = np.asarray(x, float)
    if x.size < 2 or np.allclose(np.var(x), 0.0):
        bins = min(50, max(10, int(np.sqrt(max(x.size, 1)))))
        cnt, edg = np.histogram(x, bins=bins, density=True)
        mid = 0.5 * (edg[1:] + edg[:-1])
        return np.interp(grid, mid, cnt, left=0.0, right=0.0)
    return gaussian_kde(x)(grid)

def _nice_label(v):
    # NaN handled outside; for numbers, ':g' drops trailing zeros
    if isinstance(v, (float, np.floating, int, np.integer)):
        return f"{float(v):g}"
    return str(v)
        
# def plot_W_by_covariate(W_tilde, 
#                         covariate, 
#                         include_nan=True, 
#                         covariate_label="group", 
#                         ylabel="Density", 
#                         xlabel=r"$\widetilde{W}$ value",
#                         panel_title=None,   # <- per-panel title (str/callable/seq/None)
#                         suptitle=None):     # <- overall title (str or None)
#     """
#     Plot density of each column of W_tilde with one curve per unique group value in covariate.
#     NaN is treated as its own group when include_nan is True.
#     """
    
#     def _title_for(k, K):
#         if panel_title is None:
#             return f"Source {k+1} by {covariate_label}"
#         if isinstance(panel_title, str):
#             return panel_title.format(k=k, k1=k+1, K=K, covariate_label=covariate_label)
#         if callable(panel_title):
#             return str(panel_title(k, covariate_label, K))
#         try:
#             return str(panel_title[k])
#         except Exception:
#             return f"Source {k+1} by {covariate_label}"
        
#     W = np.asarray(W_tilde, float)
#     g = np.asarray(covariate)

#     if W.ndim != 2 or g.shape[0] != W.shape[0]:
#         raise ValueError("shape mismatch")

#     # build group masks, including NaN as its own group
#     if g.dtype.kind in "f" or np.issubdtype(g.dtype, np.number):
#         isnan = np.isnan(g)
#         cats = np.unique(g[~isnan]).tolist()
#         if include_nan and isnan.any():
#             cats.append(np.nan)
#         masks = []
#         labels = []
#         for c in cats:
#             if isinstance(c, float) and np.isnan(c):
#                 m = isnan
#                 lab = "NaN"
#             else:
#                 m = (~isnan) & (g == c)
#                 lab = _nice_label(c)
#             masks.append(m)
#             labels.append(lab)
#     else:
#         isnan = np.array([val != val for val in g])
#         cats = np.unique(g[~isnan]).tolist()
#         if include_nan and isnan.any():
#             cats.append(np.nan)
#         masks = []
#         labels = []
#         for c in cats:
#             if c is np.nan:
#                 m = isnan
#                 lab = "NaN"
#             else:
#                 m = (~isnan) & (g == c)
#                 lab = str(c)
#             masks.append(m)
#             labels.append(lab)

#     K = W.shape[1]
#     cols = min(4, K)
#     rows = int(np.ceil(K / cols))
#     fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False)

#     for k in range(K):
#         ax = axes[k // cols, k % cols]
#         base_mask = np.isfinite(W[:, k])  # do not prefilter on group

#         # gather data per group to set grid
#         series = [W[base_mask & m, k] for m in masks if np.any(base_mask & m)]
#         if not series:
#             ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
#             ax.set_title(_title_for(k, K))
#             continue

#         allx = np.concatenate(series)
#         lo = np.nanpercentile(allx, 0.5)
#         hi = np.nanpercentile(allx, 99.5)
#         if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
#             lo = np.nanmin(allx); hi = np.nanmax(allx)
#             if np.isclose(lo, hi):
#                 lo, hi = lo - 1.0, hi + 1.0
#         grid = np.linspace(lo, hi, 400)

#         # plot each group including NaN
#         for m, lab in zip(masks, labels):
#             idx = base_mask & m
#             x = W[idx, k]
#             if x.size:
#                 ax.plot(grid, _safe_kde(x, grid), label=f"{lab} (n={x.size})")

#         ax.set_title(_title_for(k, K))
#         ax.set_xlabel(xlabel)
#         ax.set_ylabel(ylabel)
#         ax.legend(fontsize=8)

#     for idx in range(K, rows * cols):
#         axes[idx // cols, idx % cols].axis("off")

#     if suptitle:
#         fig.suptitle(str(suptitle), y=0.995)
        
#     plt.tight_layout()
#     plt.show()

def plot_W_by_covariate(
    W_tilde, 
    covariate, 
    include_nan=True, 
    covariate_label="group", 
    ylabel="Density", 
    xlabel=r"$\widetilde{W}$ value",
    panel_title=None,   # per-panel title (str/callable/seq/None)
    suptitle=None,      # overall title (str or None)
    savepath=None,      # if not None, save the figure here
    label_map=None,     # {raw_value: "Nice label"}
    color_map=None,     # {raw_value: "color"} 
    linestyle_map=None,    # {raw_value: "linestyle"}
    legend_outside=False,
    legend_loc="bottom",
    legend_position_adjust=0.5
):
    """
    Plot density of each column of W_tilde with one curve per unique group value in covariate.

    - NaN is treated as its own group when include_nan=True.
    - label_map maps raw covariate values to nicer labels.
    - color_map maps *raw values* to fixed colors.
    - legend_outside puts one shared legend outside.
    """
    import numpy as np, matplotlib.pyplot as plt

    def _title_for(k, K):
        if panel_title is None:
            return f"Source {k+1} by {covariate_label}"
        if isinstance(panel_title, str):
            return panel_title.format(k=k, k1=k+1, K=K, covariate_label=covariate_label)
        if callable(panel_title):
            return str(panel_title(k, covariate_label, K))
        try:
            return str(panel_title[k])
        except Exception:
            return f"Source {k+1} by {covariate_label}"

    W = np.asarray(W_tilde, float)
    g = np.asarray(covariate)
    if W.ndim != 2 or g.shape[0] != W.shape[0]:
        raise ValueError("shape mismatch")

    # collect unique categories
    if g.dtype.kind in "f" or np.issubdtype(g.dtype, np.number):
        isnan = np.isnan(g)
        cats = np.unique(g[~isnan]).tolist()
        if include_nan and isnan.any():
            cats.append(np.nan)
    else:
        isnan = np.array([val != val for val in g])
        cats = np.unique(g[~isnan]).tolist()
        if include_nan and isnan.any():
            cats.append(np.nan)

    masks, labels = [], []
    for c in cats:
        if isinstance(c, float) and np.isnan(c):
            m = isnan
            raw_val = np.nan
            lab = "NaN"
        else:
            m = (~isnan) & (g == c)
            raw_val = c
            lab = str(c)
        if label_map is not None:
            lab = label_map.get(raw_val, lab)
        masks.append((m, raw_val))   # keep raw value with mask
        labels.append(lab)

    # figure layout
    K = W.shape[1]
    cols = min(4, K)
    rows = int(np.ceil(K / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False)

    all_handles, all_labels = [], []

    for k in range(K):
        ax = axes[k // cols, k % cols]
        base_mask = np.isfinite(W[:, k])

        # determine plotting grid
        series = [W[base_mask & m, k] for (m, _) in masks if np.any(base_mask & m)]
        if not series:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        allx = np.concatenate(series)
        lo = np.nanpercentile(allx, 0.5)
        hi = np.nanpercentile(allx, 99.5)
        if not np.isfinite(lo) or not np.isfinite(hi) or np.isclose(lo, hi):
            lo = np.nanmin(allx); hi = np.nanmax(allx)
            if np.isclose(lo, hi):
                lo, hi = lo - 1.0, hi + 1.0
        grid = np.linspace(lo, hi, 400)

        # plot each group
        for (m, raw_val), lab in zip(masks, labels):
            idx = base_mask & m
            x = W[idx, k]
            if x.size:
                color = None
                if color_map is not None and raw_val in color_map:
                    color = color_map[raw_val]
                linestyle = "-"
                if linestyle_map is not None and raw_val in linestyle_map:
                    linestyle = linestyle_map[raw_val]
                h, = ax.plot(grid, _safe_kde(x, grid),
                             label=f"{lab} (n={x.size})", 
                             color=color, linestyle=linestyle)
                if k == 0:
                    all_handles.append(h)
                    all_labels.append(f"{lab} (n={x.size})")

        ax.set_title(_title_for(k, K))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if not legend_outside:
            ax.legend(fontsize=8)

    # shared legend
    if legend_loc == "bottom":
        fig.legend(all_handles, all_labels, loc="lower center",
               ncol=len(all_labels), frameon=False,
               bbox_to_anchor=(0.5, -legend_position_adjust))  # y offset
    elif legend_loc == "top":
        fig.legend(all_handles, all_labels, loc="upper center",
                   ncol=len(all_labels), frameon=False,
                   bbox_to_anchor=(0.5, 1+legend_position_adjust))  # y offset
    elif legend_loc == "right":
        fig.legend(all_handles, all_labels, loc="center left",
                   frameon=False,
                   bbox_to_anchor=(1+legend_position_adjust, 0.5))  # x offset
    elif legend_loc == "left":
        fig.legend(all_handles, all_labels, loc="center right",
                   frameon=False,
                   bbox_to_anchor=(-legend_position_adjust, 0.5))  # x offset

    if suptitle:
        fig.suptitle(str(suptitle), y=0.995)
    plt.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")
        
    plt.show()

def boxplot_W_quantiles_by_covariate(
    df,                               # W_tilde quantiles (n_reps, G, Q, K)
    q_levels=None,                    # defaults to results_boots["q_levels"] or linspace
    groups=None,                      # defaults to results_boots["groups"] or range(G)
    k=None,                           # int to plot a single source; None => facet all sources
    ncols=3,
    showfliers=False,
    figsize_per_panel=(6, 4),
    xlabel="Percentile level",
    ylabel="Intensity",
    panel_title=None, 
    covariate_label="bulldozer", 
    meanline=True,                    
    mean_marker=True,                 
    mean_linewidth=2.0,               
    mean_markersize=3,                
    dodge=True,
    savepath=None,
    label_map=None,        # {raw_value: "Nice Label"}
    color_map=None,        # {raw_value: "color"}
    mean_linestyle_map=None, # {raw_value: "linestyle"}
    legend_outside=False,
    legend_loc="bottom",
    legend_position_adjust=0.05,
):
    import numpy as np, matplotlib.pyplot as plt, math

    def _title_for(k, K):
        if panel_title is None:
            return f"Source {k+1} by {covariate_label}"
        if isinstance(panel_title, str):
            return panel_title.format(k=k, k1=k+1, K=K, covariate_label=covariate_label)
        if callable(panel_title):
            return str(panel_title(k, covariate_label, K))
        try:
            return str(panel_title[k])
        except Exception:
            return f"Source {k+1} by {covariate_label}"
    
    A = np.asarray(df)
    if A.ndim != 4:
        raise ValueError(f"data must be 4D (n_reps, G, Q, K); got shape {A.shape}")
    n_reps, G, Q, K = A.shape

    # quantile levels
    q_levels = np.asarray(q_levels).astype(float)
    if q_levels.size != Q:
        raise ValueError(f"q_levels must have length {Q}.")

    if len(groups) != G:
        raise ValueError(f"groups must have length {G}.")

    # labels
    if label_map is not None:
        group_labels = [label_map.get(g, str(g)) for g in groups]
    else:
        group_labels = [str(g) for g in groups]

    # colors
    if color_map is not None:
        colors = [color_map.get(g, None) for g in groups]
    else:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % cmap.N) for i in range(G)]

    # which sources to plot
    ks = [k] if k is not None else list(range(K))
    n_panels = len(ks)
    ncols = min(ncols, n_panels)
    nrows = math.ceil(n_panels / ncols)

    W, H = figsize_per_panel 
    fig, axes = plt.subplots(nrows, ncols, figsize=(W * ncols, H * nrows), squeeze=False)
    
    all_handles, all_labels = [], []

    # iterate panels
    for ax, kk in zip(axes.ravel(), ks):
        xs = np.arange(Q, dtype=float)  
        group_width = 0.8
        box_w = group_width / max(G, 1)

        for gi, gval in enumerate(groups):
            data = [A[:, gi, qi, kk] for qi in range(Q)]
            if dodge:
                positions = xs - group_width / 2 + (gi + 0.5) * box_w
            else:
                positions = xs

            bp = ax.boxplot(
                data,
                positions=positions,
                widths=box_w * 0.9,
                showfliers=showfliers,
                manage_ticks=False,
                patch_artist=True,
            )
            col = colors[gi]

            # color boxplot parts
            for b in bp["boxes"]:
                b.set_facecolor(col)
                b.set_edgecolor(col)
                b.set_alpha(0.7)
            for part in ("whiskers", "caps", "medians", "fliers", "means"):
                if part in bp:
                    for artist in bp[part]:
                        artist.set_color(col)

            # --- connect means ---
            if meanline:
                means = np.array([np.nanmean(arr) for arr in data], dtype=float)
                ls = mean_linestyle_map.get(gval, "-") if mean_linestyle_map else "-"
                ax.plot(
                    positions, means,
                    linewidth=mean_linewidth,
                    color=col,
                    alpha=0.95,
                    linestyle=ls,
                    zorder=3
                )
                if mean_marker:
                    ax.plot(
                        positions, means,
                        linestyle="none",
                        marker="o",
                        markersize=mean_markersize,
                        color=col,
                        alpha=0.95,
                        zorder=4
                    )

            if kk == ks[0]:
                all_handles.append(bp["boxes"][0])
                all_labels.append(group_labels[gi])

        ax.set_xticks(xs)
        ax.set_xticklabels([f"{q:.2f}" for q in q_levels], fontsize=11)
        for label in ax.get_xticklabels():
            label.set_rotation(30)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(_title_for(kk, K), fontsize=14)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

        if not legend_outside:
            ax.legend(fontsize=8)     

    # shared vs per-axis legend
    if legend_outside and all_handles:
        if legend_loc == "bottom":
            fig.legend(all_handles, all_labels, loc="lower center",
                       ncol=len(all_labels), frameon=False,
                       bbox_to_anchor=(0.5, -legend_position_adjust), 
                       fontsize=12)
        elif legend_loc == "top":
            fig.legend(all_handles, all_labels, loc="upper center",
                       ncol=len(all_labels), frameon=False,
                       bbox_to_anchor=(0.5, 1+legend_position_adjust), 
                       fontsize=12)
        elif legend_loc == "right":
            fig.legend(all_handles, all_labels, loc="center left",
                       frameon=False, bbox_to_anchor=(1+legend_position_adjust, 0.5), 
                       fontsize=12)
        elif legend_loc == "left":
            fig.legend(all_handles, all_labels, loc="center right",
                       frameon=False, bbox_to_anchor=(-legend_position_adjust, 0.5), 
                       fontsize=12)

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches="tight")

    plt.tight_layout()
    plt.show()

# def boxplot_W_quantiles_by_covariate(
#     df,                               # W_tilde quantiles (n_reps, G, Q, K)
#     q_levels=None,                    # defaults to results_boots["q_levels"] or linspace
#     groups=None,                      # defaults to results_boots["groups"] or range(G)
#     group_labels=None,                # pretty labels for legend; default => "0","1",... (int formatting)
#     group_colors=None,
#     k=None,                           # int to plot a single source; None => facet all sources
#     ncols=3,
#     showfliers=False,
#     figsize_per_panel=(6, 4),
#     covariate_label="bulldozer", 
#     meanline=True,                    # draw mean lines per group
#     mean_marker=True,                 # add small markers on the mean line
#     mean_linewidth=2.0,               # line width
#     mean_markersize=3,                # marker size
#     dodge=True                        # dodge boxplots by group
# ):
#     A = np.asarray(df)
#     if A.ndim != 4:
#         raise ValueError(f"data must be 4D (n_reps, G, Q, K); got shape {A.shape}")
#     n_reps, G, Q, K = A.shape

#     # quantile levels
#     q_levels = np.asarray(q_levels).astype(float)
#     if q_levels.size != Q:
#         raise ValueError(f"q_levels must have length {Q}.")

#     # group ids
#     if len(groups) != G:
#         raise ValueError(f"groups must have length {G}.")
#     # labels
#     if group_labels is None:
#         # nice int formatting for 0.0/1.0 -> "0"/"1"
#         group_labels = [_nice_label(g) for g in groups]
#     if len(group_labels) != G:
#         raise ValueError(f"group_labels must have length {G}.")
        
#     # colors
#     def _build_colors():
#         # allow dict mapping or list
#         if isinstance(group_colors, dict):
#             cmap = plt.get_cmap("tab10")
#             return [group_colors.get(g, cmap(i % cmap.N)) for i, g in enumerate(groups)]
#         elif isinstance(group_colors, (list, tuple, np.ndarray)):
#             cols = list(group_colors)
#             if len(cols) < G:
#                 # repeat if shorter than G
#                 cmap = plt.get_cmap("tab10")
#                 cols += [cmap(i % cmap.N) for i in range(G - len(cols))]
#             return cols[:G]
#         else:
#             cmap = plt.get_cmap("tab10")  # nice distinct colors up to 10; use 'tab20' for more
#             return [cmap(i % cmap.N) for i in range(G)]
#     colors = _build_colors()

#     # which sources to plot
#     ks = [k] if k is not None else list(range(K))
#     n_panels = len(ks)
#     ncols = min(ncols, n_panels)
#     nrows = math.ceil(n_panels / ncols)
#     W, H = figsize_per_panel
#     fig, axes = plt.subplots(nrows, ncols, figsize=(W * ncols, H * nrows), squeeze=False)

#     # iterate panels (sources)
#     for ax, kk in zip(axes.ravel(), ks):
#         xs = np.arange(Q, dtype=float)  # base positions for quantiles
#         group_width = 0.8
#         box_w = group_width / max(G, 1)

#         handles = []
#         labels = []
#         positions = xs

#         # one boxplot series per group (at each quantile)
#         for gi in range(G):
#             # data for this group across quantiles: list length Q, each is (n_reps,)
#             data = [A[:, gi, qi, kk] for qi in range(Q)]

#             # positions for this group's boxes around each quantile tick
#             if dodge:
#                 positions = xs - group_width / 2 + (gi + 0.5) * box_w
            
#             bp = ax.boxplot(
#                 data,
#                 positions=positions,
#                 widths=box_w * 0.9,
#                 showfliers=showfliers,
#                 manage_ticks=False,
#                 patch_artist=True,   # use default colors
#             )
#             col = colors[gi]
#             # color the pieces
#             for b in bp["boxes"]:
#                 b.set_facecolor(col)
#                 b.set_edgecolor(col)
#                 b.set_alpha(0.7)
#             for part in ("whiskers", "caps", "medians", "fliers", "means"):
#                 if part in bp:
#                     for artist in bp[part]:
#                         artist.set_color(col)

#             # --- connect means across quantiles for this group ---
#             if meanline:
#                 means = np.array([np.nanmean(arr) for arr in data], dtype=float)
#                 ax.plot(
#                     positions, means,
#                     linewidth=mean_linewidth,
#                     color=col,
#                     alpha=0.95,
#                     zorder=3
#                 )
#                 if mean_marker:
#                     ax.plot(
#                         positions, means,
#                         linestyle="none",
#                         marker="o",
#                         markersize=mean_markersize,
#                         color=col,
#                         alpha=0.95,
#                         zorder=4
#                     )
                    
#             handles.append(bp["boxes"][0])
#             labels.append(group_labels[gi])

#         ax.set_xticks(xs)
#         ax.set_xticklabels([f"{q:.2f}" for q in q_levels])
#         ax.set_xlabel("Percentile level")
#         ax.set_ylabel(r"$\widetilde{W}$ value")
#         ax.set_title(f"Source {kk+1} by {covariate_label}")
#         ax.grid(axis="y", linestyle=":", alpha=0.4)
#         if handles:
#             ax.legend(handles, labels, loc="best", frameon=True)

#     for ax in axes.ravel()[n_panels:]:
#         ax.axis("off")

#     plt.tight_layout()
#     plt.show()

def plot_coefficients_by_term_and_response(
    results_all_df, 
    figsize=(12, 8),
    savepath=None,
    sharex_by_col=True,   # share x within each term column
    color_map=None,       # {location: color}
    shape_map=None,       # {location: marker}
):
    """
    Facet by response (rows) and term (columns), y-axis = location.
    
    results_all_df must contain:
        - term
        - response
        - coef
        - ci_low
        - ci_high
        - location
    """
    df = results_all_df.copy()
    # df["location"] = df["location"].astype(str)

    g = sns.FacetGrid(
        df,
        row="response", col="term",
        sharex=False, sharey=False,
        height=2.4, aspect=1.2, 
    )

    def facet_plot(data, **kwargs):
        ax = plt.gca()
        for _, row in data.iterrows():
            loc = row["location"]
            col = row["color"]
            c = color_map.get(col, None) if color_map else "C0"
            m = shape_map.get(loc, "o") if shape_map else "o"

            ax.scatter(row["coef"], row["location"], 
                       color=c, marker=m, s=40, zorder=3)
            ax.plot([row.ci_low, row.ci_high], [row["location"], row["location"]],
                    color=c, alpha=1, lw=1.5)

        ax.axvline(0, ls="--", lw=1, color="k", alpha=0.4)

    g.map_dataframe(facet_plot)

    g.set_axis_labels("Coefficient (95% CI)", "")

    # column suptitles
    for ax, term in zip(g.axes[0], g.col_names):
        ax.set_title("")
        ax.annotate(term, xy=(0.5, 1.2), xycoords="axes fraction",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    g.set_titles("{row_name}")

    # --- sharex by column ---
    if sharex_by_col:
        for ci, term in enumerate(g.col_names):
            # gather all axes in this column
            axs = g.axes[:, ci]
            xmins, xmaxs = [], []
            for ax in axs:
                if ax is not None:
                    xmin, xmax = ax.get_xlim()
                    xmins.append(xmin)
                    xmaxs.append(xmax)
            if xmins and xmaxs:
                xmin, xmax = min(xmins), max(xmaxs)
                for ax in axs:
                    if ax is not None:
                        ax.set_xlim(xmin, xmax)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()

def plot_coefficients_by_response(
    plot_df,
    figsize=(12, 4),
    height=2.6,
    aspect=1.2,
    sharex_by_col=True,
    color_map=None,     # {response: color}
    shape_map=None,     # {response: marker}
    savepath=None,
):
    """
    plot_df columns required:
      response, term, coef, se, lo, hi, label_y, label_x
    Facet by term (columns). Panel title = label_x.
    y-axis: response (displayed via label_y). x-axis: coef with [lo, hi] errorbar.
    """

    df = plot_df.copy()

    # --- Build y labels (use label_y text per response) ---
    # Use label_y if provided; otherwise fallback to response as string
    if "label_y" in df.columns:
        # Construct a single label per response (first non-null wins)
        lab_map = (
            df.dropna(subset=["label_y"])
              .groupby("response", as_index=True)["label_y"]
              .first()
              .to_dict()
        )
    else:
        lab_map = {}
    df["_y_label"] = df["response"].map(lambda r: lab_map.get(r, str(r)))

    # Make y a categorical to preserve order of appearance within each panel
    # (You can pre-order by providing a Categorical in plot_df if you prefer.)
    # We'll let each facet show only the labels present there.
    # x is numeric: coef with CI given by [lo, hi]

    # --- Facet grid by term (columns) ---
    g = sns.FacetGrid(
        df,
        col="term",
        sharex=False,   # we will optionally synchronize within each column below
        sharey=False,
        height=height,
        aspect=aspect
    )

    # If no color_map provided, build one from responses using tab10
    unique_resps = list(df["response"].unique())
    if color_map is None:
        cmap = plt.get_cmap("tab10")
        color_map = {r: cmap(i % cmap.N) for i, r in enumerate(unique_resps)}

    # Default marker per response if not provided
    if shape_map is None:
        base_markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]
        shape_map = {r: base_markers[i % len(base_markers)] for i, r in enumerate(unique_resps)}

    def _facet_plot(data, **kwargs):
        ax = plt.gca()

        # Ensure y is categorical with labels present in this panel, keep current order
        present = list(data["_y_label"].drop_duplicates())
        ax.set_yticks(np.arange(len(present)))
        ax.set_yticklabels(present)

        # Map label positions for plotting
        y_pos = data["_y_label"].map({lbl: i for i, lbl in enumerate(present)})

        # Draw CI as horizontal bars and the point
        for (_, row), y in zip(data.iterrows(), y_pos):
            resp = row["response"]
            c = color_map.get(resp, "C0")
            m = shape_map.get(resp, "o")

            # error bar (lo, hi) at y
            ax.hlines(y, row["lo"], row["hi"], color=c, lw=1.5, alpha=1, zorder=2)
            # point at coef
            ax.scatter(row["coef"], y, color=c, marker=m, s=40, zorder=3, edgecolors="none")

        # reference line at 0
        ax.axvline(0, ls="--", lw=1, color="k", alpha=0.4)
        # thin grid for readability
        ax.grid(axis="x", linestyle=":", alpha=0.4)
        ax.set_xlabel("Coefficient")
        ax.set_ylabel("")

    g.map_dataframe(_facet_plot)

    # Replace panel titles with label_x (one per term)
    for ax, term in zip(g.axes.flat, g.col_names):
        labelx = df.loc[df["term"] == term, "label_x"].dropna()
        title = str(labelx.iloc[0]) if len(labelx) else str(term)
        ax.set_title(title)

    if savepath:
        plt.savefig(savepath, dpi=200, bbox_inches="tight")
    plt.show()
    
def plot_facet_errorbars(
    plot_df,
    ncols=3,
    sharex=False,
    label_response="label_y",    # fallback to 'response' if not present
    label_covariates="label_x",    # fallback to 'term' if not present
    responses_order=None,
    covariates_order=None,
    capsize=3,
    insig_color='0.8', # insignificant
    sig_color='0.0', # significant
    zero_line_color='red', 
    suptitle=None
):
    """
    Faceted error-bar plot: one panel per response; y-axis = covariates; x = coef with 95% CI.
    CIs that include 0 are gray; others are black.
    Returns (fig, axes).
    """
    import numpy as np
    import math
    import matplotlib.pyplot as plt
    from pandas.api.types import is_categorical_dtype

    # responses & facet titles
    # ----- facet order -----
    if responses_order is not None:
        seen = plot_df["response"].unique().tolist()
        # keep only those present, then append any leftovers (optional)
        responses = [r for r in responses_order if r in seen] + [r for r in seen if r not in set(responses_order)]
    elif is_categorical_dtype(plot_df["response"]):
        # honor categorical category order
        responses = [c for c in plot_df["response"].cat.categories
                     if (plot_df["response"] == c).any()]
    else:
        responses = plot_df["response"].unique().tolist()
    
    # facet titles: try label_response -> fallback to 'response'
    if isinstance(label_response, str) and (label_response in plot_df.columns):
        # build mapping response -> first non-null label in that response
        tmp = (plot_df[["response", label_response]]
               .dropna(subset=[label_response])
               .drop_duplicates(subset=["response"]))
        labels_dic = {r: lab for r, lab in zip(tmp["response"], tmp[label_response])}
        # ensure every response has a title
        for r in responses:
            labels_dic.setdefault(r, r)
    else:
        labels_dic = {r: r for r in responses}

    # grid shape
    n_y = len(responses)
    nrows = math.ceil(n_y / ncols)

    # dynamic height from expected rows per panel
    # infer from the first panel
    first = (plot_df[plot_df["response"] == responses[0]].dropna(subset=["term"]))
    rows_per_panel = max(1, first["term"].nunique())    

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4*ncols, max(2.5, 0.45*rows_per_panel)*nrows),
        sharex=sharex
    )
    axes = np.array(axes).reshape(nrows, ncols)

    # global x-limits if sharing x
    if sharex:
        xmin = np.nanmin(plot_df["lo"].values)
        xmax = np.nanmax(plot_df["hi"].values)
        pad = 0.05 * (xmax - xmin) if np.isfinite(xmax - xmin) and xmax > xmin else 1.0
        xlim_global = (xmin - pad, xmax + pad)
    else:
        xlim_global = None

    last_i = -1
    for i, y in enumerate(responses):
        r, c = divmod(i, ncols)
        ax = axes[r, c]

        sub = (plot_df[plot_df["response"] == y]
               .dropna(subset=["term"])
               .sort_values("term"))
        if covariates_order is not None:
            # keep only terms present, in your specified order
            present = [t for t in covariates_order if t in sub["term"].values]
            if not present: 
                ax.axis('off'); 
                continue
            sub = (sub.set_index("term").loc[present].reset_index())
        else:
            sub = sub.sort_values("term")

        if sub.empty:
            ax.axis('off')
            continue

        # choose y labels column
        ylabels = sub[label_covariates] if (label_covariates in sub.columns) else sub["term"]

        idx    = np.arange(len(sub))
        coefs  = sub["coef"].to_numpy()
        lo_ep  = sub["lo"].to_numpy()
        hi_ep  = sub["hi"].to_numpy()
        left_err  = coefs - lo_ep
        right_err = hi_ep - coefs

        # gray vs black mask
        crosses_zero = (lo_ep <= 0) & (hi_ep >= 0)
        keep_black   = ~crosses_zero

        # plot gray first, then black
        if np.any(crosses_zero):
            ax.errorbar(coefs[crosses_zero], idx[crosses_zero],
                        xerr=[left_err[crosses_zero], right_err[crosses_zero]],
                        fmt='o', capsize=capsize, color=insig_color)
        if np.any(keep_black):
            ax.errorbar(coefs[keep_black], idx[keep_black],
                        xerr=[left_err[keep_black], right_err[keep_black]],
                        fmt='o', capsize=capsize, color=sig_color)

        ax.axvline(0, linestyle='--', linewidth=1, color=zero_line_color)
        ax.set_yticks(idx)
        ax.set_yticklabels(ylabels)
        ax.set_title(labels_dic[y])
        ax.invert_yaxis()

        last_i = i

    # hide any leftover empty axes
    for j in range(last_i + 1, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].axis('off')

    # if shared x, apply once at the end
    if sharex and xlim_global is not None:
        for ax in axes.ravel():
            if ax.has_data():
                ax.set_xlim(*xlim_global)
                
    fig.supylabel("Covariate")
    fig.supxlabel("Coefficient (95% CI)")
    
    if suptitle:
        fig.suptitle(str(suptitle), fontsize=16)
    fig.tight_layout()
    plt.show()

##########
# Others #
##########
def summarize_bootstrap_phi(
    C_hat_boots: np.ndarray,
    tau: float = 0.03,
    ci_levels: Tuple[float, float] = (2.5, 97.5),
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    Summarize Phi-based bootstrap stability, assuming sources are ALREADY ALIGNED across bootstraps.

    Parameters
    ----------
    C_hat_boots : (B, J, K) array
        Bootstrap estimates of C_hat for each replicate (per pollutant j and source k).
        For each bootstrap b, we form Phi^(b) by transposing to (K, J) and column-normalizing.
    tau : float
        Relevance floor for computing average CV (exclude cells with mean Phi <= tau).
    ci_levels : (low, high)
        Percentile levels for gap CIs between top-1 and top-2 source shares per pollutant.
    eps : float
        Numerical floor for safe division.

    Returns
    -------
    dict with:
      - phi_mean : (K, J) mean Phi
      - phi_se   : (K, J) standard error of Phi
      - phi_cv   : (K, J) CV = SE/mean (NaN where mean<=tau)
      - cv_mean  : scalar mean CV over entries with mean>tau
      - rank_stability_per_pollutant : (J,) fraction of bootstraps preserving the top source
      - R        : scalar average rank stability across pollutants
      - gap_ci   : (J, 2) [low, high] CI of (top1 - top2) gap
      - G        : fraction of pollutants with lower CI > 0 (top source separated)
      - row_stability : (K,) mean (1 - cosine) of each source row vs consensus
      - row_stability_mean : scalar average of row_stability
    """
    if C_hat_boots.ndim != 3:
        raise ValueError("C_hat_boots must have shape (B, J, K).")
    B, J, K = C_hat_boots.shape

    # Build Phi^(b): transpose to (K, J) and column-normalize per pollutant
    phi_boots = np.empty((B, K, J), dtype=float)
    for b in range(B):
        A = C_hat_boots[b].T  # (K, J)
        X = np.maximum(A, 0.0)
        colsum = X.sum(axis=0, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            X = np.where(colsum > eps, X / colsum, 0.0)
        phi_boots[b] = X

    # Mean and SE across bootstraps
    phi_mean = np.mean(phi_boots, axis=0)                      # (K, J)
    phi_se   = np.std(phi_boots,  axis=0, ddof=1) / np.sqrt(max(B - 1, 1))

    # Cellwise CV over relevant entries (> tau)
    mask_rel = phi_mean > tau
    with np.errstate(divide="ignore", invalid="ignore"):
        phi_cv = np.where(mask_rel, phi_se / phi_mean, np.nan)
    cv_vals = phi_cv[~np.isnan(phi_cv)]
    cv_mean = float(np.mean(cv_vals)) if cv_vals.size else float("nan")

    # Rank stability R(K): consensus top source per pollutant from phi_mean
    top_k = np.argmax(phi_mean, axis=0)                        # (J,)
    rank_match = np.zeros((B, J), dtype=bool)
    for b in range(B):
        rank_match[b] = (np.argmax(phi_boots[b], axis=0) == top_k)
    rank_stability_per_pollutant = rank_match.mean(axis=0)     # (J,)
    R = float(rank_stability_per_pollutant.mean())

    # Gap separation G(K): top1 - top2 per bootstrap, CI per pollutant
    gaps = np.zeros((B, J), dtype=float)
    for b in range(B):
        sorted_vals = np.sort(phi_boots[b], axis=0)[::-1]      # desc over k
        gaps[b] = sorted_vals[0] - sorted_vals[1]
    lo_pct, hi_pct = ci_levels
    gap_ci_lo = np.percentile(gaps, lo_pct, axis=0)
    gap_ci_hi = np.percentile(gaps, hi_pct, axis=0)
    gap_ci = np.vstack([gap_ci_lo, gap_ci_hi]).T               # (J, 2)
    G = float(np.mean(gap_ci_lo > 0))

    # Row stability: 1 - cosine similarity of each source row vs mean row
    row_stability = np.zeros(K, dtype=float)
    m = phi_mean
    m_norm = np.linalg.norm(m, axis=1) + eps                   # (K,)
    for k in range(K):
        A = phi_boots[:, k, :]                                 # (B, J)
        a_norm = np.linalg.norm(A, axis=1) + eps               # (B,)
        cos = (A @ m[k]) / (a_norm * m_norm[k])
        row_stability[k] = float(np.mean(1.0 - np.clip(cos, -1.0, 1.0)))
    row_stability_mean = float(np.mean(row_stability))

    return {
        "phi_mean": phi_mean,
        "phi_se": phi_se,
        "phi_cv": phi_cv,
        "cv_mean": cv_mean,
        "rank_stability_per_pollutant": rank_stability_per_pollutant,
        "R": R,
        "gap_ci": gap_ci,
        "G": G,
        "row_stability": row_stability,
        "row_stability_mean": row_stability_mean,
    }
    
# find the best permutation match
def permute_estimates_to_match_truth(H_true, H_hat, mu_hat, C_hat):
    """
    Permute only the estimated factors so that rows of H_hat best match rows of H_true.
    True quantities are not modified. Returns permuted copies and the permutation.
    Assumes components correspond to rows of H_* and to either rows or columns of C_*.
    """
    H_true = np.asarray(H_true)
    H_hat  = np.asarray(H_hat)
    mu_hat = np.asarray(mu_hat)

    # cost between true and estimated rows
    cost = np.linalg.norm(H_true[:, None, :] - H_hat[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)

    # order estimates to follow the natural order of true rows
    order = col_ind[np.argsort(row_ind)]              # length = min(K_true, K_hat)

    # permute estimates
    H_hat_perm  = H_hat[order]
    mu_hat_perm = mu_hat[order]

    # permute C_hat along its component axis
    C_hat = np.asarray(C_hat)
    if C_hat.ndim != 2:
        raise ValueError("C_hat must be a 2D array")
    K_hat = H_hat.shape[0]
    if C_hat.shape[0] == K_hat:            # components are rows
        C_hat_perm = C_hat[order, :]
    elif C_hat.shape[1] == K_hat:          # components are columns
        C_hat_perm = C_hat[:, order]
    else:
        raise ValueError("C_hat shape does not match H_hat along any axis")

    return H_hat_perm, mu_hat_perm, C_hat_perm, order

# performance measure
def frobenius_dist(C, C_hat, relative=True, eps=1e-12):
    """
    Frobenius distance between C (true) and C_hat (estimate).
    If relative=True: ||C_hat - C||_F / ||C||_F  (scale-invariant).
    Else: ||C_hat - C||_F.
    """
    C = np.asarray(C); C_hat = np.asarray(C_hat)
    if C.shape != C_hat.shape:
        raise ValueError("C and C_hat must have the same shape.")
    diff_norm = np.linalg.norm(C_hat - C, ord="fro")
    if not relative:
        return float(diff_norm)
    base = np.linalg.norm(C, ord="fro")
    base = base if base > eps else eps
    return float(diff_norm / base)


def rms_sad(H, H_hat, eps=1e-12):
    """
    Compare rows of H (true) and H_hat (already permuted estimate) in order.

    Returns:
      {
        'sam_k':  array of length K (angles per row),
        'rms_sad': scalar Root Mean Square (RMS) of Spectral Angle Distance (SAD),
      }

    Args:
      eps: small number for numerical safety.
    """
    H  = np.asarray(H, dtype=float)
    Hh = np.asarray(H_hat, dtype=float)
    if H.shape != Hh.shape:
        raise ValueError("H_star and H_star_hat must have the same shape (K, J).")

    # --- Spectral Angle (per row) ---
    num   = np.sum(H * Hh, axis=1)
    normH = np.maximum(np.linalg.norm(H,  axis=1), eps)
    normE = np.maximum(np.linalg.norm(Hh, axis=1), eps)
    cosang = np.clip(num / (normH * normE), -1.0, 1.0) # clipping = trimming within [-1,1]
    angles = np.arccos(cosang)
    rms_sad = float(np.sqrt(np.mean(angles**2)))

    return {
        'sam_per_row': angles,
        'rms_sad': rms_sad,
    }

def nrmse(H, H_hat, eps=1e-12):
    """
    Compare rows of H (true) and H_hat (already permuted estimate) in order.

    Returns:
      {
        'nrmse_k': array of length K,
        'nrmse':  scalar aggregate of NRMSEs (mean by default)
      }

    Args:
      eps: small number for numerical safety.
    """
    H  = np.asarray(H, dtype=float)
    Hh = np.asarray(H_hat, dtype=float)
    if H.shape != Hh.shape:
        raise ValueError("H_star and H_star_hat must have the same shape (K, J).")

    # --- NRMSE (per row) ---
    normH = np.maximum(np.linalg.norm(H,  axis=1), eps)
    diff        = Hh - H
    rmse_rows   = np.sqrt(np.mean(diff**2, axis=1))
    nrmse_rows  = rmse_rows / normH
    nrmse_overall = float(np.mean(nrmse_rows))

    return {
        'nrmse_per_row': nrmse_rows,
        'nrmse': nrmse_overall,
    }

# compute quantiles
def quantiles_by_group(M, g, q_levels, include_nan=False):
    """
    M: (n, K) matrix (e.g., W_tilde)
    g: (n,) covariate (e.g., bulldozer per row)
    q_levels: array-like of quantiles in [0,1], e.g. [0.05, 0.5, 0.95]
    include_nan: if True and categories is None, treat NaN as its own group

    Returns
    -------
    QG: (G, Q, K) array of quantiles per group
    """
    M = np.asarray(M, float)
    g = np.asarray(g)
    q_levels = np.asarray(q_levels, float)
    Q = q_levels.size
    K = M.shape[1]

    if g.dtype.kind in "f":
        isnan = np.isnan(g)
        cats = np.unique(g[~isnan]).tolist()
        if include_nan and isnan.any():
            cats.append(np.nan)
    else:
        cats = np.unique(g).tolist()
        
    G = len(cats)
    QG = np.full((G, Q, K), np.nan, dtype=float)

    for gi, c in enumerate(cats):
        if isinstance(c, float) and np.isnan(c):
            mask = np.isnan(g)
        else:
            mask = (g == c)
        if np.any(mask):
            # quantiles across rows within this group
            QG[gi] = np.nanquantile(M[mask], q_levels, axis=0)
    return QG

# run lms 
def run_multireg(df, formula_rhs, ys, alpha=0.05, save_all=True, add_main_plus=False):
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf

    all_rows = []
    models = {}

    for y in ys:
        use_cols = [y] + [v.strip() for v in formula_rhs.replace('*', '+').replace(':', '+').split('+')]
        dfm = df[use_cols].replace([np.inf, -np.inf], np.nan).dropna()

        model = smf.ols(f"{y} ~ {formula_rhs}", data=dfm).fit(cov_type="HC3")
        models[y] = model

        if save_all:
            coefs = model.params.rename("coef")
            ses   = model.bse.rename("se")
            pvals = model.pvalues.rename("pval")
            cis   = model.conf_int(alpha=alpha)
            cis.columns = ["ci_low", "ci_high"]
            rows = pd.concat([coefs, ses, pvals, cis], axis=1)
        else:
            rows = model.params.to_frame(name="coef")

        # --- add the main + interaction row using the model's robust vcov ---
        if add_main_plus:
            # Hypothesis string (adjust names if your factor levels differ)
            hyp = "bulldozer[T.1] + downwind[T.1] + bulldozer[T.1]:downwind[T.1] = 0"
            con = model.t_test(hyp)  # uses model.cov_params() (HC3 here)

            # Extract pieces
            eff = float(np.asarray(con.effect).ravel()[0])
            se  = float(np.asarray(con.sd).ravel()[0])
            p   = float(con.pvalue)
            lo, hi = map(float, con.conf_int(alpha=alpha)[0])

            # Append row (match columns depending on save_all)
            term_name = "main_plus_interaction"
            if save_all:
                rows.loc[term_name, ["coef", "se", "pval", "ci_low", "ci_high"]] = [eff, se, p, lo, hi]
            else:
                rows.loc[term_name, "coef"] = eff

        # finalize tidy rows
        rows.insert(0, "response", y)
        all_rows.append(rows.reset_index().rename(columns={"index": "term"}))

    results_df = pd.concat(all_rows, ignore_index=True)
    return results_df
