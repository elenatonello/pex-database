"""Generic plot tools by type of data. They know nothing about diagnostics:
the diagnostic-specific wrappers (diag_plots.py) pass in getter functions.

  time_traces  stacked panels, signals vs time, one line per shot
  profile_grid rows = quantities, columns = times; profiles of several shots overlaid
  map_grid     rows = shots, columns = times; 2D maps (triangular mesh or R,Z grid)
  xy_plot      one quantity vs another, points joined in time order

A "getter" is a function shot -> data; if it raises shotdb.Unavailable the
shot is skipped for that panel (and reported once on the console).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.collections import PolyCollection

from pexdb.shotdb import Unavailable

_reported = set()
CAPSIZE = 2.5      # length of the horizontal caps on error bars [pt]; 0 = no caps


def _warn(shot, panel, err):
    key = (shot.shot, panel)
    if key not in _reported:
        _reported.add(key)
        print(f"  skip {shot.label}: {panel}: {err}")


def _legend(ax, shots, line=True):
    handles = [plt.Line2D([], [], color=s.color(), marker=s.marker, ls="-" if line else "none",
                          label=s.label)
               for s in shots]
    ax.legend(handles=handles, fontsize=8, frameon=False)


def draw_wall(ax, shot, vessel_color="0.85", tiles_color="k"):
    """Draw the stored wall: vessel as a patch between the inner and outer contours
    (or a line if only one contour exists), tiles as a line."""
    w = shot.wall()
    vi, vo = w.get("vessel_inner"), w.get("vessel_outer")
    if vi is not None and vo is not None:
        ring = np.concatenate([np.column_stack(vo), np.column_stack(vi)[::-1]])
        ax.add_patch(plt.Polygon(ring, closed=True, fc=vessel_color, ec="0.4", lw=0.6, zorder=0))
    for n, (r, z) in w.items():
        if n in ("vessel_inner", "vessel_outer") and vi is not None and vo is not None:
            continue
        ax.plot(r, z, color=tiles_color if n == "tiles" else "0.4", lw=1.3 if n == "tiles" else 1,
                zorder=1)
    ax.autoscale_view()


# ---------------------------------------------------------------------------
def time_traces(shots, items, t_range=None, sharex=True, color_by="auto", err_style="bar",
                capsize=CAPSIZE, figsize=None):
    """items: list of (path, ylabel[, scale[, err_path]]); 1D time series.
    Markers only; errors (if err_path given) as err_style 'bar' | 'shade' | None."""
    fig, axes = plt.subplots(len(items), 1, sharex=sharex, squeeze=False,
                             figsize=figsize or (7, 1.9 * len(items) + 0.6))
    for ax, item in zip(axes[:, 0], items):
        item = list(item)
        path, ylabel = item[:2]
        scale = item[2] if len(item) > 2 else 1
        err = item[3] if len(item) > 3 else None
        for s in shots:
            try:
                y = s[path] * scale
                e = s[err] * scale if err else None
            except Unavailable as ex:
                _warn(s, path, ex)
                continue
            c = s.color(color_by)
            if e is not None and err_style == "bar":
                ax.errorbar(s.t, y, yerr=e, color=c, marker=s.marker, ms=4, ls="none",
                            elinewidth=0.8, capsize=capsize, capthick=0.8)
            else:
                ax.plot(s.t, y, color=c, marker=s.marker, ms=4, ls="none")
                if e is not None and err_style == "shade":
                    ax.fill_between(s.t, y - e, y + e, color=c, alpha=0.2, lw=0)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if t_range:
            ax.set_xlim(t_range)
    axes[-1, 0].set_xlabel("t [s]")
    _legend(axes[0, 0], shots, line=False)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
def profile_grid(shots, rows, times, xlabel, color_by="auto", err_style="bar",
                 capsize=CAPSIZE, figsize=None, ms=4, shade=None):
    """rows: list of dicts {label, get, [scale], [ylim], [xlim]} where
    get(shot) -> (x, y, yerr[, t]) with x, y, yerr shaped (time, space) or
    (space,) for x only; yerr may be None; t is the time axis of the data
    (default: the shot's common 40-frame axis, e.g. Langmuir probes give their
    own native axis). One column per entry of `times`.

    Points are drawn as markers only (no line). err_style: 'bar' = error bars,
    'shade' = shaded band y +- err (drawn through the points in x order),
    None = no errors. ms = marker size; shade = alpha of the marker fill (None = solid
    markers), the edge stays solid."""
    fig, axes = plt.subplots(len(rows), len(times), sharex="col", squeeze=False,
                             figsize=figsize or (3.3 * len(times) + 0.8, 2.3 * len(rows) + 0.6))
    for i, row in enumerate(rows):
        scale = row.get("scale", 1)
        for j, tt in enumerate(times):
            ax = axes[i, j]
            for s in shots:
                try:
                    x, y, e, *tax = row["get"](s)
                except Unavailable as ex:
                    _warn(s, row["label"], ex)
                    continue
                k = int(np.argmin(abs((tax[0] if tax else s.t) - tt)))
                xs = x[k] if x.ndim == 2 else x
                ys = y[k] * scale
                es = None if e is None else e[k] * scale
                ok = ~np.isnan(xs) & ~np.isnan(ys)
                if not ok.any():
                    continue
                o = np.argsort(xs[ok])
                c = s.color(color_by)
                xo, yo = xs[ok][o], ys[ok][o]
                mk = dict(marker=s.marker, ms=ms)
                if shade is not None:
                    mk.update(mfc=to_rgba(c, shade), mec=c, mew=1.3)
                if es is not None and err_style == "bar":
                    ax.errorbar(xo, yo, yerr=es[ok][o], color=c, ls="none", **mk,
                                elinewidth=0.8, capsize=capsize, capthick=0.8)
                else:
                    ax.plot(xo, yo, color=c, ls="none", **mk)
                    if es is not None and err_style == "shade":
                        ax.fill_between(xo, yo - es[ok][o], yo + es[ok][o], color=c,
                                        alpha=0.2, lw=0)
            ax.grid(alpha=0.3)
            if "ylim" in row:
                ax.set_ylim(row["ylim"])
            if "xlim" in row:
                ax.set_xlim(row["xlim"])
            if j == 0:
                ax.set_ylabel(row["label"])
            if i == 0:
                ax.set_title(f"t = {tt:.2f} s", fontsize=10)
        for j in range(len(times)):
            axes[-1, j].set_xlabel(xlabel)
    _legend(axes[0, -1], shots, line=False)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
def map_grid(shots, getter, times, label="", cmap="inferno", vmax=None, log=False,
             overlay_equilibrium=True, figsize=None):
    """getter(shot) -> dict(r_values, z_values, values) for a triangular mesh
    (r_values, z_values: (n_tri, 3); values: (n_tri, t)) or
    dict(r, z, values) for a regular grid (values: (R, Z, t)).
    One row per shot, one column per time; common colour scale."""
    data = []
    for s in shots:
        try:
            data.append(getter(s))
        except Unavailable as ex:
            _warn(s, label or "map", ex)
            data.append(None)
    vals = [d["values"][..., s.frame(t)] for d, s in zip(data, shots) if d for t in times]
    if not vals:
        raise Unavailable("no map data in any shot")
    allv = np.concatenate([v[np.isfinite(v)] for v in vals])
    lo, hi = (np.nanpercentile(allv[allv > 0], 1), vmax or np.nanpercentile(allv, 99.5)) \
        if log else (0, vmax or np.nanpercentile(allv, 99.5))
    norm = plt.matplotlib.colors.LogNorm(lo, hi) if log else plt.Normalize(lo, hi)

    fig, axes = plt.subplots(len(shots), len(times), squeeze=False, sharex=True, sharey=True,
                             figsize=figsize or (2.6 * len(times) + 1.4, 3.8 * len(shots) + 0.4))
    for i, (s, d) in enumerate(zip(shots, data)):
        for j, tt in enumerate(times):
            ax = axes[i, j]
            ax.set_aspect("equal")
            k = s.frame(tt)
            if d is not None:
                v = d["values"][..., k]
                if "r_values" in d:
                    verts = np.stack([d["r_values"], d["z_values"]], axis=-1)
                    pc = PolyCollection(verts, array=np.where(np.isfinite(v), v, np.nan),
                                        cmap=cmap, norm=norm, edgecolors="none")
                    ax.add_collection(pc)
                else:
                    pc = ax.pcolormesh(d["r"], d["z"], v.T, cmap=cmap, norm=norm, shading="auto")
            draw_wall(ax, s)
            if overlay_equilibrium and s.has("equilibrium/r_contour"):
                ax.plot(s["equilibrium/r_contour"][:, k], s["equilibrium/z_contour"][:, k],
                        color="c", lw=1)
                ax.plot(s["equilibrium/r_axis"][k], s["equilibrium/z_axis"][k], "c+")
            ax.autoscale_view()
            if i == 0:
                ax.set_title(f"t = {tt:.2f} s", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{s.label}\nZ [m]", fontsize=8)
            if i == len(shots) - 1:
                ax.set_xlabel("R [m]")
    fig.subplots_adjust(right=0.88, wspace=0.05)
    cax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, label=label)
    return fig, axes


# ---------------------------------------------------------------------------
def xy_plot(shots, x_get, y_get, xlabel, ylabel, yerr_get=None, t_range=(0.6, 1.6),
            color_by="auto", ax=None, annotate=True, err_style="bar", capsize=CAPSIZE):
    """Scatter of y vs x for every frame (markers only); first/last frame
    labelled with their time. Error bars if yerr_get is given and err_style='bar'."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure
    for s in shots:
        try:
            x, y = x_get(s), y_get(s)
            e = yerr_get(s) if yerr_get else None
        except Unavailable as ex:
            _warn(s, ylabel, ex)
            continue
        m = (s.t >= t_range[0]) & (s.t <= t_range[1]) & np.isfinite(x) & np.isfinite(y)
        c = s.color(color_by)
        ax.scatter(x[m], y[m], color=c, marker=s.marker, s=18, zorder=3)
        if e is not None and err_style == "bar":
            ax.errorbar(x[m], y[m], yerr=e[m], fmt="none", ecolor=c, elinewidth=0.8, alpha=0.6, capsize=capsize, capthick=0.8)
        if annotate and m.any():
            for idx in (np.flatnonzero(m)[0], np.flatnonzero(m)[-1]):
                ax.annotate(f"{s.t[idx]:.2f}s", (x[idx], y[idx]), fontsize=7, color=c,
                            xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    _legend(ax, shots, line=False)
    return fig, ax
