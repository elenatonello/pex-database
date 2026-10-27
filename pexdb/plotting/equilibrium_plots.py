"""Wall structure and equilibrium of a shot: flux surfaces (dashed) and separatrix (solid).

    from pexdb import shotdb
    from pexdb.plotting import equilibrium_plots as ep
    s = shotdb.load("shot_88306.h5")[0]
    fig, ax = ep.plot_equilibrium(s, t=1.0)                     # whole poloidal cross-section
    fig, ax = ep.plot_equilibrium(s, t=1.0, zoom="lower")       # lower divertor
    fig, ax = ep.plot_equilibrium(s, colors="tab:red")          # any colour for the equilibrium
    fig, ax = ep.plot_equilibrium(s, colors=dict(separatrix="m", surfaces="c", wall="k"), frame=False)

    python3 -m pexdb.plotting.equilibrium_plots $PEX_DB_DIR/shot_88306.h5 --t 1.0 --save figures/eq

Drawn at the equilibrium frame closest to t: the wall (vessel band between the inner and outer contour, tiles as
a thick line; the tiles of a SILO shot include the baffles), the separatrix (psi = 0 of the LIUQE map: plasma
boundary and both divertor legs), dashed flux surfaces rho = `core` (inside) and `sol` (outside the
separatrix; rho = sqrt of the normalised poloidal flux), the magnetic axis (+) and the lower X-point (x).
Separatrix and flux surfaces are clipped to the tile outline. The LIUQE map is coarse (28 x 65): it is
spline-interpolated before contouring. The X-point is read from equilibrium/{r_xpt, z_xpt} when the file has
them, otherwise computed from psi.
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

from pexdb import shotdb
from pexdb import ciii_front as cf

# colours: any matplotlib colour. plot_equilibrium(colors=...) takes one colour (used for the separatrix, the flux
# surfaces and the markers) or a dict with any of these keys (the others keep these defaults)
DEFAULT_COLORS = dict(wall="k", vessel="0.88", vessel_edge="0.45", separatrix="k", surfaces="0.55", markers="k",
                      gap=None)      # gap: fill colour of the region between the vessel and the tiles (None: not filled)
CORE = (0.2, 0.4, 0.6, 0.8)
SOL = (1.02, 1.05, 1.1, 1.2, 1.3, 1.4)


def _lower_xpoint(s, R, Z, psi, z_axis):
    if s.has("equilibrium/r_xpt") and s.has("equilibrium/z_xpt"):
        return np.asarray(s["equilibrium/r_xpt"], float), np.asarray(s["equilibrium/z_xpt"], float)
    xp = np.array([cf.lower_xpoint(R, Z, psi[:, :, k], z_axis[k]) for k in range(psi.shape[2])]).T
    return xp[0], xp[1]


def plot_equilibrium(shot, t=1.0, zoom="full", core=CORE, sol=SOL, colors=None, ax=None, title=False,
                     legend=False, frame=True, lw_sep=2.6, lw_wall=2.4, lw_surf=1.0, alpha_surf=1.0):
    """Plot the wall and the equilibrium of `shot` (a shotdb.Shot) at time t [s]. Returns (fig, ax).
    zoom: 'full' (whole cross-section) or 'lower' (lower divertor). core / sol: rho of the dashed flux
    surfaces inside / outside the separatrix (empty = none). colors: one matplotlib colour (separatrix, flux
    surfaces and markers) or a dict with keys wall, vessel, vessel_edge, separatrix, surfaces, markers, gap
    (defaults: DEFAULT_COLORS; `gap` = fill colour of the region between the vessel and the wall tiles, baffle
    structures included, None = unfilled). lw_*: line widths; alpha_surf: opacity of the dashed flux surfaces.
    ax: draw on an existing axis. title: '#shot (baffle), t' above the plot. legend: separatrix / flux
    surfaces / wall tiles. frame=False: no box, axes, ticks or labels (only the drawing)."""
    c = dict(DEFAULT_COLORS)
    if isinstance(colors, str):
        c.update(separatrix=colors, surfaces=colors, markers=colors)
    else:
        c.update(colors or {})
    s = cf._native(shot)                           # ELM-mode files: the equilibrium on its own (5 ms) grid
    te = np.asarray(s.time_of("equilibrium/psi"), float)
    k = int(np.argmin(abs(te - t)))
    if abs(te[k] - t) > 0.05:
        print(f"  WARNING: shot {s.shot}: closest equilibrium frame is at t = {te[k]:.3f} s (asked {t:g} s)")
    Rc, Zc = np.asarray(s["equilibrium/r"], float), np.asarray(s["equilibrium/z"], float)
    psi_all = np.asarray(s["equilibrium/psi"], float)
    R, Z = np.linspace(Rc[0], Rc[-1], 500), np.linspace(Zc[0], Zc[-1], 700)
    psi = RectBivariateSpline(Rc, Zc, psi_all[:, :, k])(R, Z)
    rho = RectBivariateSpline(Rc, Zc, np.asarray(s["equilibrium/rho"], float)[:, :, k])(R, Z)
    r_ax, z_ax = np.asarray(s["equilibrium/r_axis"], float), np.asarray(s["equilibrium/z_axis"], float)
    rx, zx = _lower_xpoint(s, Rc, Zc, psi_all, z_ax)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.0, 7.6) if zoom == "full" else (6.4, 7.0))
    else:
        fig = ax.figure
    wall = s.wall()
    vi, vo = wall.get("vessel_inner"), wall.get("vessel_outer")
    if vi is not None and vo is not None:
        ring = np.concatenate([np.column_stack(vo), np.column_stack(vi)[::-1]])
        ax.add_patch(plt.Polygon(ring, closed=True, fc=c["vessel"], ec="none", zorder=0, clip_on=False))
        for v in (vi, vo):                       # the two contours as lines (no closing seam between them)
            ax.plot(*v, color=c["vessel_edge"], lw=0.8, zorder=1, clip_on=False)
    tiles = np.column_stack(wall["tiles"])
    if c["gap"] is not None and vi is not None:           # vessel (inner contour) minus the tile outline
        from matplotlib.path import Path
        from matplotlib.patches import PathPatch
        ring_i = np.column_stack(vi)
        verts = np.concatenate([ring_i, ring_i[:1], tiles, tiles[:1]])
        codes = np.concatenate([[Path.MOVETO], [Path.LINETO] * (len(ring_i) - 1), [Path.CLOSEPOLY],
                                [Path.MOVETO], [Path.LINETO] * (len(tiles) - 1), [Path.CLOSEPOLY]])
        ax.add_patch(PathPatch(Path(verts, codes), fc=c["gap"], ec="none", zorder=0.5, clip_on=False))
    inside = plt.Polygon(tiles, closed=True, fc="none", ec="none")      # plasma-facing region (baffles excluded)
    ax.add_patch(inside)
    levels = [r for r in core if r < 1] + [r for r in sol if r > 1]
    if levels:
        cs = ax.contour(R, Z, rho.T, levels=sorted(levels), colors=c["surfaces"], linewidths=lw_surf,
                        linestyles="--", alpha=alpha_surf, zorder=2)
        cs.set_clip_path(inside)
    cs = ax.contour(R, Z, psi.T, levels=[0.0], colors=c["separatrix"], linewidths=lw_sep, zorder=4)
    cs.set_clip_path(inside)
    ax.plot(*tiles.T, color=c["wall"], lw=lw_wall, zorder=5, clip_on=False)
    ax.plot(r_ax[k], z_ax[k], "+", color=c["markers"], ms=11, mew=2, zorder=6)
    if np.isfinite(rx[k]):
        ax.plot(rx[k], zx[k], "x", color=c["markers"], ms=10, mew=2.5, zorder=6)
    ax.set_aspect("equal")
    if zoom == "lower":
        ax.set_xlim(0.60, 1.15)
        ax.set_ylim(-0.77, -0.05)
    elif zoom == "full":                          # the whole vessel (outer contour: R 0.585-1.18, |Z| <= 0.79) + margin
        vmin, vmax = (0.585, 1.18)
        ax.set_xlim(min(0.55, vmin - 0.03), max(1.25, vmax + 0.03))
        ax.set_ylim(-0.83, 0.83)
    else:
        raise ValueError("zoom must be 'full' or 'lower'")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    if not frame:
        ax.axis("off")
    if title:
        ax.set_title(f"#{s.shot} ({s.baffle}),  t = {te[k]:.2f} s")
    if legend:
        h = [plt.Line2D([], [], color=c["separatrix"], lw=lw_sep, label="separatrix"),
             plt.Line2D([], [], color=c["surfaces"], lw=1, ls="--", label=r"flux surfaces ($\rho$)"),
             plt.Line2D([], [], color=c["wall"], lw=lw_wall, label="wall tiles")]
        ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    return fig, ax


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("source", nargs="+", help="shot_*.h5 file(s)")
    ap.add_argument("--t", type=float, default=1.0)
    ap.add_argument("--zoom", choices=("full", "lower"), default="full")
    ap.add_argument("--color", default="k", help="colour of the separatrix, flux surfaces and markers (any matplotlib colour)")
    ap.add_argument("--wall-color", default="k")
    ap.add_argument("--no-frame", action="store_true", help="no box / axes")
    ap.add_argument("--save", help="prefix: writes <prefix>_<shot>.png")
    a = ap.parse_args()
    for s in shotdb.load(a.source):
        fig, ax = plot_equilibrium(s, a.t, a.zoom, colors=dict(separatrix=a.color, surfaces=a.color, markers=a.color, wall=a.wall_color),
                                   title=True, legend=True, frame=not a.no_frame)
        if a.save:
            os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
            fig.savefig(f"{a.save}_{s.shot}.png", dpi=200)
            print("wrote", f"{a.save}_{s.shot}.png")
    if not a.save:
        plt.show()


if __name__ == "__main__":
    main()
