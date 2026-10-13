#!/usr/bin/env python3
"""2D map of the perpendicular heat flux q_perp on the VIR (outer) target:
R (y) vs time (x), colour = q_perp, one panel per shot, common colour scale.

Titles are '#shot BAFFLE' in the baffle colour (SILO red, NINO blue).
qperp is assumed to be stored in W/m^2 and is shown in MW/m^2 (SCALE below).

Usage:
  python3 -m pexdb.vir_map $PEX_DB_DIR                          # show all shots
  python3 -m pexdb.vir_map $PEX_DB_DIR/shot_89*.h5 --save figures/vir_map.png
  python3 -m pexdb.vir_map shot_89437.h5 --t-range 0.5 1.8 --target HIR
"""
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from pexdb import shotdb
from pexdb.plotting import plot_types as pt
from pexdb.shotdb import Unavailable

SCALE = 1e-6          # W/m^2 -> MW/m^2 (units of qperp are not stored in the database)
FONT = 12


def vir_map(shots, t_range=(0.6, 1.6), target="VIR", ncol=4, cmap="inferno"):
    """q_perp(R, t) of camera_ir/<target> for each shot. Returns the figure."""
    data = []
    for s in shots:
        try:
            q = s[f"camera_ir/{target}/qperp"]                   # (position, t)
            r = s[f"camera_ir/{target}/r"]
            if r.ndim > 1:                                       # R varying in time: use the median
                r = np.nanmedian(r, axis=r.shape.index(len(s.t)) if len(s.t) in r.shape else 0)
            if q.shape[0] != r.size:
                q = q.T
            o = np.argsort(r)
            data.append((r[o], q[o] * SCALE))
        except Unavailable as ex:
            pt._warn(s, f"{target} q_perp map", ex)
            data.append(None)
    vals = [d[1][:, (s.t >= t_range[0]) & (s.t <= t_range[1])] for s, d in zip(shots, data) if d]
    vals = np.concatenate([v[np.isfinite(v)] for v in vals]) if vals else np.array([])
    if vals.size == 0:
        raise Unavailable(f"no {target} q_perp data in any shot")
    vmax = np.percentile(vals, 99.5)

    n = len(shots)
    nc = min(n, ncol)
    nr = -(-n // nc)
    with plt.rc_context({"font.size": FONT}):
        fig, axes = plt.subplots(nr, nc, squeeze=False, sharex=True, sharey=True,
                                 figsize=(3.2 * nc + 0.9, 2.9 * nr))
        mesh = None
        for ax, s, d in zip(axes.ravel(), shots, data):
            ax.set_title(f"#{s.shot} {s.baffle or '?'}", fontsize=FONT, color=s.color("baffle"))
            if d is None:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color="gray")
                continue
            r, q = d
            m = (s.t >= t_range[0]) & (s.t <= t_range[1])
            mesh = ax.pcolormesh(s.t[m], r, q[:, m], cmap=cmap, vmin=0, vmax=vmax,
                                 shading="nearest", rasterized=True)
        for ax in axes.ravel()[n:]:
            ax.set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel("t [s]")
        for ax in axes[:, 0]:
            ax.set_ylabel("R [m]")
        fig.tight_layout()
        if mesh is not None:
            fig.colorbar(mesh, ax=axes, label=rf"$q_\perp$ {target} [MW m$^{{-2}}$]", pad=0.02)
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("source", nargs="+", help="folder(s), file(s) or glob(s) of shot_*.h5")
    ap.add_argument("--t-range", type=float, nargs=2, default=(0.6, 1.6), metavar=("T0", "T1"))
    ap.add_argument("--target", choices=("VIR", "HIR"), default="VIR", help="IR camera (default VIR)")
    ap.add_argument("--ncol", type=int, default=4, help="panels per row")
    ap.add_argument("--save", help="output PNG file (default: show on screen)")
    args = ap.parse_args()
    if not args.save and matplotlib.get_backend().lower() in {"agg", "pdf", "ps", "svg", "pgf",
                                                              "cairo", "template"}:
        raise SystemExit("matplotlib backend cannot open windows (no display?). Use --save file.png "
                         "or set MPLBACKEND=TkAgg / QtAgg / MacOSX.")
    fig = vir_map(shotdb.load(args.source), tuple(args.t_range), args.target, args.ncol)
    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"  wrote {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
