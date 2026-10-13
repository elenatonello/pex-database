#!/usr/bin/env python3
"""Compare SILO (red) and NINO (blue) shots against the core Thomson density.

  Figure 1 (power):    total target power HIR+VIR (filled) and input power
                       P_in = OH+ECRH+NBI - P_rad,core (open) [kW] vs ne_core (TS)
  Figure 2 (pressure): divertor pressure [mPa] vs ne_core (TS)
                       BAR baratron (filled) and APG gauge (open)

  Figure 3 (upstream): ne, Te (Thomson) and Ti (CXRS) profiles at selected times,
                       one column per time (x = dru [mm] cut at --xmin, default -70; --x rho for rho)

ne_core = Thomson line-average density (thomson_scattering/av_dens).
Colour = baffle (from the shot's 'baffle' attribute), marker = shot.

Usage:
  python3 -m pexdb.compare_baffles $PEX_DB_DIR                 # all shot_*.h5 in folder
  python3 -m pexdb.compare_baffles $PEX_DB_DIR/shot_8*.h5 --save figures/baffle --t-range 0.6 1.6
  python3 -m pexdb.compare_baffles $PEX_DB_DIR                 # no --save: just show
  python3 -m pexdb.compare_baffles $PEX_DB_DIR --save figures/baffle --ask   # show, then ask
"""
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from pexdb import shotdb
from pexdb.plotting import plot_types as pt
from pexdb.plotting import diag_plots as dp
from pexdb.shotdb import Unavailable

PRAD = "bolometer/power_core"     # core radiated power subtracted from the heating
XLABEL = r"$n_{e,core}$ (TS) [$10^{19}$ m$^{-3}$]"


def _panel(ax, shots, get, ylabel, t_range, marker=None, filled=True):
    """Scatter get(shot) [already scaled] vs ne_core for every shot, coloured by baffle."""
    for s in shots:
        try:
            x = s["thomson_scattering/av_dens"] * 1e-19
            y = get(s)
        except Unavailable as ex:
            pt._warn(s, ylabel, ex)
            continue
        m = (s.t >= t_range[0]) & (s.t <= t_range[1]) & np.isfinite(x) & np.isfinite(y)
        c = s.color("baffle")
        ax.scatter(x[m], y[m], s=18, marker=marker or s.marker,
                   **(dict(color=c) if filled else dict(facecolors="none", edgecolors=c)))
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def _legend(ax, shots, loc="best"):
    ax.legend(handles=[plt.Line2D([], [], color=s.color("baffle"), marker=s.marker, ls="none",
                                  label=f"#{s.shot} {s.baffle or '?'}") for s in shots],
              fontsize=7, frameon=False, loc=loc)


def power_vs_necore(shots, t_range=(0.6, 1.6)):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    _panel(ax, shots, lambda s: (s["camera_ir/HIR/power"] + s["camera_ir/VIR/power"]) * 1e-3,
           "power [kW]", t_range)
    _panel(ax, shots, lambda s: (dp.heating_total(s) - s[PRAD]) * 1e-3, "power [kW]", t_range, filled=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)                    # headroom for the legends
    leg = ax.legend(handles=[
        plt.Line2D([], [], color="gray", marker="o", ls="none", label="targets (IR HIR+VIR)"),
        plt.Line2D([], [], color="gray", marker="o", mfc="none", ls="none",
                   label=r"$P_{in}$ = OH+ECRH+NBI $-$ $P_{rad,core}$")],
        fontsize=8, frameon=False, loc="upper left")
    ax.add_artist(leg)
    _legend(ax, shots, loc="upper right")
    fig.tight_layout()
    return fig


def pressure_vs_necore(shots, t_range=(0.6, 1.6)):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    _panel(ax, shots, lambda s: s["barometry/barometer/pressure_div"] * 1e3,
           "divertor pressure [mPa]", t_range)
    _panel(ax, shots, lambda s: s["barometry/pressure_gauges/pressure_div"] * 1e3,
           "divertor pressure [mPa]", t_range, filled=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)                    # headroom for the legends
    leg = ax.legend(handles=[
        plt.Line2D([], [], color="gray", marker="o", ls="none", label="BAR (baratron)"),
        plt.Line2D([], [], color="gray", marker="o", mfc="none", ls="none", label="APG (gauge)")],
        fontsize=8, frameon=False, loc="upper left")
    ax.add_artist(leg)
    _legend(ax, shots, loc="upper right")
    fig.tight_layout()
    return fig


def upstream_profiles(shots, times=(0.8, 1.0, 1.2), x="dsep", xmin=None):
    """xmin: lower limit of the radial axis, in mm for x='dsep' (rho for x='rho')."""
    return dp.profiles_ne_te_ti(shots, times, x=x, color_by="baffle",
                                xrange=None if xmin is None else (xmin, None))[0]


def compare(shots, t_range=(0.6, 1.6), times=(0.8, 1.0, 1.2), x="dsep", xmin=None, save=None):
    """shots: list of shotdb.Shot (those without a SILO/NINO baffle are dropped)."""
    keep = [s for s in shots if s.baffle in shotdb.BAFFLE_COLORS]
    for s in shots:
        if s not in keep:
            print(f"  skip {s.label}: baffle '{s.baffle or '?'}' is neither NINO nor SILO")
    figs = {"power": power_vs_necore(keep, t_range), "pressure": pressure_vs_necore(keep, t_range),
            "upstream": upstream_profiles(keep, times, x, xmin)}
    if save:
        for name, fig in figs.items():
            fig.savefig(f"{save}_{name}.png", dpi=150)
    return figs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("source", nargs="+", help="folder(s), file(s) or glob(s) of shot_*.h5")
    ap.add_argument("--t-range", type=float, nargs=2, default=(0.6, 1.6), metavar=("T0", "T1"))
    ap.add_argument("--times", type=float, nargs="+", default=(0.8, 1.0, 1.2),
                    help="times [s] of the upstream profiles")
    ap.add_argument("--x", choices=("rho", "dsep"), default="dsep",
                    help="radial coordinate of the upstream profiles: dsep = dru, distance to the "
                         "separatrix at the midplane in mm (default, drops Ti), or rho")
    ap.add_argument("--xmin", type=float, default=None,
                    help="lower limit of the radial axis (mm for dsep, default -70; rho for rho)")
    ap.add_argument("--save", help="filename prefix: writes <prefix>_{power,pressure,upstream}.png")
    ap.add_argument("--ask", action="store_true",
                    help="with --save: show the figures first and save only if you answer y")
    args = ap.parse_args()
    shots = shotdb.load(args.source)
    print("  ".join(f"#{s.shot}:{s.baffle or '?'}" for s in shots))
    if args.ask or not args.save:
        bk = matplotlib.get_backend()
        if bk.lower() in {"agg", "pdf", "ps", "svg", "pgf", "cairo", "template"}:
            raise SystemExit(f"matplotlib backend '{bk}' cannot open windows (no display?). "
                             "Use --save <prefix> and view the PNGs, ssh -X, or set "
                             "MPLBACKEND=TkAgg / QtAgg / MacOSX.")
    xmin = args.xmin if args.xmin is not None else (-70.0 if args.x == "dsep" else None)
    figs = compare(shots, tuple(args.t_range), tuple(args.times), args.x, xmin,
                   None if args.ask else args.save)
    if args.ask and args.save:
        plt.show(block=False)
        plt.pause(0.5)                                  # let the windows draw
        if input(f"Save to {args.save}_*.png? [y/N] ").strip().lower() == "y":
            for name, fig in figs.items():
                fig.savefig(f"{args.save}_{name}.png", dpi=150)
                print(f"  wrote {args.save}_{name}.png")
    elif not args.save:
        plt.show()


if __name__ == "__main__":
    main()
