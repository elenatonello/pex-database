#!/usr/bin/env python3
"""Compare SILO (red) and NINO (blue) shots against the core Thomson density.

  Figure 1 (fraction): fraction of the input power reaching the targets,
                       (HIR+VIR) / P_in, with P_in = OH+ECRH+NBI - P_rad,core, vs ne_core (TS)
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
FIG = (7.5, 5.5)
FONT = 15                          # base font size [pt]; markers / legends scale with it
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
        ax.scatter(x[m], y[m], s=60, marker=marker or s.marker,
                   **(dict(color=c) if filled else dict(facecolors="none", edgecolors=c)))
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def _legend(ax, shots, loc="best"):
    ax.legend(handles=[plt.Line2D([], [], color=s.color("baffle"), marker=s.marker, ls="none", ms=9,
                                  label=f"#{s.shot} {s.baffle or '?'}") for s in shots],
              fontsize=FONT - 3, frameon=False, loc=loc)


def fraction_vs_necore(shots, t_range=(0.6, 1.6)):
    """Fraction of the input power that reaches the targets: (HIR+VIR) / P_in."""
    fig, ax = plt.subplots(figsize=FIG)
    _panel(ax, shots, lambda s: _target_fraction(s),
           r"$(P_{HIR}+P_{VIR})\,/\,P_{in}$", t_range)
    ax.set_ylim(bottom=0)
    _legend(ax, shots)
    fig.tight_layout()
    return fig


def _target_fraction(s):
    p_in = dp.heating_total(s) - s[PRAD]
    p_t = s["camera_ir/HIR/power"] + s["camera_ir/VIR/power"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p_in > 0, p_t / p_in, np.nan)             # no heating -> no fraction


def pressure_vs_necore(shots, t_range=(0.6, 1.6)):
    fig, ax = plt.subplots(figsize=FIG)
    _panel(ax, shots, lambda s: s["barometry/barometer/pressure_div"] * 1e3,
           "divertor pressure [mPa]", t_range)
    _panel(ax, shots, lambda s: s["barometry/pressure_gauges/pressure_div"] * 1e3,
           "divertor pressure [mPa]", t_range, filled=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)                    # headroom for the legends
    leg = ax.legend(handles=[
        plt.Line2D([], [], color="gray", marker="o", ls="none", ms=9, label="BAR (baratron)"),
        plt.Line2D([], [], color="gray", marker="o", mfc="none", ls="none", ms=9, label="APG (gauge)")],
        fontsize=FONT - 3, frameon=False, loc="upper left")
    ax.add_artist(leg)
    _legend(ax, shots, loc="upper right")
    fig.tight_layout()
    return fig


def upstream_profiles(shots, times=(0.8, 1.0, 1.2), x="dsep", xmin=None):
    """xmin: lower limit of the radial axis, in mm for x='dsep' (rho for x='rho')."""
    rows = 3 if x == "rho" else 2
    fig, axes = dp.profiles_ne_te_ti(shots, times, x=x, color_by="baffle", ms=9, capsize=3,
                                     xrange=None if xmin is None else (xmin, None),
                                     figsize=(5.2 * len(times), 3.6 * rows))
    for ax in axes[0]:
        ax.set_title(ax.get_title(), fontsize=FONT)
    _legend(axes[0, -1], shots)                                  # replaces the small default
    fig.tight_layout()
    return fig


def compare(shots, t_range=(0.6, 1.6), times=(0.8, 1.0, 1.2), x="dsep", xmin=None, save=None):
    """shots: list of shotdb.Shot (those without a SILO/NINO baffle are dropped)."""
    keep = [s for s in shots if s.baffle in shotdb.BAFFLE_COLORS]
    for s in shots:
        if s not in keep:
            print(f"  skip {s.label}: baffle '{s.baffle or '?'}' is neither NINO nor SILO")
    with plt.rc_context({"font.size": FONT}):
        figs = {"fraction": fraction_vs_necore(keep, t_range),
                "pressure": pressure_vs_necore(keep, t_range),
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
    ap.add_argument("--save", help="filename prefix: writes <prefix>_{fraction,pressure,upstream}.png")
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
