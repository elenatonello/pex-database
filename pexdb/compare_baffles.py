#!/usr/bin/env python3
"""Compare SILO (red) and NINO (blue) shots, screened by discharge type.

Every shot is put in a category = (baffle, sign of B0, plasma current in kA), taken from
the file (attribute 'baffle', median of equilibrium/b0 and |equilibrium/ip| over --t-range).
Colour = baffle; marker shape = (B0 sign, Ip); shots of one category look identical.

  Figure 1 (fraction): fraction of the input power reaching the targets,
                       (HIR+VIR) / P_in, with P_in = OH+ECRH+NBI - P_rad,core, vs ne_core (TS)
  Figure 2 (pressure): divertor pressure [mPa] vs ne_core (TS)
                       BAR baratron (shaded markers) and APG gauge (open markers)
  Figure 3 (upstream): ne, Te (Thomson) and Ti (CXRS) profiles at selected times,
                       one column per time (x = dru [mm] cut at --xmin, default -70; --x rho for rho)

ne_core = Thomson line-average density (thomson_scattering/av_dens).
Shaded bands = 16-84 percentile of each baffle (all categories pooled) in bins of ne_core (--no-bands to remove).

Usage:
  python3 -m pexdb.compare_baffles $PEX_DB_DIR                 # no --save: just show
  python3 -m pexdb.compare_baffles $PEX_DB_DIR --save figures/baffle --ask   # show, then ask
  python3 -m pexdb.compare_baffles $PEX_DB_DIR/shot_8*.h5 --save figures/baffle --t-range 0.6 1.6
"""
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

from pexdb import shotdb
from pexdb.plotting import plot_types as pt
from pexdb.plotting import diag_plots as dp
from pexdb.shotdb import Unavailable

PRAD = "bolometer/power_core"     # core radiated power subtracted from the heating
FIG = (5.6, 5.2)                  # single-panel figures [in]: nearly square
FONT = 12                         # base font size [pt]
MS = 11                           # marker size in the line plots [pt]
S = 110                           # marker area in the scatter plots [pt^2]
SHADE = 0.35                      # alpha of the marker fill (edge is solid)
XLABEL = r"$n_{e,core}$ (TS) [$10^{19}$ m$^{-3}$]"


# ---------------------------------------------------------------------------
# Classification: baffle x B0 sign x Ip
# ---------------------------------------------------------------------------
def classify(shots, t_range=(0.6, 1.6)):
    """Set s.b0_sign ('+', '-', '?'), s.ip_ka (rounded to 10 kA, None if unknown), s.cat
    (baffle, b0_sign, ip_ka) and s.index (marker index: one per (b0_sign, ip_ka) pair)."""
    for s in shots:
        m = (s.t >= t_range[0]) & (s.t <= t_range[1])

        def med(path):
            try:
                v = s[path][m]
                return float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
            except Unavailable:
                return np.nan
        b0, ip = med("equilibrium/b0"), med("equilibrium/ip")
        s.b0_sign = "?" if not np.isfinite(b0) or b0 == 0 else "+" if b0 > 0 else "-"
        s.ip_ka = int(round(abs(ip) * 1e-3 / 10) * 10) if np.isfinite(ip) else None
        s.cat = (s.baffle, s.b0_sign, s.ip_ka)
    pairs = sorted({s.cat[1:] for s in shots}, key=lambda p: (p[0], p[1] or 0))
    for s in shots:
        s.index = pairs.index(s.cat[1:])
    return shots


def cat_label(cat, n=None):
    baffle, b0, ip = cat
    return (f"{baffle or '?'} B0{b0} " + (f"{ip}kA" if ip else "Ip?")
            + (f" ({n})" if n else ""))


def print_table(shots):
    print("  shot   baffle  B0   Ip[kA]")
    for s in shots:
        print(f"  {s.shot:<6d} {s.baffle or '?':<6s}  {s.b0_sign}    {s.ip_ka or '?'}")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def _band(ax, x, y, color, nbins=6, minpts=3):
    """Shaded 16-84 percentile band of y in bins of x, with the median as a thin line."""
    if len(x) < 2 * minpts or np.ptp(x) == 0:
        return
    edges = np.linspace(x.min(), x.max(), nbins + 1)
    xc, lo, md, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        k = (x >= a) & ((x <= b) if b == edges[-1] else (x < b))
        if k.sum() >= minpts:
            xc.append(x[k].mean())
            lo.append(np.percentile(y[k], 16))
            md.append(np.median(y[k]))
            hi.append(np.percentile(y[k], 84))
    if len(xc) >= 2:
        ax.fill_between(xc, lo, hi, color=color, alpha=0.15, lw=0, zorder=1)
        ax.plot(xc, md, color=color, lw=1.2, alpha=0.6, zorder=1)


def _panel(ax, shots, get, ylabel, t_range, filled=True, bands=True):
    """Scatter get(shot) [already scaled] vs ne_core. Shaded markers with solid edge
    (filled=False: open markers), colour = baffle, shape = (B0 sign, Ip) category."""
    pooled = {}
    for s in shots:
        try:
            x = s["thomson_scattering/av_dens"] * 1e-19
            y = get(s)
        except Unavailable as ex:
            pt._warn(s, ylabel, ex)
            continue
        m = (s.t >= t_range[0]) & (s.t <= t_range[1]) & np.isfinite(x) & np.isfinite(y)
        c = s.color("baffle")
        ax.scatter(x[m], y[m], s=S, marker=s.marker, edgecolors=c, linewidths=1.5, zorder=3,
                   facecolors=to_rgba(c, SHADE) if filled else "none")
        xs, ys = pooled.setdefault(s.baffle, ([], []))
        xs.append(x[m])
        ys.append(y[m])
    if bands and filled:
        for baffle, (xs, ys) in pooled.items():                  # one band per baffle (all categories)
            _band(ax, np.concatenate(xs), np.concatenate(ys), shotdb.BAFFLE_COLORS[baffle])
    ax.set_xlabel(XLABEL)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def _legend(ax, shots, loc="best", ncol=1):
    """One entry per category (not per shot), with the number of shots."""
    cats = {}
    for s in shots:
        cats.setdefault(s.cat, [s, 0])[1] += 1
    handles = [plt.Line2D([], [], color=s.color("baffle"), marker=s.marker, ls="none", ms=MS * 0.8,
                          mfc=to_rgba(s.color("baffle"), SHADE), mew=1.5, label=cat_label(c, n))
               for c, (s, n) in sorted(cats.items(), key=lambda kv: (kv[0][0] or "", kv[1][0].index))]
    return ax.legend(handles=handles, fontsize=FONT - 3, frameon=False, loc=loc, ncol=ncol,
                     columnspacing=1.0, handletextpad=0.3)


def _target_fraction(s):
    p_in = dp.heating_total(s) - s[PRAD]
    p_t = s["camera_ir/HIR/power"] + s["camera_ir/VIR/power"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p_in > 0, p_t / p_in, np.nan)             # no heating -> no fraction


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fraction_vs_necore(shots, t_range=(0.6, 1.6), bands=True):
    """Fraction of the input power that reaches the targets: (HIR+VIR) / P_in."""
    fig, ax = plt.subplots(figsize=FIG)
    _panel(ax, shots, _target_fraction, r"$(P_{HIR}+P_{VIR})\,/\,P_{in}$", t_range, bands=bands)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.45)                   # headroom for the legend
    _legend(ax, shots, loc="upper center", ncol=2)
    fig.tight_layout()
    return fig


def pressure_vs_necore(shots, t_range=(0.6, 1.6), bands=True):
    fig, ax = plt.subplots(figsize=FIG)
    _panel(ax, shots, lambda s: s["barometry/barometer/pressure_div"] * 1e3,
           "divertor pressure [mPa]", t_range, bands=bands)
    _panel(ax, shots, lambda s: s["barometry/pressure_gauges/pressure_div"] * 1e3,
           "divertor pressure [mPa]", t_range, filled=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.7)                    # headroom for the legends
    leg = ax.legend(handles=[
        plt.Line2D([], [], color="gray", marker="o", ls="none", ms=MS * 0.8,
                   mfc=to_rgba("gray", SHADE), mew=1.5, label="BAR (baratron)"),
        plt.Line2D([], [], color="gray", marker="o", mfc="none", ls="none", ms=MS * 0.8, mew=1.5,
                   label="APG (gauge)")],
        fontsize=FONT - 3, frameon=False, loc="upper left")
    ax.add_artist(leg)
    _legend(ax, shots, loc="upper right", ncol=1)
    fig.tight_layout()
    return fig


def upstream_profiles(shots, times=(0.8, 1.0, 1.2), x="dsep", xmin=None):
    """xmin: lower limit of the radial axis, in mm for x='dsep' (rho for x='rho')."""
    rows = 3 if x == "rho" else 2
    fig, axes = dp.profiles_ne_te_ti(shots, times, x=x, color_by="baffle", ms=MS, shade=SHADE,
                                     err_style="shade",
                                     xrange=None if xmin is None else (xmin, None),
                                     figsize=(3.7 * len(times), 3.3 * rows))
    for ax in axes[0]:
        ax.set_title(ax.get_title(), fontsize=FONT)
    _legend(axes[0, -1], shots)                                  # replaces the default per-shot one
    fig.tight_layout()
    return fig


def compare(shots, t_range=(0.6, 1.6), times=(0.8, 1.0, 1.2), x="dsep", xmin=None, bands=True,
            save=None):
    """shots: list of shotdb.Shot (those without a SILO/NINO baffle are dropped)."""
    keep = [s for s in shots if s.baffle in shotdb.BAFFLE_COLORS]
    for s in shots:
        if s not in keep:
            print(f"  skip {s.label}: baffle '{s.baffle or '?'}' is neither NINO nor SILO")
    classify(keep, t_range)
    print_table(keep)
    with plt.rc_context({"font.size": FONT}):
        figs = {"fraction": fraction_vs_necore(keep, t_range, bands),
                "pressure": pressure_vs_necore(keep, t_range, bands),
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
    ap.add_argument("--no-bands", action="store_true", help="no shaded percentile bands")
    ap.add_argument("--save", help="filename prefix: writes <prefix>_{fraction,pressure,upstream}.png")
    ap.add_argument("--ask", action="store_true",
                    help="with --save: show the figures first and save only if you answer y")
    args = ap.parse_args()
    shots = shotdb.load(args.source)
    if args.ask or not args.save:
        bk = matplotlib.get_backend()
        if bk.lower() in {"agg", "pdf", "ps", "svg", "pgf", "cairo", "template"}:
            raise SystemExit(f"matplotlib backend '{bk}' cannot open windows (no display?). "
                             "Use --save <prefix> and view the PNGs, ssh -X, or set "
                             "MPLBACKEND=TkAgg / QtAgg / MacOSX.")
    xmin = args.xmin if args.xmin is not None else (-70.0 if args.x == "dsep" else None)
    figs = compare(shots, tuple(args.t_range), tuple(args.times), args.x, xmin,
                   not args.no_bands, None if args.ask else args.save)
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
