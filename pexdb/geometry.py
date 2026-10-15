#!/usr/bin/env python3
"""Poloidal geometry of the diagnostics, in two figures (one panel per shot, with wall and LCFS):

  Figure 1 (plasma):    Thomson, CXRS, Langmuir probes (inner / outer target / embedded), FIR chords,
                        IR cameras VIR and HIR
  Figure 2 (radiation): bolometer lines of sight, MANTIS field of view, DSS lines of sight,
                        pressure gauge ports, gas valve

A diagnostic missing in a shot is listed under its panel.
The layers of each figure are the lists in SETS below; `python3 -m pexdb.geometry --list` prints all names.

Usage:
  python3 -m pexdb.geometry $PEX_DB_DIR/shot_89437.h5 $PEX_DB_DIR/shot_89837.h5
  python3 -m pexdb.geometry $PEX_DB_DIR --save figures/geom --t 1.1     # writes figures/geom_{plasma,radiation}.png
  python3 -m pexdb.geometry $PEX_DB_DIR --save figures/geom --format svg   # or pdf
"""
import argparse
import matplotlib
import matplotlib.pyplot as plt

from pexdb import shotdb
from pexdb.plotting import diag_plots as dp

SETS = {
    "plasma": ["Thomson", "CXRS", "LP outer target", "LP inner target", "LP embedded", "FIR chords",
               "IR HIR", "IR VIR"],
    "radiation": ["bolometer LOS", "MANTIS FOV", "DSS LOS", "MANTIS FOV", "pressure gauge ports", "gas valve"],
}


def geometry(shots, t=1.0, sets=SETS):
    """{name: fig}, one plot_geometry figure per entry of `sets` (name -> layer labels)."""
    figs = {}
    for name, layers in sets.items():
        fig, _ = dp.plot_geometry(shots, t=t, layers=layers)
        fig.suptitle(name, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))             # keep the title clear of the panel titles
        figs[name] = fig
    return figs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("source", nargs="*", help="folder(s), file(s) or glob(s) of shot_*.h5")
    ap.add_argument("--list", action="store_true", help="print the layer names usable in SETS and exit")
    ap.add_argument("--t", type=float, default=1.0, help="time of the LCFS [s] (default 1.0)")
    ap.add_argument("--save", help="filename prefix: writes <prefix>_{plasma,radiation}.<format>")
    ap.add_argument("--format", default="png", choices=("png", "svg", "pdf"),
                    help="file format of the saved figures (default png)")
    args = ap.parse_args()
    if args.list:
        print("\n".join(label for label, *_ in dp._geometry_layers(None)))
        return
    if not args.source:
        ap.error("give at least one shot file or folder (or --list)")
    if not args.save and matplotlib.get_backend().lower() in {"agg", "pdf", "ps", "svg", "pgf",
                                                              "cairo", "template"}:
        raise SystemExit("matplotlib backend cannot open windows (no display?). Use --save <prefix> "
                         "or set MPLBACKEND=TkAgg / QtAgg / MacOSX.")
    figs = geometry(shotdb.load(args.source), args.t)
    if args.save:
        for name, fig in figs.items():
            fig.savefig(f"{args.save}_{name}.{args.format}", dpi=150)
            print(f"  wrote {args.save}_{name}.{args.format}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
