"""Plots that show what pexdb.ciii_front computes.

    from pexdb import shotdb
    from pexdb.plotting import ciii_plots
    s = shotdb.load("shot_88306.h5")[0]
    ciii_plots.ciii_profiles(s, times=(0.8, 1.0, 1.2, 1.4))   # emission along the two branches, front marked
    ciii_plots.ciii_frames(s, times=(0.8, 1.0, 1.2, 1.4))     # emissivity maps with separatrix and front
    ciii_plots.ciii_vs_dss(s)                                 # outer-leg front: MANTIS vs DSS

    python3 -m pexdb.plotting.ciii_plots $PEX_DB_DIR/shot_88306.h5 --save figures/ciii

Profiles: rows = inner / outer branch (leg from the target to the X-point, then the core separatrix above
the X-point, grey), columns = times. Raw bin means (grey line), smoothed profile (black), X-point (dotted),
front (red), the front stored in OLD database files (cyan, legacy).
Frames: CIII emissivity in the lower divertor, separatrix, X-point, front (red; circle = inner,
square = outer) and, if an old file has them, the legacy stored fronts (green rings).
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.ndimage import gaussian_filter1d, median_filter

from pexdb import shotdb, ciii_front as cf

ZDIV = -0.05          # lower-divertor view: z < ZDIV


def ciii_profiles(shot, times=(0.8, 1.0, 1.2, 1.4, 1.5), result=None):
    """Profiles of the CIII emissivity along the inner and outer branch at `times`. result: output of
    ciii_front.ciii_front(shot) (computed if None). Returns the figure."""
    r = result or cf.ciii_front(shot)
    t, sc = r["t"], r["s"]
    times = [tt for tt in times if tt <= t[-1] + 1e-9]
    fig, ax = plt.subplots(2, len(times), figsize=(3.6 * len(times), 7), sharex=True, squeeze=False)
    for i_leg, leg in enumerate(("inner", "outer")):
        P = median_filter(r["I"][leg], size=(1, cf.TMED)) if cf.TMED > 1 else r["I"][leg]
        for c, tt in enumerate(times):
            i = int(np.argmin(abs(t - tt)))
            x = ax[i_leg, c]
            l, ext = r["l"][leg][i], r["ext"][leg][i]
            m = sc <= ext
            x.axvspan(l, ext, color="0.9", zorder=0)
            x.plot(sc[m], np.nan_to_num(r["I"][leg][:, i])[m], color="0.6", lw=1, label="raw")
            x.plot(sc[m], gaussian_filter1d(np.nan_to_num(P[:, i]), cf.SIGMA)[m], "k", lw=2, label="smoothed")
            x.axvline(l, color="k", ls=":", lw=1, label="X-point")
            x.axvline(r[leg][i], color="C3", lw=2,
                      label=f"front {r[leg][i]:.2f} m" + (" (X-pt ref.)" if r["xmode_" + leg][i] == 1 else ""))
            st = r["stored"][leg][i]
            if np.isfinite(st):
                x.axvline(min(st, ext), color="c", lw=1.5, ls="--", label=f"old stored {st:.2f}")
            x.set_title(f"{leg} branch, t = {t[i]:.2f} s", fontsize=10)
            x.set_xlim(0, ext + 0.02)
            x.set_ylim(bottom=0)
            x.grid(alpha=0.3)
            x.legend(fontsize=7, frameon=False, loc="upper left")
            if i_leg == 1:
                x.set_xlabel("s from the target [m]")
        ax[i_leg, 0].set_ylabel("mean CIII emissivity\n(band around the branch)")
    fig.suptitle(f"shot {shot.shot} ({shot.baffle}): CIII profiles along the branches "
                 "(grey area: above the X-point, core separatrix)", y=0.995)
    fig.tight_layout()
    return fig


def ciii_frames(shot, times=(0.7, 0.85, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5), result=None, log=False):
    """Contact sheet of the CIII emissivity maps (lower divertor) with separatrix, X-point and front."""
    r = result or cf.ciii_front(shot)
    s = cf._native(shot)
    C, L = cf.build_CL(s)
    wall = s.wall().get("tiles")
    g = C.grid
    verts = np.stack([g.R_values, g.Z_values], axis=-1)
    div = g.Z_mean < ZDIV
    t = C.t
    pick = [int(np.argmin(abs(t - tt))) for tt in times if tt <= t[-1] + 1e-9]
    vmax = np.nanpercentile(C.emi[np.ix_(div, pick)], 99.8)
    norm = plt.matplotlib.colors.LogNorm(vmax * 1e-3, vmax) if log else plt.matplotlib.colors.Normalize(0, vmax)
    nc = min(4, len(pick))
    nr = int(np.ceil(len(pick) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.4 * nc, 6.6 * nr), sharex=True, sharey=True, squeeze=False)
    for ax, i in zip(axes.ravel(), pick):
        pc = PolyCollection(verts[div], cmap="inferno", norm=norm, edgecolors="none")
        pc.set_array(C.emi[div, i])
        ax.add_collection(pc)
        if wall is not None:
            ax.plot(*wall, color="w", lw=1.2)
        k = int(np.argmin(abs(L.t - t[i])))
        br = cf.branches(L, k)
        ax.plot(*br["xpt"], "x", color="c", ms=8, mew=2)
        txt = []
        for name, mk in (("inner", "o"), ("outer", "s")):
            if name not in br:
                continue
            ax.plot(br[name][0], br[name][1], color="c", lw=1)
            pt = cf.point_at(br[name], r[name][i])
            if pt is not None:
                ax.plot(*pt, mk, mfc="r", mec="w", ms=9, mew=1)
            st = r["stored"][name][i]
            if np.isfinite(st):
                pt = cf.point_at(br[name], st)
                ax.plot(*pt, mk, mfc="none", mec="lime", ms=14, mew=2.5)
            txt.append(f"{name[:3]}: front {r[name][i]:.2f} m (l={r['l'][name][i]:.2f})")
        ax.set_xlim(0.62, 1.15)
        ax.set_ylim(-0.76, ZDIV)
        ax.set_aspect("equal")
        ax.set_facecolor("k")
        ax.set_title(f"t = {t[i]:.3f} s\n" + "\n".join(txt), fontsize=10)
    for ax in axes.ravel()[len(pick):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Z [m]")
    for ax in axes[-1]:
        ax.set_xlabel("R [m]")
    fig.suptitle(f"shot {shot.shot} ({shot.baffle}): CIII emissivity, lower divertor", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=4.5)
    return fig


def ciii_vs_dss(shot, result=None):
    """Outer-leg CIII front against time: MANTIS (recomputed), DSS measured and DSS synthetic (MANTIS
    integrated along the DSS lines of sight). Dotted: X-point; dash-dot: highest chord the DSS can use
    (it cannot see a front above it). Returns the figure (None if the shot has no DSS CIII)."""
    r = result or cf.ciii_front(shot, dss=True)
    d = r.get("dss")
    if d is None and result is not None:                 # result computed without dss=True
        d = cf.dss_front(shot)
    if d is None:
        print(f"  shot {shot.shot}: no DSS CIII triplet")
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.plot(r["t"], r["outer"], "-", color="C3", lw=2.2, label="MANTIS")
    ax.plot(d["t"], d["front_synthetic"], "--", color="C3", lw=1.5, label="DSS synthetic (from MANTIS)")
    ax.plot(d["t"], d["front"], "ko-", ms=3.5, lw=1.2, label="DSS measured")
    ax.plot(r["t"], r["l"]["outer"], ":", color="0.5", lw=1)
    ax.plot(d["t"], d["s_top"], "-.", color="0.4", lw=1)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("outer-leg CIII front [m from the target]")
    ax.set_title(f"shot {shot.shot} ({shot.baffle})")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("source", nargs="+", help="shot_*.h5 file(s)")
    ap.add_argument("--times", type=float, nargs="+", default=(0.8, 1.0, 1.2, 1.4, 1.5))
    ap.add_argument("--save", help="prefix: writes <prefix>_<shot>_profiles.png, _frames.png, _dss.png")
    a = ap.parse_args()
    for s in shotdb.load(a.source):
        r = cf.ciii_front(s, dss=True)
        figs = {"profiles": ciii_profiles(s, a.times, r), "frames": ciii_frames(s, a.times, r),
                "dss": ciii_vs_dss(s, r)}
        for name, fig in figs.items():
            if fig is None:
                continue
            if a.save:
                os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
                out = f"{a.save}_{s.shot}_{name}.png"
                fig.savefig(out, dpi=130)
                print("wrote", out)
    if not a.save:
        plt.show()


if __name__ == "__main__":
    main()
