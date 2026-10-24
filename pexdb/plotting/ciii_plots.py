"""Plots that show what pexdb.ciii_front computes.

    from pexdb import shotdb
    from pexdb.plotting import ciii_plots
    s = shotdb.load("shot_88306.h5")[0]
    ciii_plots.ciii_profiles(s, times=(0.8, 1.0, 1.2, 1.4))   # emission along the two branches, front marked
    ciii_plots.ciii_frames(s, times=(0.8, 1.0, 1.2, 1.4))     # emissivity maps with separatrix and front
    ciii_plots.ciii_frames(s, quantity="both", front=False)   # emissivity and C2+ parallel velocity (CIS)
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


def ciii_frames(shot, times=(0.7, 0.85, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5), result=None, log=False,
                quantity="emi", front=True, separatrix=True, vmax=None, vlim=None, mask_frac=0.03):
    """Maps of the CIII camera in the lower divertor, one column per time.

    quantity   'emi' (emissivity), 'vpar' (parallel velocity of the emitting ion, C2+ for CIII, from CIS;
               same triangles as emi) or 'both' (emi on the top row, vpar below).
    front      mark the CIII front (red; circle = inner, square = outer; needs the front computation,
               result = ciii_front(shot) is computed if not given). Legacy stored fronts of old files: green.
    separatrix draw the whole separatrix (psi = 0 contour of the equilibrium: plasma boundary and both legs)
               and the X-point (the X-point is also drawn when front=True).
    vmax       upper limit of the emissivity colour scale (default: 99.8 percentile of the plotted frames);
               log: logarithmic emissivity scale.
    vlim       limit of the symmetric vpar colour scale (default: 98th percentile of |vpar|).
    mask_frac  vpar is only shown where the emissivity of that frame exceeds mask_frac x its maximum
               (CIS gives no meaningful velocity without light); None shows it everywhere.
    Returns the figure. With 'both' the number of columns is the number of times (up to 6)."""
    if quantity not in ("emi", "vpar", "both"):
        raise ValueError("quantity must be 'emi', 'vpar' or 'both'")
    s = cf._native(shot)
    C, L = cf.build_CL(s)
    have_vpar = hasattr(C, "vpar")
    if quantity in ("vpar", "both") and not have_vpar:
        raise ValueError(f"shot {shot.shot}: no camera_visible/CIII_cam1/vpar in this file (the MANTIS struct "
                         "had no vpar, or the database was built with a version of build_database.py "
                         "without it)")
    r = (result or cf.ciii_front(shot)) if front else None
    wall = s.wall().get("tiles")
    g = C.grid
    verts = np.stack([g.R_values, g.Z_values], axis=-1)
    div = g.Z_mean < ZDIV
    t = C.t
    pick = [int(np.argmin(abs(t - tt))) for tt in times if tt <= t[-1] + 1e-9]
    mpl = plt.matplotlib
    vmax_e = vmax or np.nanpercentile(C.emi[np.ix_(div, pick)], 99.8)
    norm_e = mpl.colors.LogNorm(vmax_e * 1e-3, vmax_e) if log else mpl.colors.Normalize(0, vmax_e)
    if have_vpar and quantity != "emi":
        vl = vlim or float(np.nanpercentile(np.abs(C.vpar[np.ix_(div, pick)]), 98))
        norm_v = mpl.colors.Normalize(-vl, vl)
    unit = s.attr("camera_visible/CIII_cam1", "vpar_units", "as stored")
    qs = ["emi", "vpar"] if quantity == "both" else [quantity]
    if quantity == "both":
        pick = pick[:6]
        nr, nc = 2, len(pick)
        order = [(q, i) for q in qs for i in pick]
    else:
        nc = min(4, len(pick))
        nr = int(np.ceil(len(pick) / nc))
        order = [(quantity, i) for i in pick]
    fig, axes = plt.subplots(nr, nc, figsize=(4.0 * nc + 0.9, 6.0 * nr), sharex=True, sharey=True,
                             squeeze=False, layout="constrained")
    pcs = {}
    for ax, (q, i) in zip(axes.ravel(), order):
        if q == "emi":
            pc = PolyCollection(verts[div], cmap="inferno", norm=norm_e, edgecolors="none")
            pc.set_array(C.emi[div, i])
            ax.set_facecolor("k")
            wall_col = "w"
        else:
            v = np.array(C.vpar[div, i], float)
            if mask_frac is not None:
                v[C.emi[div, i] < mask_frac * np.nanmax(C.emi[div, i])] = np.nan
            pc = PolyCollection(verts[div], cmap="RdBu_r", norm=norm_v, edgecolors="none")
            pc.set_array(np.ma.masked_invalid(v))
            ax.set_facecolor("0.85")
            wall_col = "k"
        ax.add_collection(pc)
        pcs[q] = pc
        if wall is not None:
            ax.plot(*wall, color=wall_col, lw=1.2)
        txt = []
        if front or separatrix:
            k = int(np.argmin(abs(L.t - t[i])))
            br = cf.branches(L, k)
            sep_col = "c" if q == "emi" else "k"
            ax.plot(*br["xpt"], "x", color=sep_col, ms=8, mew=2)
            if separatrix:                       # the whole separatrix (psi = 0), not only the legs of the profile
                for ln in cf.separatrix_lines(L, k):
                    ax.plot(ln[:, 0], ln[:, 1], color=sep_col, lw=1)
            for name, mk in (("inner", "o"), ("outer", "s")):
                if name not in br:
                    continue
                if front:
                    pt = cf.point_at(br[name], r[name][i])
                    if pt is not None:
                        ax.plot(*pt, mk, mfc="r", mec="w" if q == "emi" else "k", ms=9, mew=1)
                    st = r["stored"][name][i]
                    if np.isfinite(st):
                        ax.plot(*cf.point_at(br[name], st), mk, mfc="none", mec="lime", ms=14, mew=2.5)
                    txt.append(f"{name[:3]}: front {r[name][i]:.2f} m (l={r['l'][name][i]:.2f})")
        ax.set_xlim(0.62, 1.15)
        ax.set_ylim(-0.76, ZDIV)
        ax.set_aspect("equal")
        title = f"t = {t[i]:.3f} s" + ("\n" + "\n".join(txt) if txt and q == qs[0] else "")
        ax.set_title(title, fontsize=10)
    shown = axes[:, :].ravel().tolist()
    labels = {"emi": "CIII emissivity", "vpar": f"v$_\\parallel$ of the emitting ion ({unit})"}
    for q, pc in pcs.items():                       # one colorbar per quantity, spanning its row(s)
        rows = axes[qs.index(q)] if quantity == "both" else axes.ravel()[:len(order)]
        fig.colorbar(pc, ax=list(rows), shrink=0.8, pad=0.01).set_label(labels[q])
    for ax in axes.ravel()[len(order):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("Z [m]")
    for ax in axes[-1]:
        ax.set_xlabel("R [m]")
    names = {"emi": "emissivity", "vpar": "parallel velocity", "both": "emissivity (top) and parallel velocity (bottom)"}
    fig.suptitle(f"shot {shot.shot} ({shot.baffle}): CIII {names[quantity]}, lower divertor")
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
