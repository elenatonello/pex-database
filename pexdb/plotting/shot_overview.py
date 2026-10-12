"""Overview of one or several shots.

    from pexdb import shotdb
    from pexdb.plotting import shot_overview
    shots = shotdb.load("shots/")
    figs = shot_overview.shot_overview(shots, times=(0.8, 1.0, 1.2), save="out/ov")

shot_overview() returns a dict of figures: geometry, power, vs_density, power_balance, vir_vs_hir,
upstream, outer_target, inner_target (all built from diag_plots).
shot_page() is the previous single-shot one-page view (time traces + profiles).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pexdb.shotdb import Unavailable, Shot
from pexdb.plotting import diag_plots as dp


def _lines(ax, shot, specs, ylabel):
    """specs: list of (path, label, scale); plots those that are available."""
    drawn = False
    for path, label, scale in specs:
        try:
            y = shot[path]
        except Unavailable:
            continue
        if y.ndim > 1:                       # e.g. interferometer channels: mean over good ones
            y = np.nanmean(y, axis=1)
        ax.plot(shot.t, y * scale, marker="o", ms=3, ls="none", label=label)
        drawn = True
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if drawn:
        ax.legend(fontsize=7, frameon=False, ncol=2)
    else:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color="gray")


def shot_page(shot, times=(0.8, 1.1, 1.4), figsize=(13, 9)):
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(8, 4, figure=fig, hspace=0.35, wspace=0.55, bottom=0.12, top=0.94)
    c = shot.color()

    panels = [
        ([("equilibrium/ip", "Ip", 1e-3)], "Ip [kA]"),
        ([("heating/power_ohmic", "ohmic", 1e-3), ("heating/power_ecrh", "ECRH", 1e-3),
          ("heating/power_nbi", "NBI", 1e-3)], "heating [kW]"),
        ([("interferometer/av_dens/data", "mean line-av. density", 1e-19),
          ("thomson_scattering/av_dens", "line-av. density (TS)", 1e-19),
          ("thomson_scattering/ne_u", "ne_u (TS)", 1e-19)], r"$n_e$ [$10^{19}$ m$^{-3}$]"),
        ([("bolometer/power_tot", "tot", 1e-3), ("bolometer/power_core", "core", 1e-3),
          ("bolometer/power_div", "div", 1e-3), ("bolometer/power_xpt", "xpt", 1e-3)], "P_rad [kW]"),
        ([("gas_injection/valve/V1/flow_rate", "V1", 1)], r"gas [Pa m$^3$/s]"),
        ([("barometry/barometer/pressure_div", "baratron div", 1), ("barometry/barometer/pressure_mid", "baratron mid", 1),
          ("barometry/pressure_gauges/pressure_div", "gauge div", 1)], "pressure [Pa]"),
        ([("thomson_scattering/te_u", "Te_u", 1)], r"$T_{e,u}$ [eV]"),
        ([("camera_ir/HIR/power", "HIR", 1e-3), ("camera_ir/VIR/power", "VIR", 1e-3)], "IR power [kW]"),
    ]
    first = None
    for i, (specs, ylabel) in enumerate(panels):
        ax = fig.add_subplot(gs[i, :2], sharex=first)
        first = first or ax
        _lines(ax, shot, specs, ylabel)
        if i < len(panels) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_xlabel("t [s]")
    for tt in times:
        for a in fig.axes[:len(panels)]:
            a.axvline(tt, color="gray", lw=0.6, ls="--")

    # profiles at the chosen times
    prof = [("thomson_scattering/rho", "thomson_scattering/ne", 1e-19, r"$n_e$ [$10^{19}$ m$^{-3}$]", False),
            ("thomson_scattering/rho", "thomson_scattering/te", 1, r"$T_e$ [eV]", False),
            ("charge_exchange/rho", "charge_exchange/Ti/data", 1, r"$T_i$ [eV]", True)]
    cols = plt.cm.viridis(np.linspace(0.1, 0.9, len(times)))
    for i, (xp, yp, sc, lab, transp) in enumerate(prof):
        ax = fig.add_subplot(gs[i * 8 // 3:(i + 1) * 8 // 3, 2:])
        try:
            x, y = shot[xp], shot[yp]
            if transp:
                x, y = x.T, y.T
            for tt, col in zip(times, cols):
                k = shot.frame(tt)
                o = np.argsort(np.where(np.isnan(x[k]), np.inf, x[k]))
                ax.plot(x[k][o], y[k][o] * sc, "o", ms=3, color=col, label=f"{tt:.2f} s")
            ax.legend(fontsize=7, frameon=False)
        except Unavailable:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", color="gray")
        ax.set_ylabel(lab)
        ax.set_xlabel(r"$\rho$")
        ax.grid(alpha=0.3)

    # availability
    ok = [d for d, a in shot.diagnostics.items() if a]
    bad = [d for d, a in shot.diagnostics.items() if not a]
    fig.text(0.02, 0.01, "available: " + ", ".join(ok) + "\nmissing: " + (", ".join(bad) or "-"),
             va="bottom", fontsize=7)
    fig.suptitle(f"{shot.label}   regime={shot.regime or '?'}   baffle={shot.baffle or '?'}",
                 color=c, fontsize=12)
    return fig


def shot_overview(shots, times=(0.8, 1.0, 1.2), x="rho", color_by="auto", save=None):
    """Geometry of all diagnostics, power balance, pressures and separatrix density
    vs line-average density, and upstream / outer-target / inner-target profiles.
    shots: a Shot or a list. x: radial coordinate of ALL profiles (Thomson, CXRS and
    probes), 'rho' or 'dsep' (distance to the separatrix at the midplane; CXRS
    has no dsep, so the Ti row is dropped). save: filename prefix; each figure is written to
    <save>_<name>.png. Returns {name: fig}. A figure that cannot be built (missing
    diagnostic in every shot, unexpected data) is reported and skipped; the others
    are still made."""
    shots = [shots] if isinstance(shots, Shot) else list(shots)
    makers = {
        "geometry": lambda: dp.plot_geometry(shots, t=times[len(times) // 2]),
        "power": lambda: dp.power_summary(shots, color_by=color_by),
        "vs_density": lambda: dp.vs_density(shots, color_by=color_by),
        "power_balance": lambda: dp.power_balance_vs_density(shots, color_by=color_by),
        "vir_vs_hir": lambda: dp.vir_vs_hir(shots, color_by=color_by),
        "upstream": lambda: dp.profiles_ne_te_ti(shots, times, x=x, color_by=color_by),
        "outer_target": lambda: dp.lp_target_profiles(shots, times, target="outer", x=x,
                                                      color_by=color_by),
        "inner_target": lambda: dp.lp_target_profiles(shots, times, target="inner", x=x,
                                                      color_by=color_by),
    }
    figs = {}
    for name, make in makers.items():                 # a failing figure is skipped, not fatal
        before = set(plt.get_fignums())
        try:
            figs[name] = make()[0]
        except Exception as ex:
            for n in set(plt.get_fignums()) - before:     # drop the half-built figure
                plt.close(n)
            print(f"  skip figure '{name}': {type(ex).__name__}: {ex}")
    if save:
        for name, fig in figs.items():
            fig.savefig(f"{save}_{name}.png", dpi=150)
    return figs
