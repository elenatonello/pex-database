"""Diagnostic-specific plots built on plot_types.py.

All functions take a list of shotdb.Shot and return (fig, axes). Typical use:

    from pexdb import shotdb
    from pexdb.plotting import diag_plots as dp
    shots = shotdb.load("shots/")
    dp.profiles_ne_te_ti(shots, times=(0.8, 1.0, 1.2))
    dp.lp_target_profiles(shots, times=(0.8, 1.0, 1.2), target="outer")
    dp.mantis_maps(shots, times=(0.8, 1.0, 1.2))
    dp.upstream_vs_time(shots)
    dp.upstream_vs_pressure(shots)
    dp.plot_geometry(shots)
    dp.power_summary(shots)
    dp.vs_density(shots)
    dp.power_balance_vs_density(shots)
    dp.vir_vs_hir(shots)
    dp.ir_qperp_profiles(shots, times=(0.8, 1.0, 1.2))
    dp.dss_profiles(shots, lines=("CII_6582", "Ti"), x="z")
    dp.dss_chords(shots, chords=(5, 15, 25), lines=("CII_6582", "Ti"), x="nesep")
"""
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from pexdb.plotting import plot_types as pt
from pexdb.shotdb import Unavailable


# ---------------------------------------------------------------------------
# Thomson scattering + CXRS profiles
# ---------------------------------------------------------------------------
def profiles_ne_te_ti(shots, times=(0.8, 1.0, 1.2), x="rho", quantities=("ne", "te", "ti"),
                      xrange=None, color_by="auto", **kw):
    """ne, Te (Thomson) and Ti (CXRS) profiles, one column per time.

    x = 'rho' (TS and CXRS) or 'dsep' (distance to the separatrix at the
    midplane in mm, TS only: the Ti row is then dropped).
    quantities: any of 'ne', 'te', 'ti' (rows, in that order).
    xrange: (min, max) in the units of x (None = open end) keeps only points
    inside, e.g. x='dsep', xrange=(-7, None) for dsep >= -7 mm."""
    xs = 1 if x == "rho" else 1e3

    def inside(xv, yv):
        if xrange is None:
            return yv
        m = np.ones(xv.shape, bool)
        if xrange[0] is not None:
            m &= xv >= xrange[0]
        if xrange[1] is not None:
            m &= xv <= xrange[1]
        return np.where(m, yv, np.nan)

    def ts(y, err):
        xp = "thomson_scattering/rho" if x == "rho" else \
             "thomson_scattering/distance_separatrix_midplane"

        def g(s):
            xv = s[xp] * xs
            return xv, inside(xv, s[y]), inside(xv, s[err])
        return g

    def cx(s):
        xv = s["charge_exchange/rho"].T
        return xv, inside(xv, s["charge_exchange/Ti/data"].T), \
               inside(xv, s["charge_exchange/Ti/error"].T)

    avail = {"ne": {"label": r"$n_e$ [$10^{19}$ m$^{-3}$]",
                    "get": ts("thomson_scattering/ne", "thomson_scattering/dne"), "scale": 1e-19},
             "te": {"label": r"$T_e$ [eV]",
                    "get": ts("thomson_scattering/te", "thomson_scattering/dte")}}
    if x == "rho":
        avail["ti"] = {"label": r"$T_i$ [eV]", "get": cx}
    rows = [avail[q] for q in quantities if q in avail]
    xlabel = r"$\rho$" if x == "rho" else r"$R-R_{sep}$ midplane [mm]"
    return pt.profile_grid(shots, rows, times, xlabel, color_by=color_by, **kw)


# ---------------------------------------------------------------------------
# Langmuir probes at the targets
# ---------------------------------------------------------------------------
def despike(y, factor=3.0, floor=15.0, half=2):
    """NaN out spikes in y (probe, t): a point is a spike if it exceeds both
    `floor` and `factor` x the median of its neighbours (+-`half` probes in
    probe order, same time). Only those points are removed; the rest of the
    profile is untouched."""
    y = y.copy()
    n = y.shape[0]
    pad = np.full((half, y.shape[1]), np.nan)
    ext = np.concatenate([pad, y, pad])
    neigh = np.stack([ext[half + k: half + k + n] for k in range(-half, half + 1) if k != 0])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)         # all-NaN neighbourhoods
        med = np.nanmedian(neigh, axis=0)
    spike = (y > floor) & (y > factor * np.where(np.isnan(med), 0, med))
    y[spike] = np.nan
    return y


LP_QUANTITIES = {
    "t_e": (r"$T_e$ [eV]", 1),
    "n_e": (r"$n_e$ [$10^{18}$ m$^{-3}$]", 1e-18),
    "j_i_saturation": (r"$j_{sat}$ [kA m$^{-2}$]", 1e-3),
    "v_floating": (r"$V_{fl}$ [V]", 1),
}


def lp_target_profiles(shots, times=(0.8, 1.0, 1.2), target="outer",
                       quantities=("t_e", "n_e", "j_i_saturation"), x="rho",
                       spike_factor=3.0, spike_floor=15.0, te_range=(0, None), low_ne=1e17,
                       low_ne_factor=0.1, color_by="auto", **kw):
    """Langmuir-probe profiles on the 'inner' or 'outer' target (or 'embedded'
    = all probes), against distance to the separatrix (x='dsep', mm) or rho.

    Quality control (applied to the plot only; the database keeps the raw data):
      t_e is multiplied by `low_ne_factor` (0.1) where n_e < `low_ne` [m^-3]
      (1e17; Te is unreliable there); low_ne=None disables it.
      t_e spikes are then removed point by point (see despike): a point is dropped if
      it is above `spike_floor` [eV] AND above `spike_factor` x the median of
      the neighbouring probes. spike_factor=None disables the filter.
      te_range (min, max) [eV]: t_e outside this range is masked; None = open end.
      t_e, n_e, j_i_saturation <= 0 are always masked.
    The y range is set from the 0-99.5 percentiles of the plotted data."""
    grp = "embedded" if target == "all" else f"{target}_target"
    base = f"langmuir_probes/{grp}/"
    xp = "distance_separatrix_midplane" if x == "dsep" else "rho"
    xs, xlabel = (1e3, r"$R-R_{sep}$ midplane [mm]") if x == "dsep" else (1, r"$\rho$")

    def getter(q):
        def g(s):
            y = s[base + q].copy()                          # (probe, t)
            if q in ("t_e", "n_e", "j_i_saturation"):
                y[y <= 0] = np.nan
            if q == "t_e":
                if low_ne is not None:
                    ne = s[base + "n_e"]
                    y[(ne < low_ne) & (y > 0)] *= low_ne_factor
                if spike_factor is not None:
                    y = despike(y, spike_factor, spike_floor)
                if te_range[0] is not None:
                    y[y < te_range[0]] = np.nan
                if te_range[1] is not None:
                    y[y > te_range[1]] = np.nan
            return s[base + xp].T * xs, y.T, None, s["langmuir_probes/t"]
        return g

    rows = []
    for q in quantities:
        label, scale = LP_QUANTITIES[q]
        row = {"label": label, "scale": scale, "get": getter(q)}
        vals = []
        for s in shots:
            try:
                vals.append(np.ravel(row["get"](s)[1]) * scale)
            except Unavailable:
                pass
        if vals:
            v = np.concatenate(vals)
            v = v[np.isfinite(v)]
            if v.size:
                row["ylim"] = (min(0, np.percentile(v, 1)), 1.1 * np.percentile(v, 99.5))
        rows.append(row)
    fig, axes = pt.profile_grid(shots, rows, times, xlabel, color_by=color_by, **kw)
    cuts = (f"Te spikes removed (>{spike_floor:g} eV and >{spike_factor:g}x neighbours)"
            if spike_factor is not None else "")
    fig.suptitle(f"Langmuir probes, {target} target   {cuts}", fontsize=10)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# MANTIS maps
# ---------------------------------------------------------------------------
def mantis_maps(shots, times=(0.8, 1.0, 1.2), camera="CIII_cam1", vmax=None, log=False, **kw):
    """Reconstructed emissivity maps, one row per shot, with the LCFS and wall."""
    base = f"camera_visible/{camera}/"

    def getter(s):
        return {"r_values": s[base + "grid/r_values"], "z_values": s[base + "grid/z_values"],
                "values": s[base + "emi"]}
    return pt.map_grid(shots, getter, times, label=f"{camera} emissivity", vmax=vmax, log=log, **kw)


# ---------------------------------------------------------------------------
# Upstream (separatrix) Thomson values
# ---------------------------------------------------------------------------
def upstream_vs_time(shots, color_by="auto", t_range=(0.6, 1.6), err_style="bar",
                     capsize=pt.CAPSIZE):
    """Upstream ne_u and Te_u against time (with errors) for all shots."""
    items = [("thomson_scattering/ne_u", r"$n_{e,u}$ [$10^{19}$ m$^{-3}$]", 1e-19,
              "thomson_scattering/dne_u"),
             ("thomson_scattering/te_u", r"$T_{e,u}$ [eV]", 1, "thomson_scattering/dte_u")]
    return pt.time_traces(shots, items, t_range=t_range, color_by=color_by, err_style=err_style,
                          capsize=capsize)


def upstream_vs_pressure(shots, pressure="barometry/barometer/pressure_div",
                         xlabel="divertor pressure (baratron)", color_by="auto",
                         t_range=(0.6, 1.6), err_style="bar", capsize=pt.CAPSIZE):
    """ne_u and Te_u against the divertor baratron pressure (scatter, markers only)."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (q, dq, lab, sc) in zip(axes, [
            ("ne_u", "dne_u", r"$n_{e,u}$ [$10^{19}$ m$^{-3}$]", 1e-19),
            ("te_u", "dte_u", r"$T_{e,u}$ [eV]", 1)]):
        pt.xy_plot(shots, lambda s: s[pressure],
                   lambda s, q=q, sc=sc: s["thomson_scattering/" + q] * sc,
                   xlabel, lab, yerr_get=lambda s, d=dq, sc=sc: s["thomson_scattering/" + d] * sc,
                   t_range=t_range, color_by=color_by, ax=ax, err_style=err_style, capsize=capsize)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Geometry check
# ---------------------------------------------------------------------------
GEOMETRY_LAYERS = ("fir", "bolo", "gauges", "lp", "ts", "cxrs")


def plot_geometry(shots, layers=GEOMETRY_LAYERS, with_equilibrium=True, t=1.0):
    """Poloidal cross-section per shot: wall (vessel patch + tiles), LCFS at time t and
    the geometry of the diagnostics in `layers` (subset of GEOMETRY_LAYERS):
      fir     interferometer chords (dotted red = bad channel)
      bolo    bolometer lines of sight, detector (squares) and aperture (dots)
      gauges  ports of the pressure gauges / barometer (filled patches)
      lp      Langmuir probes   ts  Thomson points   cxrs  CXRS measurement points
    """
    fig, axes = plt.subplots(1, len(shots), squeeze=False, figsize=(4.4 * len(shots) + 0.5, 7))
    for ax, s in zip(axes[0], shots):
        ax.set_aspect("equal")
        pt.draw_wall(ax, s)
        if with_equilibrium and s.has("equilibrium/r_contour"):
            k = s.frame(t)
            ax.plot(s["equilibrium/r_contour"][:, k], s["equilibrium/z_contour"][:, k], color="0.3", lw=1.4)
        legend = []

        def lines(name, r0, z0, r1, z1, color, ls="-", lw=0.7):
            for a_, b_, c_, d_ in zip(r0, z0, r1, z1):
                ax.plot([a_, c_], [b_, d_], color=color, ls=ls, lw=lw)
            legend.append(plt.Line2D([], [], color=color, lw=1.2, label=name))

        def points(name, r, z, color, marker, ms=18):
            ax.scatter(r, z, s=ms, c=color, marker=marker, zorder=4)
            legend.append(plt.Line2D([], [], color=color, marker=marker, ls="none", label=name))

        if "fir" in layers and s.has("interferometer/line_of_sight/first_point/r"):
            p = "interferometer/line_of_sight/"
            ok = s["interferometer/ch_ok"].astype(bool)
            for good in (True, False):
                m = ok == good
                if m.any():
                    lines("FIR" + ("" if good else " (bad ch.)"),
                          s[p + "first_point/r"][m], s[p + "first_point/z"][m],
                          s[p + "second_point/r"][m], s[p + "second_point/z"][m],
                          "tab:orange" if good else "tab:red", "-" if good else ":", 0.9)
        if "bolo" in layers and s.has("bolometer/line_of_sight/first_point/r"):
            p = "bolometer/"
            has = ~np.isnan(s[p + "channels"]).all(axis=0)         # channels with data in this shot
            for good in (True, False):
                m = has == good
                if m.any():
                    lines("bolometer LOS" + ("" if good else " (no data)"),
                          s[p + "line_of_sight/first_point/r"][m], s[p + "line_of_sight/first_point/z"][m],
                          s[p + "line_of_sight/second_point/r"][m], s[p + "line_of_sight/second_point/z"][m],
                          "tab:green" if good else "0.7", "-" if good else ":", 0.5)
            points("bolo detector", s[p + "detector/r"], s[p + "detector/z"], "tab:green", "s", 10)
            points("bolo aperture", s[p + "aperture/r"], s[p + "aperture/z"], "darkgreen", ".", 10)
        if "gauges" in layers:
            # each geometry entry is the outline of the port the gauge is connected to
            for grp, col in (("barometer", "tab:purple"), ("pressure_gauges", "tab:brown")):
                base = f"barometry/{grp}/position/"
                found = False
                for nm in ("mic", "top", "bot", "div"):
                    if s.has(base + nm + "/r"):
                        xy = np.column_stack([s[base + nm + "/r"], s[base + nm + "/z"]])
                        ax.add_patch(plt.Polygon(xy, closed=True, fc=col, ec=col, alpha=0.55,
                                                 lw=0.8, zorder=4))
                        ax.annotate(nm, xy.mean(axis=0), fontsize=6, color=col,
                                    xytext=(5, 5), textcoords="offset points")
                        found = True
                if found:
                    legend.append(plt.Rectangle((0, 0), 1, 1, fc=col, alpha=0.55,
                                                label=f"{grp} ports"))
        if "lp" in layers and s.has("langmuir_probes/embedded/position/r"):
            points("LP", s["langmuir_probes/embedded/position/r"], s["langmuir_probes/embedded/position/z"],
                   "tab:blue", "o", 8)
        if "ts" in layers and s.has("thomson_scattering/r"):
            points("TS", s["thomson_scattering/r"], s["thomson_scattering/z"], "tab:pink", "|", 20)
        if "cxrs" in layers and s.has("charge_exchange/r"):
            points("CXRS", s["charge_exchange/r"], s["charge_exchange/z"], "tab:olive", "x", 25)
        ax.set_title(s.label, fontsize=9)
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.grid(alpha=0.3)
        if legend:
            ax.legend(handles=legend, fontsize=6, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False)
        if not s.wall():
            ax.text(0.5, 0.5, "no wall data", transform=ax.transAxes, ha="center", color="gray")
    fig.tight_layout()
    return fig, axes


def plot_wall(shots, **kw):
    """Wall (+ LCFS and FIR chords) only; see plot_geometry."""
    return plot_geometry(shots, layers=kw.pop("layers", ("fir",)), **kw)


# ---------------------------------------------------------------------------
# Geometry of all diagnostics
# ---------------------------------------------------------------------------
def _rz(s, base):
    return s[base + "/r"], s[base + "/z"]


def _segments(s, base):
    """Line segments first_point -> second_point of a line-of-sight group."""
    r1, z1 = _rz(s, base + "/first_point")
    r2, z2 = _rz(s, base + "/second_point")
    return np.stack([np.column_stack([r1, z1]), np.column_stack([r2, z2])], axis=1)


def _valves(s):
    """(name, r, z) of the gas valves with a stored exit position."""
    out = []
    if s.has("gas_injection"):
        with h5py.File(s.path, "r") as f:
            for n, v in f["gas_injection/valve"].items():
                a = v["exit_position"].attrs if "exit_position" in v else {}
                if "r" in a:
                    out.append((n, float(a["r"]), float(a["z"])))
    return out


def _geometry_layers(s):
    """(label, diagnostic group, draw(ax, color)) for every diagnostic with a
    geometry. draw raises Unavailable if the shot has no such data."""
    def pts(base, **st):
        return lambda ax, c: ax.plot(*_rz(s, base), ls="none", color=c, **st)[0]

    def seg(base, **st):
        return lambda ax, c: ax.add_collection(LineCollection(_segments(s, base), colors=c, **st))

    def line(base, **st):
        return lambda ax, c: ax.plot(*_rz(s, base), color=c, **st)[0]

    def ir_camera(cam, **st):
        """IR target profile (thick line) plus, when stored: the outline of the port the
        camera looks through (closed contour, transparent fill) and the field of view, a
        shaded triangle from the detector to the two ends of the target profile, shown
        only where it is inside the vessel (clipped to the inner vessel contour; the
        detector itself is not drawn)."""
        def draw(ax, c):
            r, z = _rz(s, f"camera_ir/{cam}")
            art = ax.plot(r, z, color=c, **st)[0]
            if s.has(f"camera_ir/{cam}/port/outline/r"):
                pr, pz = _rz(s, f"camera_ir/{cam}/port/outline")
                ax.add_patch(plt.Polygon(np.column_stack([pr, pz]), closed=True, fc=c, ec=c,
                                         alpha=0.5, lw=1.2))
            if s.has(f"camera_ir/{cam}/detector/r"):
                dr, dz = (float(v[0]) for v in _rz(s, f"camera_ir/{cam}/detector"))
                ok = np.isfinite(r) & np.isfinite(z)
                if ok.sum() >= 2:                                  # first / last valid target point
                    ends = np.array([[r[ok][0], z[ok][0]], [r[ok][-1], z[ok][-1]]])
                    # add_artist (not add_patch): the unclipped triangle, which reaches the
                    # detector outside the vessel, must not stretch the axis limits
                    fov = ax.add_artist(plt.Polygon(np.vstack([[dr, dz], ends]), closed=True, fc=c,
                                                    ec="none", alpha=0.18, zorder=0))
                    vi = s.wall().get("vessel_inner")
                    if vi is not None:                             # keep only the part inside the vessel
                        fov.set_clip_path(plt.Polygon(np.column_stack(vi), closed=True,
                                                      transform=ax.transData))
            return art
        return draw

    def pressure(grp, **st):
        """Gauge ports: closed contour (edge) filled with a transparent colour, no markers."""
        def draw(ax, c):
            patches = []
            for pos in ("bot", "div", "mic", "top"):
                r, z = _rz(s, f"barometry/{grp}/position/{pos}")
                patches.append(ax.add_patch(plt.Polygon(np.column_stack([r, z]), closed=True,
                                                        fc=c, ec=c, alpha=0.5, lw=1.2, **st)))
                up = z.mean() > 0 or pos == "mic"
                ax.annotate(pos, (r.mean(), z.mean()), fontsize=7, color=c,
                            xytext=(6, 6 if up else -10), textcoords="offset points",
                            bbox=dict(fc="w", ec="none", alpha=0.7, pad=0.5))
            return patches[0]                   # carries the legend label
        return draw

    def valves(ax, c):
        v = _valves(s)
        if not v:
            raise Unavailable("no valve exit position")
        for n, r, z in v:
            art = ax.plot(r, z, "*", ms=10, color=c)[0]
            ax.annotate(n, (r, z), fontsize=7, xytext=(4, 4), textcoords="offset points")
        return art

    def mantis(ax, c):
        b = "camera_visible/CIII_cam1/grid/"
        r, z = s[b + "r_values"], s[b + "z_values"]
        return ax.add_patch(plt.Rectangle((np.nanmin(r), np.nanmin(z)), np.nanmax(r) - np.nanmin(r),
                                          np.nanmax(z) - np.nanmin(z), fill=False, ec=c, ls="--",
                                          lw=1))

    return [
        ("Thomson", "thomson_scattering", "tab:red", pts("thomson_scattering", marker=".", ms=4)),
        ("CXRS", "charge_exchange", "tab:orange", pts("charge_exchange", marker="x", ms=5)),
        ("FIR chords", "interferometer", "tab:green",
         seg("interferometer/line_of_sight", lw=1.2)),
        ("bolometer LOS", "bolometer", "tab:blue",
         seg("bolometer/line_of_sight", lw=0.3, alpha=0.5)),
        ("DSS LOS", "spectrometer_visible", "tab:orange",
         seg("spectrometer_visible/line_of_sight", lw=0.8)),
        ("LP outer target", "langmuir_probes", "tab:purple",
         pts("langmuir_probes/outer_target/position", marker="o", ms=3, zorder=4)),
        ("LP inner target", "langmuir_probes", "tab:pink",
         pts("langmuir_probes/inner_target/position", marker="o", ms=3, zorder=4)),
        ("LP embedded", "langmuir_probes", "0.5",
         pts("langmuir_probes/embedded/position", marker=".", ms=2, zorder=4)),
        ("pressure gauge ports", "barometry", "tab:brown", pressure("pressure_gauges")),
        ("IR HIR", "camera_ir", "darkcyan", ir_camera("HIR", lw=3, alpha=0.7)),
        ("IR VIR", "camera_ir", "teal", ir_camera("VIR", lw=3, alpha=0.7)),
        ("gas valve", "gas_injection", "tab:olive", valves),
        ("MANTIS FOV", "camera_visible", "tab:cyan", mantis),
    ]


def _geometry_extent(shots, margin=0.04):
    """(rmin, rmax, zmin, zmax) common to all panels: the outer vessel and the bolometer
    detectors of every shot, whatever layers are drawn, so the boxes always have the same size."""
    r, z = [], []
    for s in shots:
        w = s.wall()
        for name in ("vessel_outer", "vessel", "tiles"):
            if name in w:
                r.append(np.nanmin(w[name][0])); r.append(np.nanmax(w[name][0]))
                z.append(np.nanmin(w[name][1])); z.append(np.nanmax(w[name][1]))
                break
        if s.has("bolometer/detector/r"):
            r += [np.nanmin(s["bolometer/detector/r"]), np.nanmax(s["bolometer/detector/r"])]
            z += [np.nanmin(s["bolometer/detector/z"]), np.nanmax(s["bolometer/detector/z"])]
    if not r:
        return None
    return min(r) - margin, max(r) + margin, min(z) - margin, max(z) + margin


def plot_geometry(shots, t=1.0, with_equilibrium=True, layers=None):
    """Poloidal cross-section per shot with the position / line of sight of every
    available diagnostic (Thomson, CXRS, FIR, bolometer, Langmuir probes,
    DSS lines of sight, pressure gauge ports, IR cameras, gas valve, MANTIS field of view), the
    wall and the LCFS at time t. Diagnostics missing in a shot are listed below
    its panel. All panels share the same axis limits (vessel + bolometer extent), so the
    boxes have the same size whatever is drawn.

    layers: draw only these entries (None = all), by label: 'Thomson', 'CXRS',
    'FIR chords', 'bolometer LOS', 'DSS LOS', 'LP outer target', 'LP inner target',
    'LP embedded', 'pressure gauge ports', 'IR HIR', 'IR VIR', 'gas valve', 'MANTIS FOV'."""
    if layers is not None:
        known = [l[0] for l in _geometry_layers(shots[0])]
        bad = [l for l in layers if l not in known]
        if bad:
            raise ValueError(f"unknown geometry layer(s) {bad}; choose from {known}")
    extent = _geometry_extent(shots)
    fig, axes = plt.subplots(1, len(shots), squeeze=False,
                             figsize=(4.4 * len(shots) + 0.5, 8))
    for ax, s in zip(axes[0], shots):
        ax.set_aspect("equal", adjustable="box")
        pt.draw_wall(ax, s)
        missing = []
        for label, grp, color, draw in _geometry_layers(s):
            if layers is not None and label not in layers:
                continue
            try:
                if not s.diagnostics.get(grp, False):
                    raise Unavailable(grp)
                draw(ax, color).set_label(label)
            except Unavailable:
                missing.append(label)
        if with_equilibrium and s.has("equilibrium/r_contour"):
            k = s.frame(t)
            ax.plot(s["equilibrium/r_contour"][:, k], s["equilibrium/z_contour"][:, k], color="0.3", lw=1.4,
                    label=f"LCFS t={t:.2f}s")
        if extent:
            ax.set_xlim(extent[:2])
            ax.set_ylim(extent[2:])
        else:
            ax.autoscale_view()
        ax.set_title(s.label, fontsize=9, color=s.color())
        ax.set_xlabel("R [m]")
        ax.set_ylabel("Z [m]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, frameon=False, ncol=2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.08))
        if missing:
            ax.text(0.5, -0.25, "no data: " + ", ".join(missing), transform=ax.transAxes,
                    ha="center", va="top", fontsize=7, color="gray")
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Power balance
# ---------------------------------------------------------------------------
def heating_total(s):
    """Input power ohmic + ECRH + NBI [W] (MHD term excluded)."""
    parts = [s["heating/" + k] for k in ("power_ohmic", "power_ecrh", "power_nbi")
             if s.has("heating/" + k)]
    if not parts:
        raise Unavailable(f"shot {s.shot}: no heating data")
    return np.nansum(parts, axis=0)


def power_summary(shots, color_by="auto", t_range=(0.6, 1.6)):
    """All powers [kW] on one plot, colour = shot:
    P_in = ohmic+ECRH+NBI (filled), core radiated power (bolometer power_core,
    open), NBI alone (x, only if non-zero) and P_in - P_rad,core (dashed line)."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    nbi_used = False
    for s in shots:
        c = s.color(color_by)
        try:
            pin = heating_total(s) * 1e-3
        except Unavailable as ex:
            pt._warn(s, "P_in", ex)
            continue
        ax.plot(s.t, pin, color=c, marker=s.marker, ms=4, ls="none")
        if s.has("heating/power_nbi") and np.nanmax(s["heating/power_nbi"]) > 0:
            ax.plot(s.t, s["heating/power_nbi"] * 1e-3, color=c, marker="x", ms=4, ls="none")
            nbi_used = True
        try:
            prad = s["bolometer/power_core"] * 1e-3
        except Unavailable as ex:
            pt._warn(s, "P_rad,core", ex)
            continue
        ax.plot(s.t, prad, color=c, marker=s.marker, ms=4, ls="none", mfc="none")
        ax.plot(s.t, pin - prad, color=c, ls="--", lw=1)
    ax.set_ylabel("power [kW]")
    ax.set_xlabel("t [s]")
    ax.set_xlim(t_range)
    ax.grid(alpha=0.3)
    quantities = [plt.Line2D([], [], color="gray", marker="o", ls="none",
                             label=r"$P_{in}$ (OH+ECRH+NBI)"),
                  plt.Line2D([], [], color="gray", marker="o", mfc="none", ls="none",
                             label=r"$P_{rad,core}$"),
                  plt.Line2D([], [], color="gray", ls="--", label=r"$P_{in}-P_{rad,core}$")]
    if nbi_used:
        quantities.insert(1, plt.Line2D([], [], color="gray", marker="x", ls="none",
                                        label=r"$P_{NBI}$"))
    ax.set_ylim(top=ax.get_ylim()[1] * 1.25)                  # headroom for the legends
    leg = ax.legend(handles=quantities, fontsize=8, frameon=False, loc="upper left", ncol=2)
    ax.add_artist(leg)
    ax.legend(handles=[plt.Line2D([], [], color=s.color(color_by), marker=s.marker, ls="none",
                                  label=s.label) for s in shots],
              fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    return fig, ax


def power_balance_vs_density(shots, x="necore", color_by="auto", t_range=(0.6, 1.6)):
    """Total input power (ohmic+ECRH+NBI, filled), total power on the targets
    (IR HIR+VIR, open) and total radiated power (bolometer power_tot, x) [kW] and
    their output sum targets + P_rad (dashed line, in time order), colour = shot, against the Thomson line-average density (x='necore') or
    time (x='t')."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for s in shots:
        c = s.color(color_by)
        try:
            xv = s.t if x == "t" else s["thomson_scattering/av_dens"] * 1e-19
        except Unavailable as ex:
            pt._warn(s, "TS line-average density", ex)
            continue
        m = (s.t >= t_range[0]) & (s.t <= t_range[1])
        for get, kw in ((lambda: heating_total(s), dict(marker=s.marker, color=c)),
                        (lambda: s["camera_ir/HIR/power"] + s["camera_ir/VIR/power"],
                         dict(marker=s.marker, facecolors="none", edgecolors=c)),
                        (lambda: s["bolometer/power_tot"], dict(marker="x", color=c))):
            try:
                ax.scatter(xv[m], (get() * 1e-3)[m], s=18, **kw)
            except Unavailable as ex:
                pt._warn(s, "power", ex)
        try:                                           # output: targets + total radiation
            out = (s["camera_ir/HIR/power"] + s["camera_ir/VIR/power"] + s["bolometer/power_tot"]) * 1e-3
            ok = m & np.isfinite(xv) & np.isfinite(out)
            ax.plot(xv[ok], out[ok], color=c, ls="--", lw=1)
        except Unavailable as ex:
            pt._warn(s, "targets + P_rad", ex)
    ax.set_xlabel("t [s]" if x == "t" else r"$\bar n_e$ (TS) [$10^{19}$ m$^{-3}$]")
    ax.set_ylabel("power [kW]")
    ax.grid(alpha=0.3)
    ax.set_ylim(top=ax.get_ylim()[1] * 1.3)                   # headroom for the legends
    leg = ax.legend(handles=[
        plt.Line2D([], [], color="gray", marker="o", ls="none", label=r"$P_{in}$ (OH+ECRH+NBI)"),
        plt.Line2D([], [], color="gray", marker="o", mfc="none", ls="none",
                   label="targets (IR HIR+VIR)"),
        plt.Line2D([], [], color="gray", marker="x", ls="none", label=r"$P_{rad}$ total (bolometer)"),
        plt.Line2D([], [], color="gray", ls="--", label=r"targets + $P_{rad}$")],
        fontsize=8, frameon=False, loc="upper left")
    ax.add_artist(leg)
    ax.legend(handles=[plt.Line2D([], [], color=s.color(color_by), marker=s.marker, ls="none",
                                  label=s.label) for s in shots],
              fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    return fig, ax


def vir_vs_hir(shots, color_by="auto", t_range=(0.6, 1.6)):
    """Power on the outer target (IR VIR) against the inner target (IR HIR) [kW];
    first / last frame labelled with their time; dashed line VIR = HIR."""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    pt.xy_plot(shots, lambda s: s["camera_ir/HIR/power"] * 1e-3,
               lambda s: s["camera_ir/VIR/power"] * 1e-3, "inner target, IR HIR [kW]",
               "outer target, IR VIR [kW]", t_range=t_range, color_by=color_by, ax=ax)
    lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, "k--", lw=0.8, zorder=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig, ax


IR_QPERP_SCALE = 1e-6      # W/m^2 -> MW/m^2 (units of qperp are not stored in the database; assumed W/m^2)


def ir_qperp_profiles(shots, times=(0.8, 1.0, 1.2), targets=("HIR", "VIR"), color_by="auto", **kw):
    """Perpendicular heat flux q_perp [MW/m^2] from the IR cameras against the distance to
    the separatrix at the midplane [mm], one row per camera (HIR = inner, VIR = outer
    target), one column per time. The IR cameras only provide dsep (no rho)."""
    names = {"HIR": "inner target (HIR)", "VIR": "outer target (VIR)"}

    def getter(cam):
        def g(s):
            q = s[f"camera_ir/{cam}/qperp"]                      # (position, t)
            d = s[f"camera_ir/{cam}/distance_separatrix_midplane"]
            return d.T * 1e3, q.T, None
        return g

    rows = []
    for cam in targets:
        row = {"label": rf"{names.get(cam, cam)}" + "\n" + r"$q_\perp$ [MW m$^{-2}$]",
               "scale": IR_QPERP_SCALE, "get": getter(cam)}
        vals = []
        for s in shots:
            try:
                vals.append(np.ravel(row["get"](s)[1]) * IR_QPERP_SCALE)
            except Unavailable:
                pass
        if vals:
            v = np.concatenate(vals)
            v = v[np.isfinite(v)]
            if v.size:
                row["ylim"] = (min(0, np.percentile(v, 1)), 1.1 * np.percentile(v, 99.5))
        rows.append(row)
    return pt.profile_grid(shots, rows, times, r"$R-R_{sep}$ midplane [mm]", color_by=color_by, **kw)


# ---------------------------------------------------------------------------
# DSS divertor spectroscopy
# ---------------------------------------------------------------------------
DSS_LINES = {   # name in the database -> (label, scale)
    "CII_6582": (r"C II 6582 [$10^{18}$ a.u.]", 1e-18),
    "CIII_trip": (r"C III triplet [$10^{18}$ a.u.]", 1e-18),
    "D3_2": (r"D$_\alpha$ [$10^{18}$ a.u.]", 1e-18),
    "Ti": (r"$T_i$ (CIII) [eV]", 1),
}


def _dss_base(s, line):
    """Path of the DSS system (spectrometer_visible/sys<N>) that holds `line`."""
    if not s.diagnostics.get("spectrometer_visible", False):
        raise Unavailable(f"shot {s.shot}: DSS not available")
    with h5py.File(s.path, "r") as f:
        systems = sorted((k for k in f["spectrometer_visible"] if k.startswith("sys")),
                         key=lambda k: int(k[3:]))
    for sysn in systems:
        base = f"spectrometer_visible/{sysn}/"
        if s.has(base + line):
            return base
    raise Unavailable(f"shot {s.shot}: DSS line '{line}' not available")


def dss_profiles(shots, times=(0.8, 1.0, 1.2), lines=("CII_6582", "CIII_trip", "Ti"),
                 x="channel", color_by="auto", **kw):
    """DSS divertor spectroscopy profiles along the lines of sight, one row per
    line, one column per time.

    lines: one name or several of DSS_LINES: 'CII_6582', 'CIII_trip', 'D3_2'
    (D-alpha) and 'Ti' (CIII ion temperature, with errors). Only lines that
    hold data in a shot are drawn.
    x: 'channel' (chord number, 1 at the bottom), or 'z' / 'r' (position of the
    chord along the divertor leg [m], NaN for chords without a position)."""
    lines = [lines] if isinstance(lines, str) else list(lines)
    bad = [l for l in lines if l not in DSS_LINES]
    if bad:
        raise ValueError(f"unknown DSS line(s) {bad}; choose from {list(DSS_LINES)}")
    if x not in ("channel", "z", "r"):
        raise ValueError("x must be 'channel', 'z' or 'r'")

    def getter(line):
        def g(s):
            base = _dss_base(s, line)
            y = s[base + line].T                                     # (t, channel)
            xv = np.broadcast_to(np.arange(1, y.shape[1] + 1), y.shape) if x == "channel" \
                else s[base + x].T
            e = s[base + "Ti_err"].T if line == "Ti" and s.has(base + "Ti_err") else None
            return xv, y, e
        return g

    rows = [{"label": DSS_LINES[l][0], "scale": DSS_LINES[l][1], "get": getter(l)} for l in lines]
    xlabel = {"channel": "DSS channel", "z": "Z of chord [m]", "r": "R of chord [m]"}[x]
    return pt.profile_grid(shots, rows, times, xlabel, color_by=color_by, **kw)


DSS_X = {   # name -> (path, scale, label)
    "t": (None, 1, "t [s]"),
    "nesep": ("thomson_scattering/ne_u", 1e-19, r"$n_{e,sep}$ (TS) [$10^{19}$ m$^{-3}$]"),
    "necore": ("thomson_scattering/av_dens", 1e-19,
               r"$\bar n_e$ core (TS) [$10^{19}$ m$^{-3}$]"),
    "pressure": ("barometry/pressure_gauges/pressure_div", 1e3, "divertor pressure [mPa]"),
}


def dss_chords(shots, chords=(5, 15, 25), lines=("CII_6582",), x="t", pressure=None,
               t_range=(0.6, 1.6), color_by="auto", figsize=None):
    """DSS signal of selected chords against time or a plasma parameter.
    One row per line, one column per chord, colour = shot.

    chords: chord numbers (1-based, as the 'channel' axis of dss_profiles).
    lines: name(s) from DSS_LINES ('CII_6582', 'CIII_trip', 'D3_2', 'Ti' with errors).
    x: 't', 'nesep' (TS upstream ne_u), 'necore' (TS line-average density) or
       'pressure' (divertor pressure, path set by `pressure=`, default the APG
       divertor gauge; scaled to mPa).
    A shot / line without data is skipped with a message."""
    chords = [chords] if np.isscalar(chords) else list(chords)
    lines = [lines] if isinstance(lines, str) else list(lines)
    bad = [l for l in lines if l not in DSS_LINES]
    if bad:
        raise ValueError(f"unknown DSS line(s) {bad}; choose from {list(DSS_LINES)}")
    if x not in DSS_X:
        raise ValueError(f"x must be one of {list(DSS_X)}")
    xpath, xscale, xlabel = DSS_X[x]
    if x == "pressure" and pressure:
        xpath = pressure
    fig, axes = plt.subplots(len(lines), len(chords), sharex="col", sharey="row", squeeze=False,
                             figsize=figsize or (3.4 * len(chords) + 0.8, 2.4 * len(lines) + 0.8))
    zpos = {}
    for i, line in enumerate(lines):
        label, scale = DSS_LINES[line]
        for j, ch in enumerate(chords):
            ax = axes[i, j]
            for s in shots:
                try:
                    base = _dss_base(s, line)
                    arr = s[base + line]
                    if not 1 <= ch <= arr.shape[0]:
                        raise ValueError(f"chord {ch} outside 1..{arr.shape[0]}")
                    y = arr[ch - 1] * scale
                    e = s[base + "Ti_err"][ch - 1] * scale if line == "Ti" and \
                        s.has(base + "Ti_err") else None
                    xv = s.t if xpath is None else s[xpath] * xscale
                    if ch not in zpos and s.has(base + "z"):
                        zpos[ch] = np.nanmedian(s[base + "z"][ch - 1])
                except Unavailable as ex:
                    pt._warn(s, f"{line} chord {ch}", ex)
                    continue
                m = (s.t >= t_range[0]) & (s.t <= t_range[1]) & np.isfinite(xv) & np.isfinite(y)
                c = s.color(color_by)
                if e is not None:
                    ax.errorbar(xv[m], y[m], yerr=e[m], color=c, marker=s.marker, ms=4, ls="none",
                                elinewidth=0.8, capsize=pt.CAPSIZE, capthick=0.8)
                else:
                    ax.plot(xv[m], y[m], color=c, marker=s.marker, ms=4, ls="none")
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel(label)
    for j, ch in enumerate(chords):
        z = zpos.get(ch)
        axes[0, j].set_title(f"chord {ch}" + (f"  (Z = {z:.2f} m)" if z is not None and np.isfinite(z)
                                              else ""), fontsize=10)
        axes[-1, j].set_xlabel(xlabel)
    pt._legend(axes[0, -1], shots, line=False)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Pressures and separatrix density against the Thomson line-average density
# ---------------------------------------------------------------------------
PRESSURES = {   # panel -> [(path, instrument, marker)]
    "divertor": [("barometry/barometer/pressure_div", "baratron", "o"),
                 ("barometry/pressure_gauges/pressure_div", "APG gauge", "s")],
}
GAS = {   # flow quantity -> (label, scale)
    "flow_rate": (r"gas flow [Pa m$^3$ s$^{-1}$]", 1),
    "flow_particles": (r"gas flow [$10^{21}$ particles s$^{-1}$]", 1e-21),
}


def gas_flow(s, key="flow_rate", valve=None):
    """Gas flow of valve `valve` (name, e.g. 'V1'), or the sum over all valves of the shot
    (None); NaN-aware. key: 'flow_rate' [Pa m^3/s] or 'flow_particles' [particles/s]."""
    if not s.diagnostics.get("gas_injection", False):
        raise Unavailable(f"shot {s.shot}: gas_injection not available")
    with h5py.File(s.path, "r") as f:
        names = list(f["gas_injection/valve"]) if "valve" in f["gas_injection"] else []
    names = [valve] if valve else names
    flows = [s[f"gas_injection/valve/{n}/{key}"] for n in names
             if s.has(f"gas_injection/valve/{n}/{key}")]
    if not flows:
        raise Unavailable(f"shot {s.shot}: no gas valve data")
    return np.nansum(flows, axis=0)


def vs_density(shots, color_by="auto", t_range=(0.6, 1.6), err_style="bar", gas="flow_rate",
               valve=None):
    """Divertor pressure (baratron = circles, APG gauge = squares), valve gas flow
    (`gas`: 'flow_rate' or 'flow_particles'; `valve`: name, default the sum of all valves)
    and upstream ne_u (with errors) against the Thomson line-average density; colour = shot."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    xlabel = r"$\bar n_e$ (TS) [$10^{19}$ m$^{-3}$]"
    gas_label, gas_scale = GAS[gas]
    panels = [(axes[0], "divertor pressure [mPa]", PRESSURES["divertor"]),
              (axes[1], gas_label, [(None, None, "o")])]
    for ax, ylabel, items in panels:
        for s in shots:
            try:
                x = s["thomson_scattering/av_dens"] * 1e-19
            except Unavailable as ex:
                pt._warn(s, "TS line-average density", ex)
                continue
            m = (s.t >= t_range[0]) & (s.t <= t_range[1])
            for path, _, mk in items:
                try:
                    y = gas_flow(s, gas, valve) * gas_scale if path is None else s[path] * 1e3
                    ax.scatter(x[m], y[m], color=s.color(color_by),
                               marker=s.marker if path is None else mk, s=18)
                except Unavailable as ex:
                    pt._warn(s, path or "gas flow", ex)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if items[0][0] is None:
            pt._legend(ax, shots, line=False)
        else:
            ax.legend(handles=[plt.Line2D([], [], color="gray", marker=mk, ls="none", label=lab)
                               for _, lab, mk in items], fontsize=8, frameon=False,
                      loc="upper left")
    pt.xy_plot(shots, lambda s: s["thomson_scattering/av_dens"] * 1e-19,
               lambda s: s["thomson_scattering/ne_u"] * 1e-19, xlabel,
               r"$n_{e,u}$ (separatrix, TS) [$10^{19}$ m$^{-3}$]",
               yerr_get=lambda s: s["thomson_scattering/dne_u"] * 1e-19, t_range=t_range,
               color_by=color_by, ax=axes[2], err_style=err_style, annotate=False)
    fig.tight_layout()
    return fig, axes
