#!/usr/bin/env python3
"""H-mode / ELM analysis of the ELM-mode databases (shot_<N>.h5 built with the ELM time base).

Figures (one set per shot):
  time_traces_zoom    fast signals over a short span with every ELM marked
  time_traces_window  the whole database window: IR power, stored energy, heating, density,
                      instantaneous ELM frequency
  elm_statistics      frequency, inter-ELM intervals, ELM sizes (IR energy, W_MHD loss) and how the
                      size depends on the time since the previous ELM
  elm_average         ELM-synchronised average (median and 16-84% band) of the fast signals
  ts_phase            Thomson ne, Te profiles binned by the phase of the ELM cycle
  pedestal_phase      pedestal ne, Te against the time since the last ELM

ELM times: the peaks of the IR power (default camera_ir/HIR/power; 0.1 ms) near the ELMs found by the
builder (root dataset elm_times), so they are good to the grid step, not just 1 ms. Units are those
of the database; IR power and energies are in W / J as assumed there.

Python:
    from pexdb import shotdb, hmode_analysis as ha
    s = shotdb.load("shot_87672.h5")[0]                 # common view: everything on the fast grid
    times = ha.elm_times(s)
    tab = ha.elm_table(s, times)                        # dict of arrays: interval, IR peak/energy, dW...
    figs = ha.analyse(s, save="figures/hm")             # all figures
Command line:
    python3 -m pexdb.hmode_analysis $PEX_DB_DIR/shot_87672.h5 --save figures/hm [--t0 0.9 --span 0.06]
"""
import argparse
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from pexdb import shotdb

PRE = (-4e-3, -1.5e-3)        # [s] baseline window before an ELM (relative to its IR peak)
PEAK = (-1e-3, 1.5e-3)        # [s] where the ELM peak is searched
ENERGY = (-1e-3, 4e-3)        # [s] integration window for the energy of an ELM
W_PRE, W_POST = (-4e-3, -0.5e-3), (0.5e-3, 3e-3)   # [s] windows to measure the W_MHD loss
EPOCH = (-5e-3, 15e-3)        # [s] window of the ELM-synchronised averages
ISOLATED = (4.5e-3, 3.5e-3)   # [s] an ELM is "isolated" if the previous / next ELM is farther than this
                              # (otherwise the baseline and integration windows above overlap a neighbour)


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def _need_elm_file(s):
    if s.time_mode != "elm":
        raise ValueError(f"shot {s.shot}: not an ELM-mode file (time_mode={s.time_mode!r}); the ELM "
                         "analysis needs the 0.1 ms fast grid. Build it with --mode=elm (or let "
                         "--mode=auto detect the ELMs).")


def _sl(t, a, b):
    """Slice of the sorted time axis t covering [a, b]."""
    return slice(int(np.searchsorted(t, a)), int(np.searchsorted(t, b, side="right")))


def _get(s, path):
    """Array at `path` (common view) or None if the shot does not have it."""
    return s[path] if s.has(path) else None


def _trapz(y, t):
    f = getattr(np, "trapezoid", None) or np.trapz
    return float(f(y, t))


# --------------------------------------------------------------------------------------------------
# ELM times and sizes
# --------------------------------------------------------------------------------------------------
def elm_times(s, signal="camera_ir/HIR/power", edge=6e-3):
    """ELM times [s] inside the file's window: the maxima of `signal` (fast grid) within +-1.5 ms of
    the ELMs found by the builder. Falls back to a spike search on `signal` itself if the file has no
    elm_times. ELMs closer than `edge` to the ends of the window are dropped (no room for the averages)."""
    _need_elm_file(s)
    t, y = s.t, s[signal]
    coarse = s.elm_times[(s.elm_times >= t[0]) & (s.elm_times <= t[-1])]
    if coarse.size == 0:
        from pexdb import build_database as bd
        coarse = bd.spike_train(t, y, 1.0, (t[0], t[-1]))["times"]
    out = []
    for tc in coarse:
        k = _sl(t, tc - 1.5e-3, tc + 1.5e-3)
        if k.stop > k.start and np.isfinite(y[k]).any():
            out.append(t[k][np.nanargmax(y[k])])
    out = np.unique(np.round(out, 6))
    if out.size:                                          # two coarse times can map to one peak
        out = out[np.concatenate([[True], np.diff(out) > 2e-3])]
    return out[(out >= t[0] + edge) & (out <= t[-1] - edge)]


def _size(t, y, tk, pre=PRE, peak=PEAK, energy=ENERGY):
    """(peak above baseline, energy above baseline) of an ELM at tk in the fast trace y."""
    a = y[_sl(t, tk + pre[0], tk + pre[1])]
    base = np.nanmedian(a) if a.size and np.isfinite(a).any() else np.nan
    pk = y[_sl(t, tk + peak[0], tk + peak[1])]
    e = _sl(t, tk + energy[0], tk + energy[1])
    return (np.nanmax(pk) - base if pk.size else np.nan,
            _trapz(y[e] - base, t[e]) if e.stop - e.start > 2 else np.nan)


def elm_table(s, times):
    """Per-ELM quantities as arrays (NaN where a diagnostic is missing):
      t, interval_prev, interval_next [s]
      ir_hir_peak [W], ir_hir_energy [J]   IR power of HIR (inner target) above its pre-ELM level
      ir_vir_peak, ir_vir_energy           same for VIR (outer target)
      apg_peak                             divertor pressure peak above its pre-ELM level
      dW_stored  [J]    W_MHD before minus after the ELM (positive = loss), from heating/energy_mhd
      dW_power   [J]    minus the integral of dW_MHD/dt over the ELM (needs heating/power_mhd)
      isolated          True if the previous ELM is > ISOLATED[0] and the next > ISOLATED[1] away; the
                        sizes of the other ELMs are contaminated by their neighbours (use _iso)"""
    t = s.t
    n = len(times)
    tab = {"t": np.asarray(times, float)}
    iv = np.diff(times) if n > 1 else np.array([])
    tab["interval_prev"] = np.concatenate([[np.nan], iv])
    tab["interval_next"] = np.concatenate([iv, [np.nan]])
    for key, path in (("ir_hir", "camera_ir/HIR/power"), ("ir_vir", "camera_ir/VIR/power")):
        y = _get(s, path)
        res = np.array([_size(t, y, tk) for tk in times]) if y is not None else np.full((n, 2), np.nan)
        tab[key + "_peak"], tab[key + "_energy"] = res[:, 0], res[:, 1]
    p = _get(s, "barometry/pressure_gauges/pressure_div")
    tab["apg_peak"] = (np.array([_size(t, p, tk)[0] for tk in times]) if p is not None
                       else np.full(n, np.nan))
    w = _get(s, "heating/energy_mhd")
    if w is not None:
        tab["dW_stored"] = np.array([np.nanmean(w[_sl(t, tk + W_PRE[0], tk + W_PRE[1])]) -
                                     np.nanmean(w[_sl(t, tk + W_POST[0], tk + W_POST[1])]) for tk in times])
    else:
        tab["dW_stored"] = np.full(n, np.nan)
    pw = _get(s, "heating/power_mhd")
    if pw is not None:
        tab["dW_power"] = np.array([-_trapz(pw[_sl(t, tk - 1e-3, tk + 3e-3)], t[_sl(t, tk - 1e-3, tk + 3e-3)])
                                    if np.isfinite(pw[_sl(t, tk - 1e-3, tk + 3e-3)]).all() else np.nan
                                    for tk in times])
    else:
        tab["dW_power"] = np.full(n, np.nan)
    tab["isolated"] = (tab["interval_prev"] > ISOLATED[0]) & (tab["interval_next"] > ISOLATED[1])
    return tab


def _iso(tab, key):
    """tab[key] of the isolated ELMs only (NaN for the others)."""
    return np.where(tab["isolated"], tab[key], np.nan)


def elm_report(s, times, tab=None):
    """Text summary of the ELMs of a shot."""
    tab = tab or elm_table(s, times)
    iv = tab["interval_prev"][np.isfinite(tab["interval_prev"])]
    lines = [f"{s.label}: {times.size} ELMs between {times[0]:.3f} and {times[-1]:.3f} s"]
    if iv.size:
        lines.append(f"  inter-ELM interval {1e3 * np.mean(iv):.2f} +- {1e3 * np.std(iv):.2f} ms "
                     f"(median {1e3 * np.median(iv):.2f}, cv {np.std(iv) / np.mean(iv):.2f}) -> "
                     f"f = {1 / np.mean(iv):.0f} Hz")

    lines.append(f"  isolated ELMs (sizes below): {int(tab['isolated'].sum())} of {times.size}")

    def stat(label, key, scale, unit):
        v = _iso(tab, key)
        v = v[np.isfinite(v)]
        if v.size:
            lines.append(f"  {label:34s} {scale * np.mean(v):8.3g} +- {scale * np.std(v):.2g} {unit}   (n={v.size})")
    stat("IR HIR energy per ELM (inner)", "ir_hir_energy", 1e-3, "kJ")
    stat("IR VIR energy per ELM (outer)", "ir_vir_energy", 1e-3, "kJ")
    stat("W_MHD loss per ELM (stored energy)", "dW_stored", 1e-3, "kJ")
    stat("W_MHD loss per ELM (dEMHD integral)", "dW_power", 1e-3, "kJ")
    if "heating/power_mhd" in [p for _, p, _ in TRACES] and s.has("heating/power_mhd"):
        lines.append("  note: dEMHD has a strong ~1.2 kHz ripple (check its spectrum); single-ELM dEMHD is noisy")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------------
TRACES = [("IR HIR power [kW]", "camera_ir/HIR/power", 1e-3),
          ("IR VIR power [kW]", "camera_ir/VIR/power", 1e-3),
          ("divertor pressure (gauge)", "barometry/pressure_gauges/pressure_div", 1),
          (r"$dW_{MHD}/dt$ [MW]", "heating/power_mhd", 1e-6),
          (r"$W_{MHD}$ [kJ]", "heating/energy_mhd", 1e-3)]


def plot_time_traces(s, times, t0=None, span=0.05):
    """Fast signals over [t0, t0 + span] with every ELM marked (red lines), and the whole window.
    Returns {'time_traces_zoom': fig, 'time_traces_window': fig}."""
    t = s.t
    if t0 is None:                                       # middle of the ELMs where dW_MHD/dt has data
        pm = _get(s, "heating/power_mhd")
        sel = times
        if pm is not None:
            ok = np.isfinite(pm)[np.clip(np.searchsorted(t, times), 0, t.size - 1)]
            sel = times[ok] if ok.sum() >= 3 else times
        t0 = float(np.median(sel)) - span / 2
    t0 = float(np.clip(t0, t[0], t[-1] - span))
    k = _sl(t, t0, t0 + span)
    avail = [(lab, p, sc) for lab, p, sc in TRACES if s.has(p) and np.isfinite(s[p][k]).any()]
    figs = {}

    fig, axes = plt.subplots(len(avail), 1, sharex=True, figsize=(10, 1.7 * len(avail) + 0.8), squeeze=False)
    for ax, (lab, p, sc) in zip(axes[:, 0], avail):
        ax.plot(1e3 * (t[k] - t0), s[p][k] * sc, lw=0.8, color="k")
        for tk in times[(times >= t0) & (times <= t0 + span)]:
            ax.axvline(1e3 * (tk - t0), color="r", lw=0.7, alpha=0.7)
        ax.set_ylabel(lab, fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel(f"t - {t0:.3f} s [ms]")
    axes[0, 0].set_title(f"{s.label}: red lines = ELMs ({int(((times >= t0) & (times <= t0 + span)).sum())} in "
                         f"{1e3 * span:.0f} ms)", fontsize=9)
    fig.tight_layout()
    figs["time_traces_zoom"] = fig

    panels = [("IR power [kW]", [("camera_ir/HIR/power", "HIR", 1e-3), ("camera_ir/VIR/power", "VIR", 1e-3)]),
              (r"$W_{MHD}$ [kJ]", [("heating/energy_mhd", "", 1e-3)]),
              ("heating [kW]", [("heating/power_ohmic", "ohmic", 1e-3), ("heating/power_nbi", "NBI", 1e-3),
                                ("heating/power_ecrh", "ECRH", 1e-3)]),
              (r"$\bar n_e$ (TS) [$10^{19}$ m$^{-3}$]", [("thomson_scattering/av_dens", "", 1e-19)])]
    fig, axes = plt.subplots(len(panels) + 1, 1, sharex=True, figsize=(11, 1.9 * (len(panels) + 1) + 0.8))
    for ax, (lab, items) in zip(axes[:-1], panels):
        for p, name, sc in items:
            if s.has(p) and np.isfinite(s[p]).any():
                ax.plot(t, s[p] * sc, lw=0.6, label=name or None)
        top = ax.get_ylim()[1]
        ax.plot(times, np.full(times.size, top), "|", color="r", ms=6)
        ax.set_ylabel(lab, fontsize=8)
        ax.grid(alpha=0.3)
        if any(n for _, n, _ in items):
            ax.legend(fontsize=7, ncol=3, frameon=False, loc="lower left")
    if times.size > 2:
        ax = axes[-1]
        ax.plot(0.5 * (times[1:] + times[:-1]), 1 / np.diff(times), ".", color="k", ms=4)
        ax.set_ylabel("ELM frequency [Hz]", fontsize=8)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("t [s]")
    axes[0].set_title(f"{s.label}: ELMs marked by red ticks", fontsize=9)
    fig.tight_layout()
    figs["time_traces_window"] = fig
    return figs


def plot_elm_statistics(s, times, tab=None):
    """ELM frequency, intervals and sizes, and size against the previous interval."""
    tab = tab or elm_table(s, times)
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    iv = tab["interval_prev"]
    ok = np.isfinite(iv)
    a = ax[0, 0]
    a.plot(tab["t"][ok], 1 / iv[ok], ".", color="k")
    if ok.sum() > 5:
        w = min(9, ok.sum() // 2 * 2 - 1)
        sm = np.convolve(1 / iv[ok], np.ones(w) / w, "same")
        a.plot(tab["t"][ok][w // 2:-(w // 2)], sm[w // 2:-(w // 2)], color="r", lw=1.2, label=f"{w}-ELM mean")
        a.legend(fontsize=7, frameon=False)
    a.set_xlabel("t [s]"); a.set_ylabel("1 / interval [Hz]"); a.set_title("instantaneous ELM frequency", fontsize=9)
    a = ax[0, 1]
    if ok.any():
        a.hist(1e3 * iv[ok], bins=20, color="0.6", ec="k")
        a.axvline(1e3 * np.mean(iv[ok]), color="r")
        a.set_title(f"intervals: {1e3 * np.mean(iv[ok]):.1f} ms, cv {np.std(iv[ok]) / np.mean(iv[ok]):.2f}", fontsize=9)
    a.set_xlabel("interval to the previous ELM [ms]"); a.set_ylabel("ELMs")
    a = ax[0, 2]
    for key, lab, c in (("ir_hir_energy", "IR HIR (inner)", "tab:blue"), ("ir_vir_energy", "IR VIR (outer)", "tab:orange"),
                        ("dW_stored", r"$\Delta W_{MHD}$", "tab:green")):
        v = _iso(tab, key)
        v = v[np.isfinite(v)]
        if v.size:
            a.hist(1e-3 * v, bins=15, alpha=0.5, label=f"{lab}: {1e-3 * np.mean(v):.2f} kJ", color=c)
    a.set_xlabel("energy per ELM [kJ]"); a.set_ylabel("ELMs"); a.legend(fontsize=7, frameon=False)
    a.set_title(f"ELM size ({int(tab['isolated'].sum())} isolated ELMs)", fontsize=9)
    for a, key, lab in ((ax[1, 0], "ir_hir_energy", "IR HIR energy [kJ]"), (ax[1, 1], "dW_stored", r"$\Delta W_{MHD}$ [kJ]")):
        y = _iso(tab, key)
        m = np.isfinite(iv) & np.isfinite(y)
        a.plot(1e3 * iv[m], 1e-3 * y[m], ".", color="k")
        if m.sum() > 4:
            c = np.corrcoef(iv[m], y[m])[0, 1]
            p = np.polyfit(1e3 * iv[m], 1e-3 * y[m], 1)
            xs = np.array([1e3 * iv[m].min(), 1e3 * iv[m].max()])
            a.plot(xs, np.polyval(p, xs), "r-", lw=1)
            a.set_title(f"r = {c:.2f} (n={m.sum()})", fontsize=9)
        a.set_xlabel("interval to the previous ELM [ms]"); a.set_ylabel(lab)
    a = ax[1, 2]
    eh, ev = _iso(tab, "ir_hir_energy"), _iso(tab, "ir_vir_energy")
    m = np.isfinite(eh) & np.isfinite(ev)
    a.plot(1e-3 * eh[m], 1e-3 * ev[m], ".", color="k")
    if m.sum() > 4:
        a.set_title(f"inner vs outer target, r = {np.corrcoef(eh[m], ev[m])[0, 1]:.2f}", fontsize=9)
    a.set_xlabel("IR HIR energy [kJ]"); a.set_ylabel("IR VIR energy [kJ]")
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.suptitle(s.label, fontsize=10)
    fig.tight_layout()
    return {"elm_statistics": fig}


def _stack(t, y, times, lo, hi, subtract_pre=False):
    """(tau, array (n_elm, n_tau)) of y around each ELM, on the grid samples (no interpolation)."""
    dt = float(np.median(np.diff(t)))
    a, b = int(round(lo / dt)), int(round(hi / dt))
    tau = np.arange(a, b + 1) * dt
    rows = []
    for tk in times:
        i = int(np.argmin(np.abs(t - tk)))
        if i + a >= 0 and i + b < t.size:
            seg = y[i + a: i + b + 1].astype(float)
            if subtract_pre:
                pre = y[_sl(t, tk + W_PRE[0], tk + W_PRE[1])]
                seg = seg - np.nanmean(pre)
            rows.append(seg)
    return tau, np.array(rows)


def plot_elm_average(s, times, lo=EPOCH[0], hi=EPOCH[1], max_lines=30):
    """ELM-synchronised average of the fast signals: median, 16-84% band and some single ELMs."""
    t = s.t
    sigs = [("IR HIR power [kW]", "camera_ir/HIR/power", 1e-3, False),
            ("IR VIR power [kW]", "camera_ir/VIR/power", 1e-3, False),
            ("divertor pressure", "barometry/pressure_gauges/pressure_div", 1, False),
            (r"$dW_{MHD}/dt$ [MW]", "heating/power_mhd", 1e-6, False),
            (r"$W_{MHD}$ change [kJ]", "heating/energy_mhd", 1e-3, True)]
    sigs = [x for x in sigs if s.has(x[1]) and np.isfinite(s[x[1]]).any()]
    fig, ax = plt.subplots(len(sigs), 1, sharex=True, figsize=(8, 1.9 * len(sigs) + 0.8), squeeze=False)
    for a, (lab, p, sc, sub) in zip(ax[:, 0], sigs):
        tau, st = _stack(t, s[p], times, lo, hi, sub)
        st = st * sc
        st = st[np.isfinite(st).mean(axis=1) > 0.9]
        if st.shape[0] < 3:
            a.text(0.5, 0.5, "not enough ELMs with data", transform=a.transAxes, ha="center", color="gray")
            a.set_ylabel(lab, fontsize=8)
            continue
        for row in st[:max_lines]:
            a.plot(1e3 * tau, row, color="0.8", lw=0.4)
        q16, q50, q84 = np.nanpercentile(st, [16, 50, 84], axis=0)
        a.fill_between(1e3 * tau, q16, q84, color="tab:blue", alpha=0.3, lw=0)
        a.plot(1e3 * tau, q50, color="tab:blue", lw=1.4)
        a.axvline(0, color="r", lw=0.6)
        a.set_ylabel(lab, fontsize=8)
        a.grid(alpha=0.3)
        a.text(0.99, 0.92, f"n = {st.shape[0]}", transform=a.transAxes, ha="right", va="top", fontsize=7)
    ax[-1, 0].set_xlabel("time from the ELM [ms]")
    ax[0, 0].set_title(f"{s.label}: ELM-synchronised average (median, 16-84%)", fontsize=9)
    fig.tight_layout()
    return {"elm_average": fig}


def _phase_samples(sn, times, path):
    """Real (measured) samples of a slow diagnostic with the phase of the ELM cycle they were taken at.
    Returns (frame indices k, time since the previous ELM [s], cycle fraction in [0,1)). `sn` must be
    loaded with grid='native'."""
    group = path.rsplit("/", 1)[0]
    ts, meas = sn[group + "/t_sample"], sn[group + "/measured"].astype(bool)
    k = np.flatnonzero(meas & np.isfinite(ts))
    tsam = ts[k]
    j = np.searchsorted(times, tsam, side="right") - 1
    ok = (j >= 0) & (j < times.size - 1)
    k, tsam, j = k[ok], tsam[ok], j[ok]
    since = tsam - times[j]
    frac = since / (times[j + 1] - times[j])
    return k, since, frac


def plot_ts_phase(sn, times, nbins=4, rho_edges=None):
    """Thomson ne and Te profiles (against rho) averaged in bins of the ELM-cycle fraction (0 = ELM,
    1 = next ELM). Only real TS samples are used, each once. `sn` loaded with grid='native'."""
    for need in ("thomson_scattering/ne", "thomson_scattering/te", "thomson_scattering/rho",
                 "thomson_scattering/t_sample", "thomson_scattering/measured"):
        if not sn.has(need):
            return {}
    k, since, frac = _phase_samples(sn, times, "thomson_scattering/te")
    edges = np.linspace(0.0, 1.2, 25) if rho_edges is None else rho_edges
    ctr = 0.5 * (edges[1:] + edges[:-1])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    cols = plt.cm.viridis(np.linspace(0.05, 0.9, nbins))
    for a, q, lab, sc in ((ax[0], "ne", r"$n_e$ [$10^{19}$ m$^{-3}$]", 1e-19), (ax[1], "te", r"$T_e$ [eV]", 1)):
        Q, R = sn["thomson_scattering/" + q][k] * sc, sn["thomson_scattering/rho"][k]
        for b in range(nbins):
            sel = (frac >= b / nbins) & (frac < (b + 1) / nbins)
            prof = []
            for i in np.flatnonzero(sel):
                idx = np.digitize(R[i], edges) - 1
                prof.append([np.nanmean(Q[i][idx == c]) if np.any((idx == c) & np.isfinite(Q[i])) else np.nan
                             for c in range(ctr.size)])
            if not prof:
                continue
            prof = np.array(prof)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)           # rho bins without data
                m = np.nanmean(prof, axis=0)
                e = np.nanstd(prof, axis=0) / np.sqrt(np.maximum(np.sum(np.isfinite(prof), axis=0), 1))
            a.errorbar(ctr, m, yerr=e, color=cols[b], marker="o", ms=3, lw=0.8, capsize=2,
                       label=f"phase {b / nbins:.2f}-{(b + 1) / nbins:.2f}  (n={sel.sum()})")
        a.set_xlabel(r"$\rho$"); a.set_ylabel(lab); a.grid(alpha=0.3)
    ax[0].legend(fontsize=7, frameon=False)
    ax[0].set_title(f"{sn.label}: profiles in bins of the ELM cycle", fontsize=9)
    fig.tight_layout()
    return {"ts_phase": fig}


def plot_pedestal_phase(sn, times, rho_range=(0.9, 1.0)):
    """Mean TS ne and Te over rho_range (pedestal) of every real TS sample against the time since the
    previous ELM, with bin means. `sn` loaded with grid='native'."""
    for need in ("thomson_scattering/ne", "thomson_scattering/te", "thomson_scattering/rho"):
        if not sn.has(need):
            return {}
    k, since, frac = _phase_samples(sn, times, "thomson_scattering/te")
    R = sn["thomson_scattering/rho"][k]
    inside = (R >= rho_range[0]) & (R <= rho_range[1])
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, q, lab, sc in ((ax[0], "ne", r"pedestal $n_e$ [$10^{19}$ m$^{-3}$]", 1e-19), (ax[1], "te", "pedestal $T_e$ [eV]", 1)):
        Q = np.where(inside, sn["thomson_scattering/" + q][k] * sc, np.nan)
        with np.errstate(all="ignore"):
            v = np.nanmean(Q, axis=1)
        m = np.isfinite(v)
        a.plot(1e3 * since[m], v[m], ".", color="0.4")
        if m.sum() > 6:
            edges = np.linspace(0, np.nanpercentile(since[m], 98), 6)
            idx = np.digitize(since[m], edges) - 1
            xc = 0.5 * (edges[1:] + edges[:-1])
            ym = [np.mean(v[m][idx == i]) if np.any(idx == i) else np.nan for i in range(xc.size)]
            ye = [np.std(v[m][idx == i]) / np.sqrt(max(np.sum(idx == i), 1)) if np.any(idx == i) else np.nan
                  for i in range(xc.size)]
            a.errorbar(1e3 * xc, ym, yerr=ye, color="r", marker="s", capsize=3, lw=1.2, label="bin mean")
            a.legend(fontsize=7, frameon=False)
        a.set_xlabel("time since the previous ELM [ms]"); a.set_ylabel(lab); a.grid(alpha=0.3)
    ax[0].set_title(f"{sn.label}: {rho_range[0]}<rho<{rho_range[1]}, {m.sum()} TS samples", fontsize=9)
    fig.tight_layout()
    return {"pedestal_phase": fig}


# --------------------------------------------------------------------------------------------------
def analyse(s, sn=None, save=None, fmt="png", t0=None, span=0.05, signal="camera_ir/HIR/power"):
    """Run the whole analysis for one shot; prints the ELM report and returns {name: figure}.
    s: common-view Shot, sn: same file with grid='native' (loaded here if None).
    save: filename prefix; writes <prefix>_<shot>_<figure>.<fmt>."""
    sn = sn or shotdb.load(s.path, grid="native")[0]
    times = elm_times(s, signal)
    if times.size < 5:
        raise ValueError(f"shot {s.shot}: only {times.size} ELMs found in the window {s.t[0]:.3f}-{s.t[-1]:.3f} s")
    tab = elm_table(s, times)
    print(elm_report(s, times, tab))
    figs = {}
    figs.update(plot_time_traces(s, times, t0, span))
    figs.update(plot_elm_statistics(s, times, tab))
    figs.update(plot_elm_average(s, times))
    figs.update(plot_ts_phase(sn, times))
    figs.update(plot_pedestal_phase(sn, times))
    if save:
        for name, fig in figs.items():
            f = f"{save}_{s.shot}_{name}.{fmt}"
            fig.savefig(f, dpi=150)
            print(f"  wrote {f}")
    return figs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("source", nargs="+", help="ELM-mode shot_*.h5 file(s) or folder(s)")
    ap.add_argument("--save", help="filename prefix: writes <prefix>_<shot>_<figure>.<format>")
    ap.add_argument("--format", default="png", choices=("png", "svg", "pdf"))
    ap.add_argument("--t0", type=float, help="start [s] of the zoomed time traces (default: middle of the ELMs)")
    ap.add_argument("--span", type=float, default=0.05, help="length [s] of the zoomed time traces (default 0.05)")
    ap.add_argument("--signal", default="camera_ir/HIR/power", help="signal whose peaks give the ELM times")
    args = ap.parse_args()
    if not args.save and matplotlib.get_backend().lower() in {"agg", "pdf", "ps", "svg", "pgf", "cairo", "template"}:
        raise SystemExit("matplotlib backend cannot open windows (no display?). Use --save <prefix> "
                         "or set MPLBACKEND=TkAgg / QtAgg / MacOSX.")
    for s in shotdb.load(args.source):
        try:
            analyse(s, save=args.save, fmt=args.format, t0=args.t0, span=args.span, signal=args.signal)
        except ValueError as e:
            print(f"  skip {s.label}: {e}")
    if not args.save:
        plt.show()


if __name__ == "__main__":
    main()
