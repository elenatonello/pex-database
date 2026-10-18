#!/usr/bin/env python3
"""Build a uniform-in-time HDF5 database (one file per shot) from STRUCexp .mat files.

Every time-dependent quantity of every diagnostic is resampled on the same
N_FRAMES-point time base between T_START and T_END. Each frame is the
NaN-aware average of all native samples falling in [t - WINDOW/2, t + WINDOW/2)
(WINDOW = 10 ms, i.e. +-5 ms around the frame time; the frame spacing is
25.6 ms, so windows do not touch). This is done automatically only when the
diagnostic is sampled fast enough (>= MIN_SAMPLES_PER_WINDOW samples per window);
slower diagnostics (old TS system, CXRS, DSS, MANTIS...) are linearly interpolated
instead (see resample(), interpolate()), otherwise many frames would be empty.
The method used is stored in the `time_resampling` attribute of each group.

Top-level groups are named after the IMAS IDS they correspond to (see IDS_MAP);
the original TCV name is kept in each group's `tcv_name` attribute.

Langmuir probes are NOT resampled: all native samples in [T_START, T_END] are\nkept on their own time vector (see crop()).

Missing diagnostics: every diagnostic is a top-level HDF5 group carrying an
attribute `available` (1/0). If the source is absent, flagged unavailable,
empty, or fails to convert, the group is still written, with available=0, an
`reason` attribute, and no datasets. Check `available` before reading.

Layout: a dict leaf is an HDF5 group, an array is a dataset, a string/number
is an attribute-free scalar dataset. Time is the first axis, except where
noted in the `layout` attribute of a group or in the docs below.

Usage: python3 -m pexdb.build_database STRUCTexp.mat [out_dir] [--strict] [--mode=auto|default|elm]
         [--regime=L|H] [--t0=0.6] [--t1=1.0] [--dt-fast=1e-4] [--dt-slow=5e-3] [--mhd=mhd.mat]  (mat file with t_dMHD, dEMHD and optionally EMHD)
       python3 -m pexdb.build_database STRUCTexp.mat --detect      # prints ELMY=0|1 only
The time base is chosen automatically (--mode=auto, default): ELMy H-mode shots (detected from the
data, see detect_elms) get the ELM-resolved base (0.6-1.0 s; heating, valve, pressure gauges and IR on a
0.1 ms grid, everything else on a 5 ms grid, see configure()); all others the 40-frame base.
--mode=elm / default force one; --elm is the same as --mode=elm. --regime forces the L/H label (and
the base when --mode=auto). For L-mode shots the NBI power is stored as a constant mean over the
window (see mean_power()).
Output: <out_dir>/shot_<shot>.h5
"""
import os
import sys
import warnings
import numpy as np
import h5py
import scipy.io as sio

# Time base. The functions below read these module globals at call time.
#   default mode: one grid for every diagnostic (L-mode: 40 frames, 0.6-1.6 s, 10 ms window)
#   ELM mode    : two nested grids on a shorter window (see configure()): a fast grid for the
#                 fast diagnostics and a coarser slow grid for the slow ones, whose samples are
#                 held on the nearest frames instead of being interpolated.
T_START, T_END, N_FRAMES = 0.6, 1.6, 40
T_FRAMES = np.linspace(T_START, T_END, N_FRAMES)
DT = T_FRAMES[1] - T_FRAMES[0]            # frame spacing
WINDOW = 0.010                            # total averaging window width [s], centred on each frame
MODE = "default"                          # "default" or "elm"
CURRENT_TIER = "fast"                     # grid in use: "fast" or "slow"
GRIDS = {}                                # tier -> (frames, window); filled by configure()
# ELM mode: only these are on the fast grid (HEATING, VALVE, BAR, APG, IR, plus the probe Jsat); all the
# others are on the slow grid. Of those, the ones sampled slower than the slow frames hold their
# samples (nearest-sample hold, t_sample / measured); the faster ones are averaged in each bin.
SLOW_DIAGNOSTICS = ("BOLO", "FIR", "LIUQE", "TS", "CXRS", "MANTIS", "DSS")
HOLD_DIAGNOSTICS = ("TS", "CXRS", "IR", "MANTIS", "DSS")
CURRENT_HOLD = False                      # True while converting a HOLD_DIAGNOSTICS entry


def configure(mode="default", t_start=None, t_end=None, dt_fast=1e-4, dt_slow=5e-3):
    """Set the time grids. mode='default': one grid (the module defaults, or t_start/t_end with
    40 frames). mode='elm': fast grid with step dt_fast and slow grid with step dt_slow between
    t_start and t_end (defaults 0.6-1.0 s, 0.1 ms -> 4001 frames and 5 ms -> 81 frames); window =
    frame step (contiguous bins). dt_slow should be a multiple of dt_fast so the slow frames are a
    subset of the fast ones."""
    global MODE, T_START, T_END, GRIDS
    MODE = mode
    if mode == "elm":
        T_START = 0.6 if t_start is None else t_start
        T_END = 1.0 if t_end is None else t_end
        GRIDS = {}
        for tier, dt in (("fast", dt_fast), ("slow", dt_slow)):
            n = int(round((T_END - T_START) / dt)) + 1
            GRIDS[tier] = (np.linspace(T_START, T_END, n), dt)
        if abs(dt_slow / dt_fast - round(dt_slow / dt_fast)) > 1e-6:
            print(f"  WARNING: dt_slow ({dt_slow}) is not a multiple of dt_fast ({dt_fast})")
    else:
        T_START = 0.6 if t_start is None else t_start
        T_END = 1.6 if t_end is None else t_end
        GRIDS = {tier: (np.linspace(T_START, T_END, 40), 0.010) for tier in ("fast", "slow")}
    set_grid("fast")


def set_hold(flag):
    global CURRENT_HOLD
    CURRENT_HOLD = bool(flag)


def set_grid(tier):
    """Make `tier` ('fast' or 'slow') the active grid for the d_* functions."""
    global CURRENT_TIER, T_FRAMES, N_FRAMES, DT, WINDOW
    if not GRIDS:
        return
    CURRENT_TIER = tier
    T_FRAMES, WINDOW = GRIDS[tier]
    N_FRAMES = len(T_FRAMES)
    DT = T_FRAMES[1] - T_FRAMES[0]


configure("default")                      # default grid until configure() is called


class Missing(Exception):
    """Raised when a diagnostic has no usable data in the source file."""


MIN_SAMPLES_PER_WINDOW = 2     # window average only if the window holds >= this many native samples
_METHOD_LOG = []               # (method, native step) of every resampling call; see build()


_ALIGN_NOTED = set()


def align(y, t, axis=0):
    """Trim `y` (along `axis`) and `t` to their common length when they differ by a few samples
    (some traces come with one value more than time stamps, e.g. dEMHD); the extra samples at the
    end are dropped. Larger mismatches are left alone (they will raise)."""
    y, t = np.asarray(y), np.asarray(t).ravel()
    n = y.shape[axis] if y.ndim else 0
    if n != t.size and y.ndim and abs(n - t.size) <= 3 and min(n, t.size) > 0:
        k = min(n, t.size)
        if (n, t.size) not in _ALIGN_NOTED:
            _ALIGN_NOTED.add((n, t.size))
            print(f"  note: trace with {n} values but {t.size} time stamps: trimmed to {k}")
        return np.take(y, np.arange(k), axis=axis), t[:k]
    return y, t


def native_step(t):
    """Median sampling step of the time vector [s] (inf if it cannot be determined)."""
    d = np.diff(np.asarray(t, dtype=float).ravel())
    d = d[np.isfinite(d) & (d > 0)]
    return float(np.median(d)) if d.size else np.inf


def held_index(t):
    """Index of the native sample nearest to each frame (-1 if farther than 2 native steps,
    i.e. across a gap in the time vector). Uses the time vector only, not the values."""
    t = np.asarray(t, dtype=float).ravel()
    n = t.size
    p = np.searchsorted(t, T_FRAMES)
    left, right = np.clip(p - 1, 0, n - 1), np.clip(p, 0, n - 1)
    dl, dr = np.abs(T_FRAMES - t[left]), np.abs(t[right] - T_FRAMES)
    j = np.where(dl <= dr, left, right)
    j[np.minimum(dl, dr) > 2 * native_step(t)] = -1
    return j


def hold(arr, t, axis):
    """Nearest-sample hold onto T_FRAMES: each frame takes the value of the native sample
    closest in time (repeating it over the frames around that sample). No interpolation,
    so values before and after an ELM are never mixed. Frames farther than 2 native steps
    from any sample are NaN. See sample_info() for the matching `t_sample` / `measured`."""
    arr = np.asarray(arr, dtype=float)
    t = np.asarray(t, dtype=float).ravel()
    if arr.size == 0 or t.size == 0:
        raise Missing("empty array")
    _METHOD_LOG.append(("nearest-sample hold", native_step(t)))
    j = held_index(t)
    a = np.moveaxis(arr, axis, 0)
    out = np.full((N_FRAMES,) + a.shape[1:], np.nan)
    ok = j >= 0
    out[ok] = a[j[ok]]
    return np.moveaxis(out, 0, axis)


def sample_info(t):
    """For slow diagnostics in ELM mode: {'t_sample', 'measured'} describing the held data.
      t_sample  time [s] of the native sample each frame holds (NaN if none)
      measured  1 on the single frame nearest to each sample, 0 on its repeats; use it so that
                statistics / error bars do not count one sample many times.
    Empty dict outside ELM mode."""
    if MODE != "elm" or not CURRENT_HOLD:
        return {}
    t = np.asarray(t, dtype=float).ravel()
    if native_step(t) / WINDOW < 0.5:                      # averaged in bins: nothing is repeated
        return {}
    j = held_index(t)
    ts = np.where(j >= 0, t[np.clip(j, 0, t.size - 1)], np.nan)
    owner = np.rint((ts - T_FRAMES[0]) / DT)                       # frame nearest to the sample
    measured = (j >= 0) & (owner == np.arange(N_FRAMES)) & (np.abs(ts - T_FRAMES) <= DT / 2 + 1e-9)
    return {"t_sample": ts, "measured": measured.astype(np.uint8)}


def resample(arr, t, axis, method="auto"):
    """Resample `arr` along `axis` (native times `t`) onto T_FRAMES.

    method='auto' (default) picks the method from the native sampling step (default mode):
      window average  if the +-WINDOW/2 window holds >= MIN_SAMPLES_PER_WINDOW samples
                      (NaN-aware mean of the samples in [t - WINDOW/2, t + WINDOW/2));
      interpolation   otherwise (slow diagnostic, e.g. TS 10-25 ms, CXRS, DSS): a
                      window would often contain no sample at all, see interpolate().
    'mean' / 'interpolate' / 'hold' force one of them; 'slow' = hold in ELM mode, interpolate
    otherwise. In ELM mode, 'auto' on the slow grid means hold."""
    arr = np.asarray(arr, dtype=float)
    t = np.asarray(t, dtype=float).ravel()
    if arr.size == 0 or t.size == 0:
        raise Missing("empty array")
    arr, t = align(arr, t, axis)
    if arr.ndim == 0 or arr.shape[axis] != t.size:
        raise ValueError(f"array {arr.shape} has {arr.shape[axis] if arr.ndim else 'no'} values along axis {axis} "
                         f"but its time vector has {t.size} stamps (a trace on another time base?)")
    step = native_step(t)
    if method == "slow":
        method = "hold" if MODE == "elm" else "interpolate"
    if method == "auto" and MODE == "elm":
        # average when the diagnostic is much faster than the bins; otherwise keep real samples:
        # nearest-sample hold for the diagnostics in HOLD_DIAGNOSTICS (never mixes samples before
        # and after an ELM), linear interpolation for the smooth ones (heating, pressure)
        ratio = step / WINDOW
        method = "mean" if ratio < 0.5 else "hold" if CURRENT_HOLD else "interpolate"
    if method == "auto":
        method = "mean" if round(WINDOW / step, 6) >= MIN_SAMPLES_PER_WINDOW else "interpolate"
    if method == "hold":
        return hold(arr, t, axis)
    if method == "interpolate":
        return interpolate(arr, t, axis)
    _METHOD_LOG.append(("window average", step))
    arr = np.moveaxis(arr, axis, 0)
    out = np.full((N_FRAMES,) + arr.shape[1:], np.nan)
    for i, tf in enumerate(T_FRAMES):
        sel = (t >= tf - WINDOW / 2) & (t < tf + WINDOW / 2)
        if sel.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
                out[i] = np.nanmean(arr[sel], axis=0)
    return np.moveaxis(out, 0, axis)


def resample_slow(arr, t, axis):
    """Slow diagnostic: nearest-sample hold in ELM mode, linear interpolation otherwise."""
    return resample(arr, t, axis, method="slow")


def interpolate(arr, t, axis):
    """Linear interpolation of `arr` along `axis` onto T_FRAMES, for slow diagnostics.

    Each 1D series is interpolated using only its valid (non-NaN) samples.
    A frame is NaN if it lies outside the valid range, or if the two valid
    samples around it are more than 2 native sampling steps apart (no
    interpolation across long gaps of missing data). A frame time that coincides
    with a valid sample takes that sample. Vectorised over all series (chunked).
    """
    arr = np.asarray(arr, dtype=float)
    t = np.asarray(t, dtype=float).ravel()
    if arr.size == 0 or t.size == 0:
        raise Missing("empty array")
    step = native_step(t)
    _METHOD_LOG.append(("interpolation", step))
    max_gap = 2 * step
    a = np.moveaxis(arr, axis, 0)
    n_t = a.shape[0]
    a2 = a.reshape(n_t, -1)
    out2 = np.full((N_FRAMES, a2.shape[1]), np.nan)
    p = np.searchsorted(t, T_FRAMES)                       # first native sample at or after each frame
    left_i, right_i = np.clip(p - 1, 0, n_t - 1), np.clip(p, 0, n_t - 1)
    has_left, has_right = (p >= 1)[:, None], (p < n_t)[:, None]
    idx = np.arange(n_t)[:, None]
    tf = T_FRAMES[:, None]
    for s0 in range(0, a2.shape[1], 2048):
        y = a2[:, s0:s0 + 2048]
        valid = ~np.isnan(y)
        last = np.maximum.accumulate(np.where(valid, idx, -1), axis=0)            # last valid index <= i
        nxt = np.minimum.accumulate(np.where(valid, idx, n_t)[::-1], axis=0)[::-1]  # next valid index >= i
        L = np.where(has_left, last[left_i], -1)           # nearest valid sample before the frame
        R = np.where(has_right, nxt[right_i], n_t)         # nearest valid sample at/after the frame
        Lc, Rc = np.clip(L, 0, n_t - 1), np.clip(R, 0, n_t - 1)
        tL, tR = t[Lc], t[Rc]
        yL, yR = np.take_along_axis(y, Lc, 0), np.take_along_axis(y, Rc, 0)
        gap = tR - tL
        with np.errstate(invalid="ignore", divide="ignore"):
            val = yL + np.where(gap > 0, (tf - tL) / gap, 0.0) * (yR - yL)
        good = (L >= 0) & (R < n_t) & (gap <= max_gap)
        exact = (R < n_t) & (tR == tf)                      # frame time = a valid sample
        out2[:, s0:s0 + 2048] = np.where(exact, yR, np.where(good, val, np.nan))
    out = out2.reshape((N_FRAMES,) + a.shape[1:])
    return np.moveaxis(out, 0, axis)


def crop(arr, t, axis, eps=1e-6):
    """Keep the native samples with T_START <= t <= T_END (no averaging).
    Returns (cropped array, cropped time)."""
    arr, t = np.asarray(arr, dtype=float), np.asarray(t, dtype=float).ravel()
    k = (t >= T_START - eps) & (t <= T_END + eps)
    if not k.any():
        raise Missing("no samples in the time window")
    return np.take(arr, np.flatnonzero(k), axis=axis), t[k]


def time_for(obj, n, names):
    """The time vector among obj.<names> that has exactly n samples (first match, in the order
    given); if none matches, the first one present, so that resample() reports the mismatch."""
    for k in names:
        if hasattr(obj, k) and np.size(getattr(obj, k)) == n:
            return getattr(obj, k)
    for k in names:
        if hasattr(obj, k):
            return getattr(obj, k)
    raise Missing(f"none of the time vectors {names} exists")


def rs_as(obj, mapping, t, axis):
    """Like rs() but renames: mapping = {output_name: source_field}."""
    return {out: resample(getattr(obj, src), t, axis) for out, src in mapping.items()}


def rs(obj, names, t, axis):
    """Resample several fields of a MATLAB struct that share one time axis."""
    return {n: resample(getattr(obj, n), t, axis) for n in names}


def get(obj, name):
    """Struct field, or Missing if the field/struct is absent."""
    if obj is None or not hasattr(obj, name):
        raise Missing(f"no field '{name}'")
    return getattr(obj, name)


def need_available(obj, label):
    if obj is None or getattr(obj, "available", 1) == 0:
        raise Missing(f"{label} not available")
    return obj


# --------------------------------------------------------------------------
# One function per diagnostic: (matfile dict) -> dict
# --------------------------------------------------------------------------
def bolo_geometry(b):
    """Geometry of ALL channels in BOLO.geometry.channel (IDS-style structs), ordered
    by identifier. Returns (identifiers, dict) or (None, {}) if absent."""
    g = getattr(b, "geometry", None)
    ch = getattr(g, "channel", None)
    if ch is None:
        return None, {}
    ident = np.array([c.identifier for c in ch]).astype(int)
    o = np.argsort(ident)
    ch, ident = [ch[i] for i in o], ident[o]

    def arr(f):
        return np.array([f(c) for c in ch], float)
    part = lambda name: {
        "r": arr(lambda c: getattr(c, name).centre.r),
        "z": arr(lambda c: getattr(c, name).centre.z),
        "width_x1": arr(lambda c: getattr(c, name).x1_width),
        "width_x2": arr(lambda c: getattr(c, name).x2_width),
        "surface": arr(lambda c: getattr(c, name).surface)}
    return ident, {
        "channel_name": [str(c.name) for c in ch],
        "detector": part("detector"), "aperture": part("aperture"),
        "etendue": arr(lambda c: c.etendue), "etendue_units": "m^2.sr",
        "line_of_sight": {
            "first_point": {"r": arr(lambda c: c.line_of_sight.first_point.r),
                            "z": arr(lambda c: c.line_of_sight.first_point.z)},
            "second_point": {"r": arr(lambda c: c.line_of_sight.second_point.r),
                             "z": arr(lambda c: c.line_of_sight.second_point.z)}}}


def d_bolo(m):
    b = need_available(get(get(m, "EdgeData"), "BOLO"), "BOLO")
    # the bolometer traces are not always on one time base (t, t_LOS, t_tot, t_emi may differ in
    # length in some shots): each array takes the time vector that has its length
    sig = resample(b.signals, time_for(b, np.shape(b.signals)[0], ("t", "t_LOS", "t_tot", "t_emi")), 0)
    ident, geometry = bolo_geometry(b)
    if ident is None:                                      # no geometry: measured channels only
        chn, channels = np.asarray(b.chn), sig
    else:                                                  # all channels, NaN where not measured
        chn = ident
        channels = np.full((N_FRAMES, ident.size), np.nan)
        pos = {n: i for i, n in enumerate(ident)}
        miss = [int(n) for n in np.asarray(b.chn) if int(n) not in pos]
        if miss:                                           # geometry does not cover every measured channel
            print(f"  WARNING: bolometer channels {miss} have no geometry: storing the measured channels "
                  f"without geometry")
            chn, channels, geometry = np.asarray(b.chn), sig, {}
        else:
            channels[:, [pos[int(n)] for n in np.asarray(b.chn)]] = sig
    return {
        "t": T_FRAMES,
        "channels": channels,                             # (t, channel); NaN = no data
        "channels_units": str(b.dim),
        "chn": chn,
        "emi": resample(b.emi, time_for(b, np.shape(b.emi)[2], ("t_emi", "t", "t_LOS", "t_tot")), 2),   # (R, Z, t)
        "emi_units": "W/m3",                              # assumed
        "r": b.R, "z": b.Z,
        **{out: resample(getattr(b, src), time_for(b, np.size(getattr(b, src)), ("t", "t_tot", "t_LOS", "t_emi")), 0)
           for out, src in {"power_core": "p_cor", "power_div": "p_div", "power_xpt": "p_xpt"}.items()},
        "power_tot": resample(b.p_tot, time_for(b, np.size(b.p_tot), ("t_tot", "t", "t_LOS", "t_emi")), 0),
        "power_units": "W",                               # assumed
        **geometry,
    }


FIR_Z_TOP, FIR_Z_BOTTOM = 0.8, -0.8      # [m] vertical extent of the FIR chords (user-supplied)


def d_fir(m):
    fd = get(m, "FIRdata")
    ok = np.asarray(get(fd, "ok_ch")).astype(bool)        # bad channels -> NaN

    def fir(sub):
        return {"data": resample(np.where(ok, sub.data, np.nan), sub.time, 0),
                "error": resample(np.where(ok, sub.error, np.nan), sub.time, 0)}
    r = np.asarray(fd.R, float).ravel()
    n = r.size
    top, bottom = np.full(n, FIR_Z_TOP), np.full(n, FIR_Z_BOTTOM)
    return {"t": T_FRAMES, "r": r, "ch_ok": ok.astype(np.uint8),
            # 14 vertical chords at radius r, from z = FIR_Z_TOP down to FIR_Z_BOTTOM
            "line_of_sight": {"first_point": {"r": r, "z": top},
                              "second_point": {"r": r, "z": bottom}},
            "av_dens": fir(get(fd, "av_dens")),           # (t, channel)
            "int_dens": fir(get(fd, "int_dens"))}


KB = 1.380649e-23      # J/K
T_GAS = 273.15         # K, ASSUMED gas temperature for particles/s -> Pa.m3/s


def d_valve(m):
    """IDS layout: gas_injection/valve/<name> (exit position included)."""
    valve = need_available(get(get(m, "EdgeData"), "VALVE"), "VALVE")
    out = {"valve": {}}
    for k in [n for n in valve._fieldnames if n != "available"]:
        v = getattr(valve, k)
        try:
            flow = resample(v.flow, v.t, 0)                # particles/s (molecules assumed)
        except (Missing, ValueError, AttributeError) as e:  # one bad valve must not hide the others
            out["valve"][k] = Missing_group(str(e))
            print(f"  gas_injection/valve/{k}: unavailable ({e})")
            continue
        out["valve"][k] = {
            "t": T_FRAMES,
            "species_name": str(v.gas),
            "exit_position": {"r": float(v.R), "z": float(v.Z)},
            "flow_rate": flow * KB * T_GAS,                # Pa.m^3.s^-1
            "flow_rate_units": "Pa.m^3.s^-1",
            "flow_particles": flow,                        # original TCV quantity
            "flow_particles_units": str(v.units)}
    if all(isinstance(v, Missing_group) for v in out["valve"].values()):      # (Missing_group is a dict too)
        raise Missing("no valves" if not out["valve"] else
                      "no valve has data: " + "; ".join(f"{k}: {v.reason}" for k, v in out["valve"].items()))
    return out


def d_ts(m):
    s = need_available(get(get(m, "EdgeData"), "TS"), "TS")
    return {"t": T_FRAMES, "r": s.R, "z": s.Z,            # fields: (t, point) or (t,)
            **rs(s, ["te", "dte", "ne", "dne", "rho",
                     "ne_u", "dne_u", "te_u", "dte_u"], s.t, 0),
            "distance_separatrix_midplane": resample(s.dru, s.t, 0),
            # line-averaged density computed from TS; same name as the FIR quantity
            "av_dens": resample(s.nel, s.t, 0),
            **sample_info(s.t)}


def d_lp(m):
    """embedded = all probes; inner_target / outer_target = subsets of them.

    NOT resampled: every native sample (10 ms) between T_START and T_END is kept,
    on the group's own time vector `t`, so arrays are (probe, t_native)."""
    L = get(m, "LPdata")
    probes = np.asarray(get(L, "probes")).astype(int)
    fit = get(L, "FourParamsFit")
    pos = np.asarray(L.pos)
    t_lp = crop(L.time, L.time, 0)[1]

    fast = MODE == "elm" and hasattr(L, "Jsat2") and hasattr(L, "time2")

    def subset(sel):
        d = {"probes": probes[sel],
             "position": {"r": pos[sel, 0], "z": pos[sel, 1]}}
        if fast:                       # 1 ms ion saturation current density, on the fast grid
            d["fast"] = {"t": T_FRAMES,
                         "j_i_saturation": resample(np.asarray(L.Jsat2)[sel], L.time2, 1),
                         "j_i_saturation_sigma": resample(np.asarray(L.Jsat2_std)[sel], L.time2, 1)}
        for out_n, arr in [("b_field_angle", L.ang),
                           ("distance_separatrix_midplane", L.dsep_mid),
                           ("t_e", fit.Te), ("n_e", fit.dens),
                           ("v_floating", fit.Vfl), ("j_i_saturation", fit.Jsat),
                           ("rho", L.rho_psi), ("Pperp", fit.Pperp)]:   # last two: no IDS field
            d[out_n] = crop(np.asarray(arr)[sel], L.time, 1)[0]         # (probe, t)
        return d

    return {"t": t_lp, "time_resampling": "none (native samples)",
            "embedded": subset(np.ones(probes.shape, bool)),
            "inner_target": subset((probes >= 136) & (probes <= 190)),
            "outer_target": subset((probes >= 1) & (probes <= 42))}


# Port each IR camera looks through: which poloidal axis the port *thickness* runs along.
# HIR sits in the outer wall (thickness along R, width along Z); VIR in the top of the
# vessel (thickness along Z, width along R).
IR_PORT_THICKNESS_AXIS = {"HIR": "r", "VIR": "z"}


def ir_cam_geometry(IR, cam):
    """EdgeData.IR.geometry.<cam> (or .geom), or None."""
    geom = next((getattr(IR, n) for n in ("geometry", "geom") if hasattr(IR, n)), None)
    return next((getattr(geom, n) for n in (cam, cam.lower()) if hasattr(geom, n)), None)


def ir_detector(IR, cam):
    """Detector position {r, z} [m] from geometry.<cam>.Rd / .Zd, or None if absent."""
    g = ir_cam_geometry(IR, cam)
    try:
        rd, zd = (float(np.squeeze(getattr(g, f))) for f in ("Rd", "Zd"))
    except (AttributeError, TypeError, ValueError):
        return None
    return {"r": np.array([rd]), "z": np.array([zd]), "units": "m"}


def ir_port(IR, cam):
    """Port geometry of an IR camera from EdgeData.IR.geometry.<cam> (fields Rp, Zp = centre
    of the port, Dp = width, Lp = thickness, all in m). Returns None if absent/incomplete.

    Output (consistent with the other geometry: arrays, `outline` as for the gauge ports):
      port/{r, z}            centre            port/{width, thickness}
      port/outline/{r, z}    the 4 corners of the port rectangle (closed contour)."""
    g = ir_cam_geometry(IR, cam)
    try:
        rp, zp, dp, lp = (float(np.squeeze(getattr(g, f))) for f in ("Rp", "Zp", "Dp", "Lp"))
    except (AttributeError, TypeError, ValueError):
        return None
    axis = IR_PORT_THICKNESS_AXIS[cam]
    hr, hz = (lp / 2, dp / 2) if axis == "r" else (dp / 2, lp / 2)      # half sizes along r, z
    return {"r": np.array([rp]), "z": np.array([zp]),
            "width": np.array([dp]), "thickness": np.array([lp]), "units": "m",
            "thickness_axis": axis,
            "outline": {"r": np.array([rp - hr, rp + hr, rp + hr, rp - hr]),
                        "z": np.array([zp - hz, zp - hz, zp + hz, zp + hz])}}


def d_ir(m):
    IR = get(get(m, "EdgeData"), "IR")
    out = {}
    for k, name in (("HIR", "hir"), ("VIR", "vir")):
        try:
            i = need_available(get(IR, name), name)
            out[k] = {"t": T_FRAMES, "r": i.r, "z": i.z,
                      "power": resample(i.power, i.t, 0),
                      "qperp": resample(i.qperp, i.t, 1),   # (position, t)
                      "distance_separatrix_midplane": resample(i.dru, i.t, 1),
                      **sample_info(i.t)}
            port, det = ir_port(IR, k), ir_detector(IR, k)
            if port:
                out[k]["port"] = port
            else:
                print(f"  camera_ir/{k}: no port geometry (EdgeData.IR.geometry.{k}: Rp, Zp, Dp, Lp)")
            if det:
                out[k]["detector"] = det                  # as bolometer/detector: r, z
            else:
                print(f"  camera_ir/{k}: no detector position (EdgeData.IR.geometry.{k}: Rd, Zd)")
        except Missing as e:
            out[k] = Missing_group(str(e))
    return out


PRESSURE_UNITS = "Pa"      # as stated by the user (magnitudes ~1e-3 look more like mbar)


def gauge_positions(obj):
    """geometry.r_<name>/z_<name> point sets -> {name: {r, z}} (empty if absent)."""
    g = getattr(obj, "geometry", None)
    out = {}
    for f in getattr(g, "_fieldnames", []):
        if f.startswith("r_") and hasattr(g, "z_" + f[2:]):
            out[f[2:]] = {"r": np.asarray(getattr(g, f), float).ravel(),
                          "z": np.asarray(getattr(g, "z_" + f[2:]), float).ravel()}
    return out


def d_bar(m):
    b = need_available(get(get(m, "EdgeData"), "BAR"), "BAR")
    pos = gauge_positions(b)
    return {"t": T_FRAMES, "pressure_units": PRESSURE_UNITS,
            **({"position": pos} if pos else {}),
            **rs_as(b, {"pressure_mid": "p_mid", "pressure_div": "p_div",
                        "pressure_tmp": "p_tmp", "pressure_45d": "p_45d"}, b.t, 0)}


def d_apg(m):
    a = need_available(get(get(m, "EdgeData"), "APG"), "APG")
    pos = gauge_positions(a)
    return {"t": T_FRAMES, "pressure_units": PRESSURE_UNITS,
            **({"position": pos} if pos else {}),
            **rs_as(a, {"pressure_bot": "p_bot", "pressure_div": "p_div",
                        "pressure_mic": "p_mic", "pressure_top": "p_top"}, a.t, 0)}


def d_mantis(m):
    mt = need_available(get(get(m, "EdgeData"), "MANTIS"), "MANTIS")
    out = {}
    for k in [n for n in mt._fieldnames if n != "available"]:
        c = getattr(mt, k)
        g = c.grid
        out[k] = {"t": T_FRAMES,
                  "emi": resample(c.emi, c.t, 1),         # (triangle, t)
                  # triangle j has vertices (r_values[j,:], z_values[j,:]);
                  # MATLAB: patch(r_values', z_values', emi(:,it)')
                  "grid": {"r_values": g.R_values, "z_values": g.Z_values,
                           "r_mean": g.R_mean, "z_mean": g.Z_mean, "area": g.Area},
                  # The CIII front (c.sFront*, c.l*) is NOT stored: its definition involves analysis choices
                  # and the MANTIS routine gave wrong values. It is computed from `emi` and the equilibrium
                  # by pexdb.ciii_front.
                  **sample_info(c.t)}
    if not out:
        raise Missing("no cameras")
    return out


def lower_xpoint_series(q):
    """Lower X-point (R, Z) at every LIUQE time: of the X-points of a time slice (LIUQE gives up to two, with
    0 for a missing one) the one with the smallest Z that lies below the magnetic axis; NaN where there is
    none."""
    r, z = np.atleast_2d(np.asarray(q.r_xpts, float)), np.atleast_2d(np.asarray(q.z_xpts, float))
    nt = np.size(q.t)
    if r.shape[1] != nt and r.shape[0] == nt:                 # (t, 2) -> (2, t)
        r, z = r.T, z.T
    zax = np.asarray(q.z_axis, float).ravel()
    ok = np.isfinite(r) & np.isfinite(z) & (r > 0) & (z < zax[None, :])
    zz = np.where(ok, z, np.inf)
    i = np.argmin(zz, axis=0)
    cols = np.arange(nt)
    has = np.isfinite(zz[i, cols])
    return np.where(has, r[i, cols], np.nan), np.where(has, z[i, cols], np.nan)


def d_liuqe(m):
    q = need_available(get(get(m, "EdgeData"), "LIUQE"), "LIUQE")
    rx, zx = lower_xpoint_series(q)
    return {"t": T_FRAMES, "r": q.R, "z": q.Z,
            # lower X-point [m] (NaN if none), used by pexdb.ciii_front
            "r_xpt": resample(rx, q.t, 0), "z_xpt": resample(zx, q.t, 0),
            "psi": resample(q.psi, q.t, 2),               # (R, Z, t)
            "ip": resample(q.i_p, q.t, 0), "ip_units": "A",
            "b0": resample(q.b_zero, q.t, 0), "b0_units": "T",   # vacuum toroidal field
            "rho": resample(q.rho, q.t, 2),
            **rs(q, ["r_axis", "z_axis", "psi_axis"], q.t, 0),
            "r_contour": resample(q.r_contour, q.t, 1),   # LCFS, (point, t)
            "z_contour": resample(q.z_contour, q.t, 1)}


def d_cxrs(m):
    # slow diagnostic (24 ms sampling): interpolation (default mode) or nearest-sample hold (ELM mode)
    cd = get(m, "CXRSdata")
    C = get(cd, "calc")
    out = {"t": T_FRAMES, "r": C.R, "z": C.Z,
           "rho": resample_slow(C.rho, C.time, 1)}
    for q_ in ("ni", "Ti", "vi"):                          # (channel, t)
        out[q_] = {"data": resample_slow(getattr(C, q_), C.time, 1),
                   "error": resample_slow(getattr(C, "err_" + q_), C.time, 1),
                   "units": str(getattr(C, q_ + "_units"))}
    out.update(sample_info(C.time))
    return out


def d_wall(m):
    """Wall contours from the `wall` struct, written as wall/<name>/{r,z}.

    Recognised layouts inside `wall`:
      R_<name>, Z_<name>   vectors            e.g. R_tiles, Z_tiles   -> 'tiles'
      RZv_<name>           (2, N) = [R; Z]    e.g. RZv_ivessel        -> 'vessel_inner'
                                                   RZv_ovessel        -> 'vessel_outer'
      <name>.R, <name>.Z   sub-struct (also SHOTINFO.wall.R/Z -> 'vessel')
    """
    rename = {"ivessel": "vessel_inner", "ovessel": "vessel_outer"}
    parts = {}
    w = getattr(m, "wall", None)
    fields = getattr(w, "_fieldnames", [])
    vec = lambda x: np.asarray(x, float).ravel()
    for k in fields:
        v = getattr(w, k)
        if k.startswith("R_") and ("Z_" + k[2:]) in fields:
            parts[rename.get(k[2:], k[2:])] = {"r": vec(v), "z": vec(getattr(w, "Z_" + k[2:]))}
        elif k.startswith("RZv_") and np.ndim(v) == 2 and 2 in np.shape(v):
            a = np.asarray(v, float)
            a = a if a.shape[0] == 2 else a.T
            parts[rename.get(k[4:], k[4:])] = {"r": a[0], "z": a[1]}
        elif hasattr(v, "R") and hasattr(v, "Z"):
            parts[k] = {"r": vec(v.R), "z": vec(v.Z)}
    si = getattr(getattr(m, "SHOTINFO", None), "wall", None)
    if not parts and si is not None and hasattr(si, "R"):
        parts["vessel"] = {"r": vec(si.R), "z": vec(si.Z)}
    if not parts:
        raise Missing("no wall coordinates in struct 'wall' (MATLAB graphics handles hold no data)")
    return parts


REGIME_OVERRIDE = None     # 'L' / 'H', set by --regime=L on the command line
AUX_MIN_POWER = 5e3        # [W] a source is named in the description only above this mean power


# --------------------------------------------------------------------------
# ELM detection: is this shot an ELMy H-mode? (decided from the data, no input needed)
# --------------------------------------------------------------------------
DETECT_WINDOW = (0.6, 1.6)     # [s] analysed (clipped to what each signal covers, >= MIN_COVERAGE)
MIN_COVERAGE = 0.1             # [s]
ELM_K = 5.0                    # spike threshold, in robust sigmas of the high-passed signal
ELM_MIN_SEP = 3e-3             # [s] minimum separation between two ELMs
ELM_FREQ_RANGE = (20.0, 600.0) # [Hz] plausible ELM frequencies
ELM_MIN_COUNT = 8              # at least this many spikes
ELM_MAX_CV = 0.6               # max coefficient of variation of the intervals (a spike train is
                               # fairly regular; random intermittent turbulence has CV ~ 1)
ELM_STRONG = (15, 0.4)         # (min spikes, max cv): a train this clean is convincing on its own
ELM_COINCIDENCE = 3e-3         # [s] spikes of two signals closer than this are the same event
ELM_MIN_COINCIDENCE = 0.4      # fraction of spikes that must coincide between two signals


def _running_median(y, size):
    """Centred running median (odd `size`), edges padded with the nearest value. Plain numpy, so
    the builder does not depend on scipy.ndimage (fragile on some servers)."""
    h = size // 2
    pad = np.pad(y, h, mode="edge")
    win = np.lib.stride_tricks.as_strided(pad, shape=(y.size, size), strides=(pad.strides[0],) * 2)
    return np.median(win, axis=1)


def _find_peaks(x, height, distance):
    """Indices of the local maxima of x with x >= height, keeping the tallest of any group closer
    than `distance` samples (same rule as scipy.signal.find_peaks(x, height=, distance=); a flat top
    is reported at its first sample). Plain numpy: scipy.signal pulls in scipy.stats / integrate,
    which break on servers with an old system numpy."""
    x = np.asarray(x, float)
    if x.size < 3:
        return np.array([], int)
    pk = np.flatnonzero((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:])) + 1
    pk = pk[x[pk] >= height]
    keep = np.ones(pk.size, bool)
    for i in np.argsort(-x[pk], kind="stable"):                   # tallest first
        if not keep[i]:
            continue
        k = i - 1
        while k >= 0 and pk[i] - pk[k] < distance:
            keep[k] = False
            k -= 1
        k = i + 1
        while k < pk.size and pk[k] - pk[i] < distance:
            keep[k] = False
            k += 1
    return pk[keep]


def spike_train(t, y, sign=1.0, window=None):
    """ELM-like spikes in y(t) within `window` (default DETECT_WINDOW; sign=-1 for negative spikes).

    The signal is put on a uniform 1 ms grid (point-sampled if it is slower than that, maximum of
    each 1 ms bin if it is faster, so that narrow spikes survive), a running median (25 ms) is
    subtracted and spikes above ELM_K sigmas are picked (sigma: robust noise level of the lower half
    of the residuals). Returns a dict with the spike times, their number, frequency (from the median
    interval), regularity (cv) and 'ok' (True if it looks like an ELM train), or a dict with 'reason'
    if the signal cannot be used."""
    y, t = align(np.asarray(y, float).ravel(), np.asarray(t, float).ravel())
    good = np.isfinite(t) & np.isfinite(y)
    t, y = t[good], y[good]
    if t.size < 50:
        return {"ok": False, "reason": "too few samples"}
    w0, w1 = DETECT_WINDOW if window is None else window
    w0, w1 = max(w0, t[0]), min(w1, t[-1])                   # what the signal covers
    if w1 - w0 < MIN_COVERAGE:
        return {"ok": False, "reason": "signal does not cover the window"}
    step = np.median(np.diff(t))
    if step > 2e-3:
        return {"ok": False, "reason": f"too slow ({step * 1e3:.1f} ms) to resolve ELMs"}
    dt = 1e-3
    tu = np.arange(w0 - 0.05, w1 + 0.05, dt)                 # margin for the running median
    if step < 0.9e-3:                                        # faster than 1 ms: keep the spike maxima
        k = np.clip(np.floor((t - tu[0]) / dt).astype(int), -1, tu.size)
        ys = sign * y
        yu = np.full(tu.size, np.nan)
        ok = (k >= 0) & (k < tu.size)
        np.fmax.at(yu, k[ok], ys[ok])
        yu = np.interp(tu, tu[np.isfinite(yu)], yu[np.isfinite(yu)])
    else:
        yu = sign * np.interp(tu, t, y)
    r = yu - _running_median(yu, 25)
    # noise level from the lower half of the residuals only: ELM spikes are positive, so a high
    # ELM duty cycle cannot inflate it (a plain MAD would, and hide the spikes)
    low = r[r < np.median(r)]
    sigma = 1.4826 * np.median(np.abs(low - np.median(r))) if low.size > 10 else 0.0
    if not sigma > 0:
        return {"ok": False, "reason": "no fluctuations"}
    pk = _find_peaks(r, ELM_K * sigma, max(1, int(round(ELM_MIN_SEP / dt))))
    times = tu[pk]
    times = times[(times >= w0) & (times <= w1)]
    n = times.size
    out = {"ok": False, "times": times, "n": n, "frequency": n / (w1 - w0), "cv": np.nan}
    if n >= 3:
        iv = np.diff(times)
        out["cv"] = float(np.std(iv) / np.mean(iv))
        out["frequency"] = float(1.0 / np.median(iv))        # ELMs may only fill part of the window
    out["ok"] = bool(n >= ELM_MIN_COUNT and ELM_FREQ_RANGE[0] <= out["frequency"] <= ELM_FREQ_RANGE[1]
                     and out["cv"] <= ELM_MAX_CV)
    if not out["ok"]:
        out["reason"] = f"{n} spikes ({out['frequency']:.0f} Hz), cv {out['cv']:.2f}: not an ELM train"
    return out


def _elm_signals(m):
    """(name, t, y, sign, divertor) of the signals that can show ELMs, for those present in `m`.
    Divertor signals are preferred: core signals (dW/dt) also respond to sawteeth."""
    sigs = []
    try:                                                     # fast gauges near the target / midplane
        A = m.EdgeData.APG
        for name, field in (("APG divertor pressure", "p_div"), ("APG midplane pressure", "p_mic")):
            if hasattr(A, field):
                sigs.append((name, np.asarray(A.t, float), np.asarray(getattr(A, field), float), 1.0, True))
    except Exception:
        pass
    try:                                                     # IR cameras: power on the target (fast in some shots)
        for cam in ("hir", "vir"):
            c = getattr(m.EdgeData.IR, cam)
            if getattr(c, "available", 1) != 0 and hasattr(c, "power"):
                sigs.append((f"IR {cam.upper()} power", np.asarray(c.t, float), np.asarray(c.power, float), 1.0, True))
    except Exception:
        pass
    L = getattr(m, "LPdata", None)
    try:                                                     # outer-target ion saturation current
        probes = np.asarray(L.probes).astype(int)
        t = np.asarray(L.time2, float)
        k = (t >= DETECT_WINDOW[0]) & (t <= DETECT_WINDOW[1])
        idx = np.flatnonzero((probes >= 1) & (probes <= 42))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)                # all-NaN probes
            med = np.array([np.nanmedian(np.asarray(L.Jsat2)[i][k]) for i in idx])
            top = idx[np.argsort(-np.nan_to_num(med, nan=-np.inf))[:3]]    # the 3 probes nearest the strike point
            sigs.append(("Jsat outer target", t, np.nanmean(np.asarray(L.Jsat2)[top], axis=0), 1.0, True))
    except Exception:
        pass
    try:                                                     # divertor radiated power
        b = m.EdgeData.BOLO
        sigs.append(("bolometer divertor power", np.asarray(b.t, float), np.asarray(b.p_div, float), 1.0, True))
    except Exception:
        pass
    try:                                                     # stored-energy loss, dW_MHD/dt (core)
        h = m.heating.HEATING
        if np.size(h.t_dMHD) == np.size(h.dEMHD):            # a mismatched pair means a wrong time vector
            sigs.append(("dEMHD", np.asarray(h.t_dMHD, float), np.asarray(h.dEMHD, float), -1.0, False))
    except Exception:
        pass
    return sigs


ELM_ENERGY_TSTAT = -5.0        # mean W_MHD drop across the ELMs must be this many standard errors below 0


def elm_energy_drop(times, m):
    """Independent physical check that spikes at `times` are ELMs: the stored energy W_MHD drops
    across an ELM. Mean of (W after - W before) over the events, its t-statistic, the fraction of
    events with a drop, and the mean drop relative to W. None if W_MHD (heating.HEATING.EMHD with
    a consistent t_MHD) is not available. Random times give t ~ 0 +- 1 (min about -4)."""
    try:
        # the whole-shot 1 ms W_MHD if it was replaced by the (shorter) LIUQE trace: more ELMs, so a
        # stronger test, independent of the fetch window
        tw, w = getattr(m, "_mhd_native", None) or (m.heating.HEATING.t_MHD, m.heating.HEATING.EMHD)
        tw, w = np.asarray(tw, float).ravel(), np.asarray(w, float).ravel()
        if tw.size != w.size or tw.size < 50 or len(times) < 5:
            return None
    except Exception:
        return None
    d = []
    for tk in np.asarray(times, float):
        a = (tw >= tk - 4e-3) & (tw <= tk - 0.5e-3)
        b = (tw >= tk + 0.5e-3) & (tw <= tk + 3e-3)
        if a.sum() >= 2 and b.sum() >= 2:
            d.append(np.nanmean(w[b]) - np.nanmean(w[a]))
    d = np.array(d)
    if d.size < 5 or not np.std(d, ddof=1) > 0:
        return None
    k = (tw >= np.min(times)) & (tw <= np.max(times))
    return {"n": int(d.size), "mean_drop": float(d.mean()),
            "tstat": float(d.mean() / (d.std(ddof=1) / np.sqrt(d.size))),
            "fraction_down": float(np.mean(d < 0)),
            "relative": float(d.mean() / np.nanmean(w[k])) if k.any() else np.nan}


def detect_elms(m):
    """Is this an ELMy H-mode? Looks for a regular train of spikes (ELMs, 20-600 Hz) in divertor
    signals (fast pressure gauges, IR power, ion saturation current, radiated power) and in dW_MHD/dt, within
    DETECT_WINDOW.

    ELMy if at least two signals show an ELM train and their spikes coincide (ELM_MIN_COINCIDENCE),
    or if a single divertor signal shows a very clean train (ELM_STRONG), or if it is the only usable
    signal. Heating power is not used, so high-power L-modes are not mistaken for H-modes.
    Returns {'elmy', 'frequency' [Hz], 'times' [s] (of the best signal), 'signals': {name: result},
    'n_available', 'n_flagged', 'summary'}; cached on `m`."""
    if hasattr(m, "_elm_detection"):
        return m._elm_detection
    sigs = _elm_signals(m)
    res = {}
    for name, t, y, sg, _ in sigs:
        try:
            res[name] = spike_train(t, y, sg)
        except Exception as e:                               # a bad trace must not stop the build
            res[name] = {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    divertor = {name: dv for name, _, _, _, dv in sigs}
    avail = {k: v for k, v in res.items() if "times" in v}          # usable (even if no ELMs)
    flagged = {k: v for k, v in avail.items() if v["ok"]}
    elmy, best, energy = False, None, None
    kinds = {k: ("pressure" if k.startswith("APG") else "ir" if k.startswith("IR") else
                 "jsat" if k.startswith("Jsat") else "bolometer" if k.startswith("bolo") else "energy")
             for k in res}
    if flagged:
        best = max(flagged, key=lambda k: flagged[k]["n"])
        energy = elm_energy_drop(flagged[best]["times"], m)
        confirmed = energy is not None and energy["tstat"] <= ELM_ENERGY_TSTAT
        coincide = False
        if len(flagged) >= 2:
            others = [k for k in flagged if k != best and kinds[k] != kinds[best]]
            if others:
                tb = flagged[best]["times"]
                co = max(np.mean([np.min(np.abs(flagged[k]["times"] - x)) <= ELM_COINCIDENCE
                                  for x in tb]) for k in others)
                flagged[best]["coincidence"] = float(co)
                coincide = co >= ELM_MIN_COINCIDENCE
        # evidence: two different kinds of signal agreeing, or one train confirmed by the energy loss;
        # without any energy data a single very clean divertor train (or the only usable signal) is
        # accepted; a train whose events show NO energy loss is not taken for ELMs
        if energy is None:
            elmy = coincide or len(avail) == 1 or any(
                divertor[k] and v["n"] >= ELM_STRONG[0] and v["cv"] <= ELM_STRONG[1] for k, v in flagged.items())
        else:
            elmy = confirmed or (coincide and energy["tstat"] <= -2.0)
    parts = [f"{k}: {v['n']} spikes, {v['frequency']:.0f} Hz, cv {v['cv']:.2f}" if v.get("ok")
             else f"{k}: no ({v.get('reason', '?')})" for k, v in res.items()]
    if energy is not None:
        parts.append(f"energy: W_MHD drops {-1e-3 * energy['mean_drop']:.2f} kJ per ELM "
                     f"({-100 * energy['relative']:.1f}%), t={energy['tstat']:.1f}")
    out = {"elmy": bool(elmy), "n_available": len(avail), "n_flagged": len(flagged), "energy": energy,
           "signals": res, "summary": "; ".join(parts) or "no usable fast signal",
           "frequency": float(np.median([flagged[k]["frequency"] for k in flagged])) if elmy else np.nan,
           "times": flagged[best]["times"] if (elmy and best) else np.array([])}
    try:
        m._elm_detection = out
    except Exception:
        pass
    return out


def heating_means(m):
    """Mean power [W] of the ohmic, ECRH and NBI raw traces over the analysis window."""
    out = {}
    h = getattr(getattr(m, "heating", None), "HEATING", None)
    for src, t, p in (("ohmic", "t_OHM", "pOHM"), ("ecrh", "t_ECH", "pECH"), ("nbi", "t_NBI", "pNBI")):
        try:
            out[src] = mean_power(getattr(h, p), getattr(h, t))
        except (Missing, AttributeError):
            out[src] = 0.0
    return out


def shot_date(m):
    """(date 'YYYY-MM-DD', source text): the shot date is not stored in the file, so use the
    earliest analysis timestamp found (TS data time, CXRS analysis time in its comment)."""
    import datetime as dt
    import re
    found = []
    ts = getattr(getattr(getattr(m, "EdgeData", None), "TS", None), "date", None)
    if isinstance(ts, str) and ts.strip():
        for fmt in ("%d-%b-%Y %H:%M:%S.%f", "%d-%b-%Y %H:%M:%S"):
            try:
                found.append(("TS", dt.datetime.strptime(ts.strip(), fmt)))
                break
            except ValueError:
                pass
    cm = getattr(getattr(getattr(m, "CXRSdata", None), "calc", None), "comment", None)
    mt = re.search(r"\((\d{1,2}-[A-Za-z]{3}-\d{4} \d{2}:\d{2}:\d{2})\)", cm) if isinstance(cm, str) else None
    if mt:
        found.append(("CXRS", dt.datetime.strptime(mt.group(1), "%d-%b-%Y %H:%M:%S")))
    if not found:
        return None, None
    src, first = min(found, key=lambda x: x[1])
    return first.date().isoformat(), (f"earliest analysis timestamp ({src} {first:%Y-%m-%d %H:%M}); "
                                      "proxy for the shot date, which is not stored")


def shot_info(m):
    """Shot-level information written as root attributes of the HDF5 file.

      t_end        EdgeData.tend [s]
      baffle       wall.baffle ('SILO'/'NINO'); SHOTINFO.baffle overrides
      regime       'L' or 'H' (see below);       description  e.g. 'Ohmic L-mode'
      date         from analysis timestamps (see shot_date)
      heating_mean_{ohmic,ecrh,nbi}  mean powers [W] over the analysis window

    Regime (priority: --regime on the command line > SHOTINFO > ELM detection > heating power):
    ELM detection (detect_elms) looks for a train of ELMs in divertor signals: ELMy -> 'H'
    ('ELMy H-mode (NBI)', ...), none -> 'L' ('Ohmic L-mode', or 'heated L-mode (no ELMs)' when the
    auxiliary power exceeds the ohmic power). Only if no usable fast signal exists, the heating
    power decides: auxiliary (ECRH + NBI) below ohmic = L-mode, else H-mode.
    Also stored: elmy, elm_frequency [Hz], elm_detection (text report).
    """
    info = {}
    t_end = getattr(getattr(m, "EdgeData", None), "tend", None)
    if t_end is not None and np.size(t_end) == 1:
        info["t_end"] = float(t_end)
    date, src = shot_date(m)
    if date:
        info["date"], info["date_source"] = date, src

    si = getattr(m, "SHOTINFO", None)
    explicit = {k: str(getattr(si, k)) for k in ("description", "regime", "baffle") if hasattr(si, k)}
    baffle = explicit.get("baffle") or getattr(getattr(m, "wall", None), "baffle", None)
    if isinstance(baffle, str) and baffle.strip():
        info["baffle"] = baffle.strip().upper()

    pm = heating_means(m)
    for k, v in pm.items():
        info[f"heating_mean_{k}"] = pm[k]
    aux = pm["ecrh"] + pm["nbi"]
    if pm["ohmic"] > 0 or aux > 0:
        auto_regime = "L" if aux < pm["ohmic"] else "H"
        # name only the auxiliary sources that matter: >= 20% of the auxiliary power
        on = [n for n, k in (("NBI", "nbi"), ("ECRH", "ecrh"))
              if pm[k] > AUX_MIN_POWER and pm[k] >= 0.2 * aux]
        auto_desc = "Ohmic L-mode" if auto_regime == "L" else ("+".join(on) + " H-mode").strip()
    else:
        auto_regime, auto_desc = "", ""

    regime, source = auto_regime, "heating power (ECRH + NBI below ohmic = L-mode, else H-mode)"
    det = detect_elms(m)
    on = [n for n, k in (("NBI", "nbi"), ("ECRH", "ecrh")) if pm[k] > AUX_MIN_POWER and pm[k] >= 0.2 * aux]
    if det["n_available"] > 0:                             # decided from the data: ELMs or not
        info["elmy"] = int(det["elmy"])
        info["elm_detection"] = det["summary"]
        if det["elmy"]:
            info["elm_frequency"] = det["frequency"]
            if det.get("energy"):
                info["elm_energy_drop"] = -det["energy"]["mean_drop"]          # [J], mean W_MHD loss per ELM
                info["elm_energy_tstat"] = det["energy"]["tstat"]
            regime, source = "H", "ELM detection"
            auto_desc = "ELMy H-mode" + (f" ({'+'.join(on)})" if on else "")
        else:
            regime, source = "L", "ELM detection (no ELMs)"
            auto_desc = ("L-mode" if pm["ohmic"] + aux == 0 else
                         "Ohmic L-mode" if aux < pm["ohmic"] else "heated L-mode (no ELMs)")
        auto_regime = regime
    if explicit.get("regime"):
        regime, source = explicit["regime"].strip().upper()[:1], "SHOTINFO"
    elif "description" in explicit:
        d = explicit["description"].upper().replace(" ", "")
        guess = "H" if "H-MODE" in d or "HMODE" in d else "L" if "L-MODE" in d or "LMODE" in d else ""
        if guess:
            regime, source = guess, "SHOTINFO description"
    if REGIME_OVERRIDE:
        regime, source = REGIME_OVERRIDE, "command line"
    if regime:
        info["regime"], info["regime_source"] = regime, source
    if det["n_available"] > 0 and regime and regime != ("H" if det["elmy"] else "L"):
        print(f"  WARNING: regime {regime} ({source}) disagrees with the ELM detection "
              f"({'ELMs found' if det['elmy'] else 'no ELMs'})")
    desc = explicit.get("description") or (auto_desc if regime == auto_regime else
                                           {"L": "L-mode", "H": "H-mode"}.get(regime, ""))
    if desc:
        info["description"] = desc
    return info


# TCV field names inside heating.HEATING -> output names. (time field, data field, output suffix)
HEATING_SOURCES = [("t_OHM", "pOHM", "ohmic", "power"),
                   ("t_ECH", "pECH", "ecrh", "power"),
                   ("t_NBI", "pNBI", "nbi", "power"),
                   ("t_MHD", "EMHD", "mhd", "energy"),       # MHD stored energy
                   ("t_dMHD", "dEMHD", "mhd", "power")]      # its time derivative = power


def mean_power(p, t, t0=None, t1=None):
    """Mean of the raw signal p(t) over [t0, t1] = integral (trapezoid) / (t1 - t0).

    The signal is clipped to the window with linear interpolation at the edges, and
    taken as zero where the trace does not cover the window. NaN samples are skipped."""
    t0, t1 = T_START if t0 is None else t0, T_END if t1 is None else t1
    p, t = align(np.asarray(p, float).ravel(), np.asarray(t, float).ravel())
    ok = np.isfinite(p) & np.isfinite(t)
    p, t = p[ok], t[ok]
    if t.size < 2:
        raise Missing("NBI trace too short")
    a, b = max(t0, t[0]), min(t1, t[-1])
    if b <= a:
        return 0.0                                          # beam off during the window
    inside = (t > a) & (t < b)
    tt = np.concatenate([[a], t[inside], [b]])
    pp = np.concatenate([[np.interp(a, t, p)], p[inside], [np.interp(b, t, p)]])
    trapezoid = getattr(np, "trapezoid", None) or np.trapz      # numpy 2 renamed trapz
    return float(trapezoid(pp, tt) / (t1 - t0))


def d_heating(m):
    """heating/{t,power}_{ohmic,ecrh,nbi,mhd} (+ energy_mhd), all on the common frames.

    Sources absent from the struct are simply left out."""
    h = need_available(get(get(m, "heating"), "HEATING"), "heating")
    out = {"power_units": "W", "energy_units": "J"}           # units ASSUMED
    # step_<quantity>: native step of each trace [s]; the stored 'power_mhd' / 'energy_mhd' are
    # interpolated to the database grid, so they are not resolved below their native step
    regime = shot_info(m).get("regime", "")
    for tname, dname, src, kind in HEATING_SOURCES:
        if hasattr(h, tname) and hasattr(h, dname) and np.size(getattr(h, tname)) != np.size(getattr(h, dname)):
            print(f"  WARNING: heating {tname} ({np.size(getattr(h, tname))} stamps) and {dname} "
                  f"({np.size(getattr(h, dname))} values) differ in length: {kind}_{src} not stored")
            continue
        if hasattr(h, tname) and hasattr(h, dname) and np.size(getattr(h, dname)):
            tt = np.asarray(getattr(h, tname), float)
            if np.nanmin(tt) > T_START + 1e-6 or np.nanmax(tt) < T_END - 1e-6:
                print(f"  WARNING: heating {tname} covers {np.nanmin(tt):.3f}-{np.nanmax(tt):.3f} s only, "
                      f"the database window is {T_START}-{T_END} s: {kind}_{src} is NaN outside "
                      f"(fetch it over the same window: build_pex_database 'Window')")
            out[f"t_{src}"] = T_FRAMES
            out[f"step_{kind}_{src}"] = native_step(tt)   # [s] native sampling step of this trace
            if src == "nbi" and regime == "L":
                # L-mode: NBI is a (fast-modulated) diagnostic beam, much faster than the
                # sampling, so keep the energy: constant mean power over the window.
                p = mean_power(getattr(h, dname), getattr(h, tname))
                out["power_nbi"] = np.full(N_FRAMES, p)
                out["power_nbi_method"] = (f"L-mode: constant mean power = integral of the raw "
                                           f"trace over {T_START}-{T_END} s / {T_END - T_START:g} s")
                print(f"  NBI (L-mode): constant mean power {p:.4g} W over {T_START}-{T_END} s")
            else:
                out[f"{kind}_{src}"] = resample(getattr(h, dname), getattr(h, tname), 0)
    if not any(k.startswith("t_") for k in out):
        raise Missing("no heating sources")
    return out


DSS_LINES = ("CIII_trip", "CII_6582", "D3_2")   # lines to extract from EdgeData.DSS (D3_2 = D-alpha)
DSS_EXTRA = ("Ti", "Ti_err")               # saved in any system where they are present
DSS_Z_CHANNELS = 30                         # geom.rLos/zLos: column 0 is a (0,0) placeholder


def d_dss(m):
    """Divertor spectroscopy, EdgeData.DSS -> spectrometer_visible/sys<N>.

    Only the lines in DSS_LINES are read, and only from systems where they hold data
    (not all NaN inside the time window): a system or line that is all NaN is
    unavailable and is not saved. Ti and Ti_err are saved from any system that has
    them (in practice the CIII system), whichever line it was fitted on.
    Slow diagnostic (6-15 ms sampling): linear interpolation, as for CXRS.
    Arrays are (channel, t)."""
    D = need_available(get(get(m, "EdgeData"), "DSS"), "DSS")
    out, have = {}, set()
    for name in [n for n in D._fieldnames if n.startswith("sys")]:
        so = getattr(D, name)
        if getattr(so, "available", 1) == 0 or not hasattr(so, "t"):
            out[name] = Missing_group("system not available")
            continue
        data = {}
        for f in DSS_LINES + DSS_EXTRA:
            if hasattr(so, f):
                a = resample_slow(getattr(so, f), so.t, 1)
                if np.isfinite(a).any():                   # all-NaN = unavailable
                    data[f] = a
                    have.add(f)
        if not data:
            out[name] = Missing_group("no data (all NaN) for " + ", ".join(DSS_LINES))
            continue
        out[name] = {"t": T_FRAMES, "r": resample_slow(so.R, so.t, 1),
                     "z": resample_slow(so.Z, so.t, 1), **data, **sample_info(so.t)}
        if "Ti" in data:
            out[name]["Ti_units"] = "eV"                   # ASSUMED
    if not any(isinstance(v, dict) for v in out.values()):
        raise Missing("no DSS system has data for " + ", ".join(DSS_LINES))
    none = [ln for ln in DSS_LINES if ln not in have]
    if none:
        out["lines_without_data"] = ",".join(none)
    g = getattr(D, "geom", None)
    if g is not None and np.shape(g.rLos) == (2, DSS_Z_CHANNELS + 1):
        rl, zl = np.asarray(g.rLos, float)[:, 1:], np.asarray(g.zLos, float)[:, 1:]
        out["line_of_sight"] = {"first_point": {"r": rl[0], "z": zl[0]},     # at the port
                                "second_point": {"r": rl[1], "z": zl[1]}}    # on the wall
    return out


DIAGNOSTICS = {"BOLO": d_bolo, "FIR": d_fir, "VALVE": d_valve, "TS": d_ts,
               "LP": d_lp, "IR": d_ir, "BAR": d_bar, "APG": d_apg,
               "MANTIS": d_mantis, "LIUQE": d_liuqe, "CXRS": d_cxrs,
               "DSS": d_dss, "WALL": d_wall, "HEATING": d_heating}


# where each diagnostic stores the shot number it was loaded for
SHOT_SOURCES = {
    "BOLO": lambda m: m.EdgeData.shot, "VALVE": lambda m: m.EdgeData.shot,
    "IR": lambda m: m.EdgeData.shot, "BAR": lambda m: m.EdgeData.shot,
    "APG": lambda m: m.EdgeData.shot, "MANTIS": lambda m: m.EdgeData.shot,
    "TS": lambda m: m.EdgeData.TS.shot, "LIUQE": lambda m: m.EdgeData.LIUQE.shot,
    "FIR": lambda m: m.FIRdata.shot, "LP": lambda m: m.LPdata.shot,
    "CXRS": lambda m: m.CXRSdata.shot, "HEATING": lambda m: m.heating.shot,
    "DSS": lambda m: m.EdgeData.shot,
}


class Missing_group(dict):
    """Placeholder for an unavailable sub-group (e.g. IR/VIR)."""
    def __init__(self, reason):
        super().__init__()
        self.reason = reason


# TCV diagnostic -> (IMAS IDS name, sub-group inside the IDS or None).
# Several TCV diagnostics can feed the same IDS (barometry: BAR + APG).
IDS_MAP = {
    "BOLO": ("bolometer", None),
    "FIR": ("interferometer", None),
    "VALVE": ("gas_injection", None),
    "TS": ("thomson_scattering", None),
    "LP": ("langmuir_probes", None),
    "IR": ("camera_ir", None),                  # sub-groups HIR, VIR
    "BAR": ("barometry", "barometer"),
    "APG": ("barometry", "pressure_gauges"),
    "MANTIS": ("camera_visible", None),         # sub-group per camera
    "LIUQE": ("equilibrium", None),
    "CXRS": ("charge_exchange", None),
    "DSS": ("spectrometer_visible", None),      # divertor spectroscopy
    "WALL": ("wall", None),                     # from the user-supplied wall struct
    "HEATING": ("heating", None),               # not a single IDS (ec_launchers, nbi, summary)
}


def build(src, strict=False, mode="auto", t0=None, t1=None, dt_fast=1e-4, dt_slow=5e-3, mhd_file=None):
    """Convert the .mat into a dict of groups. mode: 'auto' (default) = ELM time base if the shot
    is an ELMy H-mode according to detect_elms(), else the default 40-frame base; 'default' / 'elm'
    force one of them (with a forced regime: --regime=H -> 'elm', --regime=L -> 'default')."""
    m = sio.loadmat(src, squeeze_me=True, struct_as_record=False)
    m = type("M", (), {k: v for k, v in m.items() if not k.startswith("__")})()
    if mhd_file:                                           # dW_MHD/dt fetched separately (ELMy shots only)
        mm = sio.loadmat(mhd_file, squeeze_me=True)
        if hasattr(getattr(m, "heating", None), "HEATING"):
            H = m.heating.HEATING
            H.t_dMHD, H.dEMHD = mm["t_dMHD"], mm["dEMHD"]
            if "EMHD" in mm:                               # LIUQE W_MHD on the same time stamps replaces
                m._mhd_native = (np.asarray(H.t_MHD), np.asarray(H.EMHD))   # the 1 ms one (as fetch_shot_data.m
                H.t_MHD, H.EMHD = mm["t_dMHD"], mm["EMHD"]                  # does); keep it for the ELM check
        else:
            print("  WARNING: --mhd given but the source has no heating struct")
    det = detect_elms(m)
    if mode == "auto":
        mode = ("elm" if REGIME_OVERRIDE == "H" else "default") if REGIME_OVERRIDE else \
               ("elm" if det["elmy"] else "default")
        print(f"  ELM detection: {'ELMy H-mode' if det['elmy'] else 'no ELMs'} "
              f"({det['summary']}) -> {mode} time base")
    configure(mode, t0, t1, dt_fast, dt_slow)
    if MODE == "elm" and det["elmy"]:                      # where do the ELMs sit relative to the window?
        te = det["times"]
        inside = te[(te >= T_START) & (te <= T_END)]
        print(f"  ELM window {T_START}-{T_END} s: {inside.size} of {te.size} detected ELMs; first ELM at "
              f"{te.min():.3f} s" + (f"  -> the window starts {te.min() - T_START:.2f} s before the first ELM "
                                    f"(H-mode transition?); use --t0 to move it" if te.min() - T_START > 0.1 else ""))
    shot = int(get(get(m, "EdgeData"), "shot")) if hasattr(m, "EdgeData") else int(m.shot)
    db = {}
    for tcv, fn in DIAGNOSTICS.items():
        ids, sub = IDS_MAP[tcv]
        _METHOD_LOG.clear()
        set_grid("slow" if (MODE == "elm" and tcv in SLOW_DIAGNOSTICS) else "fast")
        set_hold(MODE == "elm" and tcv in HOLD_DIAGNOSTICS)
        try:
            res = fn(m)
        except Missing as e:
            res = Missing_group(str(e))
        except Exception as e:                             # corrupt/unexpected content
            if strict:
                raise
            import traceback
            fr = traceback.extract_tb(e.__traceback__)[-1]
            res = Missing_group(f"conversion failed: {type(e).__name__}: {e} "
                                f"[{os.path.basename(fr.filename)}:{fr.lineno} in {fr.name}]")
        if isinstance(res, Missing_group):
            print(f"  {ids}{'/' + sub if sub else ''} ({tcv}): unavailable ({res.reason})")
        else:
            used = {meth for meth, _ in _METHOD_LOG}
            if used and isinstance(res, dict) and "time_resampling" not in res:
                res["time_resampling"] = " + ".join(sorted(used))
            slow = [(meth, st) for meth, st in _METHOD_LOG if meth in ("interpolation", "nearest-sample hold")]
            if slow:
                print(f"  {ids}{'/' + sub if sub else ''}: native step {np.median([st for _, st in slow]) * 1e3:.1f} ms "
                      f"vs {WINDOW * 1e3:g} ms window -> {slow[0][0]}")
            # record the shot each diagnostic was read from; warn on any mismatch
            try:
                src_shot = int(SHOT_SOURCES[tcv](m))
            except Exception:
                src_shot = None                            # source has no shot field
            if src_shot is not None:
                res["source_shot"] = src_shot
                if src_shot != shot:
                    print(f"  WARNING: {ids} ({tcv}) shot {src_shot} != shot {shot}")
        if sub is None:
            db[ids] = res
            res["tcv_name"] = tcv
        else:                                              # several TCV sources in one IDS
            g = db.setdefault(ids, {})
            g[sub] = res
            g["tcv_name"] = ",".join(filter(None, [g.get("tcv_name"), tcv]))
    # an IDS fed only by unavailable sources is itself unavailable
    for ids, g in list(db.items()):
        subs = [v for v in g.values() if isinstance(v, (dict, Missing_group))]
        if not isinstance(g, Missing_group) and subs and all(isinstance(v, Missing_group) for v in subs):
            db[ids] = Missing_group("; ".join(f"{k}: {v.reason}" for k, v in g.items()
                                              if isinstance(v, Missing_group)))
            db[ids]["tcv_name"] = g["tcv_name"]
    set_grid("fast")                                       # root time base = fast grid
    set_hold(False)
    info = shot_info(m)
    info["_elm_times"] = detect_elms(m)["times"]
    return shot, db, info


def write_h5(group, d):
    for k, v in d.items():
        if isinstance(v, Missing_group):
            g = group.create_group(k)
            g.attrs["available"] = 0
            g.attrs["reason"] = v.reason
            for k2, v2 in v.items():                       # e.g. tcv_name
                g.attrs[k2] = v2
        elif isinstance(v, dict):
            g = group.create_group(k)
            g.attrs["available"] = 1
            write_h5(g, v)
        elif isinstance(v, str):
            group.attrs[k] = v
        elif np.ndim(v) == 0:
            group.attrs[k] = v
        elif isinstance(v, (list, tuple)) and v and isinstance(v[0], str):
            group.create_dataset(k, data=np.array(v, dtype=h5py.string_dtype()))
        else:
            group.create_dataset(k, data=np.asarray(v), compression="gzip",
                                 compression_opts=4)


def convert(src, out_dir=".", strict=False, regime=None, mode="auto", t0=None, t1=None,
            dt_fast=1e-4, dt_slow=5e-3, mhd_file=None):
    """Convert one STRUCTexp .mat file into <out_dir>/shot_<shot>.h5; return the path.

    mode='auto'   : (default) ELM time base if the shot is an ELMy H-mode (detect_elms), else 'default'.
    mode='default': one 40-frame grid (L-mode, 0.6-1.6 s).
    mode='elm'    : fast grid (dt_fast: heating, valve, pressure gauges, IR) + slow grid (dt_slow: the rest)
                    on t0-t1 (default 0.6-1.0 s)."""
    global REGIME_OVERRIDE
    if regime:
        REGIME_OVERRIDE = regime.strip().upper()[:1]
    shot, db, info = build(src, strict, mode, t0, t1, dt_fast, dt_slow, mhd_file)
    elm_times = info.pop("_elm_times", np.array([]))
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, f"shot_{shot}.h5")
    with h5py.File(dst, "w") as f:
        f.attrs["shot"] = shot
        f.attrs["source_file"] = os.path.basename(src)
        f.attrs["time_mode"] = MODE
        f.attrs["t_start"], f.attrs["t_end_window"] = T_START, T_END
        if MODE == "elm":
            f.attrs["t_info"] = ("`t` = fast grid (heating, valve, pressure gauges, IR; frame centres [s]); all other "
                                 "diagnostics are on `t_slow` (their groups carry their own `t`); TS, CXRS, IR, "
                                 "MANTIS and DSS hold each sample on the nearest frames, see t_sample / measured")
            f.attrs["dt_fast"], f.attrs["dt_slow"] = GRIDS["fast"][1], GRIDS["slow"][1]
            f.create_dataset("t_slow", data=GRIDS["slow"][0])
            if elm_times.size:
                f.create_dataset("elm_times", data=elm_times)
        else:
            f.attrs["t_info"] = ("frame centres [s]; each value is the mean of the "
                                 "native samples in t +/- window/2")
        f.attrs["dt_frame"] = DT
        f.attrs["window"] = WINDOW
        for k, v in info.items():
            f.attrs[k] = v
        f.create_dataset("t", data=T_FRAMES)
        write_h5(f, db)
    print(f"wrote {dst}")
    return dst


def detect_only(src, regime=None):
    """Print 'ELMY=1' or 'ELMY=0' (+ the detection report) for a .mat file, without converting it.
    Used by build_pex_database.m to decide, before the slow LIUQE call, whether dEMHD is needed.
    A forced regime (--regime=H/L) decides directly."""
    if regime:
        print(f"ELMY={int(regime.strip().upper()[:1] == 'H')}  regime forced to {regime}")
        return
    m = sio.loadmat(src, variable_names=["EdgeData", "LPdata", "heating"], squeeze_me=True,
                    struct_as_record=False)
    m = type("M", (), {k: v for k, v in m.items() if not k.startswith("__")})()
    det = detect_elms(m)
    print(f"ELMY={int(det['elmy'])}  {det['summary']}")


def main(argv):
    strict = "--strict" in argv
    regime, mode, opts, mhd, detect = None, "auto", {}, None, "--detect" in argv
    for a in argv:
        if a.startswith("--regime="):                       # e.g. --regime=L (forces the regime label)
            regime = a.split("=", 1)[1]
        elif a == "--elm":                                  # force the ELM-resolved time base
            mode = "elm"
        elif a.startswith("--mode="):                       # auto (default) | default | elm
            mode = a.split("=", 1)[1]
        elif a.startswith("--mhd="):                        # .mat with t_dMHD, dEMHD (separate fetch)
            mhd = a.split("=", 1)[1]
        elif a.startswith(("--t0=", "--t1=", "--dt-fast=", "--dt-slow=")):
            k, v = a[2:].split("=", 1)
            opts[k.replace("-", "_")] = float(v)
    args = [a for a in argv if not a.startswith("--")]
    src = args[0] if args else "STRUCTexp.mat"
    if detect:
        detect_only(src, regime)
        return
    out_dir = args[1] if len(args) > 1 else "."
    convert(src, out_dir, strict, regime, mode, mhd_file=mhd, **opts)


if __name__ == "__main__":
    main(sys.argv[1:])
