#!/usr/bin/env python3
"""Build a uniform-in-time HDF5 database (one file per shot) from STRUCexp .mat files.

Every time-dependent quantity of every diagnostic is resampled on the same
N_FRAMES-point time base between T_START and T_END. Each frame is the
NaN-aware average of all native samples falling in [t - WINDOW/2, t + WINDOW/2)
(WINDOW = 10 ms, i.e. +-5 ms around the frame time; the frame spacing is
25.6 ms, so windows do not touch). Frames with no native sample are NaN.

Top-level groups are named after the IMAS IDS they correspond to (see IDS_MAP);
the original TCV name is kept in each group's `tcv_name` attribute.

Langmuir probes are NOT resampled: all native samples in [T_START, T_END] are\nkept on their own time vector (see crop()).

CXRS is sampled every 24 ms (slower than the window) so it is linearly
interpolated instead (see interpolate()).

Missing diagnostics: every diagnostic is a top-level HDF5 group carrying an
attribute `available` (1/0). If the source is absent, flagged unavailable,
empty, or fails to convert, the group is still written, with available=0, an
`reason` attribute, and no datasets. Check `available` before reading.

Layout: a dict leaf is an HDF5 group, an array is a dataset, a string/number
is an attribute-free scalar dataset. Time is the first axis, except where
noted in the `layout` attribute of a group or in the docs below.

Usage: python3 -m pexdb.build_database STRUCTexp.mat [out_dir] [--strict] [--regime=L|H]
(--regime overrides/sets the confinement regime when SHOTINFO is absent; for L-mode
shots the NBI power is stored as a constant mean over the window, see mean_power()).
Output: <out_dir>/shot_<shot>.h5
"""
import os
import sys
import warnings
import numpy as np
import h5py
import scipy.io as sio

T_START, T_END, N_FRAMES = 0.6, 1.6, 40
T_FRAMES = np.linspace(T_START, T_END, N_FRAMES)
DT = T_FRAMES[1] - T_FRAMES[0]            # frame spacing
WINDOW = 0.010                            # total averaging window width [s], centred on each frame


class Missing(Exception):
    """Raised when a diagnostic has no usable data in the source file."""


def resample(arr, t, axis):
    """Bin-average `arr` along `axis` (native times `t`) onto T_FRAMES."""
    arr = np.asarray(arr, dtype=float)
    t = np.asarray(t, dtype=float).ravel()
    if arr.size == 0 or t.size == 0:
        raise Missing("empty array")
    arr = np.moveaxis(arr, axis, 0)
    out = np.full((N_FRAMES,) + arr.shape[1:], np.nan)
    for i, tf in enumerate(T_FRAMES):
        sel = (t >= tf - WINDOW / 2) & (t < tf + WINDOW / 2)
        if sel.any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices
                out[i] = np.nanmean(arr[sel], axis=0)
    return np.moveaxis(out, 0, axis)


def interpolate(arr, t, axis):
    """Linear interpolation of `arr` along `axis` onto T_FRAMES, for slow diagnostics.

    Each 1D series is interpolated using only its valid (non-NaN) samples.
    A frame is NaN if it lies outside the valid range, or if the two valid
    samples around it are more than 2 native sampling steps apart (no
    interpolation across long gaps of missing data).
    """
    arr = np.asarray(arr, dtype=float)
    t = np.asarray(t, dtype=float).ravel()
    if arr.size == 0 or t.size == 0:
        raise Missing("empty array")
    max_gap = 2 * np.median(np.diff(t))
    a = np.moveaxis(arr, axis, 0)
    out = np.full((N_FRAMES,) + a.shape[1:], np.nan)
    for idx in np.ndindex(*a.shape[1:]):
        y = a[(slice(None),) + idx]
        ok = ~np.isnan(y)
        if ok.sum() < 2:
            continue
        tv, yv = t[ok], y[ok]
        i = np.searchsorted(tv, T_FRAMES)                 # right neighbour index
        inside = (i > 0) & (i < len(tv))
        i0, i1 = np.clip(i - 1, 0, len(tv) - 1), np.clip(i, 0, len(tv) - 1)
        good = inside & (tv[i1] - tv[i0] <= max_gap)
        val = np.interp(T_FRAMES, tv, yv)
        out[(slice(None),) + idx] = np.where(good, val, np.nan)
    return np.moveaxis(out, 0, axis)


def crop(arr, t, axis, eps=1e-6):
    """Keep the native samples with T_START <= t <= T_END (no averaging).
    Returns (cropped array, cropped time)."""
    arr, t = np.asarray(arr, dtype=float), np.asarray(t, dtype=float).ravel()
    k = (t >= T_START - eps) & (t <= T_END + eps)
    if not k.any():
        raise Missing("no samples in the time window")
    return np.take(arr, np.flatnonzero(k), axis=axis), t[k]


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
    sig = resample(b.signals, b.t, 0)                      # (t, measured channels)
    ident, geometry = bolo_geometry(b)
    if ident is None:                                      # no geometry: measured channels only
        chn, channels = np.asarray(b.chn), sig
    else:                                                  # all channels, NaN where not measured
        chn = ident
        channels = np.full((N_FRAMES, ident.size), np.nan)
        pos = {n: i for i, n in enumerate(ident)}
        miss = [int(n) for n in np.asarray(b.chn) if int(n) not in pos]
        if miss:
            raise Missing(f"measured channels without geometry: {miss}")
        channels[:, [pos[int(n)] for n in np.asarray(b.chn)]] = sig
    return {
        "t": T_FRAMES,
        "channels": channels,                             # (t, channel); NaN = no data
        "channels_units": str(b.dim),
        "chn": chn,
        "emi": resample(b.emi, b.t_emi, 2),               # (R, Z, t)
        "emi_units": "W/m3",                              # assumed
        "r": b.R, "z": b.Z,
        **rs_as(b, {"power_core": "p_cor", "power_div": "p_div", "power_xpt": "p_xpt"}, b.t, 0),
        "power_tot": resample(b.p_tot, b.t_tot, 0),
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
        flow = resample(v.flow, v.t, 0)                    # particles/s (molecules assumed)
        out["valve"][k] = {
            "t": T_FRAMES,
            "species_name": str(v.gas),
            "exit_position": {"r": float(v.R), "z": float(v.Z)},
            "flow_rate": flow * KB * T_GAS,                # Pa.m^3.s^-1
            "flow_rate_units": "Pa.m^3.s^-1",
            "flow_particles": flow,                        # original TCV quantity
            "flow_particles_units": str(v.units)}
    if not out["valve"]:
        raise Missing("no valves")
    return out


def d_ts(m):
    s = need_available(get(get(m, "EdgeData"), "TS"), "TS")
    return {"t": T_FRAMES, "r": s.R, "z": s.Z,            # fields: (t, point) or (t,)
            **rs(s, ["te", "dte", "ne", "dne", "rho",
                     "ne_u", "dne_u", "te_u", "dte_u"], s.t, 0),
            "distance_separatrix_midplane": resample(s.dru, s.t, 0),
            # line-averaged density computed from TS; same name as the FIR quantity
            "av_dens": resample(s.nel, s.t, 0)}


def d_lp(m):
    """embedded = all probes; inner_target / outer_target = subsets of them.

    NOT resampled: every native sample (10 ms) between T_START and T_END is kept,
    on the group's own time vector `t`, so arrays are (probe, t_native)."""
    L = get(m, "LPdata")
    probes = np.asarray(get(L, "probes")).astype(int)
    fit = get(L, "FourParamsFit")
    pos = np.asarray(L.pos)
    t_lp = crop(L.time, L.time, 0)[1]

    def subset(sel):
        d = {"probes": probes[sel],
             "position": {"r": pos[sel, 0], "z": pos[sel, 1]}}
        for out_n, arr in [("b_field_angle", L.ang),
                           ("distance_separatrix_midplane", L.dsep_mid),
                           ("t_e", fit.Te), ("n_e", fit.dens),
                           ("v_floating", fit.Vfl), ("j_i_saturation", fit.Jsat),
                           ("rho", L.rho_psi), ("Pperp", fit.Pperp)]:   # last two: no IDS field
            d[out_n] = crop(np.asarray(arr)[sel], L.time, 1)[0]         # (probe, t)
        return d

    return {"t": t_lp,
            "embedded": subset(np.ones(probes.shape, bool)),
            "inner_target": subset((probes >= 136) & (probes <= 190)),
            "outer_target": subset((probes >= 1) & (probes <= 42))}


def d_ir(m):
    IR = get(get(m, "EdgeData"), "IR")
    out = {}
    for k, name in (("HIR", "hir"), ("VIR", "vir")):
        try:
            i = need_available(get(IR, name), name)
            out[k] = {"t": T_FRAMES, "r": i.r, "z": i.z,
                      "power": resample(i.power, i.t, 0),
                      "qperp": resample(i.qperp, i.t, 1),   # (position, t)
                      "distance_separatrix_midplane": resample(i.dru, i.t, 1)}
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
                  **rs(c, ["lInner", "sFrontInner", "lOuter", "sFrontOuter"], c.t, 0)}
    if not out:
        raise Missing("no cameras")
    return out


def d_liuqe(m):
    q = need_available(get(get(m, "EdgeData"), "LIUQE"), "LIUQE")
    return {"t": T_FRAMES, "r": q.R, "z": q.Z,
            "psi": resample(q.psi, q.t, 2),               # (R, Z, t)
            "ip": resample(q.i_p, q.t, 0), "ip_units": "A",
            "b0": resample(q.b_zero, q.t, 0), "b0_units": "T",   # vacuum toroidal field
            "rho": resample(q.rho, q.t, 2),
            **rs(q, ["r_axis", "z_axis", "psi_axis"], q.t, 0),
            "r_contour": resample(q.r_contour, q.t, 1),   # LCFS, (point, t)
            "z_contour": resample(q.z_contour, q.t, 1)}


def d_cxrs(m):
    # slow diagnostic (24 ms sampling): linear interpolation, not window average
    cd = get(m, "CXRSdata")
    C = get(cd, "calc")
    out = {"t": T_FRAMES, "r": C.R, "z": C.Z,
           "rho": interpolate(C.rho, C.time, 1)}
    for q_ in ("ni", "Ti", "vi"):                          # (channel, t)
        out[q_] = {"data": interpolate(getattr(C, q_), C.time, 1),
                   "error": interpolate(getattr(C, "err_" + q_), C.time, 1),
                   "units": str(getattr(C, q_ + "_units"))}
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

    Regime (priority: --regime on the command line > SHOTINFO > heating power):
    by heating power, the shot is 'Ohmic L-mode' if the auxiliary power (ECRH + NBI)
    is below the ohmic power, otherwise H-mode ('NBI H-mode', 'ECRH H-mode', ...).
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
    p, t = np.asarray(p, float).ravel(), np.asarray(t, float).ravel()
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
    regime = shot_info(m).get("regime", "")
    for tname, dname, src, kind in HEATING_SOURCES:
        if hasattr(h, tname) and hasattr(h, dname) and np.size(getattr(h, dname)):
            out[f"t_{src}"] = T_FRAMES
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
    if len(out) == 2:
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
                a = interpolate(getattr(so, f), so.t, 1)
                if np.isfinite(a).any():                   # all-NaN = unavailable
                    data[f] = a
                    have.add(f)
        if not data:
            out[name] = Missing_group("no data (all NaN) for " + ", ".join(DSS_LINES))
            continue
        out[name] = {"t": T_FRAMES, "r": interpolate(so.R, so.t, 1),
                     "z": interpolate(so.Z, so.t, 1), **data}
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


def build(src, strict=False):
    m = sio.loadmat(src, squeeze_me=True, struct_as_record=False)
    m = type("M", (), {k: v for k, v in m.items() if not k.startswith("__")})()
    shot = int(get(get(m, "EdgeData"), "shot")) if hasattr(m, "EdgeData") else int(m.shot)
    db = {}
    for tcv, fn in DIAGNOSTICS.items():
        ids, sub = IDS_MAP[tcv]
        try:
            res = fn(m)
        except Missing as e:
            res = Missing_group(str(e))
        except Exception as e:                             # corrupt/unexpected content
            if strict:
                raise
            res = Missing_group(f"conversion failed: {type(e).__name__}: {e}")
        if isinstance(res, Missing_group):
            print(f"  {ids}{'/' + sub if sub else ''} ({tcv}): unavailable ({res.reason})")
        else:
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
    return shot, db, shot_info(m)


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


def convert(src, out_dir=".", strict=False, regime=None):
    """Convert one STRUCTexp .mat file into <out_dir>/shot_<shot>.h5; return the path."""
    global REGIME_OVERRIDE
    if regime:
        REGIME_OVERRIDE = regime.strip().upper()[:1]
    shot, db, info = build(src, strict)
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, f"shot_{shot}.h5")
    with h5py.File(dst, "w") as f:
        f.attrs["shot"] = shot
        f.attrs["source_file"] = os.path.basename(src)
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


def main(argv):
    strict = "--strict" in argv
    regime = None
    for a in argv:
        if a.startswith("--regime="):                       # e.g. --regime=L (when SHOTINFO is missing)
            regime = a.split("=", 1)[1]
    args = [a for a in argv if not a.startswith("--")]
    src = args[0] if args else "STRUCTexp.mat"
    out_dir = args[1] if len(args) > 1 else "."
    convert(src, out_dir, strict, regime)


if __name__ == "__main__":
    main(sys.argv[1:])
