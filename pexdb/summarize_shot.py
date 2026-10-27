#!/usr/bin/env python3
"""Print a summary of the database at three levels of detail (for talks).

  1. database : python3 -m pexdb.summarize_shot DIR              one line per shot_<N>.h5 (label of each shot)
  2. shot     : python3 -m pexdb.summarize_shot shot_89837.h5    label + diagnostics (IMAS IDS names)
  3. diagnostic: python3 -m pexdb.summarize_shot shot_89837.h5 thomson_scattering   contents of one diagnostic

Each level is printed as a tree. A shot is labelled by its description (e.g. Ohmic L-mode), plasma current and
toroidal field (signed, mean over the window) and baffle. Diagnostic groups are named after the IMAS IDS (see build_database.IDS_MAP);

Full dump (header, availability, tree, every diagnostic with NaN statistics):
    python3 -m pexdb.summarize_shot shot_89837.h5 --full [DIAG ...]
"""
import glob
import os
import sys
import warnings
import numpy as np
import h5py


def fmt(v):
    """Compact text for an attribute value."""
    if isinstance(v, (float, np.floating)):
        return f"{v:.5g}"
    if isinstance(v, bytes):
        return v.decode()
    return str(v)


def stats(a, n_t=None):
    """NaN fraction and range; n_t = length of the group's time axis (if known)."""
    a = np.asarray(a)
    if a.dtype.kind == "O":                                   # string datasets
        txt = [x.decode() if isinstance(x, bytes) else str(x) for x in a.ravel()]
        return ", ".join(txt[:4]) + (", ..." if len(txt) > 4 else "")
    if a.dtype.kind not in "fiu" or a.size == 0:
        return ""
    a = a.astype(float)
    nan = np.isnan(a).mean()
    if nan == 1:
        return "all NaN"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        s = f"NaN {nan:5.1%}  min {np.nanmin(a):10.3g}  max {np.nanmax(a):10.3g}"
    if n_t and a.ndim >= 2 and a.shape.count(n_t) == 1:       # time axis unambiguous: count empty series
        ax = a.shape.index(n_t)
        series = np.moveaxis(a, ax, 0).reshape(n_t, -1)
        empty = int(np.isnan(series).all(axis=0).sum())
        if empty:
            s += f"  [{empty}/{series.shape[1]} series empty]"
    return s


def show(group, indent, n_t=None):
    pad = "  " * indent
    if "t" in group and isinstance(group["t"], h5py.Dataset):
        n_t = group["t"].shape[0]
    attrs = {k: v for k, v in group.attrs.items() if k not in ("available", "reason")}
    if attrs:
        print(pad + "attrs: " + ", ".join(f"{k}={fmt(v)}" for k, v in attrs.items()))
    for name, obj in group.items():
        if isinstance(obj, h5py.Dataset):
            print(f"{pad}{name:<14s}{str(obj.shape):<16s}{stats(obj[()], n_t)}")
        elif obj.attrs.get("available", 1) == 0:
            print(f"{pad}{name}/  UNAVAILABLE ({obj.attrs.get('reason', '?')})")
        else:
            print(f"{pad}{name}/")
            show(obj, indent + 1, n_t)


def tree(group, prefix=""):
    """Print group contents as an ascii tree (names, shapes, availability)."""
    items = list(group.items())
    for i, (name, obj) in enumerate(items):
        last = i == len(items) - 1
        branch, cont = ("`-- ", "    ") if last else ("|-- ", "|   ")
        if isinstance(obj, h5py.Dataset):
            unit = group.attrs.get(name + "_units", "")
            print(f"{prefix}{branch}{name:<14s}{str(obj.shape):<14s}{'[' + fmt(unit) + ']' if unit != '' else ''}")
        elif obj.attrs.get("available", 1) == 0:
            print(f"{prefix}{branch}{name}/  [UNAVAILABLE: {obj.attrs.get('reason', '?')}]")
        else:
            print(f"{prefix}{branch}{name}/")
            tree(obj, prefix + cont)


def source_shots(group, found=None, path=""):
    """{path: source_shot} for every group that carries the attribute."""
    found = {} if found is None else found
    for name, obj in group.items():
        if isinstance(obj, h5py.Group):
            if "source_shot" in obj.attrs:
                found[path + name] = int(obj.attrs["source_shot"])
            source_shots(obj, found, path + name + "/")
    return found


def header(f):
    a = f.attrs
    t = f["t"][()]
    print(f"Shot {a['shot']}  (source: {a.get('source_file', '?')})")
    regime = (f"   [regime {a['regime']}" + (f", from {a['regime_source']}]" if "regime_source" in a else "]")
              if "regime" in a else "   [regime unknown]")
    print(f"  type     : {a.get('description', '?')}{regime}")
    if "elmy" in a:
        print(f"  ELMs     : {'ELMy, ' + format(a['elm_frequency'], '.0f') + ' Hz' if a['elmy'] else 'none detected'}"
              f"   ({a.get('elm_detection', '')})")
    print(f"  baffle   : {a.get('baffle', '?')}")
    print(f"  date     : {a.get('date', '?')}" + (f"   ({a['date_source']})" if "date_source" in a else ""))
    if "t_end" in a:
        print(f"  t_end    : {a['t_end']:.3f} s")
    pw = {k: a[f"heating_mean_{k}"] for k in ("ohmic", "ecrh", "nbi") if f"heating_mean_{k}" in a}
    if pw:
        print("  <power>  : " + ", ".join(f"{k} {v / 1e3:.1f} kW" for k, v in pw.items())
              + f"  (mean over {t[0]:.1f}-{t[-1]:.1f} s)")
    window = f", averaging window +-{a['window'] * 5e2:.1f} ms" if "window" in a else ""
    if a.get("time_mode") == "elm" and "t_slow" in f:
        ts = f["t_slow"][()]
        print(f"  time base: ELM mode, {t[0]:.3f}-{t[-1]:.3f} s | fast grid {len(t)} frames, "
              f"dt = {a['dt_fast'] * 1e3:.2f} ms (heating, valve, pressure gauges, IR) | slow grid {len(ts)} frames, "
              f"dt = {a['dt_slow'] * 1e3:.2f} ms (all other diagnostics; TS, CXRS, MANTIS, DSS hold each "
              f"sample on the nearest frames: t_sample / measured)\n")
    else:
        print(f"  time base: {len(t)} frames, {t[0]:.3f}-{t[-1]:.3f} s, dt = {a['dt_frame'] * 1e3:.2f} ms{window}\n")
    bad = {p: s for p, s in source_shots(f).items() if s != int(a["shot"])}
    for p, s in bad.items():
        print(f"  WARNING: {p} comes from shot {s}, not {a['shot']}")
    if bad:
        print()


def shot_type(f):
    """[(key, value)] describing the shot: description, signed mean Ip and Bt over the window, baffle."""
    a = f.attrs
    rows = [("description", a.get("description") or {"L": "L-mode", "H": "H-mode"}.get(a.get("regime", ""), "?"))]
    for label, path, scale, unit, fmt_ in (("Ip", "equilibrium/ip", 1e3, "kA", "+.0f"),
                                           ("Bt", "equilibrium/b0", 1, "T", "+.2f")):
        if path in f and np.isfinite(np.nanmean(f[path][()])):
            rows.append((label, f"{np.nanmean(f[path][()]) / scale:{fmt_}} {unit}"))
    if "baffle" in a:
        rows.append(("baffle", str(a["baffle"])))
    return rows


def flavour(f):
    """One-line label of a shot: 'Ohmic L-mode, Ip -250 kA, Bt -1.43 T, SILO'."""
    return ", ".join(f"{k} {v}" if k in ("Ip", "Bt") else v for k, v in shot_type(f))


def diagnostics(f):
    """(available, unavailable) top-level diagnostic groups."""
    diags = [k for k in f if isinstance(f[k], h5py.Group)]
    ok = [d for d in diags if f[d].attrs.get("available", 1)]
    return ok, [d for d in diags if d not in ok]


def branch(i, n):
    """(connector, continuation prefix) for the i-th of n tree children."""
    return ("`-- ", "    ") if i == n - 1 else ("|-- ", "|   ")


def level1(directory):
    files = sorted(glob.glob(os.path.join(directory, "shot_*.h5")))
    print(f"{os.path.abspath(directory)}/   ({len(files)} files, one per shot)")
    for i, path in enumerate(files):
        with h5py.File(path, "r") as f:
            print(f"{branch(i, len(files))[0]}{os.path.basename(path):<16s}{flavour(f)}")


def level2(path):
    with h5py.File(path, "r") as f:
        ok, bad = diagnostics(f)
        rows, names = shot_type(f), ok + bad
        print(os.path.basename(path))
        print("|-- shot type")
        for i, (k, v) in enumerate(rows):
            print(f"|   {branch(i, len(rows))[0]}{k:<12s}{v}")
        print("`-- diagnostics")
        for i, d in enumerate(names):
            note = "" if d in ok else f"  [UNAVAILABLE: {f[d].attrs.get('reason', '?')}]"
            print(f"    {branch(i, len(names))[0]}{d}{note}")


def level3(path, diag):
    with h5py.File(path, "r") as f:
        if diag not in f:
            sys.exit(f"{diag} not in {path}; choose from: {', '.join(diagnostics(f)[0])}")
        print(f"{os.path.basename(path)}/{diag}/")
        tree(f[diag])


def full(path, diags):
    with h5py.File(path, "r") as f:
        header(f)
        ok, bad = diagnostics(f)
        print(f"Available ({len(ok)}/{len(ok) + len(bad)}): {', '.join(ok)}")
        for d in bad:
            print(f"Unavailable: {d} ({f[d].attrs.get('reason', '?')})")
        print("\n=== STRUCTURE ===")
        print(f"{f.filename}/")
        tree(f)
        for d in (diags or ok):
            if d not in f:
                print(f"\n=== {d} ===\n  not in file")
                continue
            print(f"\n=== {d} ===")
            show(f[d], 1)


def main(argv):
    full_dump = "--full" in argv
    argv = [a for a in argv if a != "--full"]
    if not argv:
        sys.exit(__doc__)
    if os.path.isdir(argv[0]):
        level1(argv[0])
    elif full_dump:
        full(argv[0], argv[1:])
    elif len(argv) == 1:
        level2(argv[0])
    else:
        level3(argv[0], argv[1])


if __name__ == "__main__":
    main(sys.argv[1:])
