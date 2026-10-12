#!/usr/bin/env python3
"""Print a summary of a shot_<N>.h5 database file.

  1. shot info: description, regime, baffle, date, t_end, mean heating powers (and
     where regime / date come from), time base, and a warning for every diagnostic
     whose source shot differs from the shot
  2. availability of every diagnostic (with the reason when missing)
  3. the file structure as a tree (groups, dataset shapes, availability)
  4. for every diagnostic: attributes (units, source_shot, ...) and each dataset with
     shape, NaN fraction, min/max and, for time series, how many are completely empty

Usage: python3 -m pexdb.summarize_shot shot_89837.h5 [DIAG ...]   (DIAG = group names to detail)
"""
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
            print(f"{prefix}{branch}{name}  {obj.shape}")
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
    print(f"  baffle   : {a.get('baffle', '?')}")
    print(f"  date     : {a.get('date', '?')}" + (f"   ({a['date_source']})" if "date_source" in a else ""))
    if "t_end" in a:
        print(f"  t_end    : {a['t_end']:.3f} s")
    pw = {k: a[f"heating_mean_{k}"] for k in ("ohmic", "ecrh", "nbi") if f"heating_mean_{k}" in a}
    if pw:
        print("  <power>  : " + ", ".join(f"{k} {v / 1e3:.1f} kW" for k, v in pw.items())
              + f"  (mean over {t[0]:.1f}-{t[-1]:.1f} s)")
    window = f", averaging window +-{a['window'] * 5e2:.1f} ms" if "window" in a else ""
    print(f"  time base: {len(t)} frames, {t[0]:.3f}-{t[-1]:.3f} s, dt = {a['dt_frame'] * 1e3:.2f} ms{window}\n")
    bad = {p: s for p, s in source_shots(f).items() if s != int(a["shot"])}
    for p, s in bad.items():
        print(f"  WARNING: {p} comes from shot {s}, not {a['shot']}")
    if bad:
        print()


def main(argv):
    if not argv:
        sys.exit(__doc__)
    with h5py.File(argv[0], "r") as f:
        header(f)
        diags = [k for k in f if k != "t"]
        ok = [d for d in diags if f[d].attrs.get("available", 1)]
        print(f"Available ({len(ok)}/{len(diags)}): {', '.join(ok)}")
        for d in diags:
            if d not in ok:
                print(f"Unavailable: {d} ({f[d].attrs.get('reason', '?')})")
        print("\n=== STRUCTURE ===")
        print(f"{f.filename}/")
        tree(f)
        for d in (argv[1:] or ok):
            if d not in f:
                print(f"\n=== {d} ===\n  not in file")
                continue
            print(f"\n=== {d} ===")
            show(f[d], 1)


if __name__ == "__main__":
    main(sys.argv[1:])
