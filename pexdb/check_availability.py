#!/usr/bin/env python3
"""Check which diagnostics are available in shot_<N>.h5 database files.

A diagnostic is one of:
  #  OK       available, with finite measurement data
  ~  PARTIAL  available, but some of its parts are missing (e.g. one IR camera, one DSS system)
  0  EMPTY    flagged available, but every measurement dataset is NaN (coordinates / geometry ignored)
  .  MISSING  not available (the file stores the reason)

The table has one row per file: shot, mode (L or H), ELMy? (yes/no), baffle (SILO/NINO), then the
diagnostics ('?' when the file does not say; the time base, elm or default, is in --detail and the csv).

Usage:
  python3 -m pexdb.check_availability $PEX_DB_DIR                   # table: one row per shot
  python3 -m pexdb.check_availability shot_87672.h5 --detail        # reasons, parts, % finite data
  python3 -m pexdb.check_availability $PEX_DB_DIR --diag TS CXRS    # only these columns
  python3 -m pexdb.check_availability $PEX_DB_DIR --diag TS --list  # shots where TS is not OK
  python3 -m pexdb.check_availability $PEX_DB_DIR --csv status.csv

--diag takes the column names of the table (BOLO, FIR, VALVE, TS, LP, HIR, VIR, BAR, APG, MANTIS,
LIUQE, CXRS, DSS, HEAT, WALL), IDS group names, or any path inside the file
(e.g. spectrometer_visible/sys12). Exit status: 0 if every requested diagnostic is OK or PARTIAL in
every file (with --strict: only OK), 1 otherwise, so it can be used in scripts.

From Python:
  from pexdb import check_availability as ca
  ca.check_file("shot_87672.h5")            # {'TS': {'status': 'OK', 'reason': '', 'finite': 0.8, ...}, ...}
"""
import argparse
import csv
import glob
import os
import sys

import h5py
import numpy as np

# default columns: label -> path inside the file
COLUMNS = {
    "BOLO": "bolometer", "FIR": "interferometer", "VALVE": "gas_injection", "TS": "thomson_scattering",
    "LP": "langmuir_probes", "HIR": "camera_ir/HIR", "VIR": "camera_ir/VIR", "BAR": "barometry/barometer",
    "APG": "barometry/pressure_gauges", "MANTIS": "camera_visible", "LIUQE": "equilibrium",
    "CXRS": "charge_exchange", "DSS": "spectrometer_visible", "HEAT": "heating", "WALL": "wall",
}
ALIASES = {"HEATING": "HEAT", "IR": "HIR", "TCV": None}
# not measurements: coordinates, time stamps, geometry. Ignored when deciding whether data are present.
COORD_DATASETS = {"t", "t_sample", "measured", "r", "z", "probes", "chn", "channel_name", "ch_ok", "rho",
                  "distance_separatrix_midplane", "b_field_angle", "etendue", "r_values", "z_values",
                  "r_mean", "z_mean", "area", "width_x1", "width_x2", "surface"}
GEOMETRY_GROUPS = {"position", "grid", "line_of_sight", "detector", "aperture", "port", "exit_position",
                   "outline", "tiles", "vessel_inner", "vessel_outer"}
SYMBOL = {"OK": "#", "PARTIAL": "~", "EMPTY": "0", "MISSING": "."}
MAX_READ = 2_000_000          # elements read per dataset (large ones are subsampled along axis 0)


def _finite_fraction(ds):
    """Fraction of finite values of a float dataset (subsampled if large); None if not numeric."""
    if ds.dtype.kind != "f" or ds.size == 0:
        return None
    if ds.size > MAX_READ and ds.ndim >= 1 and ds.shape[0] > 1:
        step = int(np.ceil(ds.size / MAX_READ))
        a = ds[::step]
    else:
        a = ds[()]
    return float(np.isfinite(a).mean())


def _walk(g, path, acc):
    """Collect finite fractions of the measurement datasets under g, and the unavailable parts."""
    for name, obj in g.items():
        p = f"{path}/{name}" if path else name
        if isinstance(obj, h5py.Group):
            if obj.attrs.get("available", 1) == 0:
                acc["missing"].append((p, str(obj.attrs.get("reason", "?"))))
            elif name not in GEOMETRY_GROUPS:
                _walk(obj, p, acc)
        elif name not in COORD_DATASETS:
            fr = _finite_fraction(obj)
            if fr is not None:
                acc["finite"].append((p, fr))


def assess(f, path):
    """Status of the group `path` in the open file f:
    {'status', 'reason', 'finite' (mean finite fraction of the measurement data or None),
     'parts_missing' [(path, reason)], 'n_data' (number of measurement datasets)}."""
    node = f
    for part in path.split("/"):
        if part not in node:
            return {"status": "MISSING", "reason": "not in the file", "finite": None, "parts_missing": [], "n_data": 0}
        node = node[part]
        if isinstance(node, h5py.Group) and node.attrs.get("available", 1) == 0:
            return {"status": "MISSING", "reason": str(node.attrs.get("reason", "?")), "finite": None,
                    "parts_missing": [], "n_data": 0}
    acc = {"finite": [], "missing": []}
    if isinstance(node, h5py.Group):
        _walk(node, path, acc)
    fr = [x for _, x in acc["finite"]]
    res = {"status": "OK", "reason": "", "finite": float(np.mean(fr)) if fr else None,
           "parts_missing": acc["missing"], "n_data": len(fr)}
    if fr and max(fr) == 0.0:
        res["status"], res["reason"] = "EMPTY", "all measurement datasets are NaN"
    elif acc["missing"]:
        res["status"] = "PARTIAL"
        res["reason"] = "; ".join(f"{p.rsplit('/', 1)[-1]}: {r}" for p, r in acc["missing"])
    return res


def check_file(path, columns=None):
    """{label: assess(...)} for the columns (default COLUMNS) of one file."""
    with h5py.File(path, "r") as f:
        return {lab: assess(f, p) for lab, p in (columns or COLUMNS).items()}


def file_info(path):
    with h5py.File(path, "r") as f:
        a = f.attrs
        return {"shot": int(a["shot"]) if "shot" in a else os.path.basename(path),
                "time_base": str(a.get("time_mode", "default")), "mode": str(a.get("regime", "?")) or "?",
                "baffle": str(a.get("baffle", "?")) or "?",
                "elmy": ("yes" if a.get("elmy", 0) else "no") if "elmy" in a else "?"}


def resolve_columns(names):
    """--diag names -> {label: path}."""
    if not names:
        return dict(COLUMNS)
    out = {}
    for n in names:
        key = ALIASES.get(n.upper(), n.upper())
        if key in COLUMNS:
            out[key] = COLUMNS[key]
        else:
            out[n] = n                                      # an IDS group name or a path
    return out


def find_files(sources):
    files = []
    for s in sources:
        if os.path.isdir(s):
            files += sorted(glob.glob(os.path.join(s, "shot_*.h5")))
        else:
            files += sorted(glob.glob(s)) or [s]
    return files


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("source", nargs="+", help="shot_*.h5 file(s), folder(s) or glob(s)")
    ap.add_argument("--diag", nargs="+", help="only these diagnostics (table names, IDS groups or paths)")
    ap.add_argument("--detail", action="store_true", help="per file: status, % finite data, missing parts, reasons")
    ap.add_argument("--list", action="store_true", help="list the shots where a requested diagnostic is not OK")
    ap.add_argument("--csv", help="write the statuses to this csv file")
    ap.add_argument("--strict", action="store_true", help="exit status 1 also for PARTIAL / EMPTY")
    args = ap.parse_args(argv)

    cols = resolve_columns(args.diag)
    files = find_files(args.source)
    if not files:
        sys.exit("no shot_*.h5 files found")
    rows = []
    for fn in files:
        try:
            rows.append((fn, file_info(fn), check_file(fn, cols)))
        except (OSError, KeyError) as e:
            print(f"  cannot read {fn}: {e}")
    if not rows:
        sys.exit(1)

    # ---- table ----
    w = max(len(c) for c in cols) + 1
    head = (f"{'shot':>7s} {'mode':>4s} {'ELMy?':>5s} {'baffle':>6s} | " + " ".join(f"{c:>{w}s}" for c in cols)
            + " | file")
    print(head)
    print("-" * len(head))
    for fn, info, res in rows:
        print(f"{str(info['shot']):>7s} {info['mode']:>4s} {info['elmy']:>5s} {info['baffle']:>6s} | "
              + " ".join(f"{SYMBOL[res[c]['status']]:>{w}s}" for c in cols) + f" | {os.path.basename(fn)}")
    print("-" * len(head))
    print("#" + " ok   ~ partial   0 empty (available, all NaN)   . missing")
    cnt = {c: {k: 0 for k in SYMBOL} for c in cols}
    for _, _, res in rows:
        for c in cols:
            cnt[c][res[c]["status"]] += 1
    print(f"{'files ok':>22s} | " + " ".join(f"{cnt[c]['OK']:>{w}d}" for c in cols) + f" | of {len(rows)} files")

    # ---- details ----
    if args.detail:
        for fn, info, res in rows:
            print(f"\n{os.path.basename(fn)}  (shot {info['shot']}, {info['mode']}-mode, ELMy {info['elmy']}, baffle {info['baffle']}, time base {info['time_base']})")
            for c, r in res.items():
                fr = "" if r["finite"] is None else f"{100 * r['finite']:5.1f}% finite"
                print(f"  {c:8s} {r['status']:8s} {fr:14s} {r['reason']}")
    if args.list:
        print()
        for c in cols:
            bad = [str(i['shot']) for _, i, res in rows if res[c]["status"] != "OK"]
            print(f"{c}: not OK in {len(bad)} of {len(rows)} files" + (f": {', '.join(bad)}" if bad else ""))

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["file", "shot", "mode", "elmy", "baffle", "time_base"] + [f"{c}" for c in cols]
                        + [f"{c}_finite" for c in cols])
            for fn, info, res in rows:
                wr.writerow([os.path.basename(fn), info["shot"], info["mode"], info["elmy"], info["baffle"], info["time_base"]]
                            + [res[c]["status"] for c in cols]
                            + ["" if res[c]["finite"] is None else f"{res[c]['finite']:.3f}" for c in cols])
        print(f"\nwrote {args.csv}")

    ok = ("OK",) if args.strict else ("OK", "PARTIAL")
    sys.exit(0 if all(res[c]["status"] in ok for _, _, res in rows for c in cols) else 1)


if __name__ == "__main__":
    main()
