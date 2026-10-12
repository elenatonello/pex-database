"""Reader for the shot_<N>.h5 databases written by build_database.py, plus colours.

    from pexdb import shotdb
    shots = shotdb.load("shots/")                 # folder, glob, file or list
    shots[0]["thomson_scattering/te"]             # (40, 117) array, NaN = no data
    shots[0].has("charge_exchange/Ti/data")       # False if absent/unavailable

Colours follow the conventions: L-mode black, H-mode magenta; NINO and SILO
baffle colours as in the MATLAB scripts. Use `shot.color(by)` with by in
'regime' | 'baffle' | 'shot' | 'auto'.
"""
import glob
import os
import numpy as np
import h5py

REGIME_COLORS = {"L": "black", "H": "magenta"}
NINO_COL = (0 / 255, 114 / 255, 189 / 255)
SILO_COL = (230 / 255, 2 / 255, 55 / 255)
BAFFLE_COLORS = {"NINO": NINO_COL, "SILO": SILO_COL}
CYCLE = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
         "tab:brown", "tab:cyan", "tab:olive", "tab:gray", "tab:pink"]
MARKERS = ["o", "s", "^", "v", "D", "P", "X", "<", ">", "*"]


class Unavailable(KeyError):
    """Requested path is missing or belongs to an unavailable diagnostic."""


class Shot:
    def __init__(self, path, index=0):
        self.path, self.index = path, index
        self._cache = {}
        with h5py.File(path, "r") as f:
            a = f.attrs
            self.shot = int(a["shot"])
            self.t = f["t"][()]
            self.window = float(a.get("window", np.nan))
            self.description = str(a.get("description", ""))
            self.regime = str(a.get("regime", "")).upper()   # 'L', 'H' or ''
            self.baffle = str(a.get("baffle", "")).upper()   # 'NINO', 'SILO' or ''
            self.t_end = float(a["t_end"]) if "t_end" in a else np.nan   # [s]
            self.date = str(a.get("date", ""))               # YYYY-MM-DD (proxy, see date_source)
            self.diagnostics = {k: bool(f[k].attrs.get("available", 1))
                                for k in f if k != "t"}

    # ---- access -----------------------------------------------------------
    def has(self, path):
        parts = path.strip("/").split("/")
        if not self.diagnostics.get(parts[0], False):
            return False
        with h5py.File(self.path, "r") as f:
            node = f
            for p in parts:
                if p not in node or (isinstance(node[p], h5py.Group)
                                     and node[p].attrs.get("available", 1) == 0):
                    return False
                node = node[p]
        return True

    def __getitem__(self, path):
        if path not in self._cache:
            if not self.has(path):
                raise Unavailable(f"shot {self.shot}: '{path}' not available")
            with h5py.File(self.path, "r") as f:
                a = f[path.strip("/")][()]
                if isinstance(a, np.ndarray) and a.dtype.kind == "O":       # string arrays
                    a = np.array([x.decode() if isinstance(x, bytes) else x for x in a])
                self._cache[path] = a
        return self._cache[path]

    def attr(self, path, name, default=None):
        with h5py.File(self.path, "r") as f:
            return f[path.strip("/")].attrs.get(name, default)

    def wall(self):
        """dict name -> (r, z) of the stored wall contours (e.g. 'vessel', 'tiles');
        empty if none."""
        out = {}
        with h5py.File(self.path, "r") as f:
            if not self.diagnostics.get("wall", False):
                return out
            g = f["wall"]
            if "r" in g:                                  # old single-contour layout
                return {"vessel": (g["r"][()], g["z"][()])}
            for k, v in g.items():
                if isinstance(v, h5py.Group) and "r" in v:
                    out[k] = (v["r"][()], v["z"][()])
        return out

    def frame(self, time):
        """Index of the frame closest to `time` [s]."""
        return int(np.argmin(abs(self.t - time)))

    # ---- appearance -------------------------------------------------------
    @property
    def label(self):
        return f"#{self.shot}" + (f" {self.description}" if self.description else "")

    def color(self, by="auto"):
        if by == "auto":
            by = "regime" if self.regime in REGIME_COLORS else \
                 "baffle" if self.baffle in BAFFLE_COLORS else "shot"
        if by == "regime" and self.regime in REGIME_COLORS:
            return REGIME_COLORS[self.regime]
        if by == "baffle" and self.baffle in BAFFLE_COLORS:
            return BAFFLE_COLORS[self.baffle]
        return CYCLE[self.index % len(CYCLE)]

    @property
    def marker(self):
        """Distinguishes shots that share a colour (e.g. two L-mode shots)."""
        return MARKERS[self.index % len(MARKERS)]

    def __repr__(self):
        return (f"Shot({self.shot}, '{self.description}', regime={self.regime or '?'}, "
                f"baffle={self.baffle or '?'})")


def load(source):
    """Load shots from a file, folder, glob pattern, or list of those."""
    items = [source] if isinstance(source, (str, os.PathLike)) else list(source)
    paths = []
    for it in items:
        it = str(it)
        if os.path.isdir(it):
            paths += sorted(glob.glob(os.path.join(it, "shot_*.h5")))
        else:
            paths += sorted(glob.glob(it)) or [it]
    return [Shot(p, i) for i, p in enumerate(paths)]
