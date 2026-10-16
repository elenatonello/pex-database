"""Reader for the shot_<N>.h5 databases written by build_database.py, plus colours.

    from pexdb import shotdb
    shots = shotdb.load("shots/")                 # folder, glob, file or list
    shots[0]["thomson_scattering/te"]             # (40, 117) array, NaN = no data
    shots[0].has("charge_exchange/Ti/data")       # False if absent/unavailable

ELM-mode files (chosen automatically for ELMy H-modes) have two time grids: heating, valve and
pressure gauges and IR on `t` (0.1 ms); every other diagnostic on `t_slow` (5 ms). TS, CXRS, MANTIS and
DSS hold each real sample on the nearest frames (see the group's `t_sample` / `measured`); the rest
(bolometer, FIR, LIUQE) are means over each 5 ms bin. By default (grid="common") a Shot presents
slow data on the fast grid too, so any mix of diagnostics can be plotted against `t`:

    shots = shotdb.load("shots/", grid="native")   # data on their own grids instead
    s.time_of("thomson_scattering/te")             # time axis matching s["thomson_scattering/te"]

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
    def __init__(self, path, index=0, grid="common"):
        if grid not in ("common", "native"):
            raise ValueError("grid must be 'common' or 'native'")
        self.path, self.index, self.grid = path, index, grid
        self._cache = {}
        self._slow_cache = {}
        self._warned = set()
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
            self.time_mode = str(a.get("time_mode", "default"))          # 'default' or 'elm'
            self.t_slow = f["t_slow"][()] if "t_slow" in f else None
            self.elm_times = f["elm_times"][()] if "elm_times" in f else np.array([])   # [s], ELMy shots
            self.diagnostics = {k: bool(f[k].attrs.get("available", 1))
                                for k in f if isinstance(f[k], h5py.Group)}

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

    def _read(self, path):
        with h5py.File(self.path, "r") as f:
            a = f[path.strip("/")][()]
        if isinstance(a, np.ndarray) and a.dtype.kind == "O":               # string arrays
            a = np.array([x.decode() if isinstance(x, bytes) else x for x in a])
        return a

    def __getitem__(self, path):
        if path not in self._cache:
            if not self.has(path):
                raise Unavailable(f"shot {self.shot}: '{path}' not available")
            a = self._read(path)
            self._cache[path] = self._to_common(path, a) if self._is_common_view(path) else a
        return self._cache[path]

    # ---- time axes ----------------------------------------------------------
    def _ancestors(self, path):
        """Group paths containing `path`, deepest first, root excluded."""
        parts = path.strip("/").split("/")[:-1]
        return ["/".join(parts[:i]) for i in range(len(parts), 0, -1)]

    def t_of(self, path):
        """Native time vector of the group that holds `path` (nearest enclosing group with a
        't' dataset); the root `t` if there is none."""
        with h5py.File(self.path, "r") as f:
            for g in self._ancestors(path):
                if "t" in f[g] and isinstance(f[g]["t"], h5py.Dataset):
                    return f[g]["t"][()]
        return self.t

    def _slow_group(self, path):
        """Group (path) on the slow grid that `path` belongs to, or None. A slow group has a `t`
        equal to `t_slow`; it also has t_sample / measured if its samples are held (TS, CXRS, IR,
        MANTIS, DSS), otherwise (bolometer, FIR, LIUQE) its values are means over each slow bin."""
        if path not in self._slow_cache:
            found = None
            if self.time_mode == "elm" and self.t_slow is not None:
                with h5py.File(self.path, "r") as f:
                    for g in self._ancestors(path):
                        if "t" in f[g] and isinstance(f[g]["t"], h5py.Dataset):
                            tg = f[g]["t"][()]
                            if tg.size == self.t_slow.size and np.allclose(tg, self.t_slow):
                                found = g
                            break                        # the nearest group with a `t` decides
            self._slow_cache[path] = found
        return self._slow_cache[path]

    def _is_common_view(self, path):
        return self.grid == "common" and self._slow_group(path) is not None

    def time_of(self, path):
        """Time axis matching `self[path]`: the fast grid for slow data in the common view,
        otherwise the native time vector of the group."""
        return self.t if self._is_common_view(path) else self.t_of(path)

    def _to_common(self, path, a):
        """Slow-grid data -> fast grid: every fast frame takes the real sample nearest in time
        (no interpolation); NaN farther than 2 sampling steps. `t_sample` / `measured` are
        recomputed on the fast grid. Arrays without the slow time axis are returned as is."""
        g = self._slow_group(path)
        name = path.rsplit("/", 1)[-1]
        t_slow_grp = self._read(g + "/t")
        with h5py.File(self.path, "r") as f:
            held = "t_sample" in f[g]
        if not held:                                       # bin means: repeat the nearest slow frame
            if name == "t":
                return self.t
            axes = [i for i, n in enumerate(np.shape(a)) if n == t_slow_grp.size]
            if not axes or a.dtype.kind not in "fiu":
                return a
            k = np.clip(np.rint((self.t - t_slow_grp[0]) / np.median(np.diff(t_slow_grp))).astype(int),
                        0, t_slow_grp.size - 1)
            return np.take(a, k, axis=axes[0])
        tsam, meas = self._read(g + "/t_sample"), self._read(g + "/measured").astype(bool)
        src = np.flatnonzero(meas)                         # one frame per real sample
        if src.size == 0:
            return a
        ts_src = tsam[src]
        pos = np.searchsorted(ts_src, self.t)
        lo, hi = np.clip(pos - 1, 0, ts_src.size - 1), np.clip(pos, 0, ts_src.size - 1)
        j = np.where(np.abs(self.t - ts_src[lo]) <= np.abs(ts_src[hi] - self.t), lo, hi)
        step = np.median(np.diff(ts_src)) if ts_src.size > 1 else np.inf
        ok = np.abs(self.t - ts_src[j]) <= 2 * step
        if name == "t":
            return self.t
        if name == "t_sample":
            return np.where(ok, ts_src[j], np.nan)
        if name == "measured":                             # fast frame nearest to each sample
            out = np.zeros(self.t.size, np.uint8)
            near = np.argmin(np.abs(self.t[:, None] - ts_src[None, :]), axis=0)
            out[near] = 1
            return out
        axes = [i for i, n in enumerate(np.shape(a)) if n == t_slow_grp.size]
        if not axes or a.dtype.kind not in "fiu":
            return a
        ax = axes[0]
        b = np.moveaxis(np.asarray(a, dtype=float), ax, 0)
        out = np.full((self.t.size,) + b.shape[1:], np.nan)
        out[ok] = b[src[j[ok]]]
        return np.moveaxis(out, 0, ax)

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

    def covers(self, time, t=None):
        """True if `time` [s] lies inside the time axis `t` (default: the root axis), allowing one
        sampling step of tolerance at the ends."""
        t = self.t if t is None else np.asarray(t)
        tol = float(np.median(np.diff(t))) if t.size > 1 else 0.0
        return bool(t[0] - tol <= time <= t[-1] + tol)

    def frame(self, time, path=None):
        """Index of the frame closest to `time` [s], in the root time base or, if `path` is
        given, in the time axis matching self[path] (see time_of). A time outside the data
        (e.g. 1.5 s in a file that ends at 1.0 s) gives the nearest end frame, with a warning."""
        t = self.t if path is None else self.time_of(path)
        k = int(np.argmin(abs(t - time)))
        if not self.covers(time, t):
            key = (round(float(time), 4), float(t[0]), float(t[-1]))
            if key not in self._warned:
                self._warned.add(key)
                print(f"  WARNING: shot {self.shot}: t = {time:g} s is outside the data "
                      f"({t[0]:.3f}-{t[-1]:.3f} s); using the nearest frame, t = {t[k]:.3f} s")
        return k

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


def load(source, grid="common"):
    """Load shots from a file, folder, glob pattern, or list of those.
    grid='common' (default): slow data of ELM-mode files are shown on the fast grid;
    grid='native': every group on its own time axis (see Shot.time_of)."""
    items = [source] if isinstance(source, (str, os.PathLike)) else list(source)
    paths = []
    for it in items:
        it = str(it)
        if os.path.isdir(it):
            paths += sorted(glob.glob(os.path.join(it, "shot_*.h5")))
        else:
            paths += sorted(glob.glob(it)) or [it]
    return [Shot(p, i, grid) for i, p in enumerate(paths)]
