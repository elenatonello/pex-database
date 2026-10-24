"""CIII front position along the divertor legs, computed from the MANTIS emissivity.

The database stores the reconstructed CIII emissivity (camera_visible/CIII_cam1/emi on the triangle mesh)
and the equilibrium; the front is NOT stored, because its definition involves choices (below).

    from pexdb import shotdb, ciii_front
    s = shotdb.load("shot_88306.h5")[0]
    r = ciii_front.ciii_front(s)            # MANTIS front, profiles along the legs, leg lengths
    r["outer"], r["l"]["outer"]             # front [m from the target] and leg length
    r = ciii_front.ciii_front(s, dss=True)           # + front from the DSS CIII chords: r["dss"]
    r = ciii_front.ciii_front(s, uncertainty=True)   # + 16-84 % band over analysis variants (~ +40 s)

Plots of what is computed: pexdb.plotting.ciii_plots (profiles along the legs, emissivity frames).

Method, for each time frame and each leg ('inner', 'outer'):
 1. Branches. The lower separatrix (psi = 0 of the LIUQE map) below the X-point gives the two legs.
    Each leg is continued above the X-point along the core separatrix on its OWN side of the X-point
    (inner leg -> high-field side, outer leg -> low-field side), for UP = 15 cm. The coordinate s is the
    distance from the target along the branch; the X-point is at s = l (leg length).
 2. Profile. Every mesh triangle within W of a branch is given to the NEAREST branch only (nothing is counted
    twice near the X-point) and binned in s (bins DS). Each bin holds the volume-weighted mean emissivity
    of the band (weight = triangle area x R, the toroidal ring volume / 2 pi). Units: those of `emi`.
 3. Smoothing: median over TMED frames in time (3), gaussian SIGMA in s.
 4. Front. The reference is the maximum of the leg SHOULDER (s <= l - XZ: the zone around the X-point, where
    a narrow spike usually sits, is ignored). The front is the FRAC (50 %) point on the target side of that
    reference, linear interpolation between bins. If the shoulder is weaker than ALPHA times the overall
    maximum (leg emission gone, emission concentrated at the X-point or above it), the overall maximum is the
    reference (xmode = 1) and the front can be at the X-point (s ~ l) or above it (s > l).
 Choices that are not set by the data: W, DS, SIGMA, TMED, XZ, ALPHA, UP (the 50 % is fixed). The
 uncertainty is the spread of the front over an ensemble of these choices (ENSEMBLE).

DSS. The same quantity from the divertor spectrometer (CIII triplet, 30 chords crossing the outer leg): the
chord positions stored with the data (where each chord crosses the outer separatrix) are projected on the
outer branch, and the front is the 50 % point on the target side of the maximum of the profile, using only
the chords whose line of sight does not cross the inner leg and stays > 3 cm from the X-point (the others
see the X-point region and the inner leg). The highest such chord (`s_top`, ~0.33 m) limits what the DSS can
see: where the profile peaks on it (`at_limit`; true in most detached frames, the profile keeps rising
towards the X-point) the 50 % level is taken relative to that chord, so the true front may be higher. As a cross-check the
MANTIS emissivity is integrated along the DSS lines of sight (poloidal-plane chords, nearest-triangle
sampling) and treated the same way (`front_synthetic`). Outer leg only.

Not verified against an independent measurement of the front; compared with DSS CIII chords, see
README / the DSS notes. The X-point is read from equilibrium/{r_xpt, z_xpt} when the database holds it,
otherwise computed from psi (saddle point below the axis; agrees with LIUQE to a few mm in the flat top).
"""
import numpy as np
import h5py
import contourpy
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.interpolate import RectBivariateSpline
from scipy.spatial import cKDTree

W = 0.03          # half-width of the band around the separatrix [m]
DS = 0.01         # bin size along the leg [m]
SIGMA = 1.5       # gaussian smoothing along s [bins of DS]
TMED = 3          # median filter in time [frames]
FRAC = 0.5
UP = 0.15         # [m] how far the coordinate continues above the X-point, along the core separatrix
SMAX = 0.65       # [m] length of the profile array
XZ = 0.03         # [m] zone below the X-point ignored when looking for the maximum of the leg shoulder
ALPHA = 0.25      # shoulder maximum / overall maximum below which the X-point peak is the reference


def branches(L, k, n=600, up=UP):
    """Extended branches of the lower-single-null separatrix at LIUQE frame k.

    dict name -> (r, z, l): path ordered TARGET -> X-point -> up along the core separatrix, with
    l = arc length from the target end to the X-point; the X-point is under the key 'xpt'.
    Each leg continues on its OWN side of the X-point (inner leg -> high-field side, outer leg ->
    low-field side). The opposite choice (straight through the X-point) was tried and rejected: the
    bright emission left of the X-point then went to the outer profile and made the outer front jump
    to the X-point while the outer leg was still emitting down to the target."""
    psi = L.psi[:, :, k]
    spl = RectBivariateSpline(L.R, L.Z, psi)
    rr, zz = np.linspace(L.R[0], L.R[-1], n), np.linspace(L.Z[0], L.Z[-1], n)
    lines = contourpy.contour_generator(rr, zz, spl(rr, zz).T).lines(0.0)
    i = int(np.argmin(np.asarray(L.z_xpts[:, k])))
    rx, zx = float(L.r_xpts[i, k]), float(L.z_xpts[i, k])
    pts = np.vstack([np.asarray(l) for l in lines if len(l)]) if len(lines) else np.empty((0, 2))
    dist = np.hypot(pts[:, 0] - rx, pts[:, 1] - zx)
    out = {"xpt": (rx, zx)}
    for name, side in (("inner", pts[:, 0] < rx), ("outer", pts[:, 0] >= rx)):
        below = pts[side & (pts[:, 1] < zx - 0.004)]
        above = pts[side & (pts[:, 1] > zx + 0.004) & (dist < 1.6 * up)]
        if len(below) < 3:
            continue
        below = below[np.argsort(np.hypot(below[:, 0] - rx, below[:, 1] - zx))]       # X-point -> target
        above = above[np.argsort(np.hypot(above[:, 0] - rx, above[:, 1] - zx))]       # X-point -> up
        leg = np.vstack([[rx, zx], below])[::-1]                                       # target -> X-point
        l = float(np.sum(np.hypot(*np.diff(leg, axis=0).T)))
        path = np.vstack([leg, above]) if len(above) else leg
        seg = np.r_[0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
        keep = seg <= l + up
        out[name] = (path[keep, 0], path[keep, 1], l)
    return out


def separatrix_lines(L, k, n=600):
    """The whole separatrix at LIUQE frame k: list of (r, z) polylines of the psi = 0 contour of the equilibrium
    map (confined-plasma boundary and both divertor legs down to the edge of the map). `branches` uses only
    the two legs and their continuation above the X-point; this is for drawing."""
    spl = RectBivariateSpline(L.R, L.Z, L.psi[:, :, k])
    rr, zz = np.linspace(L.R[0], L.R[-1], n), np.linspace(L.Z[0], L.Z[-1], n)
    lines = contourpy.contour_generator(rr, zz, spl(rr, zz).T).lines(0.0)
    return [np.asarray(l_) for l_ in lines if len(l_) > 1]


def point_at(branch, s):
    """(r, z) at distance s from the target along an extended branch (None if s is not finite)."""
    if not np.isfinite(s):
        return None
    r, z, _ = branch
    arc = np.r_[0, np.cumsum(np.hypot(np.diff(r), np.diff(z)))]
    s = np.clip(s, 0, arc[-1])
    return np.interp(s, arc, r), np.interp(s, arc, z)


def leg_profiles(C, L, w=None, ds=None, br_cache=None):
    """Emission profiles along the two extended branches (see branches). Every mesh triangle is given to
    ONE branch only (the nearest one, if within W of it), so nothing is counted twice near the X-point.
    Returns t, s_centres [m from the target], I = {leg: (nbins, nt)} (mean emissivity in the band,
    volume weighted), length = {leg: l (target -> X-point)}, ext = {leg: l + the part above the X-point}."""
    w = W if w is None else w
    ds = DS if ds is None else ds
    br_cache = {} if br_cache is None else br_cache
    g = C.grid
    cen = np.column_stack([g.R_mean, g.Z_mean])
    wgt = g.Area * g.R_mean                      # toroidal ring volume of a triangle / 2 pi
    emi = np.asarray(C.emi)
    t = np.asarray(C.t)
    lk = np.array([int(np.argmin(abs(L.t - tt))) for tt in t])
    nb = int(SMAX / ds + 1e-9)
    sc = (np.arange(nb) + 0.5) * ds
    names = ("inner", "outer")
    I = {n: np.full((nb, t.size), np.nan) for n in names}
    length = {n: np.full(t.size, np.nan) for n in names}
    ext = {n: np.full(t.size, np.nan) for n in names}
    cache = {}
    for i in range(t.size):
        k = lk[i]
        if k not in cache:
            if k not in br_cache:
                br_cache[k] = branches(L, k)
            br = br_cache[k]
            pts, lab, sco, ls, es = [], [], [], {}, {}
            for j, name in enumerate(names):
                if name not in br:
                    continue
                r, z, l = br[name]
                arc0 = np.r_[0, np.cumsum(np.hypot(np.diff(r), np.diff(z)))]
                arc = np.arange(0, arc0[-1], 0.0005)                       # dense, 0.5 mm
                pts.append(np.column_stack([np.interp(arc, arc0, r), np.interp(arc, arc0, z)]))
                lab.append(np.full(arc.size, j))
                sco.append(arc)
                ls[name], es[name] = l, arc0[-1]
            tab = None
            if pts:
                P, lab, sco = np.vstack(pts), np.concatenate(lab), np.concatenate(sco)
                d, jn = cKDTree(P).query(cen)
                sel = d < w
                tab = (sel, lab[jn], sco[jn], ls, es)
            cache[k] = tab
        tab = cache[k]
        if tab is None:
            continue
        sel, lb, s_all, ls, es = tab
        for j, name in enumerate(names):
            if name not in ls:
                continue
            m = sel & (lb == j)
            length[name][i], ext[name][i] = ls[name], es[name]
            b = np.clip((s_all[m] / ds).astype(int), 0, nb - 1)
            num = np.bincount(b, weights=emi[m, i] * wgt[m], minlength=nb)
            den = np.bincount(b, weights=wgt[m], minlength=nb)
            with np.errstate(invalid="ignore", divide="ignore"):
                I[name][:, i] = num / den                         # mean emissivity in the band
    return t, sc, I, length, ext


def estimate(sc, prof, smax, sigma=None, frac=None, l=None, xz=None, alpha=None):
    """peak, half_up, half_dn, cog, xmode from one profile (nb,) within [0, smax].

    The front (half_dn) is the FRAC (50 %) point on the target side of the reference peak.
    Reference peak: the maximum on the leg shoulder, i.e. on s <= l - xz (the zone around the X-point,
    where the X-point spike sits, is ignored), as long as it is at least `alpha` times the overall maximum.
    If the shoulder is weaker than that (leg emission gone, the emission is concentrated at the X-point or
    above it) the overall maximum is the reference (xmode = 1) and the front can be at the X-point.
    l = None: always the overall maximum (previous method)."""
    sigma = SIGMA if sigma is None else sigma
    frac = FRAC if frac is None else frac
    xz = XZ if xz is None else xz
    alpha = ALPHA if alpha is None else alpha
    p = gaussian_filter1d(np.nan_to_num(prof), sigma)
    m = sc <= smax
    if not m.any() or p[m].max() <= 0:
        return (np.nan,) * 5
    ps, ss = p[m], sc[m]
    ip_all = int(np.argmax(ps))
    ip, xmode = ip_all, 1.0
    if l is not None:
        leg = ss <= l - xz
        if leg.any():
            ip_leg = int(np.argmax(np.where(leg, ps, -np.inf)))
            if ps[ip_leg] >= alpha * ps[ip_all]:
                ip, xmode = ip_leg, 0.0
    thr = frac * ps[ip]
    up = np.flatnonzero(ps[ip:] < thr)

    def cross(j_hi, j_lo):                      # linear interpolation of the threshold between two bins
        a, b = ps[j_hi], ps[j_lo]
        return ss[j_hi] + (a - thr) / (a - b) * (ss[j_lo] - ss[j_hi]) if a != b else ss[j_lo]
    half_up = cross(ip + up[0] - 1, ip + up[0]) if up.size else ss[-1]
    dn = np.flatnonzero(ps[:ip + 1][::-1] < thr)
    half_dn = cross(ip - dn[0] + 1, ip - dn[0]) if dn.size else 0.0
    return ss[ip], half_up, half_dn, np.sum(ps * ss) / np.sum(ps), xmode


def lower_xpoint(R, Z, psi, z_axis, n=500):
    """Lower X-point (r, z) of a LIUQE flux map psi(R, Z): the saddle (min |grad psi|) below the
    magnetic axis, among the points where psi is close to the boundary value (0)."""
    spl = RectBivariateSpline(R, Z, psi)
    rr, zz = np.linspace(R[0], R[-1], n), np.linspace(Z[0], Z[-1], n)
    RR, ZZ = np.meshgrid(rr, zz, indexing="ij")
    g2 = spl.ev(RR, ZZ, dx=1) ** 2 + spl.ev(RR, ZZ, dy=1) ** 2
    pv = spl.ev(RR, ZZ)
    cost = g2 / (np.nanmax(g2) + 1e-30) + (pv / (0.05 * (np.nanmax(np.abs(psi)) + 1e-30))) ** 2
    cost[ZZ > z_axis - 0.15] = np.inf
    k = np.unravel_index(np.argmin(cost), cost.shape)
    return rr[k[0]], zz[k[1]]


def _native(s):
    """The same shot with every group on its own time axis (needed for ELM-mode files, whose common view
    would put the MANTIS maps on the 4001-frame fast grid). Files in the default mode are unaffected."""
    if getattr(s, "grid", "native") == "native":
        return s
    return type(s)(s.path, s.index, "native")


class _NS:
    """Plain attribute container (what load() returns for the .mat structs)."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def build_CL(s):
    s = _native(s)
    """MANTIS (C) and LIUQE (L) objects, as load() gives them, from a pexdb Shot (database file);
    L also carries the lower X-point (r_xpts, z_xpts) computed from psi."""
    c = "camera_visible/CIII_cam1/"
    tc = np.asarray(s.time_of(c + "emi"), float)
    grid = _NS(R_mean=s[c + "grid/r_mean"], Z_mean=s[c + "grid/z_mean"], Area=s[c + "grid/area"],
               R_values=s[c + "grid/r_values"], Z_values=s[c + "grid/z_values"])
    C = _NS(grid=grid, emi=np.asarray(s[c + "emi"], float), t=tc,
            **{n: np.asarray(s[c + n], float) for n in ("sFrontInner", "sFrontOuter", "lInner", "lOuter", "vpar")
               if s.has(c + n)})        # legacy stored fronts (old files only); vpar if the file has it
    te = np.asarray(s.time_of("equilibrium/psi"), float)
    psi = np.asarray(s["equilibrium/psi"], float)                      # (R, Z, t)
    R, Z = np.asarray(s["equilibrium/r"], float), np.asarray(s["equilibrium/z"], float)
    if s.has("equilibrium/r_xpt") and s.has("equilibrium/z_xpt"):       # stored by the database build
        xr, xz_ = np.asarray(s["equilibrium/r_xpt"], float), np.asarray(s["equilibrium/z_xpt"], float)
        bad = ~np.isfinite(xr) | ~np.isfinite(xz_)
    else:
        xr, xz_, bad = np.full(te.size, np.nan), np.full(te.size, np.nan), np.ones(te.size, bool)
    if bad.any():                                                        # older files, or no X-point stored
        zax = np.asarray(s["equilibrium/z_axis"], float)
        for k in np.flatnonzero(bad):
            xr[k], xz_[k] = lower_xpoint(R, Z, psi[:, :, k], zax[k])
    L = _NS(psi=psi, R=R, Z=Z, t=te, r_xpts=np.vstack([xr, xr]), z_xpts=np.vstack([xz_, xz_]))
    return C, L


def ciii_front(s, w=None, ds=None, sigma=None, tmed=None, xz=None, alpha=None, dss=False,
                uncertainty=False):
    """CIII front of a pexdb Shot, computed from the MANTIS emissivity (this module's method).

    Returns a dict:
      t, s            time [s] of the MANTIS frames; bin centres [m from the target] of the profiles
      I[leg]          (nbins, nt) mean emissivity along the branch (leg = 'inner' | 'outer')
      l[leg], ext[leg]  length of the leg (target -> X-point) and of the extended branch [m]
      inner, outer    CIII front [m from the target]; > l[leg] means above the X-point
      peak_*, xmode_* position of the reference peak; 1 where the X-point peak is the reference
                      (shoulder gone), 0 where the reference is the maximum of the leg shoulder
      stored          sFront* stored in OLD database files (legacy, may be NaN / wrong), else NaN
      dss             only with dss=True: outer-leg front from the DSS CIII triplet, see dss_front
                      (None otherwise, or if the shot has no DSS CIII)
      <leg>_lo, _hi, _med, _alt   only with uncertainty=True: 16th / 84th percentile and median of the
                      front over the ENSEMBLE of analysis variants, and the median of the previous rule
                      (reference = overall maximum); [m from the target], see ciii_front_uncertainty
    Needs camera_visible/CIII_cam1 and equilibrium/{psi,r,z,z_axis} (+ r_xpt, z_xpt if stored).
    Optional w, ds, sigma [bins], tmed [frames], xz, alpha override the module constants (W, DS, SIGMA,
    TMED, XZ, ALPHA); the 50 % threshold (FRAC) is fixed.
    By default only the MANTIS front is computed (a few seconds per shot). dss=True adds the DSS front
    (~ +5 s); uncertainty=True adds the ensemble band (~ +40 s; the ensemble varies the analysis choices
    itself, whatever w, ds, sigma, ... are passed)."""
    s = _native(s)
    c = "camera_visible/CIII_cam1/"
    C, L = build_CL(s)
    cache = {}
    t, sc, I, length, ext = leg_profiles(C, L, w=w, ds=ds, br_cache=cache)
    out = {"t": t, "s": sc, "I": I, "l": length, "ext": ext,
           "stored": {"inner": getattr(C, "sFrontInner", np.full(t.size, np.nan)),
                      "outer": getattr(C, "sFrontOuter", np.full(t.size, np.nan))}}
    for name in ("inner", "outer"):
        tm = TMED if tmed is None else tmed
        P = median_filter(I[name], size=(1, tm)) if tm > 1 else I[name]
        est = np.array([estimate(sc, P[:, i], ext[name][i], sigma, None, length[name][i], xz, alpha)
                        for i in range(t.size)])
        out[name], out["peak_" + name], out["xmode_" + name] = est[:, 2], est[:, 0], est[:, 4]
    out["dss"] = dss_front(s, C, L, cache) if dss else None
    if uncertainty:
        ens = front_ensemble(C, L)
        alt = front_ensemble(C, L, global_max=True)
        for n in ("inner", "outer"):
            out[n + "_lo"], out[n + "_hi"] = np.nanpercentile(ens[n], [16, 84], axis=0)
            out[n + "_med"] = np.nanmedian(ens[n], axis=0)
            out[n + "_alt"] = np.nanmedian(alt[n], axis=0)
    return out


# ---------------------------------------------------------------------------
# DSS (divertor spectroscopy) CIII front
# ---------------------------------------------------------------------------
def dss_base(s):
    """Path of the DSS system that holds the CIII triplet, or None."""
    if not s.diagnostics.get("spectrometer_visible", False):
        return None
    with h5py.File(s.path, "r") as f:
        for k in sorted(f["spectrometer_visible"]):
            if k.startswith("sys") and "CIII_trip" in f["spectrometer_visible"][k]:
                return f"spectrometer_visible/{k}/"
    return None


def dss_los(s):
    """Line-of-sight end points (R, Z) of the 30 chords, in the order of the DSS DATA arrays (the stored
    line_of_sight geometry is in the opposite order: checked on the crossings with the separatrix)."""
    with h5py.File(s.path, "r") as f:
        g = f["spectrometer_visible/line_of_sight"]
        p1 = np.c_[g["first_point/r"][()], g["first_point/z"][()]]
        p2 = np.c_[g["second_point/r"][()], g["second_point/z"][()]]
    return p1[::-1], p2[::-1]


def synthetic_dss(C, p1, p2, n=700, tol=0.008):
    """B[j, i]: integral of the MANTIS emissivity along line of sight j at MANTIS frame i (poloidal-plane
    chords; each sample point takes the emissivity of the nearest triangle if closer than tol)."""
    g = C.grid
    tree = cKDTree(np.column_stack([g.R_mean, g.Z_mean]))
    u = np.linspace(0, 1, n)
    out = np.zeros((len(p1), C.emi.shape[1]))
    for j in range(len(p1)):
        pts = p1[j] + u[:, None] * (p2[j] - p1[j])
        d, idx = tree.query(pts)
        out[j] = np.nansum(C.emi[idx[d < tol]], axis=0) * (np.hypot(*(p2[j] - p1[j])) / n)
    return out


def chord_s(br, R, Z, maxd=0.03):
    """Distance from the target of the projection of the points (R, Z) on the branch br (NaN if farther
    than maxd or not finite)."""
    r, z, _ = br
    arc0 = np.r_[0, np.cumsum(np.hypot(np.diff(r), np.diff(z)))]
    arc = np.arange(0, arc0[-1], 0.0005)
    P = np.column_stack([np.interp(arc, arc0, r), np.interp(arc, arc0, z)])
    out = np.full(len(R), np.nan)
    ok = np.isfinite(R) & np.isfinite(Z)
    if ok.any():
        d, j = cKDTree(P).query(np.column_stack([R[ok], Z[ok]]))
        out[ok] = np.where(d < maxd, arc[j], np.nan)
    return out


def front_from_profile(sj, y, frac=None):
    """(front, at_limit) of a profile y(s) sampled at the points sj: the FRAC point on the target side of the
    maximum, linear interpolation between samples (0 if the profile never drops below it towards the target).
    at_limit: the maximum is on the sample with the largest s (the 50 % level is then relative to that sample; the front may be higher than seen).
    (NaN, False) if fewer than 5 valid samples."""
    frac = FRAC if frac is None else frac
    ok = np.isfinite(sj) & np.isfinite(y)
    if ok.sum() < 5 or np.nanmax(y[ok]) <= 0:
        return np.nan, False
    o = np.argsort(sj[ok])
    s_, y_ = sj[ok][o], y[ok][o]
    ip = int(np.argmax(y_))
    thr = frac * y_[ip]
    for k in range(ip, 0, -1):
        if y_[k - 1] < thr:
            return s_[k - 1] + (thr - y_[k - 1]) / (y_[k] - y_[k - 1]) * (s_[k] - s_[k - 1]), ip == len(y_) - 1
    return 0.0, ip == len(y_) - 1


def _seg_cross(p, q, path):
    """True if the segment p-q crosses the polyline path (n, 2)."""
    a, b = path[:-1], path[1:]

    def orient(u, v, w):
        return (v[..., 0] - u[..., 0]) * (w[..., 1] - u[..., 1]) - (v[..., 1] - u[..., 1]) * (w[..., 0] - u[..., 0])
    pp, qq = np.broadcast_to(p, a.shape), np.broadcast_to(q, a.shape)
    return bool(np.any((orient(pp, qq, a) * orient(pp, qq, b) < 0) & (orient(a, b, pp) * orient(a, b, qq) < 0)))


def _dist_point_seg(x, p, q):
    d = q - p
    u = np.clip(np.dot(x - p, d) / np.dot(d, d), 0, 1)
    return float(np.hypot(*(x - (p + u * d))))


def clean_chords(br_all, p1, p2, xmin=0.03):
    """Per chord (data order): True if its line of sight does not cross the inner branch and stays more than
    xmin from the X-point, i.e. it only sees the outer leg."""
    out = np.ones(len(p1), bool)
    xpt = np.array(br_all["xpt"])
    inner = np.column_stack(br_all["inner"][:2]) if "inner" in br_all else None
    for j in range(len(p1)):
        if inner is not None and _seg_cross(p1[j], p2[j], inner):
            out[j] = False
        if _dist_point_seg(xpt, p1[j], p2[j]) < xmin:
            out[j] = False
    return out


def dss_front(s, C=None, L=None, br_cache=None, synthetic=True):
    """CIII front of the outer leg from the DSS (see the module docstring), or None if the shot has no DSS
    CIII triplet. Returns {'t', 'front', 'front_synthetic', 's_top', 'at_limit'} on the DSS time axis
    (front, s_top in m from the target). C, L, br_cache: as built by ciii_front (computed if None)."""
    s = _native(s)
    base = dss_base(s)
    if base is None:
        return None
    if C is None or L is None:
        C, L = build_CL(s)
    br_cache = {} if br_cache is None else br_cache
    td, Id = np.asarray(s[base + "t"], float), np.asarray(s[base + "CIII_trip"], float)
    Rd, Zd = np.asarray(s[base + "r"], float), np.asarray(s[base + "z"], float)
    p1, p2 = dss_los(s)
    B = synthetic_dss(C, p1, p2) if synthetic else None
    out = {"t": td, "front": np.full(td.size, np.nan), "front_synthetic": np.full(td.size, np.nan),
           "s_top": np.full(td.size, np.nan), "at_limit": np.zeros(td.size, bool)}
    cleans = {}
    for i, tt in enumerate(td):
        k = int(np.argmin(abs(L.t - tt)))
        if k not in br_cache:
            br_cache[k] = branches(L, k)
        br = br_cache[k]
        if "outer" not in br:
            continue
        if k not in cleans:
            cleans[k] = clean_chords(br, p1, p2)
        sj = np.where(cleans[k], chord_s(br["outer"], Rd[:, i], Zd[:, i]), np.nan)
        out["front"][i], out["at_limit"][i] = front_from_profile(sj, Id[:, i])
        if B is not None:
            out["front_synthetic"][i], _ = front_from_profile(sj, B[:, int(np.argmin(abs(C.t - tt)))])
        out["s_top"][i] = np.nanmax(sj) if np.isfinite(sj).any() else np.nan
    return out


ENSEMBLE = {"w": (0.02, 0.03, 0.045), "ds": (0.007, 0.01, 0.014),       # band width, bin size [m]
            "sigma_m": (0.01, 0.015, 0.025), "tmed": (1, 3, 5),         # smoothing along s [m], in time [frames]
            "xz": (0.02, 0.03, 0.04), "alpha": (0.15, 0.25, 0.35)}      # X-point zone [m], "shoulder gone" criterion
# the threshold is NOT varied: the front is always the 50 % point (FRAC)


def front_ensemble(C, L, leg_list=("inner", "outer"), global_max=False):
    """Fronts of the ENSEMBLE variants: {leg: array (n_variants, nt)} [m from the target].
    global_max=True: the previous rule (reference = overall maximum), for comparison."""
    out = {n: [] for n in leg_list}
    cache = {}
    for w in ENSEMBLE["w"]:
        for ds in ENSEMBLE["ds"]:
            t, sc, I, length, ext = leg_profiles(C, L, w=w, ds=ds, br_cache=cache)
            for tm in ENSEMBLE["tmed"]:
                P = {n: (median_filter(I[n], size=(1, tm)) if tm > 1 else I[n]) for n in leg_list}
                for sg in ENSEMBLE["sigma_m"]:
                    for xz in (ENSEMBLE["xz"] if not global_max else (XZ,)):
                        for al in (ENSEMBLE["alpha"] if not global_max else (1e9,)):
                            for n in leg_list:
                                out[n].append([estimate(sc, P[n][:, i], ext[n][i], sg / ds, FRAC, length[n][i], xz, al)[2]
                                               for i in range(t.size)])
    return {n: np.array(v) for n, v in out.items()}


def ciii_front_uncertainty(s, **kw):
    """ciii_front(s, uncertainty=True, **kw): the MANTIS front plus, per leg, '<leg>_lo' / '<leg>_hi'
    (16th / 84th percentile over the ensemble of analysis variants), '<leg>_med' and '<leg>_alt' (median of
    the previous rule: reference = overall maximum) [m from the target]."""
    return ciii_front(s, uncertainty=True, **kw)


shot_front = ciii_front                      # aliases used by older scripts
front_uncertainty = ciii_front_uncertainty
