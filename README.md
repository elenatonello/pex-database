# PEX shot database

MATLAB fetches each shot from the TCV server; Python converts it to a uniform-in-time
HDF5 file `shot_<N>.h5` (one per shot). Data are never stored in this repo.

## Setup (server)
    git clone <repo> && cd <repo>
    pip install -e .          # numpy, scipy, h5py, matplotlib
    export PEX_DB_DIR=/to/the/database

## Build
In MATLAB (needs the dss/TCV paths on the MATLAB path):

    addpath(genpath('/home/tonell/matlab/git/dss')); addpath('matlab');
    build_pex_database([89437 89837], getenv('PEX_DB_DIR'));

Each shot's heavy struct is written to a temporary .mat, converted, then deleted.
Existing .h5 files are skipped unless `'Overwrite', true`.

### Time base: automatic L-mode / ELM-resolved
The builder decides itself (`'Mode', 'auto'`, the default): it looks for a regular train of ELMs
(20-600 Hz) in the divertor Jsat, divertor radiated power and dW_MHD/dt between 0.6 and 1.0 s.
Heating power is not used, so high-power L-modes are not mistaken for H-modes. The decision is stored
in the file (`elmy`, `elm_frequency`, `elm_detection`, `elm_times`) and printed during the build.

* L-mode (no ELMs): 40 frames, 0.6-1.6 s, 10 ms windows.
* ELMy H-mode: window 0.6-1.0 s. Heating, valve, pressure gauges and IR on a 0.1 ms grid (4001 frames),
  all other diagnostics on a 5 ms grid (81 frames, every 10th fast frame). TS, CXRS, MANTIS and
  DSS are not interpolated: each real sample is held on the nearest frames, and `t_sample` (time of
  the sample a frame holds) and `measured` (1 on one frame per real sample, 0 on repeats) are stored
  in their groups. Bolometer, FIR and LIUQE are means over each 5 ms bin.

`'Mode', 'default'` / `'elm'` force one of them; `'Regime', 'L'|'H'` forces the label (and the base
in auto mode). `shotdb.load(path)` shows slow data on the fast grid by default; `grid="native"` gives
each group on its own grid (`shot.time_of(path)`).

### H-mode / ELM analysis
    python -m pexdb.hmode_analysis $PEX_DB_DIR/shot_87672.h5 --save figures/hm [--t0 0.9 --span 0.06]
    from pexdb import shotdb, hmode_analysis as ha
    s = shotdb.load("shot_87672.h5")[0]; times = ha.elm_times(s); tab = ha.elm_table(s, times)

For ELM-mode files (detected ELMy H-modes). ELM times are the IR power peaks (0.1 ms). Figures: zoomed and
whole-window time traces with the ELMs marked, ELM statistics (frequency, intervals, energy per ELM from the IR
and from W_MHD, size vs interval), ELM-synchronised averages, and Thomson ne/Te profiles and pedestal against
the time since the last ELM (only real TS samples, each once, via `t_sample` / `measured`). Sizes use isolated
ELMs only (neighbours farther than 4.5 ms before / 3.5 ms after).

### CIII front (MANTIS)
The database stores the reconstructed CIII emissivity (`camera_visible/CIII_cam1/emi` on the triangle mesh + `grid`)
and the equilibrium (`equilibrium/psi`, lower X-point `r_xpt`, `z_xpt`), **not** the front position: its definition
involves analysis choices, so it is computed from the emissivity when needed.

    from pexdb import shotdb, ciii_front
    s = shotdb.load("shot_88306.h5")[0]
    r = ciii_front.ciii_front(s)             # r["outer"], r["inner"]: front [m from the target]; r["l"]: leg lengths
    r = ciii_front.ciii_front(s, dss=True)          # + front from the DSS CIII chords, r["dss"] (~ +5 s)
    r = ciii_front.ciii_front(s, uncertainty=True)  # + 16-84 % band over analysis variants (~ +40 s per shot)
    python -m pexdb.plotting.ciii_plots $PEX_DB_DIR/shot_88306.h5 --save figures/ciii   # profiles, emissivity frames, MANTIS vs DSS

The method (read the docstring of `pexdb/ciii_front.py`): emission profile along each divertor leg (separatrix
from the target to the X-point, continued above the X-point), reference = maximum of the leg shoulder (the zone
around the X-point is ignored), front = 50 % point on the target side of it; if the shoulder is gone the X-point
peak is the reference and the front can be at / above the X-point. Parameters are constants at the top of the
module. With `dss=True`, `r["dss"]` holds the outer-leg front from the divertor spectrometer (CIII triplet chords, only chords that do
not see the inner leg / X-point, so it cannot see a front above ~0.33 m from the target) and the same obtained by
integrating the MANTIS emissivity along the DSS lines of sight. Files built before `r_xpt` was stored get the X-point from `psi`; older files may also hold the legacy
`sFront*` / `l*` of the MANTIS routine, which are not used (they were wrong in several shots). Not yet validated
against an independent measurement.

