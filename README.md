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
