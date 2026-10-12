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

## Inspect / plot
    python -m pexdb.summarize_shot $PEX_DB_DIR/shot_89437.h5
    from pexdb import shotdb
    from pexdb.plotting import shot_overview

