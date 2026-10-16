function failed = build_pex_database(shots, outDir, varargin)
%BUILD_PEX_DATABASE  Fetch each shot from the TCV server and write shot_<N>.h5.
%   failed = build_pex_database(shots, outDir)
%
%   For every shot the heavy MATLAB struct is saved to a temporary .mat file,
%   converted by build_database.py into <outDir>/shot_<N>.h5, and then deleted,
%   so at most one struct is ever on disk (and in memory).
%
%   Name-value options:
%     'Overwrite' false   re-build shots whose .h5 already exists
%     'Python'    'python3'   python executable (must have numpy, scipy, h5py)
%     'Script'    <repo>/pexdb/build_database.py
%     'Regime'    ''      'L' or 'H', forwarded as --regime= when SHOTINFO is missing
%     'Baffle'    ''      force 'NINO' or 'SILO' for all shots (default: config/baffles.csv)
%     'Mode'      'auto'  time base: 'auto' (default) = ELM-resolved base for ELMy H-mode shots,
%                         detected from the data, else the 40-frame base; or force 'default' / 'elm'.
%                         ELM base: window 0.6-1.0 s; heating, valve, pressure gauges and IR on a 0.1 ms
%                         grid, all other diagnostics on a 5 ms grid. Use separate output folders for
%                         ELM and L-mode databases if you force modes.
%     'Elm'       false   shortcut for 'Mode','elm'
%     'Window'    []      time window [t0 t1] [s]. If given, it is passed to build_database.py as
%                         --t0 / --t1 and is also the window of the LIUQE dW_MHD/dt fetch. If empty
%                         the database defaults apply and dW_MHD/dt is fetched over [0.6 1.0]
%     'MhdStep'   5e-4    time step of dW_MHD/dt [s] (the LIUQE step we can afford; the database stores it
%                         interpolated, with its native step in heating/step_power_mhd)
%
%   dW_MHD/dt (LIUQE, slow) is only fetched for ELMy H-mode shots: the shot is first fetched without
%   it, build_database.py --detect decides (from the divertor signals) whether it is ELMy, and only
%   then LIUQE is run. With 'Mode','elm' it is always fetched, with 'default' never.
%     'Args'      ''      extra arguments for build_database.py, e.g. '--t0=0.6 --t1=1.2 --dt-slow=5e-3'
%
%   Example:
%     build_pex_database([89437 89837 88306], '/to/the/database');

p = inputParser;
addParameter(p, 'Overwrite', false);
addParameter(p, 'Python', 'python3');
addParameter(p, 'Script', fullfile(fileparts(fileparts(mfilename('fullpath'))), 'pexdb', 'build_database.py'));
addParameter(p, 'Regime', '');
addParameter(p, 'Baffle', '');
addParameter(p, 'Mode', 'auto');
addParameter(p, 'Elm', false);
addParameter(p, 'Args', '');
addParameter(p, 'Window', []);
addParameter(p, 'MhdStep', 5e-4);
parse(p, varargin{:});
o = p.Results;

if ~isfolder(outDir), mkdir(outDir); end
failed = [];

for shot = shots(:)'
    h5 = fullfile(outDir, sprintf('shot_%d.h5', shot));
    if isfile(h5) && ~o.Overwrite
        fprintf('[%d] exists, skipping\n', shot);
        continue
    end

    tmp = [tempname() '.mat'];
    tmpMhd = [tempname() '.mat'];
    cleanup = onCleanup(@() delete_files({tmp, tmpMhd}));   % removed even on error / Ctrl-C
    try
        fprintf('[%d] fetching...\n', shot);
        S = fetch_shot_data(shot, o.Baffle);           % without dW_MHD/dt: only ELMy shots need it
        save(tmp, '-struct', 'S', '-v7');              % -v7: readable by scipy.io.loadmat
        clear S                                        % free memory before calling python

        mode = o.Mode;
        if o.Elm, mode = 'elm'; end
        regimeArg = '';
        if ~isempty(o.Regime), regimeArg = [' --regime=' o.Regime]; end

        % ELMy H-mode shots need dW_MHD/dt (LIUQE, slow): decide first, cheaply
        if strcmp(mode, 'auto')
            [st, out] = system(['LD_LIBRARY_PATH= ' sprintf('%s "%s" "%s" --detect', o.Python, o.Script, tmp) regimeArg]);
            fprintf('%s', out);
            if st ~= 0, error('ELM detection (build_database.py --detect) failed with status %d', st); end
            needMhd = contains(out, 'ELMY=1');
        else
            needMhd = strcmp(mode, 'elm');
        end
        mhdArg = '';
        if needMhd
            win = o.Window;
            if isempty(win), win = [0.6 1.0]; end
            fprintf('[%d] ELMy: LIUQE dW_MHD/dt on %g-%g s, step %g s...\n', shot, win(1), win(2), o.MhdStep);
            M = fetch_mhd_energy(shot, win, o.MhdStep);
            if ~isempty(M)
                save(tmpMhd, '-struct', 'M', '-v7');
                mhdArg = [' --mhd="' tmpMhd '"'];
            end
        end

        cmd = sprintf('%s "%s" "%s" "%s"%s --mode=%s%s', o.Python, o.Script, tmp, outDir, regimeArg, mode, mhdArg);
        if ~isempty(o.Window)                           % explicit window: same for fetch and database
            cmd = [cmd sprintf(' --t0=%g --t1=%g', o.Window(1), o.Window(2))];
        end
        if ~isempty(o.Args), cmd = [cmd ' ' o.Args]; end
        % MATLAB's bundled libstdc++ breaks system python -> blank LD_LIBRARY_PATH
        [status, out] = system(['LD_LIBRARY_PATH= ' cmd]);
        fprintf('%s', out);
        if status ~= 0, error('build_database.py exited with status %d', status); end
    catch err
        warning('[%d] FAILED: %s', shot, err.message);
        failed(end+1) = shot; %#ok<AGROW>
    end
    clear cleanup                                      % deletes tmp now
end

if ~isempty(failed)
    fprintf('Failed shots: %s\n', mat2str(failed));
end
end

function delete_files(files)
    for i = 1:numel(files)
        if isfile(files{i}), delete(files{i}); end
    end
end
