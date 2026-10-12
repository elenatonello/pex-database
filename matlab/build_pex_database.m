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
%
%   Example:
%     build_pex_database([89437 89837 88306], '/to/the/database');

p = inputParser;
addParameter(p, 'Overwrite', false);
addParameter(p, 'Python', 'python3');
addParameter(p, 'Script', fullfile(fileparts(fileparts(mfilename('fullpath'))), 'pexdb', 'build_database.py'));
addParameter(p, 'Regime', '');
addParameter(p, 'Baffle', '');
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
    cleanup = onCleanup(@() delete_if_exists(tmp));    % removed even on error / Ctrl-C
    try
        fprintf('[%d] fetching...\n', shot);
        S = fetch_shot_data(shot, o.Baffle);
        save(tmp, '-struct', 'S', '-v7');              % -v7: readable by scipy.io.loadmat
        clear S                                        % free memory before calling python

        cmd = sprintf('%s "%s" "%s" "%s"', o.Python, o.Script, tmp, outDir);
        if ~isempty(o.Regime), cmd = [cmd ' --regime=' o.Regime]; end
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

function delete_if_exists(f)
    if isfile(f), delete(f); end
end
