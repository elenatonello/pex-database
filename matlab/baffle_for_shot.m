function baffle = baffle_for_shot(shot)
%BAFFLE_FOR_SHOT  'NINO' or 'SILO' for a shot, from config/baffles.csv ('' if not listed).
%   The table has columns first_shot,last_shot,baffle; a row may cover a range of shots.
%   Add new shots (or ranges) to the csv, not here.

f = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'config', 'baffles.csv');
T = readtable(f, 'TextType', 'char');
i = find(shot >= T.first_shot & shot <= T.last_shot, 1);
if isempty(i)
    baffle = '';
    warning('baffle_for_shot:unknown', ...
        'Shot %d is not in %s: baffle left empty. Add it to the table.', shot, f);
else
    baffle = upper(strtrim(T.baffle{i}));
end
end
