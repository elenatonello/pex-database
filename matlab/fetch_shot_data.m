function S = fetch_shot_data(shot, baffle, window, dtMhd)
%FETCH_SHOT_DATA  Retrieve all experimental data needed by build_database.py for one shot.
%   S = fetch_shot_data(shot) returns a struct with the fields
%   CXRSdata, LPdata, EdgeData, FIRdata, heating, wall, shot
%   (the same variables that used to be saved in STRUCTexp.mat).
%   S = fetch_shot_data(shot, baffle) forces the baffle ('NINO'/'SILO');
%   by default it is looked up in config/baffles.csv.
%   S = fetch_shot_data(shot, baffle, window, dtMhd) with dtMhd > 0 also computes the MHD
%   stored-energy derivative heating.HEATING.dEMHD / t_dMHD from LIUQE on [t0 t1] = window
%   (default [0.6 1.0]) with step dtMhd [s] (see fetch_mhd_energy). Default dtMhd = 0: not
%   computed (L-mode shots do not need it, and LIUQE at 0.1 ms is slow); build_pex_database
%   fetches it separately, only for ELMy shots.

if nargin < 2 || isempty(baffle), baffle = baffle_for_shot(shot); end
if nargin < 3 || isempty(window), window = [0.6 1.0]; end
if nargin < 4 || isempty(dtMhd),  dtMhd  = 0; end

CXRSdata    = CXRS_load_MDS(shot, 1, 'result', [], 0, 0);
LPdata      = LP_MDSdld(shot);
DSSdata_fit = DSS_GetFitResultsMDS(shot);

EdgeData = get_edge_data(shot, [], 'l', {'TS', 'MANTIS', 'DSS', 'APG', 'BAR', 'VALVE', 'IR', 'BOLO'});
FIRdata  = fir_getdata(shot);
heating  = get_edge_data(shot, [], '', {'HEATING'});
if dtMhd > 0
    M = fetch_mhd_energy(shot, window, dtMhd);
    if ~isempty(M)
        heating.HEATING.t_dMHD = M.t_dMHD;
        heating.HEATING.dEMHD  = M.dEMHD;
    end
end

bolo_geom        = bolou_load_geometry(shot);
[~, ~, apg_geom] = apg_ports(shot);
[~, ~, bar_geom] = bar_ports(shot);

EdgeData.BOLO.geometry = bolo_geom;
EdgeData.APG.geometry  = apg_geom;
EdgeData.BAR.geometry  = bar_geom;
EdgeData.DSS.geom      = dss_geom();
EdgeData.DSS           = dss_addTi(EdgeData.DSS, DSSdata_fit);
EdgeData.IR.geom       = ir_geom();

% Wall / vessel description
mdsconnect('tcvdata.epfl.ch');
cleanup = onCleanup(@() mdsclose());          % close the tree even if something below fails
mdsopen('tcv_shot', shot);
vD = vesseldesc();
Rt = mdsdata('static(''r_tv'')');
Zt = mdsdata('static(''z_tv'')');
wall = struct;
wall.Z_tiles       = Zt;
wall.R_tiles       = Rt;
wall.RZv_ivessel   = vD.RZv_i;
wall.RZv_ovessel   = vD.RZv_o;
wall.baffle        = baffle;

S = struct('CXRSdata', CXRSdata, 'LPdata', LPdata, 'EdgeData', EdgeData, ...
           'FIRdata', FIRdata, 'heating', heating, 'wall', wall, 'shot', shot);
end


%% Local functions

function vD = vesseldesc
    % Piece-wise polynominal description of the VV
    vD.pev_m = mdsvalue('static(''perim_v'')');   % perimeter of middle contour
    vD.ppv_i = ppmak(mdsvalue('static(''pp_v:in:breaks'')')',  mdsvalue('static(''pp_v:in:coefs'')'),  2);
    vD.ppv_m = ppmak(mdsvalue('static(''pp_v:breaks'')')',     mdsvalue('static(''pp_v:coefs'')'),     2);
    vD.ppv_o = ppmak(mdsvalue('static(''pp_v:out:breaks'')')', mdsvalue('static(''pp_v:out:coefs'')'), 2);

    vD.ssv   = linspace(0, 1, 255);
    vD.RZv_i = ppual(vD.ppv_i, vD.ssv);
    vD.RZv_o = ppual(vD.ppv_o, vD.ssv);
end


% Asdex-type pressure gauges
function [ports, names, geom] = apg_ports(shot)

    if shot>=77844
        names={'APG_MIC','APG_TOP','APG_BOT','APG_DIV'};
        ports={'E07A_2', 'T07A_3', 'B07A_2', 'B09B_1'};
    elseif shot>=75540
        names={'APG_MIA/B','APG_TOP','APG_BOT'};
        ports={'E07A_2', 'T07A_3', 'B07A_2'};
    elseif shot>=75103
        names={'APG_MIA/B','APG_BOT'};
        ports={'E07A_2', 'B07A_2'};
    elseif shot>=75004
        names={'APG_MIB','APG_TOP','APG_BOT'};
        ports={'E07A_2', 'T07A_3', 'B07A_2'};
    elseif shot>=71727
        names={'APG_MID','APG_TOP','APG_BOT'};
        ports={'E07A_2', 'T07A_3', 'B07A_2'};
    elseif shot>=70888
        names={'APG_MID'};
        ports={'E07A_2'};
    else
        names={};
        ports={};
    end

    geom = struct();
    if isempty(ports), return; end

    apgs = tcv_polview('', shot, 1.0, 'Diag', {'apg'});

    % name is a char in each element of the 1xN struct array
    apg_names = arrayfun(@(a) char(a.name), apgs, 'UniformOutput', false);

    for k = 1:numel(ports)
        idx = find(strcmp(apg_names, ports{k}), 1);
        if isempty(idx)
            warning('apg_ports:noMatch', ...
                'Port %s (%s) not found in tcv_polview output.', ports{k}, names{k});
            continue
        end

        V = apgs(idx).handle.Vertices;      % N x 2 : [R Z]

        tag = lower(regexprep(names{k}, '^APG_', ''));
        tag = regexprep(tag, '[^a-z0-9]', '_');

        geom.(['r_' tag]) = V(:,1);
        geom.(['z_' tag]) = V(:,2);
    end
end


% Baratron pressure gauges
function [ports, names, geom] = bar_ports(shot)

    if shot>=63527
        % Pex upgrade 2019: Add E06A_3 (lower lateral port) & B07A_2 (floor)
        names={'DIV',   'MID',   'TMP',   '45D'};
        ports={'B09B_1','E07A_2','E06A_3','B07A_2'}; % SPC-wiki and static tree naming convention unified in 202?
    elseif shot>52049
        % B09B_1 (inner floor) & E07A_2 (outboard mid plane)
        names={'DIV',   'MID'};
        ports={'B09B_1','E07A_2'};
    else
        names={};
        ports={};
        warning('bar_ports:noGauges', ...
            'gauges did not exist for shot %d.', shot);
    end

    geom = struct();
    if isempty(ports), return; end

    apgs = tcv_polview('', shot, 1.0, 'Diag', {'p'});

    % name is a char in each element of the 1xN struct array
    apg_names = arrayfun(@(a) char(a.name), apgs, 'UniformOutput', false);

    for k = 1:numel(ports)
        idx = find(strcmp(apg_names, ports{k}), 1);
        if isempty(idx)
            warning('bar_ports:noMatch', ...
                'Port %s (%s) not found in tcv_polview output.', ports{k}, names{k});
            continue
        end

        V = apgs(idx).handle.Vertices;      % N x 2 : [R Z]

        tag = lower(names{k});              % 'DIV' -> 'div', '45D' -> '45d'
        geom.(['r_' tag]) = V(:,1);
        geom.(['z_' tag]) = V(:,2);
    end
end


function geom = dss_geom
% load data
a   = load('/home/ducker/PHD/Other/DSS_DCDs/DCD.mat');
DCD = a.DCD.Lateral.Center;
% Compute length of each line
xpar = zeros(2,30);
for k = 2:31
    xpar(1,k) = 0.01;
    if k<18
        xpar(2,k) = (DCD.RD(k) - 0.62)/abs(cos(DCD.PVD(k)));
    elseif k == 18
        xpar(2,k) = (DCD.RD(k) - 0.64)/abs(cos(DCD.PVD(k)));
    elseif k == 19
        xpar(2,k) = (DCD.RD(k) - 0.66)/abs(cos(DCD.PVD(k)));
    else
        xpar(2,k) = (-0.75 - DCD.ZD(k))/sin(DCD.PVD(k));
    end
end
% Get [r,z] coordinates of start and end points of the LOS
r12 = zeros(2,31); z12 = zeros(2,31);
for k=2:31  %loop over all available viewlines
    r12(:,k) = DCD.RD(k) - abs(xpar(:,k)*cos(DCD.PVD(k)));
    z12(:,k) = DCD.ZD(k) + xpar(:,k)*sin(DCD.PVD(k));
end

geom = struct;
geom.rLos = r12; geom.zLos = z12;
end


function DSS = dss_addTi(DSS, DSSdata_fit)
% Add Ti and Ti_err from DSSdata_fit to the matching system in DSS (DSS.sys<N>).
%   Fit Result 1 (col 2) -> sysN.Ti,   sysN.Ti_err
%   Fit Result 2 (col 3) -> sysN.Ti_2, sysN.Ti_err_2

    fitcols = [2 3];                         % 'Fit Result 1', 'Fit Result 2'
    suffix  = {'', '_2'};

    for i = 2:size(DSSdata_fit,1)            % row 1 is the header
        sysnum = DSSdata_fit{i,1};
        if ischar(sysnum) || isstring(sysnum)
            sysnum = str2double(sysnum);
        end
        if isempty(sysnum) || isnan(sysnum), continue; end

        sysname = sprintf('sys%d', sysnum);

        for k = 1:numel(fitcols)
            fit = DSSdata_fit{i, fitcols(k)};
            if isempty(fit) || ~isstruct(fit), continue; end

            if ~isfield(DSS, sysname)
                warning('dss_addTi:noSystem', ...
                    'DSS.%s does not exist, skipping Ti fit.', sysname);
                break
            end
            if ~isfield(fit, 'Ti')
                warning('dss_addTi:noTi', ...
                    'Fit result %d of %s has no field Ti.', k, sysname);
                continue
            end

            DSS.(sysname).(['Ti'     suffix{k}]) = fit.Ti;
            if isfield(fit, 'Ti_err')
                DSS.(sysname).(['Ti_err' suffix{k}]) = fit.Ti_err;
            end
        end
    end
end


function geom = ir_geom

VIRgeom=struct; 
HIRgeom=struct;

%Port geometry
HIRgeom.Rp = 1.175;
HIRgeom.Zp = -0.460;
HIRgeom.Dp = 0.230;
HIRgeom.Lp = 0.039;

VIRgeom.Rp = 0.880;
VIRgeom.Zp = 0.790;
VIRgeom.Dp = 0.090;
VIRgeom.Lp = 0.040;

% Camera location
VIRgeom.Rd = 0.880;
VIRgeom.Zd = 0.9;
HIRgeom.Rd = 1.4;
HIRgeom.Zd = -0.45;

geom = struct;
geom.VIR = VIRgeom; geom.HIR = HIRgeom;

end
