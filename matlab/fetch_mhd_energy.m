function M = fetch_mhd_energy(shot, window, dt)
%FETCH_MHD_ENERGY  Time derivative of the MHD stored energy, dW_MHD/dt [W], from LIUQE.
%   M = fetch_mhd_energy(shot, window, dt) runs LIUQE on the uniform grid window(1):dt:window(2)
%   (defaults [0.6 1.0] s and 1e-4 s) and returns a struct with the fields
%       t_dMHD   time vector [s]
%       dEMHD    dW_MHD/dt [W], same length
%   The derivative is taken between samples and interpolated back to the LIUQE times.
%   Slow (window/dt LIUQE time points), so it is only used for ELMy H-mode shots, where
%   build_database.py uses it as power_mhd and as one of the ELM-detection signals.
%   Returns [] (with a warning) if LIUQE fails.

if nargin < 2 || isempty(window), window = [0.6 1.0]; end
if nargin < 3 || isempty(dt),     dt = 5e-4; end

try
    [~, ~, LY] = liuqe(shot, window(1):dt:window(2), 'idml', 1);
    t  = LY.t(:)';
    W  = LY.Wk(:)';
    P  = diff(W) ./ diff(t);                            % derivative between samples
    tP = (t(1:end-1) + t(2:end)) / 2;                   % at the midpoints
    M  = struct('t_dMHD', t, 'dEMHD', interp1(tP, P, t, 'linear', 'extrap'), 'EMHD', W);   % [W]
catch err
    warning('fetch_mhd_energy:liuqe', '[%d] dEMHD not computed: %s', shot, err.message);
    M = [];
end
end
