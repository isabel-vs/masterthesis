clear all

n_user = 3;
% M_test = [1, 0.1, 0.3; 0.2, 0.9, 0.15; 0.3, 0.2, 2]; % inverse-Z matrix
M_1 = [[1.02296613e+00 6.35566100e-03 7.25840181e-02];
 [6.00648774e-02 3.76520141e-01 5.94862113e-01];
 [3.79902186e-04 3.82250071e-04 7.86407429e-01]];
M_2 = [[1.02295099e+00 6.37079948e-03 7.25840181e-02];
 [6.00279607e-02 3.76557058e-01 5.94862113e-01];
 [3.78963625e-04 3.83188632e-04 7.86407429e-01]];
M_3 = [[1.02295099e+00 6.35566100e-03 7.25991566e-02];
 [6.00279607e-02 3.76520141e-01 5.94899029e-01];
 [3.78963625e-04 3.82250071e-04 7.86408368e-01]];

M_block = cat(3, M_1, M_2, M_3);
%%% If M is a inverse-Z matrix, then both algorithms are optimal %%%

w = ones(n_user,1); %positive weights of users
param.s_ini = ones(n_user, 1); %initial value for Alg 1 (over SINR region)
param.r_ini = log(1 + param.s_ini); %initial value for Alg 2 (over rate region)

param.K = 2000;  % K > 0 (use constant step-size for first K iterations)
param.q = 0.9;  % 0 < q <= 1 (decay factor of step-size)
param.stopCri = 1e-4; %stopping criterion (need to be sufficiently small)
param.maxIter = 50000; %number of max iteration (need to be sufficiently large)

% [sinr_opt, sprad_sinr] = hsdm_sinr_mod(M_block, w, param);
[rate_opt, sprad_rate] = hsdm_rate_mod(M_block, w, param);

% fprintf('sum-rate by Alg 1 (over SINR region): %f\n', w' * log(1+sinr_opt));
% fprintf('optimal sinr vals: %f\n', sinr_opt);
fprintf('sum-rate by Alg 2 (over rate region): %f\n', w' * rate_opt);
fprintf('optimal rate vals: %f\n', rate_opt);

%%% spectral radius should be close to 1. %%%
%%% If not, decrease stopCri and/or increase maxIter %%%
% fprintf('final spectral radius by Alg 1: %f\n', sprad_sinr);
fprintf('final spectral radius by Alg 2: %f\n', sprad_rate);