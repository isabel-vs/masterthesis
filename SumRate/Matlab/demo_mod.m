clear all

n_user = 3;

M_1 = [[0.22915359 0.02656795 0.00770457];
        [0.02436513 0.63052893 0.00411334];
        [0.00657763 0.00088332 0.75103832]];
M_2 = [[0.22915144 0.0265701  0.00770457];
        [0.02436293 0.63053113 0.00411334];
        [0.00657504 0.00088592 0.75103832]];
M_3 = [[0.22915144 0.02656795 0.00770672];
        [0.02436293 0.63052893 0.00411554];
        [0.00657504 0.00088332 0.75104092]];


M_block = cat(3, M_1, M_2, M_3);
%%% If M is a inverse-Z matrix, then both algorithms are optimal %%%

w = ones(n_user,1); %positive weights of users
param.s_ini = ones(n_user, 1); %initial value for Alg 1 (over SINR region)
param.r_ini = log(1 + param.s_ini); %initial value for Alg 2 (over rate region)

param.K = 2000;  % K > 0 (use constant step-size for first K iterations)
param.q = 0.9;  % 0 < q <= 1 (decay factor of step-size)
param.stopCri = 1e-4; %stopping criterion (need to be sufficiently small)
param.maxIter = 10000; %number of max iteration (need to be sufficiently large)

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