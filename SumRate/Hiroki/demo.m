clear all

n_user = 3;
M = [1, 0.1, 0.3; 0.2, 0.9, 0.15; 0.3, 0.2, 2]; % inverse-Z matrix
%%% If M is a inverse-Z matrix, then borh algorithms are optimal %%%

w = ones(n_user,1); %positive weights of users
param.s_ini = ones(n_user, 1); %initial value for Alg 1 (over SINR region)
param.r_ini = log(1 + param.s_ini); %initial value for Alg 2 (over rate region)

param.K = 2000;  % K > 0 (use constant step-size for first K iterations)
param.q = 0.9;  % 0 < q <= 1 (decay factor of step-size)
param.stopCri = 1e-4; %stopping criterion (need to be sufficiently small)
param.maxIter = 10000; %number of max iteration (need to be sufficiently large)

[sinr_opt, sprad_sinr] = hsdm_sinr(M, w, param);
[rate_opt, sprad_rate] = hsdm_rate(M, w, param);

fprintf('sum-rate by Alg 1 (over SINR region): %f\n', w' * log(1+sinr_opt));
fprintf('sum-rate by Alg 2 (over rate region): %f\n', w' * rate_opt);

%%% spectral radius should be close to 1. %%%
%%% If not, decrease stopCri and/or increase maxIter %%%
fprintf('final spectral radius by Alg 1: %f\n', sprad_sinr);
fprintf('final spectral radius by Alg 2: %f\n', sprad_rate);