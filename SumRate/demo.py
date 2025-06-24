import numpy as np
from hsdm import *

M_1 = np.array([[3.60657411e-01, 1.47777277e-04],
 [7.73765393e-04, 1.30344353e-01]])
M_2 = np.array([[3.60656879e-01, 1.48309570e-04],
 [7.72546745e-04, 1.30345571e-01]])

n_user = M_1.shape[0]

M_block = np.stack((M_1, M_2), axis=2)

w = np.ones(n_user) #positive weights of users
s_ini = np.ones(n_user)
r_ini = np.log(1 + s_ini)

param = {
    's_ini': s_ini,     # initial value for Alg 1 (over SINR region)
    'r_ini': r_ini,     # initial value for Alg 2 (over rate region)

    'K': 2000,          # K > 0 (use constant step-size for first K iterations)
    'q': 0.9,           # 0 < q <= 1 (decay factor of step-size)
    'stopCri': 1e-4,    # stopping criterion (need to be sufficiently small)
    'maxIter': 10000    # number of max iteration (need to be sufficiently large)
}

sinr_opt, sprad_sinr = hsdm_sinr(M_block, w, param)
rate_opt, sprad_rate = hsdm_rate(M_block, w, param)

rate_s = np.log2(1+sinr_opt)

print("(s) opt sinr:", sinr_opt)

print("(s) sum-rate:", np.sum(rate_s))
print("(s) opt rate:", rate_s)
print("(s) spectral radius:", sprad_sinr)

print("(r) sum-rate:", np.sum(rate_opt))
print("(r) opt rates:", rate_opt)
print("(r) spectral radius:", sprad_rate)