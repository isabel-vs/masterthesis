import numpy as np
from hsdm import *

M_1 = np.array([[5.16867179e-01, 1.81536773e-05],
                [9.14085826e-03, 7.09261167e-02]])
M_2 = np.array([[5.16867093e-01, 1.82397961e-05],
                [9.13892636e-03, 7.09280486e-02]])

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

# sinr_opt, sprad_sinr = hsdm_sinr(M_block, w, param)
rate_opt, sprad_rate = hsdm_rate(M_block, w, param)
# print(sinr_opt)
# print(sprad_sinr)
print("sum-rate:", np.sum(rate_opt))
print("opt rates:", rate_opt)
print("spectral radius:", sprad_rate)