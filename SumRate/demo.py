import numpy as np
from hsdm import *

n_user = 2

# M_1 = np.array([[1.02296613e+00, 6.35566100e-03, 7.25840181e-02],
#                 [6.00648774e-02, 3.76520141e-01, 5.94862113e-01],
#                 [3.79902186e-04, 3.82250071e-04, 7.86407429e-01]])

# M_2 = np.array([[1.02295099e+00, 6.37079948e-03, 7.25840181e-02],
#                 [6.00279607e-02, 3.76557058e-01, 5.94862113e-01],
#                 [3.78963625e-04, 3.83188632e-04, 7.86407429e-01]])

# M_3 = np.array([[1.02295099e+00, 6.35566100e-03, 7.25991566e-02],
#                 [6.00279607e-02, 3.76520141e-01, 5.94899029e-01],
#                 [3.78963625e-04, 3.82250071e-04, 7.86408368e-01]])

M_1 = np.array([[0.27382275, 0.00068982],
                [0.00690268, 0.07037035]])
M_2 = np.array([[0.27382236, 0.00069021],
                [0.00690144, 0.0703716 ]])

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
print(rate_opt)
print(sprad_rate)