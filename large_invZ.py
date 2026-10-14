# test algorithms on large artificial invZ matrix

import numpy as np
from sumrate_max import *
import toolbox as funcs
import matplotlib.pyplot as plt
import pickle

matrix_size = 150
M_file = "M.pkl"

# # 1. Create a 100x100 matrix with random values between -10 and 0
# M_inv = -10 * np.random.rand(matrix_size, matrix_size)

# # 2. Make the matrix strictly diagonally dominant to ensure it is well-conditioned
# #    This also ensures the diagonal elements are positive.
# for i in range(matrix_size):
#     # Sum of the absolute values of the off-diagonal elements
#     row_sum = np.sum(np.abs(M_inv[i, :])) - np.abs(M_inv[i, i])
#     # Set the diagonal element to be positive and greater than the sum
#     M_inv[i, i] = row_sum + np.random.uniform(1, 10) # Add a small random number for robustness

# M = np.linalg.inv(M_inv)

# M = M[:,:,np.newaxis]

# with open(M_file, 'wb') as outfile:
#     pickle.dump(M, outfile) 


M = pickle.load(open(M_file, "rb"))

file = f"rate.pkl"
(sol_b, sol_c) = pickle.load(open(file, "rb"))

#set up algos
K = matrix_size
w = np.ones(K)
s_ini = np.ones(K)
r_ini = np.log(1 + s_ini)
# r_ini = 1*np.ones(K)
param_hsdm = {
    's_ini': s_ini,     # initial value for Alg 1 (over SINR region)
    'r_ini': r_ini,     # initial value for Alg 2 (over rate region)

    # 'K': 2000,          # K > 0 (use constant step-size for first K iterations)
    'K': 0,
    'q': 0.9,           # 0 < q <= 1 (decay factor of step-size)
    'mu': 1,
    'beta': 1,
    'stopCri': 1e-6,    # stopping criterion (need to be sufficiently small)
    'maxIter': 5000,    # number of max iteration (need to be sufficiently large)
    # 'solution': np.zeros((matrix_size, matrix_size))
    'solution': sol_b
}

rate_hsdm, sprad_hsdm, conv_prog, rate_prog, _ = hsdm_rate(M, w, param_hsdm)
rate_hsdmC, sprad_hsdmC, conv_progC, rate_progC = hsdm_cut(M, w, param_hsdm)

# rate_bcd, p_bcd = bcd(b, C, sigma, K, p_max)


ax2 = plt.figure().add_subplot()
ax2.semilogy(conv_prog[0:-1], label='base')
ax2.semilogy(conv_progC[0:-1], label='deepcut')
ax2.legend()
ax2.grid()
plt.show()

print("done")

# with open(file, 'wb') as outfile:
#     pickle.dump((rate_hsdm, rate_hsdmC), outfile) 
