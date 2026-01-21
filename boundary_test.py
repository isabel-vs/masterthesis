import numpy as np
import matplotlib.pyplot as plt
from sumrate_max import *
import toolbox as funcs


K = 3
p_max = np.float64(200)

i=2
np.random.seed(i)

b = 1e+4 * np.random.rand(K,1)
b = b.flatten()
C = 1e+4 * np.random.rand(K,K)
sig = 1e+2 * np.random.rand(K,1)


M = np.diag(1/b) @ C.T
u = np.diag(1/b) @ sig


A = 1/p_max * np.eye(K)
# A = np.eye(K)
M_k = np.zeros((K,K,K))
for k in range(K):
    M_k[k,:,:] = M + (u @ A[:,k][:,np.newaxis].T)

w = np.ones(K)  #positive weight (priority) of users
# p_ini = 500 + 0.9 * np.random.rand(K,1)
# p_ini = (p_ini).flatten()
# p_ini = np.array([[1], [0.01], [1]])
# s = p_ini / F(p_ini, M, u)
s = np.ones(K)
r = np.log(1 + s)


gamma_max = 0
l_max = np.nan
for l in range(K):
    gamma_l, _ = find_sprad(np.diag(s) @ M_k[l,:,:])
    
    if gamma_l > gamma_max:
        gamma_max = gamma_l
        l_max = l

gamma = gamma_max
print(gamma)
M = M_k[l_max,:,:]

s_scaled = 1 / gamma * s
r_scaled = np.log(1 + s_scaled)
print(np.sum(r_scaled))

gamma_max = 0
for l in range(K):
    gamma_l, _ = find_sprad(np.diag(s_scaled) @ M_k[l,:,:])
    
    if gamma_l > gamma_max:
        gamma_max = gamma_l
        l_max = l
print(gamma_max)

p = power_fixpoint(p_max, K, M, u, s_scaled)
print(p)
s_result = p / F(p,M,u)
r_result = np.log(1+s_result)
print(r_result)
print(np.sum(r_result))

p_n = p_max / np.max(p) * p
print(p_n)
s_n = p_n / F(p_n, M, u)
r_n = np.log(1 + s_n)
print(r_n)
print(np.sum(r_n))

# p_a = power_fixpoint_a(p_max, K, M, u, s_scaled, 1e-6)
# print(p_a)
# s_a_result = p_a / F(p_a, M, u)


# print(s_a_result)
