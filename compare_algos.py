import numpy as np
import matplotlib.pyplot as plt
from sumrate_max import *
import toolbox as funcs
import os
import pickle
import scipy.io
import time

network_params = {"realizations": 1000  #(200 originally) small scale realizations  (Critical : should be a large value if large number APs and UEs are used)
                    , "tau_c": 200 # total samples in coherence block
                    , "tau_p": 2  # samples for pilots
                    , "p_k": 100  # uplink max power (mW)
                                     }

area_params = {"delta_h": 10 # height difference
                    , "area": 50  # lxl Service Area
                    , "bandwidth": 20e6
                    , "L": 1  # APs
                    , "N": 4  # antennas per AP
                    , "K": 2  # UEs
                    , "carrier_frequency": 2000  # carrier frequency (MHz)
                    , "sigma_sf_los": 8  # Shadow fading LoS (dB)
                    , "AP_pos_rand": True
                                     }

sim_params = {"setups": 100000,
                     "batch":network_params["realizations"],
                     "v":-1,
                     "d": 0,   #rician factor (0: no LoS component)
                     "norm_type" : np.inf,
                     "tests" :5
                    }

area_block = funcs.create_areas(area_params, sim_params["tests"])
K = area_params["K"]

area = area_params["area"]

(R_block, B_block, HMean_block, kappa_block, UEpos_block, APpos_block) = area_block
# i=29
# UEpos = UEpos_block[i,:,:]
# APpos = APpos_block[i,:,:]
# fig = plt.figure()
# ax = fig.add_subplot()
# ax.scatter(np.real(UEpos), np.imag(UEpos))
# ax.scatter(np.real(APpos), np.imag(APpos))
# ax.set_xlim(0, area)
# ax.set_ylim(0, area)
# plt.show()
# print(UEpos)

L = area_params["L"]
N = area_params["N"]
K = area_params["K"]
p_max = network_params["p_k"]
# filename = f"RateRegions/Areas Sim/results_L{L}N{N}K{K}_rician0.pkl"
# filename = "test50_area_L1N1K2.pkl"
# area_block = pickle.load(open(filename, "rb"))

num_invZ = 0
i=1
for i in range(i,i+1):
    inv_interf = True
    (R_block, B_block, HMean_block, kappa_block, UEpos_block, APpos_block) = area_block

    R_spatial_corr_norm = R_block[i,:,:,:,:]
    B_channel_gain =  B_block[i,:,:]
    HMean_norm = HMean_block[i,:,:,:]
    k_rician_factor = kappa_block[i,:,:]
    UEpos = UEpos_block[i,:,:]
    APpos = APpos_block[i,:,:]

    area_values = (R_spatial_corr_norm, B_channel_gain, HMean_norm, k_rician_factor, UEpos, APpos)


    M, u, b, C, sigma, SINR, UEpos, APpos, D_DCC = funcs.run_sim(network_params, area_params, sim_params, area_values, i)

    #------------

    p_max = network_params["p_k"]
    K = area_params["K"]

    A = 1/p_max * np.eye(K)
    # A = np.eye(K)
    M_k = np.zeros((K,K,K))
    for k in range(K):
        M_k[k,:,:] = M + (u @ A[:,k][:,np.newaxis].T)


w = np.ones(K)
s_ini = np.ones(K)

# r_ini = 2* np.ones(K)
# s_ini = np.array([1e-15, 1e-15]) 
r_ini = np.log(1 + s_ini)

r_ini = np.log(1+s_ini)*np.log(2)
# p_ini = [1, 0]
p_ini = np.ones(K)

param_hsdm = {
    's_ini': s_ini,     # initial value for Alg 1 (over SINR region)
    'r_ini': r_ini,     # initial value for Alg 2 (over rate region)

    'K': 0,         # K > 0 (use constant step-size for first K iterations)
    'q': 0.9,           # 0 < q <= 1 (decay factor of step-size)
    'mu': 1,
    'beta': 0.1,
    'stopCri': 1e-8,    # stopping criterion (need to be sufficiently small)
    'maxIter': 10000,    # number of max iteration (need to be sufficiently large)
    'solution': np.zeros((K, K))
    # 'solution' : [0.30092038,0.45111856]
}

# sinr_hsdmS, sprad_hsdmS, gammaS = hsdm_sinr(M_k, w, param_hsdm)
# print(gammaS)
# rate_hsdmS = np.log(sinr_hsdmS + 1)
rate_hsdmR, sprad_hsdm,  rate_prog, gammaR = hsdm_rate(M_k, w, param_hsdm)
# print(gammaR)
# rate_hsdmC, sprad_hsdmC,  _, gammaC = hsdm_cut(M_k, w, param_hsdm)
# print(gammaC)
rate_bcd, _, _, p_bcd = bcd(b, C, sigma, K, p_max, p_ini)

# p_hsdmS = power_fixpoint(p_max, K, M, u, sinr_hsdmS)
p_hsdmR = power_fixpoint(p_max, K, M, u, np.exp(rate_hsdmR)-1)
# p_hsdmC = power_fixpoint(p_max, K, M, u, np.exp(rate_hsdmC)-1)

# print("power S:", p_hsdmS)
print("power R:", p_hsdmR)
# print("power C:", p_hsdmC)
print("power bcd:", p_bcd)

sinr_hsdmR = np.exp(rate_hsdmR) - 1
sinr_bcd = np.exp(rate_bcd) - 1
# rate_hsdmS /= np.log(2)
rate_hsdmR /= np.log(2)
# rate_hsdmC /= np.log(2)
rate_bcd /= np.log(2)

# print("rate S:", np.sum(rate_hsdmS), rate_hsdmS, gammaS)
print("rate R:", np.sum(rate_hsdmR), rate_hsdmR, gammaR)
# print("rate C:", np.sum(rate_hsdmC), rate_hsdmC, gammaC)
print("rate bcd:", np.sum(rate_bcd), rate_bcd)


p_Rn = p_max / np.max(p_hsdmR) * p_hsdmR
# p_Sn = p_max / np.max(p_hsdmS) * p_hsdmS

s_Rn = p_Rn / F(p_Rn, M, u)
# s_Sn = p_Sn / F(p_Sn, M, u)

r_Rn = np.log(1 + s_Rn)
# r_Sn = np.log(1 + s_Sn)

print(r_Rn,p_Rn)
# print(r_Sn, p_Sn)

p1 = np.ones(K) * p_max
p2 = np.array([0, p_max])
p3 = np.array([p_max, 0])

s1 = p1 / F(p1, M, u)
s2 = p2 / F(p2, M, u)
s3 = p3 / F(p3, M, u)

r1 = np.log(1+s1)
r2 = np.log(1+s2)
r3 = np.log(1+s3)

r1 /= np.log(2)
r2 /= np.log(2)
r3 /= np.log(2)


if True:

    x,y = SINR
    rate_x = np.log2(1+np.array(x) )
    rate_y = np.log2(1+np.array(y) )

    x_max, y_max = np.max(x), np.max(y)
    max_val = max(x_max, y_max)


    
    rate_x_max, rate_y_max = np.max(rate_x), np.max(rate_y)
    max_rate = max(rate_x_max, rate_y_max)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6,3))
    ax1.scatter(x,y)

    # ax1.plot([0,max_val], [0,max_val])

    ax1.set_xlabel("SINR - User 1", fontsize=10)
    ax1.set_ylabel("SINR - User 2", fontsize=10)
    ax1.set_xlim([0, max_val + 0.05*max_val])
    ax1.set_ylim([0, max_val + 0.05*max_val])

    ax2.scatter(rate_x, rate_y)

    ax2.set_xlabel("Rate [bit/s/Hz] - User 1", fontsize=10)
    ax2.set_ylabel("Rate [bit/s/Hz] - User 2", fontsize=10)
    ax2.set_xlim([0, max_rate + 0.05*max_rate])
    ax2.set_ylim([0, max_rate + 0.05*max_rate])

    # ax1.scatter(sinr_hsdmS[0], sinr_hsdmS[1], label="hsdmS")
    # ax2.plot([0, 2*rate_opt[0]], [0, 2*rate_opt[1]])

    # rate_optS = rate_hsdmS
    rate_optR = rate_hsdmR
    # rate_optC = rate_hsdmC
    # ax2.scatter(rate_optS[0], rate_optS[1], label="hsdmS")
    ax2.scatter(rate_optR[0], rate_optR[1], label="(14.2 100)")
    # ax2.scatter(rate_optC[0], rate_optC[1], label="hsdmC")
    # ax2.scatter(rate_bcd[0], rate_bcd[1], label="bcd")

    ax2.scatter(r1[0], r1[1], label="p = (100, 100)")
    ax2.scatter(r2[0], r2[1], label="p = (0, 100)")
    ax2.scatter(r3[0], r3[1], label="p = (100, 0)")

    ax1.scatter(sinr_hsdmR[0], sinr_hsdmR[1], label="14.2 100)")
    # ax1.scatter(sinr_bcd[0], sinr_bcd[1], label="bcd")

    ax1.scatter(s1[0], s1[1], label="p = (100, 100)")
    ax1.scatter(s2[0], s2[1], label="p = (0, 100)")
    ax1.scatter(s3[0], s3[1], label="p = (100, 0)")

    plt.legend()
    plt.tight_layout()
    plt.show()