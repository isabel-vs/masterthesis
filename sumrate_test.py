import numpy as np
from sumrate_max import *
import toolbox as funcs
import pickle
import time
import matplotlib.pyplot as plt

network_params = {"realizations": 200  #(200 originally) small scale realizations  (Critical : should be a large value if large number APs and UEs are used)
                    , "tau_c": 200 # total samples in coherence block
                    , "tau_p": 2  # samples for pilots
                    , "p_k": 200  # uplink max power (mW)
                                     }

area_params = {"delta_h": 10 # height difference
                    , "area": 500  # lxl Service Area
                    , "bandwidth": 20e6
                    , "L": 1  # APs
                    , "N": 1  # antennas per AP
                    , "K": 2  # UEs
                    , "carrier_frequency": 2000  # carrier frequency (MHz)
                    , "sigma_sf_los": 8  # Shadow fading LoS (dB)
                    , "AP_pos_rand": False
                                     }

sim_params = {"setups": 5000,
                     "batch":network_params["realizations"],
                     "v":-1,
                     "d": 0,   #rician factor (0: no LoS component)
                     "norm_type" : np.inf,
                     "tests" :50
                    }

L = area_params["L"]
N = area_params["N"]
K = area_params["K"]
p_max = network_params["p_k"]
tests = sim_params["tests"]

# filename = f"RateRegions/Areas Sim/results_L{L}N{N}K{K}_rician0.pkl"
filename = f"test{tests}_area_L{L}N{N}K{K}.pkl"
area_block = pickle.load(open(filename, "rb"))

(R_block, B_block, HMean_block, kappa_block, UEpos_block, APpos_block) = area_block

i = 31

R_spatial_corr_norm = R_block[i,:,:,:,:]
B_channel_gain =  B_block[i,:,:]
HMean_norm = HMean_block[i,:,:,:]
k_rician_factor = kappa_block[i,:,:]
UEpos = UEpos_block[i,:,:]
APpos = APpos_block[i,:,:]

area_values = (R_spatial_corr_norm, B_channel_gain, HMean_norm, k_rician_factor, UEpos, APpos)

M, u, b, C, sigma, SINR, UEpos, APpos, D_DCC = funcs.run_sim(network_params, area_params, sim_params, area_values, i)

A = np.eye(K)
M_k = np.zeros((K,K,K))
inv_interf = True
for k in range(K):
    M_k[:,:,k] = M + (u @ A[:,k][:,np.newaxis].T)/p_max
for k in range(K):
    if funcs.is_invZ(M_k[:,:,k]) == False:
        inv_interf = False
        break
print(inv_interf)

w = np.ones(K)
s_ini = np.ones(K)
# r_ini = np.log(1 + s_ini)
r_ini = np.ones(K)

param_hsdm = {
    's_ini': s_ini,     # initial value for Alg 1 (over SINR region)
    'r_ini': r_ini,     # initial value for Alg 2 (over rate region)

    'K': 0,          # K > 0 (use constant step-size for first K iterations)
    'q': 0.9,           # 0 < q <= 1 (decay factor of step-size)
    'mu': 0.1,
    'beta': 2,
    'stopCri': 1e-8,    # stopping criterion (need to be sufficiently small)
    'maxIter': 10000,    # number of max iteration (need to be sufficiently large)
    'solution': np.zeros((K, K))
}

sinr_hsdmS, sprad_hsdmS, gammaS = hsdm_sinr(M_k, w, param_hsdm)
rate_hsdm, sprad_hsdm, conv_prog, rate_prog, gamma = hsdm_rate(M_k, w, param_hsdm)
rate_hsdmC, sprad_hsdmC, conv_progC, rate_progC = hsdm_cut(M_k, w, param_hsdm)
rate_hsdmS = np.log(sinr_hsdmS + 1)
# rate_hsdm /= np.log(2)

print("hsdmS rate:", np.sum(rate_hsdmS), rate_hsdmS, gammaS)
print("hsdmR rate:", np.sum(rate_hsdm), rate_hsdm, gamma)
print("hsdm deepcut:", np.sum(rate_hsdmC), rate_hsdmC)


ax2 = plt.figure().add_subplot()
ax2.semilogy(conv_prog, label='base')
ax2.semilogy(conv_progC, label='deepcut')
ax2.legend()
ax2.grid()

# ax2.add_subplot()
# ax2.semilogy(conv_progC)

plt.show()

# start_bcd = time.time()
rate_bcd, p_bcd = bcd(b, C, sigma, K, p_max)
# end_bcd = time.time()
# rate_bcd /= np.log(2)
print("bcd:", np.sum(rate_bcd), rate_bcd)
# print("time bcd", (end_bcd - start_bcd))
# print("power bcd:", p_bcd)






