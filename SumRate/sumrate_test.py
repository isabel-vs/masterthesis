import numpy as np
from sumrate_max import *
import toolbox as funcs
import pickle
import time

network_params = {"realizations": 200  #(200 originally) small scale realizations  (Critical : should be a large value if large number APs and UEs are used)
                    , "tau_c": 200 # total samples in coherence block
                    , "tau_p": 2  # samples for pilots
                    , "p_k": 200  # uplink max power (mW)
                                     }

area_params = {"delta_h": 10 # height difference
                    , "area": 100  # lxl Service Area
                    , "bandwidth": 100e6
                    , "L": 10  # APs
                    , "N": 2  # antennas per AP
                    , "K": 10  # UEs
                    , "carrier_frequency": 2000  # carrier frequency (MHz)
                    , "sigma_sf_los": 8  # Shadow fading LoS (dB)
                    , "AP_pos_rand": True
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

filename = f"RateRegions/Areas Sim/results_L{L}N{N}K{K}_rician0.pkl"
area_block = pickle.load(open(filename, "rb"))

(R_block, B_block, HMean_block, kappa_block, UEpos_block, APpos_block) = area_block

i = 0

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
r_ini = np.log(1 + s_ini)
param_hsdm = {
    's_ini': s_ini,     # initial value for Alg 1 (over SINR region)
    'r_ini': r_ini,     # initial value for Alg 2 (over rate region)

    'K': 2000,          # K > 0 (use constant step-size for first K iterations)
    'q': 0.9,           # 0 < q <= 1 (decay factor of step-size)
    'stopCri': 1e-4,    # stopping criterion (need to be sufficiently small)
    'maxIter': 10000    # number of max iteration (need to be sufficiently large)
}

start_hsdm = time.time()
rate_hsdm, sprad_hsdm = hsdm_cut(M_k, w, param_hsdm)
# rate_hsdm = np.log(sinr_hsdm + 1)
end_hsdm = time.time()
# rate_hsdm /= np.log(2)
print("hsdm:", np.sum(rate_hsdm), rate_hsdm, sprad_hsdm)
print("time hsdm", (end_hsdm - start_hsdm))

p_hsdm = power_fixpoint(p_max, K, M, u, (np.exp(rate_hsdm)-1))
print("power hsdm:", p_hsdm.flatten())

# start_bcd = time.time()
# rate_bcd, p_bcd = bcd(b, C, sigma, K, p_max)
# end_bcd = time.time()
# # rate_bcd /= np.log(2)
# print("bcd:", np.sum(rate_bcd), rate_bcd)
# print("time bcd", (end_bcd - start_bcd))
# print("power bcd:", p_bcd)






