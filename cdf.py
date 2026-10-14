import numpy as np
import toolbox as funcs
from sumrate_max import *
import pickle

network_params = {"realizations": 1000  #(200 originally) small scale realizations  (Critical : should be a large value if large number APs and UEs are used)
                    , "tau_c": 200 # total samples in coherence block
                    , "tau_p": 6  # samples for pilots
                    , "p_k": 200  # uplink max power (mW)
                                     }

area_params = {"delta_h": 10 # height difference
                    , "area": 500  # lxl Service Area
                    , "bandwidth": 20e6
                    , "L": 16  # APs
                    , "N": 1  # antennas per AP
                    , "K": 2  # UEs
                    , "carrier_frequency": 2000  # carrier frequency (MHz)
                    , "sigma_sf_los": 8  # Shadow fading LoS (dB)
                    , "AP_pos_rand": False
                                     }

sim_params = {"setups": 1,
                     "batch":1,
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

w = np.ones(K)
s_ini = np.ones(K)
r_ini = np.log(1 + s_ini)
# param_hsdm = {
#     's_ini': s_ini,     # initial value for Alg 1 (over SINR region)
#     'r_ini': r_ini,     # initial value for Alg 2 (over rate region)

#     'K': 0,          # K > 0 (use constant step-size for first K iterations)
#     'q': 0.9,           # 0 < q <= 1 (decay factor of step-size)
#     'stopCri': 1e-4,    # stopping criterion (need to be sufficiently small)
#     'maxIter': 20000    # number of max iteration (need to be sufficiently large)
# }
param_hsdm = {
    's_ini': s_ini,     # initial value for Alg 1 (over SINR region)
    'r_ini': r_ini,     # initial value for Alg 2 (over rate region)

    # 'K': 2000,          # K > 0 (use constant step-size for first K iterations)
    'K': 0,
    'q': 1,           # 0 < q <= 1 (decay factor of step-size)
    'mu': 0.1,
    'beta': 1,
    'stopCri': 1e-6,    # stopping criterion (need to be sufficiently small)
    'maxIter': 1000,    # number of max iteration (need to be sufficiently large)
    'solution': np.zeros((K, K))
    # 'solution': sol_b
}

invZ = []
sumrate_hsdmS = []
sumrate_hsdmR = []
sumrate_hsdmC = []
sumrate_bcd = []
sumrate_maxPow = []
rates_hsdmS = []
rates_hsdmR = []
rates_hsdmC = []
rates_bcd = []
rates_maxPow = []
gamma_S = []
gamma_R = []
pow_hsdmS = np.zeros((K,tests))
pow_hsdmR = np.zeros((K,tests))
pow_hsdmC = np.zeros((K,tests))
pow_bcd = np.zeros((K,tests))

p_maxPow = p_max * np.ones((K,1))


for i in range(tests):
    print(i)
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

    invZ.append(inv_interf)

    sinr_hsdmS, _, g_S = hsdm_sinr(M_k, w, param_hsdm)
    rate_hsdmS = np.log(sinr_hsdmS + 1)
    p_hsdmS = power_fixpoint(p_max, K, M, u, sinr_hsdmS)

    rate_hsdmR, _, _, _, g_R = hsdm_rate(M_k, w, param_hsdm)
    p_hsdmR = power_fixpoint(p_max, K, M, u, (np.exp(rate_hsdmR)-1))

    rate_hsdmC, _, _, _ = hsdm_cut(M_k, w, param_hsdm)
    p_hsdmC = power_fixpoint(p_max, K, M, u, (np.exp(rate_hsdmC)-1))

    rate_bcd, p_bcd = bcd(b, C, sigma, K, p_max)

    sinr_maxPow = funcs.sinr(M,u,p_maxPow)
    rate_maxPow = np.log(1 + sinr_maxPow)

    rates_hsdmS.append(rate_hsdmS)
    rates_hsdmR.append(rate_hsdmR)
    rates_hsdmC.append(rate_hsdmC)
    rates_bcd.append(rate_bcd)
    rates_maxPow.append(rate_maxPow)

    sumrate_hsdmS.append(sum(rate_hsdmS))
    sumrate_hsdmR.append(sum(rate_hsdmR))
    sumrate_hsdmC.append(sum(rate_hsdmC))
    sumrate_bcd.append(sum(rate_bcd))
    sumrate_maxPow.append(sum(rate_maxPow))

    gamma_S.append(g_S)
    gamma_R.append(g_R)

    # pow_hsdmS[:,i] = p_hsdmS.flatten()
    # pow_hsdmR[:,i] = p_hsdmR.flatten()
    # pow_hsdmC[:,i] = p_hsdmC.flatten()
    # pow_bcd[:,i] = p_bcd
    pow_hsdmS[:,0] = p_hsdmS.flatten()
    pow_hsdmR[:,0] = p_hsdmR.flatten()
    pow_hsdmC[:,0] = p_hsdmC.flatten()
    pow_bcd[:,0] = p_bcd

results = {
    'invZ': invZ,
    'sumrate_hsdmS': sumrate_hsdmS,
    'sumrate_hsdmR': sumrate_hsdmR,
    'sumrate_hsdmC': sumrate_hsdmC,
    'sumrate_bcd': sumrate_bcd,
    'sumrate_maxPow': sumrate_maxPow,
    'rates_hsdmS': rates_hsdmS,
    'rates_hsdmR': rates_hsdmR,
    'rates_hsdmC': rates_hsdmC,
    'rates_bcd': rates_bcd,
    'rates_maxPow': rates_maxPow,
    'p_hsdmS': pow_hsdmS,
    'p_hsdmR': pow_hsdmR,
    'p_hsdmC': pow_hsdmC,
    'p_bcd': pow_bcd,
    'network_params': network_params,
    'area_params': area_params,
    'sim_params': sim_params,
    'gamma_S': gamma_S,
    'gamma_R': gamma_R
}

filename = f"cdf{tests}_results_L{L}N{N}K{K}.pkl"
with open(filename, 'wb') as outfile:
    pickle.dump(results, outfile)
# print(invZ)
# print(rates_hsdmR)
# print(rate_hsdmC)
# print(rate_bcd)
# print(sumrate_hsdmR)
# print(sumrate_hsdmC)
# print(sumrate_bcd)
