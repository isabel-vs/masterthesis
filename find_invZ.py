import numpy as np
import toolbox as funcs
import pickle

network_params = {"realizations": 200  #(200 originally) small scale realizations  (Critical : should be a large value if large number APs and UEs are used)
                    , "tau_c": 200 # total samples in coherence block
                    , "tau_p": 100  # samples for pilots
                    , "p_k": 200  # uplink max power (mW)
                                     }

area_params = {"delta_h": 10 # height difference
                    , "area": 500  # lxl Service Area
                    , "bandwidth": 20e6
                    , "L": 16  # APs
                    , "N": 4  # antennas per AP
                    , "K": 4  # UEs
                    , "carrier_frequency": 2000  # carrier frequency (MHz)
                    , "sigma_sf_los": 8  # Shadow fading LoS (dB)
                    , "AP_pos_rand": False
                                     }

sim_params = {"setups": 1,
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

filename = f"area_L{L}N{N}K{K}_0-49.pkl"
area_block = pickle.load(open(filename, "rb"))

(R_block, B_block, HMean_block, kappa_block, UEpos_block, APpos_block) = area_block

idx_invZ = []

for i in range(tests):
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
        if funcs.is_invZ(M_k[:,:,k]) == False:
            inv_interf = False
            break
    if inv_interf:
        idx_invZ.append(i)

print(idx_invZ)
print(len(idx_invZ))