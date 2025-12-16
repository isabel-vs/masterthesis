import numpy as np
from sumrate_max import *
import toolbox as funcs
import pickle
import time
import matplotlib.pyplot as plt

network_params = {"realizations": 1000  #(200 originally) small scale realizations  (Critical : should be a large value if large number APs and UEs are used)
                    , "tau_c": 200 # total samples in coherence block
                    , "tau_p": 3  # samples for pilots
                    , "p_k": 200  # uplink max power (mW)
                                     }

area_params = {"delta_h": 10 # height difference
                    , "area": 500  # lxl Service Area
                    , "bandwidth": 20e6
                    , "L": 1  # APs
                    , "N": 1  # antennas per AP
                    , "K": 4 # UEs
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

filename = f"cdf{tests}_results_L{L}N{N}K{K}.pkl"

cdf_results = pickle.load(open(filename, "rb"))

invZ = cdf_results['invZ']
sumrate_hsdmS = cdf_results['sumrate_hsdmS']
sumrate_hsdmR = np.array(cdf_results['sumrate_hsdmR'])
sumrate_hsdmC = cdf_results['sumrate_hsdmC']
sumrate_bcd = np.array(cdf_results['sumrate_bcd'])
sumrate_maxPow = cdf_results['sumrate_maxPow']

thresh = 0.01
diff = np.abs(sumrate_hsdmR-sumrate_bcd)
idx = np.where(diff>thresh)
print(idx)