import numpy as np
import toolbox as funcs
import pickle

area_params = {"delta_h": 10 # height difference
                    , "area": 100  # lxl Service Area
                    , "bandwidth": 20e6
                    , "L": 1  # APs
                    , "N": 1  # antennas per AP
                    , "K": 4  # UEs
                    , "carrier_frequency": 20000  # carrier frequency (MHz)
                    , "sigma_sf_los": 8  # Shadow fading LoS (dB)
                    , "AP_pos_rand": "grid"
                                     }

sim_params = {"setups": 1000,
                     "batch":200,
                     "v":-1,
                     "d": 0,   #rician factor (0: no LoS component)
                     "norm_type" : np.inf,
                     "tests" :50
                    }

L = [4]
N = [1]
K = [4]

# combos = [(16,1,6), (4,4,6), (1,16,6)]
combos = [(100, 2, 3)]
# combos = [(36, 1, 4), (1, 36, 4)]

start = 0
tests = sim_params["tests"]
APstyle = area_params["AP_pos_rand"]

# for l in L:
#     for n in N:
#         for k in K:
#             area_params["L"] = l
#             area_params["N"] = n
#             area_params["K"] = k

#             area_block = funcs.create_areas(area_params, tests, start)

#             # file = f"area_L{l}N{n}K{k}_{start}-{start+tests-1}.pkl"
#             file = f"test{tests}_area_L{l}N{n}K{k}.pkl"
#             with open(file, 'wb') as outfile:
#                 pickle.dump(area_block, outfile) 

for (l,n,k) in combos:
    area_params["L"] = l
    area_params["N"] = n
    area_params["K"] = k

    area_block = funcs.create_areas(area_params, tests, start)

    # file = f"area_L{l}N{n}K{k}_{start}-{start+tests-1}.pkl"
    file = f"test{tests}_{APstyle}_L{l}N{n}K{k}.pkl"
    with open(file, 'wb') as outfile:
        pickle.dump(area_block, outfile) 