import numpy as np
import toolbox as funcs
import sim_funcs as sim
import pickle

area_params = {"delta_h": 10 # height difference
                    , "area": 100  # lxl Service Area
                    , "bandwidth": 100e6
                    , "L": 2  # APs
                    , "N": 2  # antennas per AP
                    , "K": 3  # UEs
                    , "carrier_frequency": 2000  # carrier frequency (MHz)
                    , "sigma_sf_los": 8  # Shadow fading LoS (dB)
                    , "AP_pos_rand": True
                                     }

sim_params = {"setups": 1000,
                     "batch":200,
                     "v":-1,
                     "d": 0,   #rician factor (0: no LoS component)
                     "norm_type" : np.inf,
                     "tests" :50
                    }

L = [2, 4, 10, 20]
N = [1, 2, 4, 20]
K = [2, 3, 10, 20]
kappa = [0, 5, 100]

for l in L:
    for n in N:
        for k in K:
            for rician in kappa:
                area_params["L"] = l
                area_params["N"] = n
                area_params["K"] = k
                sim_params["d"] = rician

                area_block = sim.create_areas(area_params, sim_params["tests"])

                file = f"Areas/results_L{l}N{n}K{k}_rician{rician}.pkl"
                with open(file, 'wb') as file:
                    pickle.dump(area_block, file)