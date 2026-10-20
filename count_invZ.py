import pickle
import numpy as np
import os
import pandas as pd
import toolbox as funcs
from tqdm import tqdm # Optional: for progress bar

# ================= Configuration =================
# Directory containing the pickle files
DATA_DIR = "2026_01_15/areas"
OUTPUT_FILE = "2026_01_21/invZ/invZ_summary_noisyMMSE.csv"

# Ranges based on your description
# L from 2^2 (4) to 10^2 (100)


L_values = [i**2 for i in range(2, 11)] 
# L_fixed = 4
# K from 2 to 10
K_values = list(range(2, 5))
N_fixed = 2
# N_values = [2, 4, 8, 12, 18, 24, 32, 40, 50]

# Base Parameters (static parts)
network_params = {
    "realizations": 200, 
    "tau_c": 200, 
    "tau_p": 1, 
    "p_k": 100,
    "csi_rho": 0.8
}

# Note: L, N, K in area_params will be updated in the loop
area_params = {
    "delta_h": 10, 
    "area": 100, 
    "bandwidth": 20e6, 
    "L": 4,  # Placeholder
    "N": 2,  # Placeholder
    "K": 2,  # Placeholder
    "carrier_frequency": 20000, 
    "sigma_sf_los": 8, 
    "AP_pos_rand": "grid"
}

sim_params = {
    "setups": 1,
    "batch": network_params["realizations"],
    "v": -1,
    "d": 0, 
    "norm_type": np.inf,
    "tests": 100 # Placeholder, will be detected from data
}

p_max = network_params["p_k"]

# ================= Main Processing =================

results_data = []

# iterate over all combinations
total_combinations = len(L_values) * len(K_values)
print(f"Starting processing for {total_combinations} file combinations...")

# Using tqdm for a progress bar (remove 'tqdm' wrapper if not installed)
for L in tqdm(L_values, desc="L Loop"):
    for K in K_values:
        
        # Construct filename
        filename = f"area_grid_L{L}N{N_fixed}K{K}.pkl"
        file_path = os.path.join(DATA_DIR, filename)

        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Skipping {filename}: File not found.")
            continue

        try:
            # Load Data
            with open(file_path, "rb") as f:
                area_block = pickle.load(f)

            (R_block, B_block, HMean_block, kappa_block, UEpos_block, APpos_block) = area_block
            
            # Determine number of tests dynamically from the data shape (index 0)
            # R_block shape is usually [tests, ..., ..., ...]
            num_tests = R_block.shape[0]

            # Update dictionaries for the simulation run
            current_area_params = area_params.copy()
            current_area_params["L"] = L
            current_area_params["N"] = N_fixed
            current_area_params["K"] = K
            
            # Update sim_params to match the file's test count
            current_sim_params = sim_params.copy()
            current_sim_params["tests"] = num_tests

            count_invZ = 0
            
            # Iterate through realizations in the file
            for i in range(num_tests):
                # Extract slices
                R_spatial_corr_norm = R_block[i,:,:,:,:]
                B_channel_gain =  B_block[i,:,:]
                HMean_norm = HMean_block[i,:,:,:]
                k_rician_factor = kappa_block[i,:,:]
                UEpos = UEpos_block[i,:,:]
                APpos = APpos_block[i,:,:]

                area_values = (R_spatial_corr_norm, B_channel_gain, HMean_norm, 
                               k_rician_factor, UEpos, APpos)

                # Run simulation
                # passing current_area_params to ensure funcs uses the correct L/K dimensions
                M, u, b, C, sigma, SINR, UEpos, APpos, D_DCC = funcs.run_sim(
                    network_params, current_area_params, current_sim_params, area_values, i
                )

                # Check Invertibility (User Logic)
                A = np.eye(K)
                M_k = np.zeros((K, K, K))
                inv_interf = True
                
                for k in range(K):
                    # User formula
                    M_k[:,:,k] = M + (u @ A[:,k][:,np.newaxis].T) / p_max
                    
                    if funcs.is_invZ(M_k[:,:,k]) == False:
                        inv_interf = False
                        break
                
                if inv_interf:
                    count_invZ += 1

            # Store result
            results_data.append({
                "L": L,
                "N": N_fixed,
                "K": K,
                "invZ_count": count_invZ,
                "total_tests": num_tests,
                "percentage": (count_invZ / num_tests) * 100
            })

        except Exception as e:
            print(f"Error processing {filename}: {e}")

# ================= Export =================

df = pd.DataFrame(results_data)
df.to_csv(OUTPUT_FILE, index=False)

print("\nProcessing complete.")
print(f"Results saved to {OUTPUT_FILE}")
print(df.head())