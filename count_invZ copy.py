
import pickle
import numpy as np
import os
import pandas as pd
import toolbox as funcs
from tqdm import tqdm
import matplotlib.pyplot as plt

# ================= Configuration =================

DATA_DIR = "2026_01_15/areas"
OUTPUT_FILE = "2026_11_09/los.csv"

# Fixed parameters
L_fixed = 4
K_fixed = 2
N_fixed = 2

# Values of d to investigate
d_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Adjust as needed

# Base Parameters
network_params = {
    "realizations": 200,
    "tau_c": 200,
    "tau_p": 1,
    "p_k": 100,
    "csi_rho": 0.8
}

area_params = {
    "delta_h": 10,
    "area": 100,
    "bandwidth": 20e6,
    "L": L_fixed,
    "N": N_fixed,
    "K": K_fixed,
    "carrier_frequency": 20000,
    "sigma_sf_los": 8,
    "AP_pos_rand": "grid"
}

sim_params = {
    "setups": 1,
    "batch": network_params["realizations"],
    "v": -1,
    "d": 0,  # Updated in loop
    "norm_type": np.inf,
    "tests": 100  # Updated dynamically
}

p_max = network_params["p_k"]

# ================= Load Pre-simulated Data =================

filename = f"area_grid_L{L_fixed}N{N_fixed}K{K_fixed}.pkl"
file_path = os.path.join(DATA_DIR, filename)

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

with open(file_path, "rb") as f:
    area_block = pickle.load(f)

(R_block, B_block, HMean_block, kappa_block,
 UEpos_block, APpos_block) = area_block

num_tests = R_block.shape[0]

print(f"Loaded {filename} with {num_tests} scenarios.")
print(f"Processing {len(d_values)} values of d...")

# Update simulation parameters
current_sim_params = sim_params.copy()
current_sim_params["tests"] = num_tests

# ================= Main Processing =================

results_data = []

for d in tqdm(d_values, desc="d Loop"):

    current_sim_params["d"] = d

    count_invZ = 0
    count_Prop = 0

    # Iterate through all pre-simulated scenarios
    for i in range(num_tests):

        # Extract scenario data
        R_spatial_corr_norm = R_block[i, :, :, :, :]
        B_channel_gain = B_block[i, :, :]
        HMean_norm = HMean_block[i, :, :, :]
        k_rician_factor = kappa_block[i, :, :]
        UEpos = UEpos_block[i, :, :]
        APpos = APpos_block[i, :, :]

        area_values = (
            R_spatial_corr_norm,
            B_channel_gain,
            HMean_norm,
            k_rician_factor,
            UEpos,
            APpos
        )

        # Run simulation
        M, u, b, C, sigma, SINR, UEpos, APpos, D_DCC = funcs.run_sim(
            network_params,
            area_params,
            current_sim_params,
            area_values,
            i
        )

        # Construct matrices M_k
        A = np.eye(K_fixed)
        M_k = np.zeros((K_fixed, K_fixed, K_fixed))

        for k in range(K_fixed):
            M_k[:, :, k] = M + (u @ A[:, k][:, np.newaxis].T) / p_max

        # Check inverse-Z property
        inv_interf = True

        for k in range(K_fixed):
            if not funcs.is_invZ(M_k[:, :, k]):
                inv_interf = False
                break

        if inv_interf:
            count_invZ += 1

        # Check new property
        prop_interf = True

        for k in range(K_fixed):
            if not funcs.is_Prop(M_k[:, :, k]):
                prop_interf = False
                break

        if prop_interf:
            count_Prop += 1

    # Store results for this d
    results_data.append({
        "L": L_fixed,
        "N": N_fixed,
        "K": K_fixed,
        "d": d,
        "invZ_count": count_invZ,
        "Prop_count": count_Prop,
        "total_tests": num_tests,
        "invZ_percentage": (count_invZ / num_tests) * 100,
        "Prop_percentage": (count_Prop / num_tests) * 100
    })

# ================= Export =================

df = pd.DataFrame(results_data)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("\nProcessing complete.")
print(f"Results saved to {OUTPUT_FILE}")
print(df)

# ================= Plot =================

plt.figure(figsize=(8, 5))

# Sort by d to ensure correct line ordering
df = df.sort_values("d")

plt.plot(
    df["d"],
    df["invZ_count"],
    marker="o",
    linestyle="-",
    label="Inverse Z"
)

plt.plot(
    df["d"],
    df["Prop_count"],
    marker="s",
    linestyle="-",
    label="New Property"
)

plt.xlabel("Rician factor (LoS strength) $d$")
plt.ylabel("Number of successful scenarios")

plt.title(f"Number of successful scenarios vs. LoS strength (L={L_fixed}, N={N_fixed}, K={K_fixed})")

plt.xticks(d_values)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

plt.tight_layout()
plt.show()