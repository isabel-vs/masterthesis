# -*- coding: utf-8 -*-
"""
Simulation code for "On the Optimal Performance of Distributed Cell-Free Massive MIMO with LoS Propagation", IEEE Wireless Communications and Networking Conference (WCNC), March 2025
Author: Noor Ul Ain

License: This code is licensed under the GPLv2 license. If you in any way
use this code for research that results in publications, please cite our
paper as described in the README file

This file has all the functions required to simulate an uplink transmission in (both centralized and distributed) cellfree massive MIMO networks in the presence of los propagation.
"""

import copy
import numpy as np
# import itertools
import cmath
# import os
# import time
from scipy import integrate, linalg
from math import exp, pi, sin, cos, sqrt, radians

def create_areas(params_dict, tests, start=0):
    L = params_dict["L"]
    N = params_dict["N"]
    K = params_dict["K"]

    R_spatial_corr_norm = np.zeros((tests,L, K, N, N), dtype=complex)
    B_channel_gain = np.zeros((tests, L, K))
    HMean_norm = np.zeros((tests, L, K, N), dtype=complex)
    k_rician_factor = np.zeros((tests, L, K))
    UEpos = np.zeros((tests, K, 1), dtype=complex)
    APpos = np.zeros((tests, L, 1), dtype=complex)

    for i in range(tests):
        print("test ", i+1, "/", tests)
        np.random.seed(i+start)
        R_spatial_corr_norm[i,:,:,:,:], B_channel_gain[i,:,:], HMean_norm[i,:,:,:], k_rician_factor[i,:,:], UEpos[i,:,:], APpos[i,:,:] = simulation_area(params_dict)

    # with open('vars.pkl', 'wb') as file:
    #     pickle.dump((R_spatial_corr_norm, B_channel_gain, HMean_norm, k_rician_factor, UEpos, APpos), file)

    return (R_spatial_corr_norm, B_channel_gain, HMean_norm, k_rician_factor, UEpos, APpos)

def run_sim(network_params, area_params, simulation_params, area_values, seed):
    setups = simulation_params["setups"]
    batch = simulation_params["batch"]
    v = simulation_params["v"]
    d = simulation_params["d"]
    norm_type = simulation_params["norm_type"]

    K = area_params["K"]
    SINR = np.zeros((K,setups))

    np.random.seed(seed)
    p_max=network_params["p_k"]

    # # A network setup is generated, distances and spatial correlation matrices are calculated
    # R_spatial_corr_norm, B_channel_gain, HMean_norm, k_rician_factor, UEpos, APpos = simulation_area(network_params)

    (R_spatial_corr_norm, B_channel_gain, HMean_norm, k_rician_factor, UEpos, APpos) = area_values

    # Cooperation clusters formation and pilot assignment
    D_cellular, D_DCC, pilot_indicies = initial_access(network_params, area_params,B_channel_gain)

    # for same network setup , k-factor for all channels is set equal to 0 and in next iteration to 5 to cover both cases.
    k_rician_factor[:]=d

    # Generate random channels, channel mean(los component) and spatial correlation matrix
    H, HMean, R_spatial_corr = channel_generation(network_params, area_params, R_spatial_corr_norm,HMean_norm,k_rician_factor)

    # Estimate channels using MMSE estimator  (CSI is obtained)
    H_hat = channel_estimation(network_params, area_params, R_spatial_corr, pilot_indicies, H,
                                                        HMean, p_max)
    
    #reshaping
    H = np.moveaxis(H, -1, 0)
    H_hat = np.moveaxis(H_hat, -1, 0)
    H = np.swapaxes(H, 2, 3)
    H_hat = np.swapaxes(H_hat, 2, 3)
    K=area_params["K"]

    M, u, b, C, sigma = calculate_interf_matrix(network_params, area_params, D_DCC, H_hat, H)

    for i in range(setups):
        p_inital = np.random.uniform(low=np.nextafter(0,1), high=np.nextafter(1,2), size=(K,1))     #draw random initial power for each user ]0,1]
        p_k = p_max / np.linalg.norm(p_inital, norm_type) * p_inital   # Kx1 power vector for all UEs (scaled such that power constraint fulfilled with equality)

        # SINR[:,i][:,np.newaxis]= compute_SINR_MRC(network_params, area_params, p_k, D_DCC, H_hat, H)
        SINR[:,i][:,np.newaxis] = sinr(M,u,p_k)
        # print("Progress: ", i, "/",setups)
 
    return M, u, b, C, sigma, SINR, UEpos, APpos, D_DCC

def run_test(network_params, area_params, sim_params, area_block, tau_p):
    network_params["tau_p"] = tau_p
    (R_block, B_block, HMean_block, kappa_block, UEpos_block, APpos_block) = area_block

    num_inv = 0
    for i in range(sim_params["tests"]):
        R_spatial_corr_norm = R_block[i,:,:,:,:]
        B_channel_gain =  B_block[i,:,:]
        HMean_norm = HMean_block[i,:,:,:]
        k_rician_factor = kappa_block[i,:,:]
        UEpos = UEpos_block[i,:,:]
        APpos = APpos_block[i,:,:]

        area_values = (R_spatial_corr_norm, B_channel_gain, HMean_norm, k_rician_factor, UEpos, APpos)

        p_max=network_params["p_k"]
        K = area_params["K"]
        A = 1/p_max * np.eye(K)

        M, u, b, C, sigma, SINR, UEpos, APpos, D_DCC = run_sim(network_params, area_params, sim_params, area_values, i)
        M_k = np.zeros((K,K,K))
        for k in range(K):
            M_k[k,:,:] = M + u @ A[:,k][:,np.newaxis].T
            # M_k_inv = np.linalg.inv(M_k[k,:,:])
            if not is_invZ(M_k[k,:,:]):
                # print(i, ": no")
                break
        if k == K-1:
            # print(i, ": yes")
            num_inv += 1

    print("number of inv Z interf. matrixes:", num_inv)
    return num_inv

#------------------

def is_ZMatrix(matrix):
    off_diag_mask = ~np.eye(matrix.shape[0], dtype=bool)
    return np.all(matrix[off_diag_mask] <= 0)

def is_invZ(M):

    K = M.shape[0]

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("Input must be a square matrix")

    if np.min(M) < 0:          # check if nonnegative matrix
        return False
    if np.size(M)==(2,2):      # if 2x2 matrix, inv-Z iff det(M)>0
        if np.linalg.det(M) > 0:
            return True
        return False
    
    # check all reduced interference matrices
    d = np.diag(M)
    i, j = np.triu_indices_from(M, k=1)
    dets = d[i] * d[j] - M[i,j] * M[j,i]
    if not np.all(dets>0):
        return False

    # last check: invert matrix
    M_inv = np.linalg.inv(M)
    if is_ZMatrix(M_inv):
        return True
    return False
    

def sinr(M, u, p):
    K = M.shape[0]
    sinr = np.zeros((K,1))
    interference = M @ p + u
    for k in range(K):
        sinr[k] = p[k] / interference[k]
    return sinr


def simulation_area(params_dict):
    """
    Functions to generate one network scenario with random placement of APs and UEs.
    For each UE and AP pair, it returns k-factor, channel gain, spatial correlation, channel mean
    Input:
        network parameters
    Output:
        R: LxKxNxN Spatial correlation matrix of NxN size between each lth AP and kth UE with kappa component
        gainOverNoise_B: LxK, Channel gain between each UE and AP
        H_Mean: LxKxN LOS component with kappa component
        kappa : LxK, los strength k-factor between each UE and AP
    """

    area = params_dict["area"]
    L = params_dict["L"]
    N = params_dict["N"]
    K = params_dict["K"]
    bandwidth = params_dict["bandwidth"]
    fc = params_dict["carrier_frequency"]
    h = params_dict["delta_h"]  # 11m
    antennaSpacing = 1 / 2  # half wavelength
    sigma_sf_los = params_dict["sigma_sf_los"]  # dB
    noiseVariancedB = -174 + pow2db(bandwidth) + 7  # Compute noise power (in dBm)
    AP_pos_rand = params_dict["AP_pos_rand"]

    # Initializations to store values
    Beta = np.zeros((L, K))
    R = np.zeros((L, K, N, N), dtype=complex)
    gainOverNoise_B = np.zeros((L, K))
    H_Mean = np.zeros((L, K, N), dtype=complex)
    kappa = np.zeros((L, K))
    dist_UEs_to_APs = np.zeros((L, K))

    # Deploying APs and UEs randomly within the service area
    UEpositions = (np.random.rand(K, 1) + 1j * np.random.rand(K, 1)) * area

    if AP_pos_rand:
        APpositions = (np.random.rand(L, 1) + 1j * np.random.rand(L, 1)) * area     # rand positions AP
    else:
        grid_points = int(np.sqrt(L))
        block_size = area/grid_points
        # Create a grid of real and imaginary parts
        real_part = np.linspace(block_size/2, area-block_size/2, grid_points)
        imag_part = np.linspace(block_size/2, area-block_size/2, grid_points)
        # Create a meshgrid for the real and imaginary parts
        real_grid, imag_grid = np.meshgrid(real_part, imag_part)

        # Combine the real and imaginary parts into complex numbers
        APpositions = real_grid + 1j * imag_grid

        # Flatten the grid and reshape to (L, 1)
        APpositions = APpositions.flatten().reshape(L, 1)

    # wraping-around to mimic infinite area
    wrapHorizontal = np.tile([-area, 0, area], (3, 1))
    wrapVertical = wrapHorizontal.T
    wrapLocations = wrapHorizontal.T.flatten() + 1j * wrapVertical.T.flatten()
    APpositionsWrapped = np.tile(APpositions, [1, len(wrapLocations)]) + np.tile(wrapLocations, [L, 1])

    # Calculating the shortest distance of UE from each AP and finding nearest AP
    for k in range(0, K):
        dist_each_UE_to_APs = abs(APpositionsWrapped - np.tile(UEpositions[k, :], (L, 9))).min(axis=1)
        nearest_AP = abs(APpositionsWrapped - np.tile(UEpositions[k, :], (L, 9))).argmin(axis=1)
        dist_UEs_to_APs[:, k] = np.sqrt(h ** 2 + dist_each_UE_to_APs ** 2)

    # Calculated rician factor using equation 12
    kappa[:] = db2pow(13 - 0.03 * dist_UEs_to_APs[:])

    for k in range(0, K):
        # step 4: Calculate gain channel using equation (between 11 and 12)
        Beta[:, k] = 35.4 - 20 * np.log10(fc) - 26 * np.log10(dist_UEs_to_APs[:, k])  # lx1

        gainOverNoise_B[:, k] = Beta[:, k] - noiseVariancedB + (
                np.random.randn(L) * sigma_sf_los)  # In power:gain*shadowing/noise

        # Calculate spatial correlation matrix R
        for l in range(0, L):
            theta = np.arcsin(h / dist_UEs_to_APs[l, k])
            phi = np.angle(UEpositions[k] - APpositionsWrapped[l, nearest_AP[l]], deg=False)[0]

            R[l, k, :, :] = db2pow(gainOverNoise_B[l, k]) * calculate_local_scattering(N, phi, theta,
                                                                                       antennaSpacing)
            # calculating channel mean (LOS component) from equation 41
            H_Mean[l, k, :] = sqrt(db2pow(gainOverNoise_B[l, k])) * np.exp(
                1j * 2 * pi * antennaSpacing * np.arange(0, N) * sin(phi) * cos(theta))
    return R, gainOverNoise_B, H_Mean, kappa, UEpositions, APpositions


def initial_access(network_params, params_dict, Beta):
    """
    Function for pilot assignment to UEs and cooperation cluster formation using algorithm in section 5.2 (step 6)
    Input:
        Beta: LxK  Channel gains between L APs and K UEs
    Output:
        pilot_indicies_: Kx1: list consisting of index of assigned pilot to each UE
        D_DCC: LxK, D matrix determining which AP will serve which UE. D[l,k] =1 means lth AP will serve kth UE
        D_cellular: LxK, D matrix for cellular/smallcell. Each UE will be served by one AP
    """
    L = params_dict["L"]
    K = params_dict["K"]
    tau_p = network_params["tau_p"]

    D_cellular = np.zeros((L, K))
    pilot_indicies_ = -1 * np.ones((K, 1))

    D_DCC = np.zeros((L,K))
    for k in range(0, K):
        # gain_sorted = np.argsort(Beta[:,k])
        # strongest_APs = gain_sorted[-2:]
        # D_DCC[strongest_APs,k] = 1

        main_AP_index = Beta[:, k].argmax()  # master AP with the highest channel gain for UE k
        D_cellular[main_AP_index, k] = 1  # In cellular, only one AP with max channel gain is serving UE k
        pilot_indicies_ = pilot_assignment(k, main_AP_index, tau_p, pilot_indicies_, Beta)

    D_DCC = createDCC(D_cellular, L, tau_p, Beta, pilot_indicies_)
    return D_cellular, D_DCC, pilot_indicies_

def createDCC(D_MasterAps, L, tau_p, Beta, pilot_indicies_):  # Cooperation cluster formation using Algo 4.1 from [2]
    D_ = copy.deepcopy(D_MasterAps)
    for l_ in range(0, L):
        for t in range(0, tau_p):
            ue_mask = (pilot_indicies_[:] == t).flatten().astype(int)
            ue_mask[ue_mask == 0] = -1
            ue_index = (db2pow(Beta[l_, :]) * ue_mask).argmax()
            D_[l_, ue_index] = 1
    return D_

def channel_generation(network_params, params_dict, R, H_Mean, kappa):
    # Function to generate los channels according to defination in section 5
    # Channel realizations are drawn from Gaussian distribution ~N_c(h_mean,R) to capture los and small scale fading.
    """
    Input:
        R: LxKxNxN Spatial correlation matrix of NxN size between each lth AP and kth UE
        H_Mean: LxKxN Rician LoS component (channel mean)
        kappa: LxK Rician factor between L APs and K UEs

    Output:
        H: LxKxNxrealizations, random channel realizations
        H_Mean: LxKxN LOS component with kappa component
        R: LxKxNxN Spatial correlation matrix of NxN size between each lth AP and kth UE with kappa component
    """
    L = params_dict["L"]
    N = params_dict["N"]
    K = params_dict["K"]
    realizations = network_params["realizations"]

    H_Mean = np.tile(H_Mean[:, :, :, np.newaxis], (1, 1, 1, realizations))

    H = sqrt(0.5) * (np.random.randn(L, K, N, realizations) + 1j * np.random.randn(L, K, N, realizations))

    for l in range(0, L):
        for k in range(0, K):
            H_Mean[l, k, :, :] = sqrt(kappa[l, k] / (1 + kappa[l, k])) * H_Mean[
                                                                         l, k, :, :]
            R[l, k, :, :] = (1 / (1 + kappa[l, k])) * R[l, k, :, :]
            H[l, k, :, :] = H_Mean[l, k, :, :] + linalg.sqrtm(R[l, k, :, :]) @ H[l, k, :, :]
    return H, H_Mean, R


def channel_estimation(network_params, params_dict, R, pilot_indicies_, H, H_Mean, p_k):
    """ Function to calculate the channel estimates, see section 2B

    Input:
        R: LxKxNxN Normalized Spatial correlation matrix of NxN size between each lth AP and kth UE
        pilot_indicies_: Kx1: list consisting of index of assigned pilot to each UE
        H: LxKxNxrealizations, random channel realizations
        H_Mean: LxKxNxRealizations Rician LoS component (channel mean)
        p_k: 1 scalar value. Same uplink pilot transmit power for all UEs in mW

    Output:
        H_hat: LxKxNxrealizations,  channel estimates
    """

    L = params_dict["L"]
    N = params_dict["N"]
    K = params_dict["K"]
    tau_p = network_params["tau_p"]
    realizations = network_params["realizations"]
    p_k = np.full(K, p_k)
    H_hat = np.zeros((L, K, N, realizations), dtype=complex)
    N_tk = sqrt(0.5) * (np.random.randn(N, realizations, L, tau_p) + 1j * np.random.randn(N, realizations, L, tau_p))

    for l in range(0, L):
        for t in range(0, tau_p):
            dim_array = np.ones((1, H[l, (t == pilot_indicies_).flatten(), :, :].ndim), int).ravel()
            dim_array[0] = -1
            p_k_reshaped = p_k.reshape(dim_array)
            # received pilot signal at each AP from all UEs receiving on that pilot
            y_pilot = np.sum(
                np.sqrt(p_k_reshaped[(t == pilot_indicies_).flatten()]) * tau_p * H[l, (t == pilot_indicies_).flatten(),
                                                                                  :, :], axis=0) + sqrt(tau_p) * N_tk[:,
                                                                                                                 :, l,
                                                                                                                 t]
            y_mean = np.sum(np.sqrt(p_k_reshaped[(t == pilot_indicies_).flatten()]) * tau_p * H_Mean[l, (
                                                                                                                t == pilot_indicies_).flatten(),
                                                                                              :, :], axis=0)
            Psi = np.sum(
                p_k_reshaped[(t == pilot_indicies_).flatten()] * tau_p * R[l, (t == pilot_indicies_).flatten(), :, :],
                axis=0) + np.eye(N)

            # Go through all UEs that use pilot t
            for k in np.where(np.any(pilot_indicies_ == t, axis=1))[0]:
                # Compute the MMSE estimate
                RPsi = R[l, k, :, :] @ np.linalg.inv(Psi)
                H_hat[l, k, :, :] = H_Mean[l, k, :, :] + np.sqrt(p_k[k]) * RPsi @ (y_pilot - y_mean)

    return H_hat


def compute_SINR_MRC(network_params,params_dict, p_k, D_DCC, H_hat, H):
    N = params_dict["N"]
    K = params_dict["K"]
    realizations = network_params["realizations"]

    SINR = np.zeros((K,1), dtype=float)

    for k in range(0, K):
        # Compute the number of APs that serve UE k
        servingAP = np.asarray(np.where(D_DCC[:, k] == 1)).reshape(-1)
        La = len(servingAP)

        H_k = H[:, servingAP, :, :]  # RxLaxNxK  channels from only serving APs of UE k
        H_k = np.reshape(H_k, [H_k.shape[0], -1, H_k.shape[-1]])  # Rx(La.N)xK

        Hhat_k = H_hat[:, servingAP, :, :]  # RxLaxNxK  channel estimates  for only serving APs of UE k
        Hhat_k = np.reshape(Hhat_k, [Hhat_k.shape[0], -1, Hhat_k.shape[-1]])  # Rx(La.N)xK

        v_k = np.zeros((realizations, La * N, 1), dtype=complex)
        v_k = Hhat_k[:,:,k][:, :, np.newaxis]

        SINR[k,:] = compute_SINR_uatf(v_k, H_k, p_k, k)

    return SINR


def calculate_interf_matrix(network_params, params_dict, D_DCC, H_hat, H):
    """ calculate interference matrix (with MRC beamforming)
    """

    N = params_dict["N"]
    K = params_dict["K"]
    realizations = network_params["realizations"]

    b = np.zeros((K,1))
    C = np.zeros((K,K))
    sigma = np.zeros((K,1))

    for k in range(0, K):
        # Compute the number of APs that serve UE k
        servingAP = np.asarray(np.where(D_DCC[:, k] == 1)).reshape(-1)
        La = len(servingAP)

        H_k = H[:, servingAP, :, :]  # RxLaxNxK  channels from only serving APs of UE k
        H_k = np.reshape(H_k, [H_k.shape[0], -1, H_k.shape[-1]])  # Rx(La.N)xK

        Hhat_k = H_hat[:, servingAP, :, :]  # RxLaxNxK  channel estimates  for only serving APs of UE k
        Hhat_k = np.reshape(Hhat_k, [Hhat_k.shape[0], -1, Hhat_k.shape[-1]])  # Rx(La.N)xK

        v_k = np.zeros((realizations, La * N, 1), dtype=complex)
        v_k = Hhat_k[:,:,k][:, :, np.newaxis]

        v_k_H = H_(v_k)  # Rx1x(la.N) # taking hermition and reshaping for multiplications
        b[k] = abs(np.mean(v_k_H @ H_k[:, :, k, np.newaxis], axis=0) ** 2)
        sigma[k] = np.mean(np.sum(abs(v_k[:, :, 0] ** 2), axis=1))
        C[:,k][:,np.newaxis] = np.mean(abs((v_k_H @ H_k) ** 2), axis=0).T
        C[k,k] = C[k,k] - abs(np.mean(v_k_H @ H_k[:, :, k, np.newaxis], axis=0) ** 2)

    b = b[:,0]
    
    M = np.linalg.inv(np.diag(b)) @ C.T     #interference matrix
    u = np.linalg.inv(np.diag(b)) @ sigma

    return M, u, b, C, sigma


def compute_SINR_uatf(v_k, H_k, p_k, k):
    """
    Function to calculate SINR for UatF bound
    Input:
        v_k: Rx(la.N)x1 receive combining vectors for all serving APs of kth UE for all small scale realizations
        H_k: Rx(La.N)xK  channel of N size between serving APs of kth UE and all other UEs for all small scale realizations
        p_k: uplink transmit power of pilot for kth UE
        k: index of UE
    Output:
        SINR: 1x1 SINR for kth UE for UatF lower bound
    """
    v_k_H = H_(v_k)  # Rx1x(la.N) # taking hermition and reshaping for multiplications
    numerator = (p_k[k] * abs(np.mean(v_k_H @ H_k[:, :, k, np.newaxis], axis=0) ** 2)).flatten()
    denominator = np.sum(p_k.T * np.mean(abs((v_k_H @ H_k) ** 2), axis=0), axis=1).flatten() + np.mean(
        np.sum(abs(v_k[:, :, 0] ** 2), axis=1)) - numerator
    SINR = numerator / denominator
    return SINR


def calculate_local_scattering(N, varPhi, varTheta, varAntennaSpacing):
    """ Calculating normalized spatial correlation matrix between one AP and one UE
    Input:
        N: 1x1 total number of antennas at AP
        varPhi: 1x1 azimuth angle
        varTheta: 1x1 angle of elevation
        varAntennaSpacing: 1x1 antenna spacing from one antenna to the next in one AP
    Output:
        R: NxN Normalized spatial correlation matrix
    """
    varASD_phi = radians(5)  # angular standard deviation for phi
    varASD_theta = radians(5)  # angular standard deviation for theta
    spatial_corr_UE = np.zeros((N, 1), dtype=complex)
    spatial_corr_UE[0] = 1
    # Go through all the columns of the first row
    for n in range(1, N):
        # Calculating the joint pdf of Phi and Theta
        inside_integral_real = lambda varthetaDelta, varphiDelta: (cmath.exp(
            1j * 2 * pi * varAntennaSpacing * n * sin(varPhi + varphiDelta) * cos(varTheta + varthetaDelta)) * (
                                                                       exp(-varphiDelta ** 2 / (
                                                                               2 * varASD_phi ** 2))) * (
                                                                       exp(-varthetaDelta ** 2 / (
                                                                               2 * varASD_theta ** 2))) / (
                                                                           2 * pi * varASD_theta * varASD_phi)).real
        inside_integral_img = lambda varthetaDelta, varphiDelta: (cmath.exp(
            1j * 2 * pi * varAntennaSpacing * n * sin(varPhi + varphiDelta) * cos(varTheta + varthetaDelta)) * (
                                                                      exp(-varphiDelta ** 2 / (
                                                                              2 * varASD_phi ** 2))) * (
                                                                      exp(-varthetaDelta ** 2 / (
                                                                              2 * varASD_theta ** 2))) / (
                                                                          2 * pi * varASD_theta * varASD_phi)).imag

        res_real = integrate.dblquad(inside_integral_real, -8 * varASD_phi, 8 * varASD_phi, -8 * varASD_theta,
                                     8 * varASD_theta)
        res_imag = integrate.dblquad(inside_integral_img, -8 * varASD_phi, 8 * varASD_phi, -8 * varASD_theta,
                                     8 * varASD_theta)
        spatial_corr_UE[n] = res_real[0] + 1j * res_imag[0]

    # Compute the spatial correlation matrix by utilizing the Toeplitz structure
    R = linalg.toeplitz(np.ravel(spatial_corr_UE))  # integral gives Nx1, we need to calculate NxN matrix from it
    R = R / np.trace(R)  # normalizing
    return R


def pilot_assignment(k_, main_AP, tau_p, pilot_index, channel_gain):  # pilot assignment using Algo 4.1 from [2]
    if k_ <= tau_p - 1:
        pilot_index[k_] = k_
    else:
        pilotinterference = np.zeros((tau_p, 1))
        for t in range(0, tau_p):
            mask = (pilot_index[:] == t).flatten()
            pilotinterference[t] = np.sum(
                db2pow(channel_gain[main_AP, mask]))  # rows where UE's main_AP using same pilot for other UEs
        pilot_index[k_] = pilotinterference.argmin()  # pilot on which there's min interference is assignd to UE
    return pilot_index


def db2pow(ydb):  # converts dB to power
    return np.power(10, ydb / 10)


def pow2db(y):  # converts power to dB
    return 10 * np.log10(y)


def H_(vec):  # Hermitian of more than 2-dimensional  matrix
    dim = vec.ndim - 1
    vec_H = np.conj(np.swapaxes(vec, dim, dim - 1))
    return vec_H
