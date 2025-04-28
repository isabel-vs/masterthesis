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
import cmath
import os
from scipy import integrate, linalg
from math import exp, pi, sin, cos, sqrt, radians


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
    sigma_sf_los = 8  # dB
    noiseVariancedB = -174 + pow2db(bandwidth) + 7  # Compute noise power (in dBm)

    # Initializations to store values
    Beta = np.zeros((L, K))
    R = np.zeros((L, K, N, N), dtype=complex)
    gainOverNoise_B = np.zeros((L, K))
    H_Mean = np.zeros((L, K, N), dtype=complex)
    kappa = np.zeros((L, K))
    dist_UEs_to_APs = np.zeros((L, K))

    # Deploying APs and UEs randomly within the service area
    UEpositions = (np.random.rand(K, 1) + 1j * np.random.rand(K, 1)) * area
    APpositions = (np.random.rand(L, 1) + 1j * np.random.rand(L, 1)) * area

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
    return R, gainOverNoise_B, H_Mean, kappa


def initial_access(params_dict, Beta):
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
    tau_p = params_dict["tau_p"]

    D_cellular = np.zeros((L, K))
    pilot_indicies_ = -1 * np.ones((K, 1))

    D_DCC = np.zeros((L,K))
    for k in range(0, K):
        gain_sorted = np.argsort(Beta[:,k])
        strongest_APs = gain_sorted[-2:]
        D_DCC[strongest_APs,k] = 1

        main_AP_index = Beta[:, k].argmax()  # master AP with the highest channel gain for UE k
        D_cellular[main_AP_index, k] = 1  # In cellular, only one AP with max channel gain is serving UE k
        pilot_indicies_ = pilot_assignment(k, main_AP_index, tau_p, pilot_indicies_, Beta)

    # D_DCC = createDCC(D_cellular, L, tau_p, Beta, pilot_indicies_)
    return D_cellular, D_DCC, pilot_indicies_


def channel_generation(params_dict, R, H_Mean, kappa):
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
    realizations = params_dict["realizations"]

    H_Mean = np.tile(H_Mean[:, :, :, np.newaxis], (1, 1, 1, realizations))

    H = sqrt(0.5) * (np.random.randn(L, K, N, realizations) + 1j * np.random.randn(L, K, N, realizations))

    for l in range(0, L):
        for k in range(0, K):
            H_Mean[l, k, :, :] = sqrt(kappa[l, k] / (1 + kappa[l, k])) * H_Mean[
                                                                         l, k, :, :]
            R[l, k, :, :] = (1 / (1 + kappa[l, k])) * R[l, k, :, :]
            H[l, k, :, :] = H_Mean[l, k, :, :] + linalg.sqrtm(R[l, k, :, :]) @ H[l, k, :, :]
    return H, H_Mean, R


def channel_estimation(params_dict, R, pilot_indicies_, H, H_Mean, p_k):
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
    tau_p = params_dict["tau_p"]
    realizations = params_dict["realizations"]
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


def compute_SINR_MRC(params_dict, p_k, D_DCC, H_hat, H):
    N = params_dict["N"]
    K = params_dict["K"]
    realizations = params_dict["realizations"]

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


def calculate_interf_matrix(params_dict, D_DCC, H_hat, H):
    """ calculate interference matrix (with MRC beamforming)
    """

    N = params_dict["N"]
    K = params_dict["K"]
    realizations = params_dict["realizations"]

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
    R = linalg.toeplitz(spatial_corr_UE)  # integral gives Nx1, we need to calculate NxN matrix from it
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
