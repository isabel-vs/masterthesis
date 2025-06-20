import numpy as np
import toolbox as funcs

def create_areas(params_dict, tests):
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
        # print("test ", i+1, "/", tests)
        np.random.seed(i)
        R_spatial_corr_norm[i,:,:,:,:], B_channel_gain[i,:,:], HMean_norm[i,:,:,:], k_rician_factor[i,:,:], UEpos[i,:,:], APpos[i,:,:] = funcs.simulation_area(params_dict)

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
    # R_spatial_corr_norm, B_channel_gain, HMean_norm, k_rician_factor, UEpos, APpos = funcs.simulation_area(network_params)

    (R_spatial_corr_norm, B_channel_gain, HMean_norm, k_rician_factor, UEpos, APpos) = area_values

    # Cooperation clusters formation and pilot assignment
    D_cellular, D_DCC, pilot_indicies = funcs.initial_access(network_params, area_params,B_channel_gain)

    # for same network setup , k-factor for all channels is set equal to 0 and in next iteration to 5 to cover both cases.
    k_rician_factor[:]=d

    # Generate random channels, channel mean(los component) and spatial correlation matrix
    H, HMean, R_spatial_corr = funcs.channel_generation(network_params, area_params, R_spatial_corr_norm,HMean_norm,k_rician_factor)

    # Estimate channels using MMSE estimator  (CSI is obtained)
    H_hat = funcs.channel_estimation(network_params, area_params, R_spatial_corr, pilot_indicies, H,
                                                        HMean, p_max)
    
    #reshaping
    H = np.moveaxis(H, -1, 0)
    H_hat = np.moveaxis(H_hat, -1, 0)
    H = np.swapaxes(H, 2, 3)
    H_hat = np.swapaxes(H_hat, 2, 3)
    K=area_params["K"]

    M, u, b, C, sigma = funcs.calculate_interf_matrix(network_params, area_params, D_DCC, H_hat, H)

    for i in range(setups):
        p_inital = np.random.uniform(low=np.nextafter(0,1), high=np.nextafter(1,2), size=(K,1))     #draw random initial power for each user ]0,1]
        p_k = p_max / np.linalg.norm(p_inital, norm_type) * p_inital   # Kx1 power vector for all UEs (scaled such that power constraint fulfilled with equality)

        # SINR[:,i][:,np.newaxis]= funcs.compute_SINR_MRC(network_params, p_k, D_DCC, H_hat, H)
        SINR[:,i][:,np.newaxis] = funcs.sinr(M,u,p_k)
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
            if not funcs.is_invZ(M_k[k,:,:]):
                # print(i, ": no")
                break
        if k == K-1:
            # print(i, ": yes")
            num_inv += 1

    print("number of inv Z interf. matrixes:", num_inv)
    return num_inv