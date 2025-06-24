import numpy as np
from scipy import sparse
from scipy import linalg

def hsdm_sinr(M_block, w, param):
    """
    INPUT:
        M_block: block of L positive square (N by N) matrices (NxNxL)
        w: positive weights of rates of users
        param['s_ini'] : initial vector for algorithm (must be positive)
        param['K'] and param['q'] : step size mu_k  = 1/max(w) if k <= K and (k-K)^(-q)
        param['stopCri'] : stopping criterion for algorithm
        param['maxIter']: number of maximum iteration
    
    OUTPUT:
        sinr_opt: optimized SINR of users
        sprad: spectral radius of diag(sinr_opt) * M
    """

    s = param['s_ini']
    gamma = 0
    beta = max(w)   # Lipschitz constant of gradient
    mu = 1/beta     # step-size for first K iterations (nonexpansive if 0 < mu < 2/beta)
    K = param['K']
    q = param['q']
    stopCri = param['stopCri']
    maxIter = param['maxIter']
    L = M_block.shape[2]

    for kk in range(maxIter+1):
        s_prev = s.copy()
        gamma_prev = gamma

        gamma_max = 0
        l_max = np.nan
        for l in range(L):
            gamma_l, _ = find_sprad(np.diag(s) @ M_block[:,:,l])
            
            if gamma_l > gamma_max:
                gamma_max = gamma_l
                l_max = l
        
        gamma = gamma_max
        M = M_block[:,:,l_max]

        _, xi = find_sprad(np.diag(s) @ M)
        _, eta = find_sprad(M.T @ np.diag(s))

        xi = xi / (xi @ eta)

        if gamma > 1:
            g = gamma * xi * eta / s
            s_hat = s - (gamma - 1) * g / (g @ g)
        else:
            s_hat = s
        
        s_til = np.maximum(0, s_hat)    # disregards b in the simulations

        if kk > K:
            mu = 1 / beta * (kk - K)**(-q)
        s = s_til + mu * w / (1+s_til)
            
        if gamma < 1 + stopCri and np.abs(gamma - gamma_prev) < stopCri and np.abs(w @ np.log(1+s) - w @ np.log(1+s_prev)) < stopCri:
            break
    
    sinr_opt = s
    sprad = gamma

    return sinr_opt, sprad


def hsdm_rate(M_block, w, param):
    """
    INPUT:
        M_block: block of L positive square (N by N) matrices (NxNxL)
        w: positive weights of rates of users
        param['r_ini'] : initial vector for algorithm (must be positive)
        param['K'] and param['q'] : step size mu_k  = 1/max(w) if k <= K and (k-K)^(-q)
        param['stopCri'] : stopping criterion for algorithm
        param['maxIter']: number of maximum iteration
    
    OUTPUT:
        sinr_opt: optimized SINR of users
        sprad: spectral radius of diag(sinr_opt) * M
    """

    r = param['r_ini']
    gamma = 0
    beta = max(w)   # Lipschitz constant of gradient
    mu = 1/beta     # step-size for first K iterations (nonexpansive if 0 < mu < 2/beta)
    K = param['K']
    q = param['q']
    stopCri = param['stopCri']
    maxIter = param['maxIter']
    L = M_block.shape[2]

    for kk in range(maxIter+1):
        r_prev = r.copy()
        gamma_prev = gamma

        gamma_max = 0
        l_max = np.nan
        for l in range(L):
            gamma_l, _ = find_sprad(np.diag(np.exp(r) - 1) @ M_block[:,:,l])
            
            if gamma_l > gamma_max:
                gamma_max = gamma_l
                l_max = l
        
        gamma = gamma_max
        M = M_block[:,:,l_max]

        _, xi = find_sprad(np.diag(np.exp(r) - 1) @ M)
        _, eta = find_sprad(M.T @ np.diag(np.exp(r) - 1))

        xi = xi / (xi @ eta)

        if gamma > 1:
            g = gamma * xi * eta / r
            r_hat = r - (gamma - 1) * g / (g @ g)
        else:
            r_hat = r
        
        r_til = np.maximum(0, r_hat)    # disregards b in the simulations

        if kk > K:
            mu = 1 / beta * (kk - K)**(-q)
        r = r_til + mu * w
            
        if gamma < 1 + stopCri and np.abs(gamma - gamma_prev) < stopCri and np.abs(w @ np.log(1+r) - w @ np.log(1+r_prev)) < stopCri:
            break
    
    rate_opt = r
    sprad = gamma

    return rate_opt, sprad


def find_sprad(A):
    """
    find spectral radius of matrix A (NxN)
    """
    N = A.shape[0]
    if N>2:
        val, vec = sparse.linalg.eigs(A, k=1, which='LR')
        sprad = np.real(val[0])
        vec = np.abs(vec)
    else:
        eigenvals, eigenvecs = linalg.eig(A)
        idx = np.argmax(np.real(eigenvals))
        sprad = np.real(eigenvals[idx])
        vec = np.abs(eigenvecs[:,idx]).flatten()
    return sprad, vec