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
    # beta = max(w)   # Lipschitz constant of gradient
    # mu = 1/beta     # step-size for first K iterations (nonexpansive if 0 < mu < 2/beta)
    K = param['K']
    q = param['q']
    stopCri = param['stopCri']
    maxIter = param['maxIter']
    L = M_block.shape[2]

    beta = param['beta']
    # beta = max(w)   # Lipschitz constant of gradient
    mu = param['mu']

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

        xi = xi / (xi.T @ eta)

        if gamma > 1:
            g = gamma * xi * eta / s
            s_hat = s - (gamma - 1) * g / (g @ g)
        else:
            s_hat = s
        
        s_til = np.maximum(0, s_hat)    # disregards b in the simulations
        s_til = np.minimum(s_til, np.exp(20))

        if kk > K:
            mu = 1 / beta * (kk - K)**(-q)
        s = s_til + mu * w / (1+s_til)
        
        if gamma < 1 + stopCri and np.abs(gamma - gamma_prev) < stopCri and np.abs(w @ np.log(1+s) - w @ np.log(1+s_prev)) < stopCri:
            break

    # scale to get back to feasible region (sprad=1)
    s_scaled = 1/gamma * s

    gamma_max = 0
    l_max = np.nan
    for l in range(L):
        gamma_l, _ = find_sprad(np.diag(s_scaled) @ M_block[:,:,l])
        
        if gamma_l > gamma_max:
            gamma_max = gamma_l
            l_max = l
     
    sinr_opt = s_scaled
    sprad = gamma_max

    return sinr_opt, sprad, gamma


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
    # beta=10
    beta = param['beta']
    # beta = max(w)   # Lipschitz constant of gradient
    mu = param['mu']
    # mu = 1/beta     # step-size for first K iterations (nonexpansive if 0 < mu < 2/beta)
    # mu = 0.001
    K = param['K']
    q = param['q']
    stopCri = param['stopCri']
    maxIter = param['maxIter']
    L = M_block.shape[2]

    sol = param['solution']
    conv_prog = [np.linalg.norm(r-sol)]
    rate_prog = [r]

    for kk in range(1,maxIter+1):
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
            g = gamma * xi * eta * np.exp(r)/ (np.exp(r) - 1)
            r_hat = r - (gamma - 1) * g / (g @ g)
        else:
            r_hat = r
        
        r_til = np.maximum(0, r_hat)    # disregards b in the simulations
        r_til = np.minimum(r_til, 20)

        if kk > K:
            # mu = 1 / beta * (kk - K)**(-q)
            mu = 1 / beta * kk**(-q)
        r = r_til + mu * w
        rate_prog.append(r)
        conv_prog.append(np.linalg.norm(r-sol))
            
        if gamma < 1 + stopCri and np.abs(gamma - gamma_prev) < stopCri and np.abs(w @ np.log(1+r) - w @ np.log(1+r_prev)) < stopCri:
            break
    
    # scale to get to feasible region (sprad=1)
    alpha = 1/gamma
    s_scaled = alpha*(np.exp(r) - 1)
    r_scaled = np.log(1 + s_scaled)

    gamma_max = 0
    l_max = np.nan
    for l in range(L):
        gamma_l, _ = find_sprad(np.diag(np.exp(r_scaled) - 1) @ M_block[:,:,l])
        
        if gamma_l > gamma_max:
            gamma_max = gamma_l
            l_max = l
    
    rate_opt = r_scaled
    sprad = gamma_max

    rate_opt = r

    return rate_opt, sprad, conv_prog, rate_prog, gamma


def hsdm_cut(M_block, w, param):
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
    # beta = max(w)   # Lipschitz constant of gradient
    # beta = 10
    beta = param['beta']
    mu = param['mu']
    # mu = 1/beta     # step-size for first K iterations (nonexpansive if 0 < mu < 2/beta)
    # mu = 0.001
    K = param['K']
    q = param['q']
    stopCri = param['stopCri']
    maxIter = param['maxIter']
    L = M_block.shape[2]

    sol = param['solution']
    conv_prog = [np.linalg.norm(r-sol)]
    rate_prog = [r]

    for kk in range(1,maxIter+1):
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

        # radial projection
        alpha = 1 / gamma
        s_Pr = alpha*(np.exp(r) - 1)
        r_Pr = np.log(1 + s_Pr)

        gamma_test, xi = find_sprad(np.diag(np.exp(r_Pr) - 1) @ M)
        _, eta = find_sprad(M.T @ np.diag(np.exp(r_Pr) - 1))

        xi = xi / (xi @ eta)

        if gamma > 1:
            g = xi * eta * np.exp(r_Pr)/ (np.exp(r_Pr) - 1)
            r_hat = r - np.inner(r - r_Pr, g) / (g @ g) * g
        else:
            r_hat = r
        
        r_til = np.maximum(0, r_hat)
        r_til = np.minimum(r_til, 20)
        # r_til = r_hat

        if kk > K:
            # mu = 1 / beta * (kk - K)**(-q)
            mu = 1 / beta * kk**(-q)
        r = r_til + mu * w
        rate_prog.append(r)
        conv_prog.append(np.linalg.norm(r-sol))
            
        if gamma < 1 + stopCri and np.abs(gamma - gamma_prev) < stopCri and np.abs(w @ np.log(1+r) - w @ np.log(1+r_prev)) < stopCri:
            break
    
    # scale to get to feasible region (sprad=1)
    alpha = 1/gamma
    s_scaled = alpha*(np.exp(r) - 1)
    r_scaled = np.log(1 + s_scaled)

    gamma_max = 0
    l_max = np.nan
    for l in range(L):
        gamma_l, _ = find_sprad(np.diag(np.exp(r_scaled) - 1) @ M_block[:,:,l])
        
        if gamma_l > gamma_max:
            gamma_max = gamma_l
            l_max = l
    
    rate_opt = r_scaled
    sprad = gamma_max

    rate_opt = r

    return rate_opt, sprad, conv_prog, rate_prog


def bcd(bk, ck, sigma2, K, pmax):
    '''
    bk (Kx1)
    ck (KxK)
    sigma2 (Kx1)
    '''
    
    objec_new = np.inf
    eta = pmax * np.random.uniform(low=np.nextafter(0,1), high=np.nextafter(1,2), size=(K))

    eee = np.zeros((K))
    uuu = np.zeros((K))
    ddd = np.zeros((K))

    diff = 100

    iterr = 0
    while (diff>0.0001) or (diff<0):
        iterr +=1
        objec_old = objec_new

        for k in range(K):
            numm = np.sqrt(bk[k]*eta[k])
            denomm = eta[k]*bk[k] + ck[:,k].T @ eta + sigma2[k]

            uuu[k] = numm/denomm
            eee[k] = 1 - np.abs(numm)**2/denomm
            ddd[k] = 1/eee[k]
        
        for k in range(K):
            numm = bk[k] * ddd[k] * uuu[k]**2
            denomm  = numm
            for i in range (K):
                denomm += ddd[i] * np.abs(uuu[i])**2 * ck[k,i]
            
            eta[k] = np.min([pmax, ddd[k] * numm/(denomm**2)])
        
        objec_new = np.sum(ddd * eee - np.log(ddd))
        diff = objec_old - objec_new

    rates = np.log(ddd)
    return rates, eta


def find_sprad(A):
    """
    find spectral radius of matrix A (NxN)
    """
    N = A.shape[0]
    if N>2:
        val, vec = sparse.linalg.eigs(A, k=1, which='LR')
        sprad = np.real(val[0])
        vec = np.abs(vec).flatten()
    else:
        eigenvals, eigenvecs = linalg.eig(A)
        idx = np.argmax(np.real(eigenvals))
        sprad = np.real(eigenvals[idx])
        vec = np.abs(eigenvecs[:,idx]).flatten()
    return sprad, vec


def power_fixpoint(p_max, K, M, u, s):
    p_new = p_max * np.random.uniform(low=np.nextafter(0,1), high=np.nextafter(1,2), size=(K,1))  #initialize randomly
    thresh = 1e-8
    max_iter = 1000000
    for iter in range(max_iter):
        p_old = p_new
        T = M @ p_new + u
        p_new = np.diag(s) @ T
        if np.abs(np.max(p_new - p_old))<thresh:
            break
    return p_new