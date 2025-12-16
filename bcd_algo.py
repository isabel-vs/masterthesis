import numpy as np

def bcd(bk, ck, sigma2, K, pmax, p_ini):
    '''
    bk (Kx1)
    ck (KxK)
    sigma2 (Kx1)
    '''
    
    objec_new = np.inf
    # eta = pmax * np.random.uniform(low=np.nextafter(0,1), high=np.nextafter(1,2), size=(K))
    eta = p_ini

    eee = np.zeros((K))
    uuu = np.zeros((K))
    ddd = np.zeros((K))

    diff = 100

    iterr = 0
    while (diff>1e-8) or (diff<0):
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