import numpy as np
import matplotlib.pyplot as plt
from sumrate_max import *
import toolbox as funcs

# def F(p, M, u):
#     return M @ p + u

K = 4
p_max = np.float64(200)

sumrate_hsdmS = []
sumrate_hsdmR = []
sumrate_hsdmC = []
sumrate_bcd = []


# C = np.array([[3.804458469753567e+03, 5.307975530089727e+03, 5.688236608721927e+03, 1.621823081932428e+03],
#               [5.678216407252211e+03, 7.791672301020111e+03, 4.693906410582058e+03, 7.942845406839069e+03],
#               [7.585428956306362e+02, 9.340106842291830e+03, 1.190206950124140e+02, 3.112150420448049e+03],
#               [5.395011866660715e+02, 1.299062084737301e+03, 3.371226443988815e+03, 5.285331355062127e+03]])
C = np.array([[3.804458469753567e+03, 5.307975530089727e+03, 5.688236608721927e+03, 1.621823081932428e+03],
              [5.307975530089727e+03, 7.791672301020111e+03, 4.693906410582058e+03, 7.942845406839069e+03],
              [5.688236608721927e+03, 4.693906410582058e+03, 1.190206950124140e+02, 3.112150420448049e+03],
              [1.621823081932428e+03, 7.942845406839069e+03, 3.112150420448049e+03, 5.285331355062127e+03]])

b = np.array([[9.171936638298100e+03],
              [2.858390188203735e+03],
              [7.572002291107213e+03],
              [7.537290942784954e+03]]).flatten()

sig = np.array([[16.564872949978092],
                [60.198194140163650],
                [26.297128454014430],
                [65.407909847678230]])

p_ini = np.array([[1.862819392919508e+02],
                  [200],
                  [1.307289859181909e+02],
                  [45.372035400905820]]).flatten()

s_ini = np.array([[0.868172283874961],
                  [0.149374800389099],
                  [0.456809686568517],
                  [0.134776350082071]]).flatten()

r_ini = np.array([[0.624960564608931],
                  [0.139218142707351],
                  [0.376248898618736],
                  [0.126435583130942]]).flatten()

a = 2

for i in range(a,a+1):
    np.random.seed(i)

    # b = 1e+4 * np.random.rand(K,1)
    # b = np.random.rand(K,1)
    # b = b.flatten()
    # # C = 1e+4 * np.random.rand(K,K)
    # C = np.random.rand(K,K)
    # # sig = 1e+2 * np.random.rand(K,1)
    # sig = np.random.rand(K,1)
    # print(C)


    M = np.diag(1/b) @ C.T
    u = np.diag(1/b) @ sig


    A = 1/p_max * np.eye(K)
    # A = np.eye(K)
    M_k = np.zeros((K,K,K))
    for k in range(K):
        M_k[k,:,:] = M + (u @ A[:,k][:,np.newaxis].T)

    w = np.ones(K)  #positive weight (priority) of users
    p_ini = 0.1 + 0.9 * np.random.rand(K,1)
    p_ini = (p_max / np.max(p_ini)) * p_ini
    # p_ini = np.array([[1], [0.01], [1]])
    s_ini = p_ini.flatten() / F(p_ini.flatten(), M, u.flatten())
    r_ini = np.log(1 + s_ini)


    param_hsdm = {
        's_ini': s_ini.flatten(),     # initial value for Alg 1 (over SINR region)
        'r_ini': r_ini.flatten(),     # initial value for Alg 2 (over rate region)

        'K': 1,         # K > 0 (use constant step-size for first K iterations)
        'q': 0.9999,           # 0 < q <= 1 (decay factor of step-size)
        'stopCri': 1e-8,    # stopping criterion (need to be sufficiently small)
        'maxIter': 10000,    # number of max iteration (need to be sufficiently large)
    }


    sinr_hsdmS, sprad_hsdmS, gammaS = hsdm_sinr(M_k, w, param_hsdm)
    rate_hsdmS = np.log(sinr_hsdmS + 1)
    rate_hsdmR, sprad_hsdm, rate_prog, gammaR = hsdm_rate(M_k, w, param_hsdm)
    rate_hsdmC, sprad_hsdmC, _, gammaC = hsdm_cut(M_k, w, param_hsdm)
   
    rate_bcd, rate_ddd, sumrate, p_bcd = bcd(b, C, sig, K, p_max, p_ini)

    sumrate_hsdmS.append(np.sum(rate_hsdmS))
    sumrate_hsdmR.append(np.sum(rate_hsdmR))
    sumrate_hsdmC.append(np.sum(rate_hsdmC))
    sumrate_bcd.append(np.sum(rate_bcd))

    # p_hsdmS = power_fixpoint(p_max, K, M, u, sinr_hsdmS)
    # print(p_hsdmS)
    # p_hsdmR = power_fixpoint(p_max, K, M, u, np.exp(rate_hsdmR)-1)
    # p_hsdmC = power_fixpoint(p_max, K, M, u, np.exp(rate_hsdmC)-1)
    # print("power S:", p_hsdmS)
    # print("power R:", p_hsdmR)
    # print("power C:", p_hsdmC)
    # print("power bcd:", p_bcd)

    # p_Rn = p_max / np.max(p_hsdmR) * p_hsdmR
    # p_Sn = p_max / np.max(p_hsdmS) * p_hsdmS

    # s_Rn = p_Rn / F(p_Rn, M, u)
    # s_Sn = p_Sn / F(p_Sn, M, u)

    # r_Rn = np.log(1 + s_Rn)
    # r_Sn = np.log(1 + s_Sn)

    # print("rate_original", rate_ddd, "rate_f", rate_bcd)

    # rate_hsdmS /= np.log(2)
    # rate_hsdmR /= np.log(2)
    # rate_hsdmC /= np.log(2)
    # rate_bcd /= np.log(2)

    print("rate S:", np.sum(rate_hsdmS))
    print("rate R:", np.sum(rate_hsdmR))
    print("rate C:", np.sum(rate_hsdmC))
    print("rate bcd:", np.sum(rate_bcd))

# print("rate S:", sumrate_hsdmS, sprad_hsdmS, gammaS)
# print("rate R:", sumrate_hsdmR, sprad_hsdm, gammaR)
# print("rate C:", sumrate_hsdmC)
print("rate bcd:", rate_bcd, sumrate_bcd)
print("done")