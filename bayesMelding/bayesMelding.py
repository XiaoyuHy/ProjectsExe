
 #!/usr/bin/python -tt   #This line is to solve any difference between spaces and tabs
import numpy as np
import gpGaussLikeFuns
import computeN3Cost
import pandas as pd
import pickle
from scipy import linalg
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gamma
from timeit import default_timer
from scipy import integrate
import argparse
import os
import simData
# import ldNetCDF
from sklearn import linear_model
from itertools import chain

def deri_avg_grad_C_No_obs_noi(x, y, log_sigma, log_w, pointArealFlag = False):
    n_x = x.shape[0]
    n_y = y.shape[0]

    C = cov_mat_xy(x, y, np.exp(log_sigma), np.exp(log_w))
    grad_C = []
    avg_grad_C = []
    grad_C_log_sigma = C 
    grad_C.append(grad_C_log_sigma)
   
    tmp1 = np.sum(x * x, axis=1)
    tmp2 = np.sum(y * y, axis=1)
    tmp3 = np.dot(x, y.T)
    
    tmp_norm = tmp1.reshape(n_x, 1) + tmp2.reshape(1, n_y) - 2 * tmp3
    grad_C_log_w = grad_C_log_sigma *  (1./np.exp(log_w)**2) * tmp_norm
    grad_C.append(grad_C_log_w)
    
    if pointArealFlag:
        avg_C = np.mean(C, axis=1)
        avg_grad_C_log_sigma = np.mean(grad_C_log_sigma, axis =1)
        avg_grad_C_log_w = np.mean(grad_C_log_w, axis =1)
    else:
        avg_C = np.float(np.sum(C))/(n_x * n_y)
        avg_grad_C_log_sigma = np.float(np.sum(grad_C_log_sigma))/(n_x * n_y)
        avg_grad_C_log_w = np.float(np.sum(grad_C_log_w))/ (n_x * n_y)

    return [avg_C, avg_grad_C_log_sigma, avg_grad_C_log_w]
       
def deri_avg_grad_C_point_and_arealZs_No_obs_noi(x, y, log_sigma, log_w, b, pointArealFlag = False):
    avg_C, avg_grad_C_log_sigma, avg_grad_C_log_w = deri_avg_grad_C_No_obs_noi(x, y, log_sigma, log_w, pointArealFlag)
    if pointArealFlag:
        return [b * avg_C, b * avg_grad_C_log_sigma, b * avg_grad_C_log_w, avg_C]
    else:
        return [b**2 * avg_C, b**2 * avg_grad_C_log_sigma, b**2 * avg_grad_C_log_w, 2 * b * avg_C]

# funtion that compute areal_areal covariance and its gradients with respect to [log_sigma_deltas,log_sigmaZs, log_phiZs, b ](order in the list) if 
# deltas is NOT of GP form;else return gradients w.r.t. [log_sigma_deltas, log_phi_deltas, log_sigmaZs, log_phiZs, b ](order in the list) 
def cov_areal(areal_coordinate, log_sigma_Zs, log_phi_Zs, b, log_sigma_deltas_of_modelOut, gp_deltas_modelOut, log_phi_deltas_of_modelOut = None,  \
     areal_res=20, OMEGA = 1e-6):
    num_areas = len(areal_coordinate)
    grad_C_tildZs = []

    if gp_deltas_modelOut:
        cov_areas_tildZs = []
        cov_areas_deltas_Mo = []
        deri_areal_pars = []
        for i in range(num_areas):
            tmp0 = [deri_avg_grad_C_point_and_arealZs_No_obs_noi(areal_coordinate[i], areal_coordinate[j], log_sigma_Zs, log_phi_Zs, b) for j in range(num_areas)]
            avgCovZs = [tmp0[ii][0] for ii in range(num_areas)]
            deriSigmaZsphiZsBbias = [tmp0[ii][1:] for ii in range(num_areas)]
            cov_areas_tildZs.append(avgCovZs)

            tmp1 = [deri_avg_grad_C_No_obs_noi(areal_coordinate[i], areal_coordinate[j], log_sigma_deltas_of_modelOut, log_phi_deltas_of_modelOut) for j in range(num_areas)]
            
            avgCovDeltas = [tmp1[iii][0] for iii in range(num_areas)]
            deriSigDtsphiDts = [tmp1[iii][1:] for iii in range(num_areas)]
            cov_areas_deltas_Mo.append(avgCovDeltas)

            tmp2 = [deriSigDtsphiDts[j] + deriSigmaZsphiZsBbias[j] for j in range(num_areas)]
            assert len(tmp2) == num_areas
            deri_areal_pars.append(tmp2)

        tmp3 = list(chain.from_iterable(deri_areal_pars))
        for ii in range(len(tmp3[0])):
            tmp = np.array([tmp3[jj][ii] for jj in range(len(tmp3))]).reshape(num_areas, num_areas)
            grad_C_tildZs.append(tmp)

        covAreas = np.hstack(cov_areas_tildZs).reshape(num_areas, num_areas) + np.hstack(cov_areas_deltas_Mo).reshape(num_areas, num_areas) +\
        np.diag(np.repeat(OMEGA, num_areas))
        return [covAreas, grad_C_tildZs]
    else:
        cov_areas_tildZs = []
        deri_areal_pars = []
        grad_C_tildZs.append(np.diag(np.repeat(np.exp(log_sigma_deltas_of_modelOut), num_areas)))
        for i in range(num_areas):

            tmp1 = [deri_avg_grad_C_point_and_arealZs_No_obs_noi(areal_coordinate[i], areal_coordinate[j], log_sigma_Zs, log_phi_Zs, b) \
            for j in range(num_areas)]
            avgCovZs = [tmp1[ii][0] for ii in range(num_areas)]
            deriSigmaZsphiZsBbias = [tmp1[ii][1:] for ii in range(num_areas)]
            cov_areas_tildZs.append(avgCovZs)
            deri_areal_pars.append(deriSigmaZsphiZsBbias)

        tmp2 = list(chain.from_iterable(deri_areal_pars))
        for ii in range(len(tmp2[0])):
            tmp = np.array([tmp2[jj][ii] for jj in range(len(tmp2))]).reshape(num_areas, num_areas)
            grad_C_tildZs.append(tmp)

        covAreas = np.hstack(cov_areas_tildZs).reshape(num_areas, num_areas) + np.diag(np.repeat(OMEGA + \
            np.exp(log_sigma_deltas_of_modelOut), num_areas))
        return [covAreas, grad_C_tildZs]

# funtion that compute the point_areal covariance and its gradients with respect to [log_sigmaZs, log_phiZs, b ](order in the list)
def point_areal(X_hatZs, areal_coordinate, log_sigma_Zs, log_phi_Zs, b, pointArealFlag = True):
    num_areas = len(areal_coordinate)
    grad_C_pointAreal_upper = []
    grad_C_pointAreal_lower = []

    tmp0 = [deri_avg_grad_C_point_and_arealZs_No_obs_noi(X_hatZs, areal_coordinate[j], log_sigma_Zs, log_phi_Zs, b, pointArealFlag) \
    for j in range(num_areas)]
    avg_pointAreal_lower = np.array([tmp0[i][0] for i in range(num_areas)])
    avg_pointAreal_upper = avg_pointAreal_lower.T 
    deriSigmaZsphiZsBbias = [tmp0[i][1:] for i in range(num_areas)]
    for i in range(len(deriSigmaZsphiZsBbias[0])):
        tmp1 = np.array([deriSigmaZsphiZsBbias[j][i] for j in range(len(deriSigmaZsphiZsBbias))])
        grad_C_pointAreal_lower.append(tmp1)
        grad_C_pointAreal_upper.append(tmp1.T)

    return [avg_pointAreal_lower, avg_pointAreal_upper, grad_C_pointAreal_lower, grad_C_pointAreal_upper]

def test_fun(useCluster, SEED):
    if useCluster:
        X_hatZs_in = open('sampRealData/X_hatZs_seed' + str(SEED) + '.pkl', 'rb')
        X_hatZs = pickle.load(X_hatZs_in) 
     
        y_hatZs_in = open('sampRealData/y_hatZs_seed' + str(SEED) + '.pkl', 'rb')
        y_hatZs = pickle.load(y_hatZs_in) 

        X_tildZs_in = open('sampRealData/X_tildZs_seed' + str(SEED) + '.pkl', 'rb')
        X_tildZs = pickle.load(X_tildZs_in) 
      
        y_tildZs_in = open('sampRealData/y_tildZs_seed' + str(SEED) + '.pkl', 'rb')
        y_tildZs = pickle.load(y_tildZs_in) 
    else:
        X_hatZs, y_hatZs, X_tildZs, y_tildZs = ldNetCDF.loadNetCdf(SEED =SEED)

    print (X_hatZs.shape, y_hatZs.shape, X_tildZs.shape, y_tildZs.shape)
    

   
    theta = np.array([0.51390184, -2.30507058, -2.37142105, -0.57473654, -1.76136598,  1.94541264, 5.56365135, 5.26520738, 2.42106564])

    _, grads = log_obsZs_giv_par_with_grad(theta, X_hatZs, y_hatZs, X_tildZs, y_tildZs, gp_deltas_modelOut = True, withPrior= False, \
    a_bias_poly_deg = 2, rbf = True, OMEGA = 1e-6)
    print grads

    num_par = 1
    # initial_theta=np.concatenate((np.log(np.random.gamma(1.2, 5., 1)), np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), \
    #                  np.log(np.random.gamma(1.2, 1./0.6, 1)), np.log(np.random.gamma(1.2, 5., 1)), \
    #                  np.log(np.random.gamma(1., np.sqrt(num_par), num_par))), axis=0)
    modelBias = np.array([1.97191694,  5.47022754 , 7.22854712, -1.22452898])
    # test = minus_log_obsZs_giv_par_of_cov(initial_theta, X_hatZs, y_hatZs, X_tildZs, y_tildZs, args.withPrior, modelBias, gp_deltas_modelOut = True, \
    # a_bias_poly_deg = 2, rbf = True, OMEGA = 1e-6)
    initial_theta=np.concatenate((np.log(np.random.gamma(1.2, 5., 1)), np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), \
                        np.log(np.random.gamma(1.2, 1./0.6, 1)), np.log(np.random.gamma(1.2, 5., 1)), \
                        np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), modelBias), axis=0)
    
    res = log_obsZs_giv_par(initial_theta, X_hatZs, y_hatZs, X_tildZs, y_tildZs, gp_deltas_modelOut = True)
    print res
    log_pos, _ = log_obsZs_giv_par_with_grad(initial_theta, X_hatZs, y_hatZs, X_tildZs, y_tildZs, gp_deltas_modelOut = True)
    print log_pos
    
    exit(-1)
    covAreas, grad_C_tildZs = cov_areal(X_tildZs, np.log(1.5), [np.log(0.1)], 3., np.log(1.0), gp_deltas_modelOut=True, log_phi_deltas_of_modelOut = [np.log(0.2)],  \
     areal_res=20, OMEGA = 1e-6)
    print len(grad_C_tildZs)
    print grad_C_tildZs[0].shape
    avg_pointAreal_lower, avg_pointAreal_upper, grad_C_pointAreal_lower, grad_C_pointAreal_upper = \
    point_areal(X_hatZs, X_tildZs, np.log(1.5), [np.log(0.1)], 2., pointArealFlag = True)
    print ' shape of upper point areal is ' + str (avg_pointAreal_upper.shape)
    print avg_pointAreal_lower.shape
    print len(grad_C_pointAreal_upper)
    print grad_C_pointAreal_upper[0].shape

def cov_mat_xy(x, y, sigma, w):
    w = np.array(w)
    if len(w) == 1:
        w = np.repeat(w, x.shape[1])
    n_x = x.shape[0]
    n_y = y.shape[0]
    d = x.shape[1]
    w[w < 1e-19] = 1e-19
    W = np.eye(d)
    W[np.diag_indices(d)] = 1./w**2

    tmp1 = np.sum(np.dot(x, W) * x, axis=1)
    tmp2 = np.sum(np.dot(y, W) * y, axis=1)
    tmp3 = np.dot(x, np.dot(W, y.T))
    
    square_dist = tmp1.reshape(n_x, 1) + tmp2.reshape(1, n_y) - 2 * tmp3

    cov_of_two_vec = sigma * np.exp(- 0.5 * square_dist) # is a matrix of size (n_train, n_y)
    return cov_of_two_vec
  
def log_like_normal(w, mu, Sigma):
    l_chol = np.linalg.cholesky(Sigma)
    u = linalg.solve_triangular(l_chol.T, linalg.solve_triangular(l_chol, w, lower=True))
    log_like_normal = -np.sum(np.log(np.diag(l_chol))) - 0.5 * np.dot(w, u) - 0.5 * len(w) * np.log(2*np.pi)
    return log_like_normal

def log_like_gamma(w, shape, rate):
    res = np.sum(shape * w - rate * np.exp(w) + shape * np.log(rate) - np.log(gamma(shape)))
    return res

def log_obsZs_giv_par(theta, X_hatZs, y_hatZs, X_tildZs, y_tildZs, gp_deltas_modelOut = False, withPrior= False, \
    a_bias_poly_deg = 2, rbf = True, OMEGA = 1e-6):
    theta = np.array(theta)
    if rbf:
        num_len_scal = 1
    else:
        num_len_scal = X_hatZs.shape(1)
    if gp_deltas_modelOut:
        log_sigma_Zs = theta[0] #sigma of GP function for Zs
        log_phi_Zs = theta[1:num_len_scal+1]  # length scale of GP function for Zs
        log_obs_noi_scale = theta[num_len_scal+1:num_len_scal+2]
        log_sigma_deltas_of_modelOut = theta[num_len_scal+2:num_len_scal+3] # sigma of GP function for deltas of model output
        log_phi_deltas_of_modelOut = theta[num_len_scal+3:num_len_scal+3 + num_len_scal]  # length scale of GP function for for deltas of model output
        b = theta[num_len_scal+3 + num_len_scal:num_len_scal+3 + num_len_scal+1]
        a_bias_coefficients = theta[len(theta) - (a_bias_poly_deg+1):]
        
    else:
        log_sigma_Zs = theta[0] #sigma of GP function for Zs
        log_phi_Zs = theta[1:num_len_scal+1]  # length scale of GP function for Zs
        log_obs_noi_scale = theta[num_len_scal+1:num_len_scal+2]
        log_sigma_deltas_of_modelOut = theta[num_len_scal+2:num_len_scal+3] # sigma of Normal for deltas of model output
        b = theta[num_len_scal+3:num_len_scal+4]
        a_bias_coefficients = theta[len(theta) - (a_bias_poly_deg+1):]

    n_hatZs = X_hatZs.shape[0]
    n_tildZs = X_tildZs.shape[0]  
    n_bothZs = n_hatZs + n_tildZs

    mat = np.zeros(n_bothZs * n_bothZs).reshape(n_bothZs, n_bothZs)

    C_hatZs = gpGaussLikeFuns.cov_matrix_reg(X = X_hatZs, sigma = np.exp(log_sigma_Zs), w = np.exp(log_phi_Zs), obs_noi_scale = np.exp(log_obs_noi_scale))
    if gp_deltas_modelOut:
        C_tildZs, _ = cov_areal(X_tildZs, log_sigma_Zs, log_phi_Zs, b, log_sigma_deltas_of_modelOut, \
            gp_deltas_modelOut, log_phi_deltas_of_modelOut)
    else:
        C_tildZs, _ = cov_areal(X_tildZs, log_sigma_Zs, log_phi_Zs,b, log_sigma_deltas_of_modelOut, \
            gp_deltas_modelOut)

    mat[:n_hatZs, :n_hatZs] = C_hatZs
    mat[n_hatZs:n_hatZs + n_tildZs, n_hatZs:n_hatZs + n_tildZs] = C_tildZs

    avg_pointAreal_lower, avg_pointAreal_upper, _, _ = \
    point_areal(X_hatZs, X_tildZs, log_sigma_Zs, log_phi_Zs, b, pointArealFlag = True)
    

    mat[n_hatZs:n_hatZs + n_tildZs, :n_hatZs] = avg_pointAreal_lower
    mat[:n_hatZs, n_hatZs:n_hatZs + n_tildZs] = avg_pointAreal_upper

    mu_hatZs = np.zeros(len(y_hatZs))

    X_tildZs_mean = np.array([np.mean(X_tildZs[i], axis=0) for i in range(len(y_tildZs))])
    n_row = X_tildZs_mean.shape[0]
    tmp0 = np.repeat(1.,n_row).reshape(n_row,1)
    X_tildZs_mean_extend = np.hstack((X_tildZs_mean, tmp0))
    mu_tildZs = np.dot(X_tildZs_mean_extend, a_bias_coefficients)

    mu_hatTildZs = np.concatenate((mu_hatZs, mu_tildZs))

    y = np.concatenate((y_hatZs, y_tildZs))

    l_chol_C = gpGaussLikeFuns.compute_L_chol(mat)
    u = linalg.solve_triangular(l_chol_C.T, linalg.solve_triangular(l_chol_C, y - mu_hatTildZs, lower=True))     
    joint_log_like  = -np.sum(np.log(np.diag(l_chol_C))) - 0.5 * np.dot(y - mu_hatTildZs, u) - 0.5 * n_bothZs * np.log(2*np.pi) 

    if withPrior:

        #compute the likelihood of the gamma priors
        sigma_shape = 1.2 
        sigma_rate = 0.2 
        len_scal_shape = 1. 
        len_scal_rate = 1./np.sqrt(num_len_scal)
        obs_noi_scale_shape = 1.2
        obs_noi_scale_rate = 0.6
        b_mu =0.
        b_Sigma = np.diag([10000.])
        a_bias_coefficients_mu = np.zeros(a_bias_poly_deg +1)
        a_bias_coefficients_Sigma = np.diag(np.repeat(10000., a_bias_poly_deg + 1))
        #sigma, length_sacle, obs_noi_scale have to take positive numbers, thus taking gamma priors, whereas the mutiplicative bias b takes a normal prior
        log_prior = log_like_gamma(log_sigma_Zs, sigma_rate, sigma_shape) + log_like_gamma(log_phi_Zs, len_scal_shape, len_scal_rate) + \
        log_like_gamma(log_obs_noi_scale, obs_noi_scale_shape, obs_noi_scale_rate) + log_like_normal(b, b_mu, b_Sigma) + \
        log_like_normal(a_bias_coefficients, a_bias_coefficients_mu, a_bias_coefficients_Sigma)

        #compute the logarithm of the posterior
        log_pos = joint_log_like + log_prior
    else:
        log_pos = joint_log_like

    return log_pos

def log_obsZs_giv_par_with_grad(theta, X_hatZs, y_hatZs, X_tildZs, y_tildZs, gp_deltas_modelOut = True, withPrior= False, \
    a_bias_poly_deg = 2, rbf = True, OMEGA = 1e-6):
    theta = np.array(theta)
    if rbf:
        num_len_scal = 1
    else:
        num_len_scal = X_hatZs.shape(1)
    if gp_deltas_modelOut:
        log_sigma_Zs = theta[0] #sigma of GP function for Zs
        log_phi_Zs = theta[1:num_len_scal+1]  # length scale of GP function for Zs
        log_obs_noi_scale = theta[num_len_scal+1:num_len_scal+2]
        log_sigma_deltas_of_modelOut = theta[num_len_scal+2:num_len_scal+3] # sigma of GP function for deltas of model output
        log_phi_deltas_of_modelOut = theta[num_len_scal+3:num_len_scal+3 + num_len_scal]  # length scale of GP function for for deltas of model output
        b = theta[num_len_scal+3 + num_len_scal:num_len_scal+3 + num_len_scal+1]
        a_bias_coefficients = theta[len(theta) - (a_bias_poly_deg+1):]
        
    else:
        log_sigma_Zs = theta[0] #sigma of GP function for Zs
        log_phi_Zs = theta[1:num_len_scal+1]  # length scale of GP function for Zs
        log_obs_noi_scale = theta[num_len_scal+1:num_len_scal+2]
        log_sigma_deltas_of_modelOut = theta[num_len_scal+2:num_len_scal+3] # sigma of Normal for deltas of model output
        b = theta[num_len_scal+3:num_len_scal+4]
        a_bias_coefficients = theta[len(theta) - (a_bias_poly_deg+1):]

    n_hatZs = X_hatZs.shape[0]
    n_tildZs = X_tildZs.shape[0]  
    n_bothZs = n_hatZs + n_tildZs

    mat = np.zeros(n_bothZs * n_bothZs).reshape(n_bothZs, n_bothZs)

    C_hatZs = gpGaussLikeFuns.cov_matrix_reg(X = X_hatZs, sigma = np.exp(log_sigma_Zs), w = np.exp(log_phi_Zs), obs_noi_scale = np.exp(log_obs_noi_scale))
    if gp_deltas_modelOut:
        C_tildZs, grad_C_tildZs = cov_areal(X_tildZs, log_sigma_Zs, log_phi_Zs, b, log_sigma_deltas_of_modelOut, \
            gp_deltas_modelOut, log_phi_deltas_of_modelOut)
        grad_C_tildZs_log_sigma_dts = grad_C_tildZs[0]
        grad_C_tildZs_log_phi_dts = grad_C_tildZs[1]
        grad_C_tildZs_log_sigma_Zs = grad_C_tildZs[2]
        grad_C_tildZs_log_phi_Zs = grad_C_tildZs[3]
        grad_C_tildZs_b_bias = grad_C_tildZs[-1]

    else:
        C_tildZs, grad_C_tildZs = cov_areal(X_tildZs, log_sigma_Zs, log_phi_Zs, b, log_sigma_deltas_of_modelOut, \
            gp_deltas_modelOut)
        grad_C_tildZs_log_sigma_dts = grad_C_tildZs[0]
        grad_C_tildZs_log_sigma_Zs = grad_C_tildZs[1]
        grad_C_tildZs_log_phi_Zs = grad_C_tildZs[2]
        grad_C_tildZs_b_bias = grad_C_tildZs[-1]

    mat[:n_hatZs, :n_hatZs] = C_hatZs
    mat[n_hatZs:n_hatZs + n_tildZs, n_hatZs:n_hatZs + n_tildZs] = C_tildZs
    
    avg_pointAreal_lower, avg_pointAreal_upper, grad_C_pointAreal_lower, grad_C_pointAreal_upper = \
    point_areal(X_hatZs, X_tildZs, log_sigma_Zs, log_phi_Zs, b, pointArealFlag = True)
    grad_C_pointAreal_lower_log_sigma_Zs = grad_C_pointAreal_lower[0]
    grad_C_pointAreal_lower_log_phi_Zs = grad_C_pointAreal_lower[1]
    grad_C_pointAreal_lower_b_bias = grad_C_pointAreal_lower[-1]

    grad_C_pointAreal_upper_log_sigma_Zs = grad_C_pointAreal_upper[0]
    grad_C_pointAreal_upper_log_phi_Zs = grad_C_pointAreal_upper[1]
    grad_C_pointAreal_upper_b_bias = grad_C_pointAreal_upper[-1]

    mat[n_hatZs:n_hatZs + n_tildZs, :n_hatZs] = avg_pointAreal_lower
    mat[:n_hatZs, n_hatZs:n_hatZs + n_tildZs] = avg_pointAreal_upper

    mu_hatZs = np.zeros(len(y_hatZs))

    X_tildZs_mean = np.array([np.mean(X_tildZs[i], axis=0) for i in range(len(y_tildZs))])
    n_row = X_tildZs_mean.shape[0]
    tmp0 = np.repeat(1.,n_row).reshape(n_row,1)
    X_tildZs_mean_extend = np.hstack((X_tildZs_mean, tmp0))
    mu_tildZs = np.dot(X_tildZs_mean_extend, a_bias_coefficients)

    mu_hatTildZs = np.concatenate((mu_hatZs, mu_tildZs))

    y = np.concatenate((y_hatZs, y_tildZs))

    l_chol_C = gpGaussLikeFuns.compute_L_chol(mat)
    
    u = linalg.solve_triangular(l_chol_C.T, linalg.solve_triangular(l_chol_C, y - mu_hatTildZs, lower=True)) 

    joint_log_like  = -np.sum(np.log(np.diag(l_chol_C))) - 0.5 * np.dot(y - mu_hatTildZs, u) - 0.5 * n_bothZs * np.log(2*np.pi) 

    if withPrior:

        #compute the likelihood of the gamma priors
        sigma_shape = 1.2 
        sigma_rate = 0.2 
        len_scal_shape = 1. 
        len_scal_rate = 1./np.sqrt(num_len_scal)
        obs_noi_scale_shape = 1.2
        obs_noi_scale_rate = 0.6
        b_mu =0.
        b_Sigma = np.diag([10000.])
        a_bias_coefficients_mu = np.zeros(a_bias_poly_deg +1)
        a_bias_coefficients_Sigma = np.diag(np.repeat(10000., a_bias_poly_deg + 1))
        #sigma, length_sacle, obs_noi_scale have to take positive numbers, thus taking gamma priors, whereas the mutiplicative bias b takes a normal prior
        log_prior = log_like_gamma(log_sigma_Zs, sigma_rate, sigma_shape) + log_like_gamma(log_phi_Zs, len_scal_shape, len_scal_rate) + \
        log_like_gamma(log_obs_noi_scale, obs_noi_scale_shape, obs_noi_scale_rate) + log_like_normal(b, b_mu, b_Sigma) + \
        log_like_normal(a_bias_coefficients, a_bias_coefficients_mu, a_bias_coefficients_Sigma)

        #compute the logarithm of the posterior
        log_pos = joint_log_like + log_prior
    else:
        log_pos = joint_log_like

    # gradients of the joint covariance with respect to all the parameters
    grad_C = []

    # gradients of the covariance of Zs with respect to log_sigma_Zs, log_phi_Zs, log_obs_noi
    grad_C_hatZs = []
    grad_C_hatZs_log_sigma = C_hatZs - np.diag(np.repeat(np.exp(log_obs_noi_scale)**2, n_hatZs))
    grad_C_hatZs.append(grad_C_hatZs_log_sigma)
    if num_len_scal==1:
        tmp1 = np.sum(X_hatZs*X_hatZs, axis=1)
        tmp2 = np.dot(X_hatZs,X_hatZs.T)
        tmp3 = tmp1.reshape(n_hatZs,1) + tmp1
        tmp_norm = tmp3 - 2 * tmp2
        grad_C_hatZs_log_phi_Zs = grad_C_hatZs_log_sigma *  (1./np.exp(log_phi_Zs)**2) * tmp_norm
        grad_C_hatZs.append(grad_C_hatZs_log_phi_Zs)
    else:
        for i in range(num_len_scal):
            temp0= (X_hatZs[:,i].reshape(n_hatZs,1) - X_hatZs[:,i])**2
            temp1 =  (1./np.exp(log_phi_Zs[i])**2) * temp0
            grad_C_hatZs_log_wi = grad_C_hatZs_log_sigma * temp1
            grad_C_hatZs.append(grad_C_hatZs_log_wi)
    grad_C_hatZs_log_obs_noi = np.diag(np.repeat(2 * np.exp(log_obs_noi_scale)**2,n_hatZs))
    grad_C_hatZs.append(grad_C_hatZs_log_obs_noi)

    # gradient of the jont covariance matrix with respect to log_sigma_Zs
    tmp_mat = np.zeros(n_bothZs * n_bothZs).reshape(n_bothZs, n_bothZs)
    tmp_mat[:n_hatZs, :n_hatZs] = grad_C_hatZs_log_sigma
    tmp_mat[n_hatZs:n_hatZs + n_tildZs, n_hatZs:n_hatZs + n_tildZs] = grad_C_tildZs_log_sigma_Zs
    tmp_mat[n_hatZs:n_hatZs + n_tildZs, :n_hatZs] = grad_C_pointAreal_lower_log_sigma_Zs
    tmp_mat[:n_hatZs, n_hatZs:n_hatZs + n_tildZs] = grad_C_pointAreal_upper_log_sigma_Zs
    grad_C.append(tmp_mat)

    # gradient of the jont covariance matrix with respect to log_phi_Zs
    tmp_mat = np.zeros(n_bothZs * n_bothZs).reshape(n_bothZs, n_bothZs)
    tmp_mat[:n_hatZs, :n_hatZs] = grad_C_hatZs_log_phi_Zs
    tmp_mat[n_hatZs:n_hatZs + n_tildZs, n_hatZs:n_hatZs + n_tildZs] = grad_C_tildZs_log_phi_Zs
    tmp_mat[n_hatZs:n_hatZs + n_tildZs, :n_hatZs] = grad_C_pointAreal_lower_log_phi_Zs
    tmp_mat[:n_hatZs, n_hatZs:n_hatZs + n_tildZs] = grad_C_pointAreal_upper_log_phi_Zs
    grad_C.append(tmp_mat)

    # gradient of the jont covariance matrix with respect to log_obs_noi
    tmp_mat = np.zeros(n_bothZs * n_bothZs).reshape(n_bothZs, n_bothZs)
    tmp_mat[:n_hatZs, :n_hatZs] = grad_C_hatZs_log_obs_noi
    grad_C.append(tmp_mat)

    if gp_deltas_modelOut:
        # gradient of the jont covariance matrix with respect to log_sigma_dts
        tmp_mat = np.zeros(n_bothZs * n_bothZs).reshape(n_bothZs, n_bothZs)
        tmp_mat[n_hatZs:n_hatZs + n_tildZs, n_hatZs:n_hatZs + n_tildZs] = grad_C_tildZs_log_sigma_dts
        grad_C.append(tmp_mat)

        # gradient of the jont covariance matrix with respect to log_phi_dts
        tmp_mat = np.zeros(n_bothZs * n_bothZs).reshape(n_bothZs, n_bothZs)
        tmp_mat[n_hatZs:n_hatZs + n_tildZs, n_hatZs:n_hatZs + n_tildZs] = grad_C_tildZs_log_phi_dts
        grad_C.append(tmp_mat)
    else:
        # gradient of the jont covariance matrix with respect to log_sigma_dts
        tmp_mat = np.zeros(n_bothZs * n_bothZs).reshape(n_bothZs, n_bothZs)
        tmp_mat[n_hatZs:n_hatZs + n_tildZs, n_hatZs:n_hatZs + n_tildZs] = grad_C_tildZs_log_sigma_dts
        grad_C.append(tmp_mat)

    # gradient of the jont covariance matrix with respect to b_bias
    tmp_mat = np.zeros(n_bothZs * n_bothZs).reshape(n_bothZs, n_bothZs)
    tmp_mat[n_hatZs:n_hatZs + n_tildZs, n_hatZs:n_hatZs + n_tildZs] = grad_C_tildZs_b_bias
    tmp_mat[n_hatZs:n_hatZs + n_tildZs, :n_hatZs] = grad_C_pointAreal_lower_b_bias
    tmp_mat[:n_hatZs, n_hatZs:n_hatZs + n_tildZs] = grad_C_pointAreal_upper_b_bias
    grad_C.append(tmp_mat)

    num_covPars = len(grad_C)
    grads_par_covPars = np.zeros(num_covPars)    
    inver_C = linalg.solve_triangular(l_chol_C.T, linalg.solve_triangular(l_chol_C, np.eye(n_bothZs), lower=True))   

    for i in range(num_covPars):
        
        temp = np.dot(grad_C[i], u)
        grads_par_covPars[i] = -0.5 * np.sum(inver_C * grad_C[i]) + 0.5 * np.dot(y - mu_hatTildZs, linalg.solve_triangular(l_chol_C.T, \
            linalg.solve_triangular(l_chol_C, temp, lower=True)))
    # gradients of the jont covariance matrix with respect to a_bias
    tmp = np.zeros((n_hatZs,X_hatZs.shape[1] + 1))

    X_all_extend = np.vstack((tmp, X_tildZs_mean_extend))

    grads_a_bias = np.dot(X_all_extend.T, u)
    grads_all_pars = np.concatenate((grads_par_covPars, grads_a_bias))

    # Add penatly if all postitive parameters are not in the range of [0.1, 100]
    # if log_phi_Zs < -2.3 or log_phi_Zs > 4.6 or log_sigma_Zs < -2.3 or log_sigma_Zs > 4.6 or \
    # log_obs_noi_scale < -2.3 or log_obs_noi_scale > 4.6 or \
    #  log_sigma_deltas_of_modelOut < -2.3 or log_sigma_deltas_of_modelOut > 4.6  or \
    #  log_phi_deltas_of_modelOut < -2.3 or log_phi_deltas_of_modelOut > 4.6:
    #     # log_pos = log_pos + 0.1 * (log_phi_Zs + 20)**8
    #     # deri_log_phi_Zs_penalty = 0.8 * (log_phi_Zs + 20)**7
    #     # grads_par_covPars[1] = grads_par_covPars[1] + deri_log_phi_Zs_penalty
    #     # grads_all_pars = np.concatenate((grads_par_covPars, grads_a_bias))
    #     log_pos = log_pos + 10**20

    return [log_pos, grads_all_pars]

def minus_log_obsZs_giv_par_with_grad(theta, X_hatZs, y_hatZs, X_tildZs, y_tildZs, gp_deltas_modelOut = True, withPrior= False, \
    a_bias_poly_deg = 2, rbf = True, OMEGA = 1e-6):
    theta = np.array(theta)
    if rbf:
        num_len_scal = 1
    else:
        num_len_scal = X_hatZs.shape(1)
    if gp_deltas_modelOut:
        log_sigma_Zs = theta[0] #sigma of GP function for Zs
        log_phi_Zs = theta[1:num_len_scal+1]  # length scale of GP function for Zs
        log_obs_noi_scale = theta[num_len_scal+1:num_len_scal+2]
        log_sigma_deltas_of_modelOut = theta[num_len_scal+2:num_len_scal+3] # sigma of GP function for deltas of model output
        log_phi_deltas_of_modelOut = theta[num_len_scal+3:num_len_scal+3 + num_len_scal]  # length scale of GP function for for deltas of model output
        b = theta[num_len_scal+3 + num_len_scal:num_len_scal+3 + num_len_scal+1]
        a_bias_coefficients = theta[len(theta) - (a_bias_poly_deg+1):]
        
    else:
        log_sigma_Zs = theta[0] #sigma of GP function for Zs
        log_phi_Zs = theta[1:num_len_scal+1]  # length scale of GP function for Zs
        log_obs_noi_scale = theta[num_len_scal+1:num_len_scal+2]
        log_sigma_deltas_of_modelOut = theta[num_len_scal+2:num_len_scal+3] # sigma of Normal for deltas of model output
        b = theta[num_len_scal+3:num_len_scal+4]
        a_bias_coefficients = theta[len(theta) - (a_bias_poly_deg+1):]

    n_hatZs = X_hatZs.shape[0]
    n_tildZs = X_tildZs.shape[0]  
    n_bothZs = n_hatZs + n_tildZs

    mat = np.zeros(n_bothZs * n_bothZs).reshape(n_bothZs, n_bothZs)

    C_hatZs = gpGaussLikeFuns.cov_matrix_reg(X = X_hatZs, sigma = np.exp(log_sigma_Zs), w = np.exp(log_phi_Zs), obs_noi_scale = np.exp(log_obs_noi_scale))
    if gp_deltas_modelOut:
        C_tildZs, grad_C_tildZs = cov_areal(X_tildZs, log_sigma_Zs, log_phi_Zs, b, log_sigma_deltas_of_modelOut, \
            gp_deltas_modelOut, log_phi_deltas_of_modelOut)
        grad_C_tildZs_log_sigma_dts = grad_C_tildZs[0]
        grad_C_tildZs_log_phi_dts = grad_C_tildZs[1]
        grad_C_tildZs_log_sigma_Zs = grad_C_tildZs[2]
        grad_C_tildZs_log_phi_Zs = grad_C_tildZs[3]
        grad_C_tildZs_b_bias = grad_C_tildZs[-1]

    else:
        C_tildZs, grad_C_tildZs = cov_areal(X_tildZs, log_sigma_Zs, log_phi_Zs, b, log_sigma_deltas_of_modelOut, \
            gp_deltas_modelOut)
        grad_C_tildZs_log_sigma_dts = grad_C_tildZs[0]
        grad_C_tildZs_log_sigma_Zs = grad_C_tildZs[1]
        grad_C_tildZs_log_phi_Zs = grad_C_tildZs[2]
        grad_C_tildZs_b_bias = grad_C_tildZs[-1]

    mat[:n_hatZs, :n_hatZs] = C_hatZs
    mat[n_hatZs:n_hatZs + n_tildZs, n_hatZs:n_hatZs + n_tildZs] = C_tildZs
    
    avg_pointAreal_lower, avg_pointAreal_upper, grad_C_pointAreal_lower, grad_C_pointAreal_upper = \
    point_areal(X_hatZs, X_tildZs, log_sigma_Zs, log_phi_Zs, b, pointArealFlag = True)
    grad_C_pointAreal_lower_log_sigma_Zs = grad_C_pointAreal_lower[0]
    grad_C_pointAreal_lower_log_phi_Zs = grad_C_pointAreal_lower[1]
    grad_C_pointAreal_lower_b_bias = grad_C_pointAreal_lower[-1]

    grad_C_pointAreal_upper_log_sigma_Zs = grad_C_pointAreal_upper[0]
    grad_C_pointAreal_upper_log_phi_Zs = grad_C_pointAreal_upper[1]
    grad_C_pointAreal_upper_b_bias = grad_C_pointAreal_upper[-1]

    mat[n_hatZs:n_hatZs + n_tildZs, :n_hatZs] = avg_pointAreal_lower
    mat[:n_hatZs, n_hatZs:n_hatZs + n_tildZs] = avg_pointAreal_upper

    mu_hatZs = np.zeros(len(y_hatZs))

    X_tildZs_mean = np.array([np.mean(X_tildZs[i], axis=0) for i in range(len(y_tildZs))])
    n_row = X_tildZs_mean.shape[0]
    tmp0 = np.repeat(1.,n_row).reshape(n_row,1)
    X_tildZs_mean_extend = np.hstack((X_tildZs_mean, tmp0))
    mu_tildZs = np.dot(X_tildZs_mean_extend, a_bias_coefficients)

    mu_hatTildZs = np.concatenate((mu_hatZs, mu_tildZs))

    y = np.concatenate((y_hatZs, y_tildZs))

    l_chol_C = gpGaussLikeFuns.compute_L_chol(mat)
    u = linalg.solve_triangular(l_chol_C.T, linalg.solve_triangular(l_chol_C, y - mu_hatTildZs, lower=True))     
    joint_log_like  = -np.sum(np.log(np.diag(l_chol_C))) - 0.5 * np.dot(y - mu_hatTildZs, u) - 0.5 * n_bothZs * np.log(2*np.pi) 

    if withPrior:

        #compute the likelihood of the gamma priors
        sigma_shape = 1.2 
        sigma_rate = 0.2 
        len_scal_shape = 1. 
        len_scal_rate = 1./np.sqrt(num_len_scal)
        obs_noi_scale_shape = 1.2
        obs_noi_scale_rate = 0.6
        b_mu =0.
        b_Sigma = np.diag([10000.])
        a_bias_coefficients_mu = np.zeros(a_bias_poly_deg +1)
        a_bias_coefficients_Sigma = np.diag(np.repeat(10000., a_bias_poly_deg + 1))
        #sigma, length_sacle, obs_noi_scale have to take positive numbers, thus taking gamma priors, whereas the mutiplicative bias b takes a normal prior
        log_prior = log_like_gamma(log_sigma_Zs, sigma_rate, sigma_shape) + log_like_gamma(log_phi_Zs, len_scal_shape, len_scal_rate) + \
        log_like_gamma(log_obs_noi_scale, obs_noi_scale_shape, obs_noi_scale_rate) + log_like_normal(b, b_mu, b_Sigma) + \
        log_like_normal(a_bias_coefficients, a_bias_coefficients_mu, a_bias_coefficients_Sigma)

        #compute the logarithm of the posterior
        log_pos = joint_log_like + log_prior
    else:
        log_pos = joint_log_like

    # gradients of the joint covariance with respect to all the parameters
    grad_C = []

    # gradients of the covariance of Zs with respect to log_sigma_Zs, log_phi_Zs, log_obs_noi
    grad_C_hatZs = []
    grad_C_hatZs_log_sigma = C_hatZs - np.diag(np.repeat(np.exp(log_obs_noi_scale)**2, n_hatZs))
    grad_C_hatZs.append(grad_C_hatZs_log_sigma)
    if num_len_scal==1:
        tmp1 = np.sum(X_hatZs*X_hatZs, axis=1)
        tmp2 = np.dot(X_hatZs,X_hatZs.T)
        tmp3 = tmp1.reshape(n_hatZs,1) + tmp1
        tmp_norm = tmp3 - 2 * tmp2
        grad_C_hatZs_log_phi_Zs = grad_C_hatZs_log_sigma *  (1./np.exp(log_phi_Zs)**2) * tmp_norm
        grad_C_hatZs.append(grad_C_hatZs_log_phi_Zs)
    else:
        for i in range(num_len_scal):
            temp0= (X_hatZs[:,i].reshape(n_hatZs,1) - X_hatZs[:,i])**2
            temp1 =  (1./np.exp(log_phi_Zs[i])**2) * temp0
            grad_C_hatZs_log_wi = grad_C_hatZs_log_sigma * temp1
            grad_C_hatZs.append(grad_C_hatZs_log_wi)
    grad_C_hatZs_log_obs_noi = np.diag(np.repeat(2 * np.exp(log_obs_noi_scale)**2,n_hatZs))
    grad_C_hatZs.append(grad_C_hatZs_log_obs_noi)

    # gradient of the jont covariance matrix with respect to log_sigma_Zs
    tmp_mat = np.zeros(n_bothZs * n_bothZs).reshape(n_bothZs, n_bothZs)
    tmp_mat[:n_hatZs, :n_hatZs] = grad_C_hatZs_log_sigma
    tmp_mat[n_hatZs:n_hatZs + n_tildZs, n_hatZs:n_hatZs + n_tildZs] = grad_C_tildZs_log_sigma_Zs
    tmp_mat[n_hatZs:n_hatZs + n_tildZs, :n_hatZs] = grad_C_pointAreal_lower_log_sigma_Zs
    tmp_mat[:n_hatZs, n_hatZs:n_hatZs + n_tildZs] = grad_C_pointAreal_upper_log_sigma_Zs
    grad_C.append(tmp_mat)

    # gradient of the jont covariance matrix with respect to log_phi_Zs
    tmp_mat = np.zeros(n_bothZs * n_bothZs).reshape(n_bothZs, n_bothZs)
    tmp_mat[:n_hatZs, :n_hatZs] = grad_C_hatZs_log_phi_Zs
    tmp_mat[n_hatZs:n_hatZs + n_tildZs, n_hatZs:n_hatZs + n_tildZs] = grad_C_tildZs_log_phi_Zs
    tmp_mat[n_hatZs:n_hatZs + n_tildZs, :n_hatZs] = grad_C_pointAreal_lower_log_phi_Zs
    tmp_mat[:n_hatZs, n_hatZs:n_hatZs + n_tildZs] = grad_C_pointAreal_upper_log_phi_Zs
    grad_C.append(tmp_mat)

    # gradient of the jont covariance matrix with respect to log_obs_noi
    tmp_mat = np.zeros(n_bothZs * n_bothZs).reshape(n_bothZs, n_bothZs)
    tmp_mat[:n_hatZs, :n_hatZs] = grad_C_hatZs_log_obs_noi
    grad_C.append(tmp_mat)

    if gp_deltas_modelOut:
        # gradient of the jont covariance matrix with respect to log_sigma_dts
        tmp_mat = np.zeros(n_bothZs * n_bothZs).reshape(n_bothZs, n_bothZs)
        tmp_mat[n_hatZs:n_hatZs + n_tildZs, n_hatZs:n_hatZs + n_tildZs] = grad_C_tildZs_log_sigma_dts
        grad_C.append(tmp_mat)

        # gradient of the jont covariance matrix with respect to log_phi_dts
        tmp_mat = np.zeros(n_bothZs * n_bothZs).reshape(n_bothZs, n_bothZs)
        tmp_mat[n_hatZs:n_hatZs + n_tildZs, n_hatZs:n_hatZs + n_tildZs] = grad_C_tildZs_log_phi_dts
        grad_C.append(tmp_mat)
    else:
        # gradient of the jont covariance matrix with respect to log_sigma_dts
        tmp_mat = np.zeros(n_bothZs * n_bothZs).reshape(n_bothZs, n_bothZs)
        tmp_mat[n_hatZs:n_hatZs + n_tildZs, n_hatZs:n_hatZs + n_tildZs] = grad_C_tildZs_log_sigma_dts
        grad_C.append(tmp_mat)

    # gradient of the jont covariance matrix with respect to b_bias
    tmp_mat = np.zeros(n_bothZs * n_bothZs).reshape(n_bothZs, n_bothZs)
    tmp_mat[n_hatZs:n_hatZs + n_tildZs, n_hatZs:n_hatZs + n_tildZs] = grad_C_tildZs_b_bias
    tmp_mat[n_hatZs:n_hatZs + n_tildZs, :n_hatZs] = grad_C_pointAreal_lower_b_bias
    tmp_mat[:n_hatZs, n_hatZs:n_hatZs + n_tildZs] = grad_C_pointAreal_upper_b_bias
    grad_C.append(tmp_mat)

    num_covPars = len(grad_C)
    grads_par_covPars = np.zeros(num_covPars)    
    inver_C = linalg.solve_triangular(l_chol_C.T, linalg.solve_triangular(l_chol_C, np.eye(n_bothZs), lower=True))   

    for i in range(num_covPars):
        
        temp = np.dot(grad_C[i], u)
        grads_par_covPars[i] = -0.5 * np.sum(inver_C * grad_C[i]) + 0.5 * np.dot(y - mu_hatTildZs, linalg.solve_triangular(l_chol_C.T, \
            linalg.solve_triangular(l_chol_C, temp, lower=True)))
    # gradients of the jont covariance matrix with respect to a_bias
    tmp = np.zeros((n_hatZs,X_hatZs.shape[1] + 1))

    X_all_extend = np.vstack((tmp, X_tildZs_mean_extend))

    grads_a_bias = np.dot(X_all_extend.T, u)
    grads_all_pars = np.concatenate((grads_par_covPars, grads_a_bias))

    # Add penatly if all postitive parameters are not in the range of [0.1, 100]
    # if log_phi_Zs < -2.3 or log_phi_Zs > 4.6 or log_sigma_Zs < -2.3 or log_sigma_Zs > 4.6 or \
    # log_obs_noi_scale < -2.3 or log_obs_noi_scale > 4.6 or \
    #  log_sigma_deltas_of_modelOut < -2.3 or log_sigma_deltas_of_modelOut > 4.6  or \
    #  log_phi_deltas_of_modelOut < -2.3 or log_phi_deltas_of_modelOut > 4.6:
    #     print 'parameters are out of range (-2.3, 4.6)'
    #     # log_pos = log_pos + 0.1 * (log_phi_Zs + 20)**8
    #     # deri_log_phi_Zs_penalty = 0.8 * (log_phi_Zs + 20)**7
    #     # grads_par_covPars[1] = grads_par_covPars[1] + deri_log_phi_Zs_penalty
    #     # grads_all_pars = np.concatenate((grads_par_covPars, grads_a_bias))
    #     log_pos = log_pos + 10**20

    minus_log_pos = -log_pos
    minus_grads_all_pars = - grads_all_pars

    return [minus_log_pos, minus_grads_all_pars]


def gradsApprox(theta, X_hatZs, y_hatZs, X_tildZs, y_tildZs, gp_deltas_modelOut = False,  withPrior= False,  \
    a_bias_poly_deg = 2, rbf = True, OMEGA = 1e-6, epsilon = 1e-7):
    theta = np.array(theta)
    num_parameters = len(theta)

    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))
    # Compute gradapprox of log_w  
    for i in range(num_parameters):
        # Compute J_plus[i]. Inputs: "thetaplus, epsilon". Output = "J_plus[i]".
        # "_" is used because the function outputs two parameters but we only care about the first one
        thetaplus = np.copy(theta)  # Step 1
        thetaplus[i] = thetaplus[i] + epsilon   # Step 2
        J_plus[i] = log_obsZs_giv_par(thetaplus, X_hatZs, y_hatZs, X_tildZs, y_tildZs, gp_deltas_modelOut) # Step 3

        
        # Compute J_minus[i]. Inputs: "thetaminus, epsilon". Output = "J_minus[i]".
        thetaminus = np.copy(theta)    # Step 1
        thetaminus[i] = thetaminus[i]-epsilon   # Step 2   
        J_minus[i]= log_obsZs_giv_par(thetaminus, X_hatZs, y_hatZs, X_tildZs, y_tildZs, gp_deltas_modelOut)# Step 3
        
        # Compute gradapprox[i]
        gradapprox[i] = (J_plus[i]-J_minus[i])/(2.*epsilon)
    gradapprox = gradapprox.reshape(num_parameters)
   
    return gradapprox

def minus_log_obsZs_giv_par(theta, X_hatZs, y_hatZs, X_tildZs, y_tildZs, gp_deltas_modelOut = False, withPrior= False, \
    a_bias_poly_deg = 2, rbf = True, OMEGA = 1e-6):
    theta = np.array(theta)
    if rbf:
        num_len_scal = 1
    else:
        num_len_scal = X_hatZs.shape(1)
    if gp_deltas_modelOut:
        log_sigma_Zs = theta[0] #sigma of GP function for Zs
        log_phi_Zs = theta[1:num_len_scal+1]  # length scale of GP function for Zs
        log_obs_noi_scale = theta[num_len_scal+1:num_len_scal+2]
        log_sigma_deltas_of_modelOut = theta[num_len_scal+2:num_len_scal+3] # sigma of GP function for deltas of model output
        log_phi_deltas_of_modelOut = theta[num_len_scal+3:num_len_scal+3 + num_len_scal]  # length scale of GP function for for deltas of model output
        b = theta[num_len_scal+3 + num_len_scal:num_len_scal+3 + num_len_scal+1]
        a_bias_coefficients = theta[len(theta) - (a_bias_poly_deg+1):]
        
    else:
        log_sigma_Zs = theta[0] #sigma of GP function for Zs
        log_phi_Zs = theta[1:num_len_scal+1]  # length scale of GP function for Zs
        log_obs_noi_scale = theta[num_len_scal+1:num_len_scal+2]
        log_sigma_deltas_of_modelOut = theta[num_len_scal+2:num_len_scal+3] # sigma of Normal for deltas of model output
        b = theta[num_len_scal+3:num_len_scal+4]
        a_bias_coefficients = theta[len(theta) - (a_bias_poly_deg+1):]

    n_hatZs = X_hatZs.shape[0]
    n_tildZs = X_tildZs.shape[0]  
    n_bothZs = n_hatZs + n_tildZs

    mat = np.zeros(n_bothZs * n_bothZs).reshape(n_bothZs, n_bothZs)

    C_hatZs = gpGaussLikeFuns.cov_matrix_reg(X = X_hatZs, sigma = np.exp(log_sigma_Zs), w = np.exp(log_phi_Zs), obs_noi_scale = np.exp(log_obs_noi_scale))
    if gp_deltas_modelOut:
        C_tildZs, _ = cov_areal(X_tildZs, log_sigma_Zs, log_phi_Zs, b, log_sigma_deltas_of_modelOut, \
            gp_deltas_modelOut, log_phi_deltas_of_modelOut)
    else:
        C_tildZs, _ = cov_areal(X_tildZs, log_sigma_Zs, log_phi_Zs,b, log_sigma_deltas_of_modelOut, \
            gp_deltas_modelOut)

    mat[:n_hatZs, :n_hatZs] = C_hatZs
    mat[n_hatZs:n_hatZs + n_tildZs, n_hatZs:n_hatZs + n_tildZs] = C_tildZs

    avg_pointAreal_lower, avg_pointAreal_upper, _, _ = \
    point_areal(X_hatZs, X_tildZs, log_sigma_Zs, log_phi_Zs, b, pointArealFlag = True)
    

    mat[n_hatZs:n_hatZs + n_tildZs, :n_hatZs] = avg_pointAreal_lower
    mat[:n_hatZs, n_hatZs:n_hatZs + n_tildZs] = avg_pointAreal_upper

    mu_hatZs = np.zeros(len(y_hatZs))

    X_tildZs_mean = np.array([np.mean(X_tildZs[i], axis=0) for i in range(len(y_tildZs))])
    n_row = X_tildZs_mean.shape[0]
    tmp0 = np.repeat(1.,n_row).reshape(n_row,1)
    X_tildZs_mean_extend = np.hstack((X_tildZs_mean, tmp0))
    mu_tildZs = np.dot(X_tildZs_mean_extend, a_bias_coefficients)

    mu_hatTildZs = np.concatenate((mu_hatZs, mu_tildZs))

    y = np.concatenate((y_hatZs, y_tildZs))

    l_chol_C = gpGaussLikeFuns.compute_L_chol(mat)
    u = linalg.solve_triangular(l_chol_C.T, linalg.solve_triangular(l_chol_C, y - mu_hatTildZs, lower=True))     
    joint_log_like  = -np.sum(np.log(np.diag(l_chol_C))) - 0.5 * np.dot(y - mu_hatTildZs, u) - 0.5 * n_bothZs * np.log(2*np.pi) 

    if withPrior:

        #compute the likelihood of the gamma priors
        sigma_shape = 1.2 
        sigma_rate = 0.2 
        len_scal_shape = 1. 
        len_scal_rate = 1./np.sqrt(num_len_scal)
        obs_noi_scale_shape = 1.2
        obs_noi_scale_rate = 0.6
        b_mu =0.
        b_Sigma = np.diag([10000.])
        a_bias_coefficients_mu = np.zeros(a_bias_poly_deg +1)
        a_bias_coefficients_Sigma = np.diag(np.repeat(10000., a_bias_poly_deg + 1))
        #sigma, length_sacle, obs_noi_scale have to take positive numbers, thus taking gamma priors, whereas the mutiplicative bias b takes a normal prior
        log_prior = log_like_gamma(log_sigma_Zs, sigma_rate, sigma_shape) + log_like_gamma(log_phi_Zs, len_scal_shape, len_scal_rate) + \
        log_like_gamma(log_obs_noi_scale, obs_noi_scale_shape, obs_noi_scale_rate) + log_like_normal(b, b_mu, b_Sigma) + \
        log_like_normal(a_bias_coefficients, a_bias_coefficients_mu, a_bias_coefficients_Sigma)

        #compute the logarithm of the posterior
        log_pos = joint_log_like + log_prior
        minus_log_pos = - log_pos
    else:
        minus_log_pos = - joint_log_like

    return minus_log_pos

def minus_log_obsZs_giv_par_of_cov(theta, X_hatZs, y_hatZs, X_tildZs, y_tildZs, withPrior, modelBias, gp_deltas_modelOut = False, \
    a_bias_poly_deg = 2, rbf = True, OMEGA = 1e-6):
    theta = np.array(theta)
    if rbf:
        num_len_scal = 1
    else:
        num_len_scal = X_hatZs.shape(1)

    if gp_deltas_modelOut:
        log_sigma_Zs = theta[0] #sigma of GP function for Zs
        log_phi_Zs = theta[1:num_len_scal+1]  # length scale of GP function for Zs
        log_obs_noi_scale = theta[num_len_scal+1:num_len_scal+2]
        log_sigma_deltas_of_modelOut = theta[num_len_scal+2:num_len_scal+3] # sigma of GP function for deltas of model output
        log_phi_deltas_of_modelOut = theta[len(theta) - num_len_scal:]  # length scale of GP function for for deltas of model output
        b = modelBias[0]
        a_bias_coefficients = modelBias[1:]
    else:
        log_sigma_Zs = theta[0] #sigma of GP function for Zs
        log_phi_Zs = theta[1:num_len_scal+1]  # length scale of GP function for Zs
        log_obs_noi_scale = theta[num_len_scal+1:num_len_scal+2]
        log_sigma_deltas_of_modelOut = theta[len(theta) -1:] # sigma of Normal for deltas of model output
        b = modelBias[0]
        a_bias_coefficients = modelBias[1:]

    n_hatZs = X_hatZs.shape[0]
    n_tildZs = X_tildZs.shape[0]  
    n_bothZs = n_hatZs + n_tildZs

    mat = np.zeros(n_bothZs * n_bothZs).reshape(n_bothZs, n_bothZs)

    C_hatZs = gpGaussLikeFuns.cov_matrix_reg(X = X_hatZs, sigma = np.exp(log_sigma_Zs), w = np.exp(log_phi_Zs), obs_noi_scale = np.exp(log_obs_noi_scale))
    if gp_deltas_modelOut:
        C_tildZs, _ = cov_areal(X_tildZs, log_sigma_Zs, log_phi_Zs, b, log_sigma_deltas_of_modelOut, \
            gp_deltas_modelOut, log_phi_deltas_of_modelOut)
    else:
        C_tildZs, _ = cov_areal(X_tildZs, log_sigma_Zs, log_phi_Zs,b, log_sigma_deltas_of_modelOut, \
            gp_deltas_modelOut)

    mat[:n_hatZs, :n_hatZs] = C_hatZs
    mat[n_hatZs:n_hatZs + n_tildZs, n_hatZs:n_hatZs + n_tildZs] = C_tildZs

    avg_pointAreal_lower, avg_pointAreal_upper, _, _ = \
    point_areal(X_hatZs, X_tildZs, log_sigma_Zs, log_phi_Zs, b, pointArealFlag = True)
    

    mat[n_hatZs:n_hatZs + n_tildZs, :n_hatZs] = avg_pointAreal_lower
    mat[:n_hatZs, n_hatZs:n_hatZs + n_tildZs] = avg_pointAreal_upper

    mu_hatZs = np.zeros(len(y_hatZs))

    X_tildZs_mean = np.array([np.mean(X_tildZs[i], axis=0) for i in range(len(y_tildZs))])
    n_row = X_tildZs_mean.shape[0]
    tmp0 = np.repeat(1.,n_row).reshape(n_row,1)
    X_tildZs_mean_extend = np.hstack((X_tildZs_mean, tmp0))
    mu_tildZs = np.dot(X_tildZs_mean_extend, a_bias_coefficients)

    mu_hatTildZs = np.concatenate((mu_hatZs, mu_tildZs))

    y = np.concatenate((y_hatZs, y_tildZs))

    l_chol_C = gpGaussLikeFuns.compute_L_chol(mat)
    u = linalg.solve_triangular(l_chol_C.T, linalg.solve_triangular(l_chol_C, y - mu_hatTildZs, lower=True))     
    joint_log_like  = -np.sum(np.log(np.diag(l_chol_C))) - 0.5 * np.dot(y - mu_hatTildZs, u) - 0.5 * n_bothZs * np.log(2*np.pi) 

    if withPrior:

        #compute the likelihood of the gamma priors
        sigma_shape = 1.2 
        sigma_rate = 0.2 
        len_scal_shape = 1. 
        len_scal_rate = 1./np.sqrt(num_len_scal)
        obs_noi_scale_shape = 1.2
        obs_noi_scale_rate = 0.6
        b_mu =0.
        b_Sigma = np.diag([10000.])
        a_bias_coefficients_mu = np.zeros(a_bias_poly_deg +1)
        a_bias_coefficients_Sigma = np.diag(np.repeat(10000., a_bias_poly_deg + 1))
        #sigma, length_sacle, obs_noi_scale have to take positive numbers, thus taking gamma priors, whereas the mutiplicative bias b takes a normal prior
        log_prior = log_like_gamma(log_sigma_Zs, sigma_rate, sigma_shape) + log_like_gamma(log_phi_Zs, len_scal_shape, len_scal_rate) + \
        log_like_gamma(log_obs_noi_scale, obs_noi_scale_shape, obs_noi_scale_rate) + log_like_normal(b, b_mu, b_Sigma) + \
        log_like_normal(a_bias_coefficients, a_bias_coefficients_mu, a_bias_coefficients_Sigma)

        #compute the logarithm of the posterior
        log_pos = joint_log_like + log_prior
        minus_log_pos = - log_pos
    else:
        minus_log_pos = - joint_log_like

    return minus_log_pos

def hmcSampler(mu, hess_inv, grad_current_pos, log_like_current_pos, X_hatZs, y_hatZs, X_tildZs,\
 y_tildZs, epsilon0=0.15, minL=6, maxL=12, OMEGA = 1e-6):
    print 'diagonal of hess_inv ' + str(np.diag(hess_inv))
    while True:
        l_chol_hess_inv = np.linalg.cholesky(hess_inv)
        minusHessian = linalg.solve_triangular(l_chol_hess_inv.T, linalg.solve_triangular(l_chol_hess_inv, np.eye(len(mu)), lower=True))
        massMatrix = minusHessian

        l_chol_M = np.linalg.cholesky(massMatrix)

        current_position = mu


        current_momentum = np.dot(l_chol_M, np.random.normal(size=[len(current_position), 1])).reshape(len(current_position))
        tmp0 =  linalg.solve_triangular(l_chol_M, current_momentum, lower=True)
        current_kinetic = 0.5 * np.inner(tmp0,tmp0)

        proposed_momentum= current_momentum
        proposed_position = current_position
        grad_proposed_pos = grad_current_pos

        premature_reject = 0      
        L = np.random.randint(minL,maxL,1)
        print 'L is' + str(L)

        for k in range(L):
            print 'k is ' + str(k)
            epsilon = np.random.exponential(epsilon0) #Randomization of the stepsize
            print 'epsilon is ' + str(epsilon)
            p_half= proposed_momentum + epsilon/2. * grad_proposed_pos

            if (np.isinf(p_half) + np.isnan(p_half)).any():
                premature_reject = 1
                break
            proposed_position = proposed_position + epsilon * linalg.solve_triangular(l_chol_M.T, linalg.solve_triangular(l_chol_M, p_half, lower=True))
        
            if np.max(np.abs(proposed_position)) > 20:
                premature_reject =1 
                break

            log_like_proposed_pos, grad_proposed_pos = log_obsZs_giv_par_with_grad(proposed_position, X_hatZs, y_hatZs, X_tildZs, y_tildZs)

            proposed_momentum = p_half + epsilon/2. * grad_proposed_pos             
            if (np.isinf(proposed_momentum) + np.isnan(proposed_momentum)).any():
                premature_reject =1
                break

            tmp1 = linalg.solve_triangular(l_chol_M, proposed_momentum, lower=True)
            proposed_kinetic = 0.5 * np.inner(tmp1,tmp1)

        if premature_reject==1:
            A = -np.inf
        if premature_reject==0:
            A = np.min([0, log_like_proposed_pos - log_like_current_pos - proposed_kinetic + current_kinetic])
        if np.isnan(A):
            A = - np.inf
        print 'acceptance status is ' + str(-np.random.exponential(1) < A)
        if -np.random.exponential(1) < A:
            break
    return [proposed_position, grad_proposed_pos, log_like_proposed_pos]
   

def optimise(X_hatZs, y_hatZs, X_tildZs, y_tildZs, withPrior, gpdtsMo=False, useGradsFlag = False, repeat=3, seed =188, method='L-BFGS-B', rbf=True, OMEGA = 1e-6, \
    bounds = ((-2.3, 4.6), (-2.3, 4.6), (-2.3, 4.6), (-2.3, 4.6), (-2.3, 4.6), (-50, 50), (-50, 50), (-50, 50), (-50, 50))): 
    print 'starting optimising when withPrior is ' + str(withPrior) + ' & gpdtsMo is ' + str(gpdtsMo) + \
         '& useGradsFlag is ' + str(useGradsFlag) 
    if rbf:
        num_par=1
    else:
        num_par=X_hatZs.shape[1]  
    res = []
    count = 0
    LBFGSB_status = False
   
    while count != repeat:
        try:#find one intial value for which the optimisation works
            if gpdtsMo:
                initial_theta=np.concatenate((np.log(np.random.gamma(1.2, 3.5, 1)), np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), \
                np.log(np.random.gamma(1.2, 1./0.6, 1)), np.log(np.random.gamma(1.2, 3.5, 1)), \
                np.log(np.random.gamma(1., np.sqrt(num_par), num_par)),  np.zeros(4)), axis=0)
            else:
                initial_theta=np.concatenate((np.log(np.random.gamma(1.2, 5., 1)), np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), \
                np.log(np.random.gamma(1.2, 1./0.6, 1)), np.log(np.random.gamma(1.2, 5., 1)), np.zeros(4)), axis=0)
            print 'initial theta when withPrior is ' + str(withPrior) + ' & gpdtsMo is ' + str(gpdtsMo) +  \
            '& useGradsFlag is ' + str(useGradsFlag) +  ' :' + str(initial_theta)
            if useGradsFlag:
                tmp_res = minimize(fun=minus_log_obsZs_giv_par_with_grad, 
                               x0=initial_theta, method=method,
                               jac=True, bounds = bounds,
                               args=(X_hatZs, y_hatZs, X_tildZs, y_tildZs, gpdtsMo, withPrior),
                               options={'maxiter': 2000, 'disp': False})
            else:
                tmp_res = minimize(fun=minus_log_obsZs_giv_par, 
                               x0=initial_theta, method=method,
                               jac=False, bounds = bounds,
                               args=(X_hatZs, y_hatZs, X_tildZs, y_tildZs, gpdtsMo, withPrior),
                               options={'maxiter': 2000, 'disp': False})
            print 'The ' + str(count) + ' round of optimisation'
        except:
            continue
        if tmp_res['fun'] is not None: # if log pos at optimisation is not None, record the resutls, else, redo the otpmisation
            temp_res = [tmp_res['x'], np.copy(tmp_res['fun']), np.copy(tmp_res['hess_inv'].todense()), np.copy(tmp_res['success']), \
            np.copy(tmp_res['message']), np.copy(tmp_res['nit'])]
            res.append(temp_res)
            print 'theta from the ' + str(count) + ' round of optimisation with LBFGSB is ' + str(tmp_res['x'])
            logPosat_resOptim, grads_at_resOptim = log_obsZs_giv_par_with_grad(tmp_res['x'], X_hatZs, y_hatZs, X_tildZs, y_tildZs, \
                gp_deltas_modelOut = gpdtsMo,  withPrior= withPrior)
            print 'grads at theta from the ' + str(count) + ' round of optimisation with LBFGSB is ' + str(grads_at_resOptim)
            flag_grads_equal_zero = np.round(grads_at_resOptim, 2) == 0.
            # if gradients from the first optimisation is zero, break out of the loop
            if np.sum(flag_grads_equal_zero) == len(flag_grads_equal_zero):
                LBFGSB_status = True
                print 'LBFGSB optmisation converged successfully at the '+ str(count) + ' round of optimisation.'
                res = np.array(res)
                res = res.T
                print 'minus_log_like for repeat ' + str(count)+ ' with LBFGSB is ' + str(res[1, :])
                print 'parameters after optimisation withPrior is ' + str(withPrior) + \
                ' & gpdtsMo is ' + str(gpdtsMo) + ' with LBFGSB :'  + \
                str([np.exp(np.array(tmp_res['x'])[:-4]), np.array(tmp_res['x'])[-4:]])
                print 'covariance of pars after optimisation withPrior is ' + str(withPrior) + \
                 ' & gpdtsMo is ' + str(gpdtsMo) + ' with LBFGSB :'  + str(np.array(tmp_res['hess_inv'].todense()))
                print 'Optim status withPrior is ' + str(withPrior) + \
                ' & gpdtsMo is ' + str(gpdtsMo) + ' with LBFGSB :'  + str(np.array(tmp_res['success']))
                print 'Optim message withPrior is ' + str(withPrior) + \
                ' & gpdtsMo is ' + str(gpdtsMo) + ' with LBFGSB :'  + str(np.array(tmp_res['message']))
                print 'Optim nit withPrior is ' + str(withPrior) + \
                 ' & gpdtsMo is ' + str(gpdtsMo) + ' with LBFGSB :'  + str(np.array(tmp_res['nit']))
                break
            else:#if gradients from the LBFGSB optimisation is NOT zero, do another BFGSB optimisation
                if count == (repeat -1):#if gradients from the all LBFGSB optimisation is NOT zero, do ONE ROUND OF  NON constraint optimisation AT THE LAST STEP
                    file = 'logs/unsuccessAll' + str(repeat)+ 'Seed' + str(seed)+ str(method)
                    f1 = open(file, 'wb')
                    f1.close()

                    print 'initial theta from LBFGSB optimisation for BFGS is ' + str(tmp_res['x'])
                    tmp_res = minimize(fun=minus_log_obsZs_giv_par_with_grad, 
                                   x0=tmp_res['x'], method='BFGS',
                                   jac=True,
                                   args=(X_hatZs, y_hatZs, X_tildZs, y_tildZs, gpdtsMo, withPrior),
                                   options={'maxiter': 2000, 'disp': False})
                    print 'theta from the ' + str(count) + ' round of optimisation with BFGS is ' + str(tmp_res['x'])
                    logPosat_resOptim, grads_at_resOptim = log_obsZs_giv_par_with_grad(tmp_res['x'], X_hatZs, y_hatZs, X_tildZs, y_tildZs, \
                        gp_deltas_modelOut = gpdtsMo,  withPrior= withPrior)
                    print 'grads at theta from the ' + str(count) + ' round of optimisation with BFGS is ' + str(grads_at_resOptim)
                    flag_grads_equal_zero = np.round(grads_at_resOptim, 2) == 0.
                    # if gradients from the BFGS optimisation is zero, break out of the loop
                    if np.sum(flag_grads_equal_zero) == len(flag_grads_equal_zero):
                        print 'BFGS optmisation converged successfully at the '+ str(count) + ' round of optimisation.'
                        print 'minus_log_like for repeat ' + str(count)+ ' with BFGS is ' + str(tmp_res['fun'])
                        print 'parameters after optimisation withPrior is ' + str(withPrior) + \
                        ' & gpdtsMo is ' + str(gpdtsMo) + ' with BFGS :'  + \
                        str([np.exp(np.array(tmp_res['x'])[:-4]), np.array(tmp_res['x'])[-4:]])
                        print 'covariance of pars after optimisation withPrior is ' + str(withPrior) + \
                 ' & gpdtsMo is ' + str(gpdtsMo) + ' with BFGS :'  + str(np.array(tmp_res['hess_inv']))
                        print 'Optim status withPrior is ' + str(withPrior) + \
                        ' & gpdtsMo is ' + str(gpdtsMo) + ' with BFGS :'  + str(np.array(tmp_res['success']))
                        print 'Optim message withPrior is ' + str(withPrior) + \
                        ' & gpdtsMo is ' + str(gpdtsMo) + ' with BFGS :'  + str(np.array(tmp_res['message']))
                        print 'Optim nit withPrior is ' + str(withPrior) + \
                         ' & gpdtsMo is ' + str(gpdtsMo) + ' with BFGS :'  + str(np.array(tmp_res['nit']))
                        break
                else:    
                    count += 1        
        else:# if log pos at optimisation is not None, record the resutls, else, redo the otpmisation
            continue 
    if LBFGSB_status:
        return [np.array(tmp_res['x']), np.array(tmp_res['hess_inv'].todense())]
    else:
        return [np.array(tmp_res['x']), np.array(tmp_res['hess_inv'])]

class MH_estimator():
    def __init__(self, X_hatZs, y_hatZs, X_tildZs, y_tildZs):
        self.X_hatZs = X_hatZs
        self.y_hatZs = y_hatZs
        self.X_tildZs = X_tildZs
        self.y_tildZs = y_tildZs
    
    def burningin(self, mu, cov, nburnin=2000,  nbatch=100, OMEGA = 1e-6):
        log_like = log_obsZs_giv_par(mu, self.X_hatZs, self.y_hatZs, self.X_tildZs, self.y_tildZs)
        num_iter = nburnin / nbatch
        tmp_mu = mu[:]
        iter_index = None

        for i in range(num_iter):
            accept_count = 0
            l_cov = np.linalg.cholesky(cov)
            for j in range(nbatch):
                mu_p = np.dot(l_cov, np.random.normal(size=[len(mu), 1])).reshape(len(mu)) + tmp_mu
                if np.max(np.abs(mu_p)) > 20:
                    diff_log_like = - np.inf
                else:
                    log_like_p = log_obsZs_giv_par(mu_p, self.X_hatZs, self.y_hatZs, self.X_tildZs, self.y_tildZs)
                    diff_log_like = log_like_p - log_like
                if np.log(np.random.uniform(0, 1)) < diff_log_like:
                    tmp_mu = mu_p
                    log_like = log_like_p
                    accept_count += 1
            accept_rate = (float)(accept_count) / (float)(nbatch)
            print 'Acc_Rate for the ' + str(i) + ' iteration is ' + str(accept_rate)

            if accept_rate >= 0.20 and accept_rate <= 0.36:
                iter_index = i
                break
            if accept_rate < 0.20:
                cov *= 0.8
            if accept_rate > 0.36:
                cov /= 0.8
        return iter_index, cov, tmp_mu

    def sampler(self, cov, mu, size=1000, OMEGA = 1e-6):
        print 'starting sampling'
        sample = np.zeros([len(mu), size])
        acc_count = 0
        l_cov = np.linalg.cholesky(cov)
        log_like = log_obsZs_giv_par(mu, self.X_hatZs, self.y_hatZs, self.X_tildZs, self.y_tildZs)
        for i in range(size):
            mu_p = np.dot(l_cov, np.random.normal(size=[len(mu), 1])).reshape(len(mu)) + mu
            if np.max(np.abs(mu_p)) > 20:
                diff_log_like = - np.inf
            else:
                log_like_p = log_obsZs_giv_par(mu_p, self.X_hatZs, self.y_hatZs, self.X_tildZs, self.y_tildZs)
                diff_log_like = log_like_p - log_like
            if np.log(np.random.uniform(0, 1)) < diff_log_like:
                mu = mu_p
                log_like = log_like_p
                acc_count += 1
            sample[:,i] = mu.reshape(len(mu))
            if (i+1) % 100 ==0:
                sample_out = open('sample_par_size' + str(size) + '.pickle', 'wb')
                pickle.dump(sample[:, :i+1], sample_out)
                sample_out.close()

        acc_rate = (float)(acc_count) / size
        print 'Acc_Rate:' + str(acc_rate)
        return sample

    def estimate(self, size = 100, proposal = 'identity', useOptim = False, rbf = True):
        if useOptim:
            while True:
                try:
                    mu, hess_inv= optimise(self.X_hatZs, self.y_hatZs, self.X_tildZs, self.y_tildZs, withPrior, \
                        gpdtsMo=False, useGradsFlag = False, repeat=3, method='BFGS', rbf=True, OMEGA = 1e-6)
                    print 'diag of the hess_inv is ' + str(np.diag(hess_inv))
                    # while np.diag(hess_inv)[0] < 1e-4:
                    #     mu, hess_inv= optimize(self.X_hatZs, self.y_hatZs, self.X_tildZs, self.y_tildZs, repeat=1, method='BFGS', rbf=True, OMEGA = 1e-6)
                    break
                except:
                    continue
            # print 'exp_mu_optim :' + str(np.exp(mu))
            if proposal == 'identity':
                cov =  np.identity(len(mu))            
            if proposal == 'fullApproxCov':
                cov = hess_inv
            if proposal == 'diagOfApproxCov':
                cov = np.diag(np.diag(hess_inv))
        else:
            if rbf:
                num_par=1
            else:
                num_par=self.X_hatZs.shape[1]
            mu = np.concatenate((np.log(np.random.gamma(1.2, 5., 1)), np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), np.log(np.random.gamma(1.2, 1./0.6, 1)), np.random.normal(0., 1., 1)), axis=0)
            cov = np.identity(len(mu))

        iter = None
        nburnin = 2000
        while iter is None:
            iter, cov, mu = self.burningin(mu, cov, nburnin=nburnin)
            nburnin += 2000

        print 'log_parameters  plus mulplicative_bias after burningin is :'  + str(mu)
        print 'parameters after burningin is :'  + str([np.exp(mu[:-1]), mu[-1]])
        print 'cov of pars after burningin is :'  + str(cov)

        # cov = hess_inv

        sample = self.sampler(cov, mu, size = size)
        sample[:-1, :] = np.exp(sample[:-1, :])

        sample_mod = stats.mode(sample, axis =1)
        sample_mean = np.mean(sample, axis=1)
        sample_std = np.std(sample, axis = 1)
        assert len(sample) == len(mu)
        return [sample_mod, sample_mean, sample_std]

class Gibbs_sampler():
    def __init__(self, X_hatZs, y_hatZs, X_tildZs, y_tildZs, areal_hatZs):
        self.X_hatZs = X_hatZs
        self.y_hatZs = y_hatZs
        self.X_tildZs = X_tildZs
        self.y_tildZs = y_tildZs
        self.areal_hatZs = areal_hatZs

    def initial_model_bias(self):
        tmp0 = [np.mean(self.X_tildZs[i], axis =0) for i in range(len(self.X_tildZs))]
        tmp0 = np.array(tmp0)
        tmp1 = [np.mean(self.areal_hatZs[i]) for i in range(len(self.areal_hatZs))]
        tmp1= np.array(tmp1).reshape(len(tmp1), 1)
        X = np.hstack((tmp1, tmp0))
        regr = linear_model.LinearRegression()
        regr.fit(X, self.y_tildZs)
        coefficients = regr.coef_
        intercept = regr.intercept_
        model_bias_coefficients = np.concatenate((coefficients, [intercept]))
        return model_bias_coefficients
    # try to fix the model bias parameters to the ones obtained from linear regression, to check how intialisation can affect the optimisation
    def optim(self, withPrior, modelBias, onlyOptimCovPar = False, gpdtsMo=False, useGradsFlag = False, repeat=3, seed =188, method='BFGS', rbf=True, OMEGA = 1e-6): 
        print 'starting optimising when withPrior is ' + str(withPrior) + ' & gpdtsMo is ' + str(gpdtsMo) + \
        ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) 
        if rbf:
            num_par=1
        else:
            num_par=self.X_hatZs.shape[1]  
        res = []
        count = 0
        if onlyOptimCovPar:
            while count != repeat:
                try:
                    if gpdtsMo:
                        initial_theta=np.concatenate((np.log(np.random.gamma(1.2, 5., 1)), np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), \
                        np.log(np.random.gamma(1.2, 1./0.6, 1)), np.log(np.random.gamma(1.2, 5., 1)), \
                        np.log(np.random.gamma(1., np.sqrt(num_par), num_par))), axis=0)
                    else:
                        initial_theta=np.concatenate((np.log(np.random.gamma(1.2, 5., 1)), np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), \
                        np.log(np.random.gamma(1.2, 1./0.6, 1)), np.log(np.random.gamma(1.2, 5., 1))), axis=0)
                    print 'initial theta when withPrior is ' + str(withPrior) + ' & gpdtsMo is ' + str(gpdtsMo) + \
                    ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) + ' :' + str(initial_theta)
                    tmp_res = minimize(fun=minus_log_obsZs_giv_par_of_cov, 
                                   x0=initial_theta, method=method,
                                   jac=False,
                                   args=(self.X_hatZs, self.y_hatZs, self.X_tildZs, self.y_tildZs, withPrior, modelBias, gpdtsMo),
                                   options={'maxiter': 100, 'disp': False})
                    print 'The ' + str(count + 1) + ' round of optimisation'
                except:
                    continue
                if tmp_res['fun'] is not None:
                    count += 1
                    temp_res = [tmp_res['x'], np.copy(tmp_res['fun']), np.copy(tmp_res['hess_inv']), np.copy(tmp_res['success']), \
                    np.copy(tmp_res['message']), np.copy(tmp_res['nit'])]
                    res.append(temp_res)
                else:
                    continue      
            res = np.array(res)
            res = res.T
            print 'minus_log_like for repeat ' + str(repeat) + ' is ' + str(res[1, :])
            i = np.argmin(res[1,:])
            print 'log_cov_parameters after optimisation withPrior is ' + str(withPrior) + \
            ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) + ' :'  + str(np.array(res[0, :][i]))
            print 'cov_parameters after optimisation withPrior is ' + str(withPrior) + \
            ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) + ' :'  + str(np.exp(np.array(res[0, :][i])))
            print 'cov after optimisation withPrior is ' + str(withPrior) + \
            ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) + ' :'  + str(np.array(res[2, :][i]))
        else:
            while count != repeat:
                try:
                    if gpdtsMo:
                        initial_theta=np.concatenate((np.log(np.random.gamma(1.2, 5., 1)), np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), \
                        np.log(np.random.gamma(1.2, 1./0.6, 1)), np.log(np.random.gamma(1.2, 5., 1)), \
                        np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), modelBias), axis=0)
                    else:
                        initial_theta=np.concatenate((np.log(np.random.gamma(1.2, 5., 1)), np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), \
                        np.log(np.random.gamma(1.2, 1./0.6, 1)), np.log(np.random.gamma(1.2, 5., 1)), modelBias), axis=0)
                    print 'initial theta when withPrior is ' + str(withPrior) + ' & gpdtsMo is ' + str(gpdtsMo) +  \
                    '& useGradsFlag is ' + str(useGradsFlag) + ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) + ' :' + str(initial_theta)
                    if useGradsFlag:
                        tmp_res = minimize(fun=minus_log_obsZs_giv_par_with_grad, 
                                       x0=initial_theta, method=method,
                                       jac=True,
                                       args=(self.X_hatZs, self.y_hatZs, self.X_tildZs, self.y_tildZs, gpdtsMo, withPrior),
                                       options={'maxiter': 100, 'disp': False})
                    else:
                        tmp_res = minimize(fun=minus_log_obsZs_giv_par, 
                                       x0=initial_theta, method=method,
                                       jac=False,
                                       args=(self.X_hatZs, self.y_hatZs, self.X_tildZs, self.y_tildZs, gpdtsMo, withPrior),
                                       options={'maxiter': 100, 'disp': False})
                    print 'The ' + str(count + 1) + ' round of optimisation'
                except:
                    continue
                if tmp_res['fun'] is not None:
                    if tmp_res['fun']<0:
                        temp_res = [tmp_res['x'], np.copy(tmp_res['fun']), np.copy(tmp_res['hess_inv']), np.copy(tmp_res['success']), \
                        np.copy(tmp_res['message']), np.copy(tmp_res['nit'])]
                        res.append(temp_res)
                        file = 'logs/seed' + str(seed)+ 'successAtrep' + str(count)
                        f0 = open(file, 'wb')
                        f0.close()

                        break
                    else:
                        count += 1
                        temp_res = [tmp_res['x'], np.copy(tmp_res['fun']), np.copy(tmp_res['hess_inv']), np.copy(tmp_res['success']), \
                        np.copy(tmp_res['message']), np.copy(tmp_res['nit'])]
                        res.append(temp_res)
                else:
                    continue      
            res = np.array(res)
            res = res.T
            print 'minus_log_like for repeat ' + str(repeat) + ' is ' + str(res[1, :])
            i = np.argmin(res[1,:])
            if np.array(res[1, :][i]) >0:
                file = 'logs/unsuccessAll' + str(repeat)+ 'repSeed' + str(seed)
                f1 = open(file, 'wb')
                f1.close()
         
            print 'log_cov_parameters plus model bias after optimisation withPrior is ' + str(withPrior) + \
             ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) + ' & gpdtsMo is ' + str(gpdtsMo) + ' :'  + str(np.array(res[0, :][i]))
            print 'parameters after optimisation withPrior is ' + str(withPrior) + \
            ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) + ' & gpdtsMo is ' + str(gpdtsMo) + ' :'  + \
            str([np.exp(np.array(res[0, :][i])[:-4]), np.array(res[0, :][i])[-4:]])
            print 'covariance of pars after optimisation withPrior is ' + str(withPrior) + \
            ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) + ' & gpdtsMo is ' + str(gpdtsMo) + ' :'  + str(np.array(res[2, :][i]))
            print 'Optim status withPrior is ' + str(withPrior) + \
            ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) + ' & gpdtsMo is ' + str(gpdtsMo) + ' :'  + str(np.array(res[3, :][i]))
            print 'Optim message withPrior is ' + str(withPrior) + \
            ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) + ' & gpdtsMo is ' + str(gpdtsMo) + ' :'  + str(np.array(res[4, :][i]))
            print 'Optim nit withPrior is ' + str(withPrior) + \
            ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) + ' & gpdtsMo is ' + str(gpdtsMo) + ' :'  + str(np.array(res[5, :][i]))

            #check wheter the gradients at theta from optimisation are close to zeros (wchich indicates the convergence of the optimisation)
            _, grads_at_resOptim = log_obsZs_giv_par_with_grad(np.array(res[0, :][i]), self.X_hatZs, self.y_hatZs, self.X_tildZs, self.y_tildZs,\
             gp_deltas_modelOut = gpdtsMo,  withPrior= withPrior)
            print 'grads at theta from optimisation is ' + str(grads_at_resOptim)
            flag_grads_equal_zero = np.round(grads_at_resOptim, 2) == 0.
            if np.sum(flag_grads_equal_zero) == len(flag_grads_equal_zero):
                print 'BFGS optmisation converged successfully.'
            else:
                print 'BFGS optmisation NOT converged.'
             
        return [np.array(res[0, :][i]), np.array(res[2, :][i])]

    def sampler(self, withPrior = False, onlyOptimCovPar = False, gpdtsMo=False, useGradsFlag = False, repeat = 3, seed=188):

        model_bias = self.initial_model_bias() 
        print 'model bias from linear regression is :' + str(model_bias)
     
        mu, cov = self.optim(withPrior, model_bias, onlyOptimCovPar, gpdtsMo, useGradsFlag,repeat, seed)
        return [mu, cov] 

class HMC_estimator():
    def __init__(self, X_hatZs, y_hatZs, X_tildZs, y_tildZs, areal_tildZs):
        self.X_hatZs = X_hatZs
        self.y_hatZs = y_hatZs
        self.X_tildZs = X_tildZs
        self.y_tildZs = y_tildZs
        self.areal_tildZs = areal_tildZs

    def initial_model_bias(self):
        tmp0 = [np.mean(self.X_tildZs[i], axis =0) for i in range(len(self.X_tildZs))]
        tmp0 = np.array(tmp0)
        tmp1 = [np.mean(self.areal_tildZs[i]) for i in range(len(self.areal_tildZs))]
        tmp1= np.array(tmp1).reshape(len(tmp1), 1)
        X = np.hstack((tmp1, tmp0))
        regr = linear_model.LinearRegression()
        regr.fit(X, self.y_tildZs)
        coefficients = regr.coef_
        intercept = regr.intercept_
        model_bias_coefficients = np.concatenate((coefficients, [intercept]))
        return model_bias_coefficients

    def burningin(self, mu, cov, massMatrix, epsilon0=0.1, minL=1, maxL=10, nburnin=2000,  nbatch=100, OMEGA = 1e-6):
        tmp_mu = mu[:]
        # random initial position centered on the mode obtained from optimisation
        current_position = tmp_mu + np.dot(np.linalg.cholesky(cov), np.random.normal(size=[len(mu), 1])).reshape(len(mu))
        log_like_current_pos, grad_current_pos = gpGaussianLikeFuns.log_py_giv_par_with_grad(current_position, self.X, self.y)
        l_chol_M = np.linalg.cholesky(massMatrix)

        num_iter = nburnin / nbatch
        iter_index = None

        for i in range(num_iter):
            accept_count = 0            
            for j in range(nbatch):
                current_momentum = np.dot(l_chol_M, np.random.normal(size=[len(mu), 1])).reshape(len(mu))
                tmp0 =  linalg.solve_triangular(l_chol_M, current_momentum, lower=True)
                current_kinetic = 0.5 * np.inner(tmp0,tmp0)

                proposed_momentum= current_momentum
                proposed_position = current_position
                grad_proposed_pos = grad_current_pos

                premature_reject = 0      
                L = np.random.randint(minL,maxL,1)

                for k in range(L):
                    epsilon = np.random.exponential(epsilon0) #Randomization of the stepsize
                    p_half= proposed_momentum + epsilon/2. * grad_proposed_pos

                    if (np.isinf(p_half) + np.isnan(p_half)).any():
                        premature_reject = 1
                        break
                    proposed_position = proposed_position + epsilon * linalg.solve_triangular(l_chol_M.T, linalg.solve_triangular(l_chol_M, p_half, lower=True))
                
                    if np.max(np.abs(proposed_position)) > 20:
                        premature_reject =1 
                        break

                    log_like_proposed_pos, grad_proposed_pos = gpGaussianLikeFuns.log_py_giv_par_with_grad(proposed_position, self.X, self.y)

                    proposed_momentum = p_half + epsilon/2. * grad_current_pos
                    if (np.isinf(proposed_momentum) + np.isnan(proposed_momentum)).any():
                        premature_reject =1
                        break
                    tmp1 = linalg.solve_triangular(l_chol_M, proposed_momentum, lower=True)
                    proposed_kinetic = 0.5 * np.inner(tmp1,tmp1)

                if premature_reject==1:
                    A = -np.inf
                if premature_reject==0:
                    A = np.min([0, log_like_proposed_pos - log_like_current_pos - proposed_kinetic + current_kinetic])
                if np.isnan(A):
                    A = - np.inf

                if -np.random.exponential(1) < A:
                    current_position = proposed_position
                    grad_current_pos = grad_proposed_pos
                    log_like_current_pos = log_like_proposed_pos
                    accept_count += 1

            accept_rate = (float)(accept_count) / (float)(nbatch)

            #minAcc= 0.56, maxAcc=0.75, tune_cof = 0.8
            minAcc= 0.2
            maxAcc=0.35
            tune_cof = 0.7

            if accept_rate >= minAcc and accept_rate <= maxAcc:
                iter_index = i
                print 'tuning accept rate is :' + str(accept_rate)
                break
            if accept_rate < minAcc:
                epsilon0 *= tune_cof
            if accept_rate > maxAcc:
                epsilon0 /= tune_cof
        return iter_index, epsilon0, current_position, grad_current_pos,log_like_current_pos

    def optim(self, withPrior, modelBias, onlyOptimCovPar = False, gpdtsMo=False, repeat=5, method='BFGS', rbf=True, OMEGA = 1e-6): 
        print 'starting optimising when withPrior is ' + str(withPrior) + ' & gpdtsMo is ' + str(gpdtsMo) + \
        ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) 
        if rbf:
            num_par=1
        else:
            num_par=self.X_hatZs.shape[1]  
        res = []
        count = 0
        if onlyOptimCovPar:
            while count != repeat:
                try:
                    if gpdtsMo:
                        initial_theta=np.concatenate((np.log(np.random.gamma(1.2, 5., 1)), np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), \
                        np.log(np.random.gamma(1.2, 1./0.6, 1)), np.log(np.random.gamma(1.2, 5., 1)), \
                        np.log(np.random.gamma(1., np.sqrt(num_par), num_par))), axis=0)
                    else:
                        initial_theta=np.concatenate((np.log(np.random.gamma(1.2, 5., 1)), np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), \
                        np.log(np.random.gamma(1.2, 1./0.6, 1)), np.log(np.random.gamma(1.2, 5., 1))), axis=0)
                    print 'initial theta when withPrior is ' + str(withPrior) + ' & gpdtsMo is ' + str(gpdtsMo) + \
                    ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) + ' :' + str(initial_theta)
                    tmp_res = minimize(fun=minus_log_obsZs_giv_par_of_cov, 
                                   x0=initial_theta, method=method,
                                   jac=False,
                                   args=(self.X_hatZs, self.y_hatZs, self.X_tildZs, self.y_tildZs, withPrior, modelBias, gpdtsMo),
                                   options={'maxiter': 100, 'disp': False})
                    print 'The ' + str(count + 1) + ' round of optimisation'
                except:
                    continue
                if tmp_res['fun'] is not None:
                    count += 1
                    temp_res = [tmp_res['x'], np.copy(tmp_res['fun']), np.copy(tmp_res['hess_inv'])]
                    res.append(temp_res)
                else:
                    continue      
            res = np.array(res)
            res = res.T
            print 'minus_log_like for repeat ' + str(repeat) + ' is ' + str(res[1, :])
            i = np.argmin(res[1,:])
            print 'log_cov_parameters after optimisation withPrior is ' + str(withPrior) + \
            ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) + ' :'  + str(np.array(res[0, :][i]))
            print 'cov_parameters after optimisation withPrior is ' + str(withPrior) + \
            ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) + ' :'  + str(np.exp(np.array(res[0, :][i])))
            print 'cov after optimisation withPrior is ' + str(withPrior) + \
            ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) + ' :'  + str(np.array(res[2, :][i]))
        else:
            while count != repeat:
                try:
                    if gpdtsMo:
                        initial_theta=np.concatenate((np.log(np.random.gamma(1.2, 5., 1)), np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), \
                        np.log(np.random.gamma(1.2, 1./0.6, 1)), np.log(np.random.gamma(1.2, 5., 1)), \
                        np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), modelBias), axis=0)
                    else:
                        initial_theta=np.concatenate((np.log(np.random.gamma(1.2, 5., 1)), np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), \
                        np.log(np.random.gamma(1.2, 1./0.6, 1)), np.log(np.random.gamma(1.2, 5., 1)), modelBias), axis=0)
                    print 'initial theta when withPrior is ' + str(withPrior) + ' & gpdtsMo is ' + str(gpdtsMo) +  \
                    '& useGradsFlag is ' + str(useGradsFlag) + ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) + ' :' + str(initial_theta)
                    if useGradsFlag:
                        tmp_res = minimize(fun=minus_log_obsZs_giv_par_with_grad, 
                                       x0=initial_theta, method=method,
                                       jac=True,
                                       args=(self.X_hatZs, self.y_hatZs, self.X_tildZs, self.y_tildZs, gpdtsMo, withPrior),
                                       options={'maxiter': 200, 'disp': False})
                    else:
                        tmp_res = minimize(fun=minus_log_obsZs_giv_par, 
                                       x0=initial_theta, method=method,
                                       jac=False,
                                       args=(self.X_hatZs, self.y_hatZs, self.X_tildZs, self.y_tildZs, gpdtsMo, withPrior),
                                       options={'maxiter': 200, 'disp': False})
                    print 'The ' + str(count + 1) + ' round of optimisation'
                except:
                    continue
                if tmp_res['fun'] is not None:
                    if tmp_res['fun']<0:
                        temp_res = [tmp_res['x'], np.copy(tmp_res['fun']), np.copy(tmp_res['hess_inv']), np.copy(tmp_res['success']), \
                        np.copy(tmp_res['message']), np.copy(tmp_res['nit'])]
                        res.append(temp_res)
                        break
                    else:
                        count += 1
                        temp_res = [tmp_res['x'], np.copy(tmp_res['fun']), np.copy(tmp_res['hess_inv']), np.copy(tmp_res['success']), \
                        np.copy(tmp_res['message']), np.copy(tmp_res['nit'])]
                        res.append(temp_res)
                else:
                    continue      
            res = np.array(res)
            res = res.T
            print 'minus_log_like for repeat ' + str(repeat) + ' is ' + str(res[1, :])
            i = np.argmin(res[1,:])
         
            print 'log_cov_parameters plus model bias after optimisation withPrior is ' + str(withPrior) + \
             ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) + ' & gpdtsMo is ' + str(gpdtsMo) + ' :'  + str(np.array(res[0, :][i]))
            print 'parameters after optimisation withPrior is ' + str(withPrior) + \
            ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) + ' & gpdtsMo is ' + str(gpdtsMo) + ' :'  + \
            str([np.exp(np.array(res[0, :][i])[:-4]), np.array(res[0, :][i])[-4:]])
            print 'covariance of pars after optimisation withPrior is ' + str(withPrior) + \
            ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) + ' & gpdtsMo is ' + str(gpdtsMo) + ' :'  + str(np.array(res[2, :][i]))
            print 'Optim status withPrior is ' + str(withPrior) + \
            ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) + ' & gpdtsMo is ' + str(gpdtsMo) + ' :'  + str(np.array(res[3, :][i]))
            print 'Optim message withPrior is ' + str(withPrior) + \
            ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) + ' & gpdtsMo is ' + str(gpdtsMo) + ' :'  + str(np.array(res[4, :][i]))
            print 'Optim nit withPrior is ' + str(withPrior) + \
            ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) + ' & gpdtsMo is ' + str(gpdtsMo) + ' :'  + str(np.array(res[5, :][i]))
          
        return [np.array(res[0, :][i]), np.array(res[2, :][i])]

    def sampler(self, massMatrix, epsilon0, current_position, grad_current_pos, log_like_current_pos, gpdtsMo, size=1000, minL=1, maxL=3 , OMEGA = 1e-6):
        sample = np.zeros([len(current_position), size])
        acc_count = 0
        l_chol_M = np.linalg.cholesky(massMatrix)

        for i in range(size):
            current_momentum = np.dot(l_chol_M, np.random.normal(size=[len(current_position), 1])).reshape(len(current_position))
            tmp0 =  linalg.solve_triangular(l_chol_M, current_momentum, lower=True)
            current_kinetic = 0.5 * np.inner(tmp0,tmp0)

            proposed_momentum= current_momentum
            proposed_position = current_position
            grad_proposed_pos = grad_current_pos

            premature_reject = 0      
            L = np.random.randint(minL,maxL,1)

            for k in range(L):
                epsilon = np.random.exponential(epsilon0) #Randomization of the stepsize
                p_half= proposed_momentum + epsilon/2. * grad_proposed_pos

                if (np.isinf(p_half) + np.isnan(p_half)).any():
                    premature_reject = 1
                    break
                proposed_position = proposed_position + epsilon * linalg.solve_triangular(l_chol_M.T, linalg.solve_triangular(l_chol_M, p_half, lower=True))
            
                if np.max(np.abs(proposed_position)) > 20:
                    premature_reject =1 
                    break

                log_like_proposed_pos = log_obsZs_giv_par(proposed_position, self.X_hatZs, self.y_hatZs, self.X_tildZs, self.y_tildZs, gpdtsMo)
                grad_proposed_pos = gradsApprox(proposed_position, self.X_hatZs, self.y_hatZs, self.X_tildZs, self.y_tildZs, gpdtsMo)

                proposed_momentum = p_half + epsilon/2. * grad_proposed_pos             
                if (np.isinf(proposed_momentum) + np.isnan(proposed_momentum)).any():
                    premature_reject =1
                    break

                tmp1 = linalg.solve_triangular(l_chol_M, proposed_momentum, lower=True)
                proposed_kinetic = 0.5 * np.inner(tmp1,tmp1)

            if premature_reject==1:
                A = -np.inf
            if premature_reject==0:
                A = np.min([0, log_like_proposed_pos - log_like_current_pos - proposed_kinetic + current_kinetic])
            if np.isnan(A):
                A = - np.inf

            if -np.random.exponential(1) < A:
                current_position = proposed_position
                grad_current_pos = grad_proposed_pos
                log_like_current_pos = log_like_proposed_pos
                acc_count += 1
            sample[:,i] = current_position.reshape(len(current_position))
        acc_rate = (float)(acc_count) / size
        print 'Acc_Rate:' + str(acc_rate)
        return sample
    def estimate(self, gpdtsMo=True, useGradsFlag = False, withPrior = False, onlyOptimCovPar = False, repeat = 2, massMat='identity', burninginFlag = False, rbf=True, size = 200):
        modelBias = self.initial_model_bias() 
        print 'model bias from linear regression is :' + str(modelBias)

        if rbf:
                num_par=1
        else:
            num_par=self.X_hatZs.shape[1] 
        if gpdtsMo:
            initial_theta=np.concatenate((np.log(np.random.gamma(1.2, 5., 1)), np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), \
            np.log(np.random.gamma(1.2, 1./0.6, 1)), np.log(np.random.gamma(1.2, 5., 1)), \
            np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), modelBias), axis=0)
        else:
            initial_theta=np.concatenate((np.log(np.random.gamma(1.2, 5., 1)), np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), \
            np.log(np.random.gamma(1.2, 1./0.6, 1)), np.log(np.random.gamma(1.2, 5., 1)), modelBias), axis=0)

        if massMat == 'identity':
            massMatrix= np.identity(len(initial_theta))
        else:
            print 'starting optimizing' 
            mu, cov = self.optim(withPrior, modelBias, onlyOptimCovPar, gpdtsMo, useGradsFlag,repeat)
            l_chol_hess_inv = np.linalg.cholesky(hess_inv)
            minusHessian = linalg.solve_triangular(l_chol_hess_inv.T, linalg.solve_triangular(l_chol_hess_inv, np.eye(len(mu)), lower=True))
            if massMat == 'minusHessian':
                massMatrix = minusHessian
            if massMat == 'diagOfMinusHessian':
                massMatrix = np.diag(np.diag(minusHessian))

        if burninginFlag:
            print 'starting burninging'
            iter = None
            nburnin = 2000
            mu = initial_theta
            hess_inv = np.identity(len(initial_theta))
            while iter is None:
                iter, epsilon0, current_position, grad_current_pos,log_like_current_pos = self.burningin(mu, hess_inv, massMatrix, epsilon0=0.1, minL=1, maxL=10, nburnin=nburnin,  nbatch=100, OMEGA = 1e-6)
                nburnin += 2000
            print 'tuned step size is :'  + str(epsilon0)
        else:
            epsilon0=0.1
            
        # random initial position centered on the mode obtained from optimisation
        current_position = initial_theta
        log_like_current_pos = log_obsZs_giv_par(current_position, self.X_hatZs, self.y_hatZs, self.X_tildZs, self.y_tildZs, gpdtsMo)
        grad_current_pos = gradsApprox(current_position, self.X_hatZs, self.y_hatZs, self.X_tildZs, self.y_tildZs, gpdtsMo)

        l_chol_M = np.linalg.cholesky(massMatrix)

        print 'starting sampling'

        size = 5
        sample = self.sampler(massMatrix, epsilon0, current_position, grad_current_pos,log_like_current_pos, gpdtsMo, size)
        sample = np.mean(sample, axis=1)
        assert len(sample) == len(current_position)
        return sample

def read_Sim_Data():
    #read the samples of hatZs
    X_hatZs = np.array(pd.read_csv('dataSimulated/X_hatZs_res100_a_bias_poly_deg2SEED39_lsZs[0.1]_sigZs1.5_gpdtsMoTrue_lsdtsMo[0.2]_sigdtsMo1.0.txt', sep=" ", header=None))
    y_hatZs = np.array(pd.read_csv('dataSimulated/y_hatZs_res100_a_bias_poly_deg2SEED39_lsZs[0.1]_sigZs1.5_gpdtsMoTrue_lsdtsMo[0.2]_sigdtsMo1.0.txt', sep=" ", header=None)).reshape(X_hatZs.shape[0])

    #read the samples of tildZs
    X_tildZs_in = open('dataSimulated/X_tildZs_a_bias_poly_deg2SEED39_lsZs[0.1]_sigZs1.5_gpdtsMoTrue_lsdtsMo[0.2]_sigdtsMo1.0.pickle', 'rb')
    X_tildZs = pickle.load(X_tildZs_in)

    y_tildZs_in = open('dataSimulated/y_tildZs_a_bias_poly_deg2SEED39_lsZs[0.1]_sigZs1.5_gpdtsMoTrue_lsdtsMo[0.2]_sigdtsMo1.0.pickle', 'rb')
    y_tildZs = pickle.load(y_tildZs_in)

    areal_tildZs_in = open('dataSimulated/areal_tildZs_a_bias_poly_deg2SEED39_lsZs[0.1]_sigZs1.5_gpdtsMoTrue_lsdtsMo[0.2]_sigdtsMo1.0.pickle', 'rb')
    areal_tildZs = pickle.load(areal_tildZs_in)

    return[X_hatZs, y_hatZs, X_tildZs, y_tildZs, areal_tildZs]

def check_grads():
    input_folder = 'sampRealData/FPstart2016020612_FR_numObs_' + str(328) + '_numMo_' + str(250) \
    + '/seed' + str(120) + '/'
    X_hatZs_in = open(input_folder + 'X_hatZs.pkl', 'rb')
    X_hatZs = pickle.load(X_hatZs_in) 
 
    y_hatZs_in = open(input_folder + 'y_hatZs.pkl', 'rb')
    y_hatZs = pickle.load(y_hatZs_in) 

    X_tildZs_in = open(input_folder + 'X_tildZs.pkl', 'rb')
    X_tildZs = pickle.load(X_tildZs_in) 
  
    y_tildZs_in = open(input_folder + 'y_tildZs.pkl', 'rb')
    y_tildZs = pickle.load(y_tildZs_in)

    areal_hatZs_in = open(input_folder + 'areal_hatZs.pkl', 'rb')
    areal_hatZs = pickle.load(areal_hatZs_in)
    modelBias = np.array([ 1.97191694,  5.47022754 , 5.22854712, 2.5])
    initial_theta=np.concatenate((np.array([np.log(1.5), np.log(0.1), np.log(0.1), np.log(1.0),np.log(0.2)]),modelBias), axis=0)
    _, grads_computed = log_obsZs_giv_par_with_grad(initial_theta, X_hatZs, y_hatZs, X_tildZs, y_tildZs,gp_deltas_modelOut = True)
    grads_approx = gradsApprox(initial_theta, X_hatZs, y_hatZs, X_tildZs, y_tildZs,  withPrior= False, gp_deltas_modelOut = True)
    print 'Computed grads are ' + str(grads_computed)
    print 'approximated grads are ' + str(grads_approx)
    numerator = np.linalg.norm(grads_approx- grads_computed)                                
    denominator = np.linalg.norm(grads_approx)+np.linalg.norm(grads_computed)               
    difference = numerator/denominator                                          

    if difference > 1e-4:
        print ("\033[93m" + "There is a mistake in computing the gradients! difference = " + str(difference) + "\033[0m")
    else:
        print ("\033[92m" + "Computing the gradients works perfectly fine! difference = " + str(difference) + "\033[0m")
    
    return difference     
    
if __name__ == '__main__':
    computeN3Cost.init(0)
    p = argparse.ArgumentParser()
    p.add_argument('-SEED', type=int, dest='SEED', default=99, help='The simulation index')
    p.add_argument('-repeat', type=int, dest='repeat', default=1, help='number of repeats in optimisation')
    p.add_argument('-o', type=str, dest='output', default=None, help='Output folder')
    p.add_argument('-withPrior', dest='withPrior', default=False,  type=lambda x: (str(x).lower() == 'true'), help='flag for ML or MAP')
    p.add_argument('-fixMb', dest='fixMb', default=False,  type=lambda x: (str(x).lower() == 'true'), \
        help='flag for fixed model bias parameters from linear regression for intialisation in optimisition')
    p.add_argument('-onlyOptimCovPar', dest='onlyOptimCovPar', default=False,  type=lambda x: (str(x).lower() == 'true'), \
        help='flag for only optimising the cov parameters with fixed model bias from linear regression')
    p.add_argument('-poly_deg', type=int, dest='poly_deg', default=2, help='degree of the polynomial function of the additive model bias')
    p.add_argument('-lsZs', type=float, dest='lsZs', default=0.1, help='lengthscale of the GP covariance for Zs')
    p.add_argument('-lsdtsMo', type=float, dest='lsdtsMo', default=0.6, help='lengthscale of the GP covariance for deltas of model output')
    p.add_argument('-sigZs', type=float, dest='sigZs', default=1.5, help='sigma (marginal variance) of the GP covariance for Zs')
    p.add_argument('-sigdtsMo', type=float, dest='sigdtsMo', default=1.0, help='sigma (marginal variance) of the GP covariance for deltas of model output')
    p.add_argument('-gpdtsMo', dest='gpdtsMo', default=True,  type=lambda x: (str(x).lower() == 'true'), \
        help='flag for whether deltas of model output is a GP')
    p.add_argument('-useGradsFlag', dest='useGradsFlag', default=True,  type=lambda x: (str(x).lower() == 'true'), \
        help='flag for whether to use analytically computed gradients to do optimisation')
    p.add_argument('-useSimData', dest='useSimData', default=False,  type=lambda x: (str(x).lower() == 'true'), \
        help='flag for whether to use simulated data')
    p.add_argument('-useCluster', dest='useCluster', default=True,  type=lambda x: (str(x).lower() == 'true'), \
        help='flag for whether to run code on Uni cluster')
    p.add_argument('-oneRepPerJob', dest='oneRepPerJob', default=False,  type=lambda x: (str(x).lower() == 'true'), \
        help='flag for whether to run one repeat for each job on cluster')
    p.add_argument('-folder', type=int, dest='folder', default=0, help='The folder index')
    p.add_argument('-cntry', type=str, dest='cntry', default=None, help='Country of the geo data used')
    p.add_argument('-usecntryFlag', dest='usecntryFlag', default=True,  type=lambda x: (str(x).lower() == 'true'), \
        help='flag for whether to use data for a specific country')
    p.add_argument('-numObs', type=int, dest='numObs', default=200, help='Number of observations used in modelling')
    p.add_argument('-numMo', type=int, dest='numMo', default=150, help='Number of model outputs used in modelling')
    p.add_argument('-crossValFlag', dest='crossValFlag', default=False,  type=lambda x: (str(x).lower() == 'true'), \
        help='whether to validate the model using cross validation')
    p.add_argument('-idxFold', type=int, dest='idxFold', default=9, help='the index for the fold for cross validation')
    args = p.parse_args()
    if args.output is None: args.output = os.getcwd()
    if args.useSimData:  
        output_folder = args.output + '/SEED_' + str(args.SEED) + '_withPrior_' + str(args.withPrior) + '_fixMb_' + str(args.fixMb) + '_onlyOptimCovPar_' + str(args.onlyOptimCovPar) + \
        '_poly_deg_' + str(args.poly_deg) + '_lsZs_' + str(args.lsZs) + '_lsdtsMo_' + str(args.lsdtsMo) \
        + '_sigZs_' + str(args.sigZs) + '_sigdtsMo_' + str(args.sigdtsMo) + '_gpdtsMo_' + str(args.gpdtsMo) + \
        '_useGradsFlag_' + str(args.useGradsFlag) + '_repeat' + str(args.repeat)
    else: 
        if args.usecntryFlag:
            if args.oneRepPerJob:
                output_folder = args.output + '/cntry_' + str(args.cntry) + '_numObs_' + str(args.numObs) + '_numMo_' + str(args.numMo) + \
                '/folder_' + str(args.folder) + '/SEED_' + str(args.SEED) + '_withPrior_' + str(args.withPrior) + '_poly_deg_' + str(args.poly_deg) + \
                '_repeat' + str(args.repeat) 
            else:
                output_folder = args.output + '/cntry_' + str(args.cntry) + '_numObs_' + str(args.numObs) + '_numMo_' + str(args.numMo) + \
                '/SEED_' + str(args.SEED) + '_withPrior_' + str(args.withPrior) + '_poly_deg_' + str(args.poly_deg) + '_repeat' + str(args.repeat) 
        else:
            if args.oneRepPerJob:
                output_folder = args.output + '/numObs_' + str(args.numObs) + '_numMo_' + str(args.numMo) + '/folder_' + str(args.folder) + \
                '/SEED_' + str(args.SEED) + '_withPrior_' + str(args.withPrior) + '_poly_deg_' + str(args.poly_deg) + '_repeat' + str(args.repeat)
            else:
                output_folder = args.output + '/numObs_' + str(args.numObs) + '_numMo_' + str(args.numMo) + \
                '/SEED_' + str(args.SEED) + '_withPrior_' + str(args.withPrior) + '_poly_deg_' + str(args.poly_deg) + '_repeat' + str(args.repeat)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_folder += '/'
    print 'Output: ' + output_folder

    # check_grads()
    # test_fun(args.useCluster, args.SEED)
    # exit(-1)

    start = default_timer()
    np.random.seed(args.SEED)

    # X_hatZs, y_hatZs, X_tildZs, y_tildZs, areal_tildZs = simData.sim_hatTildZs_With_Plots(SEED = args.SEED, phi_Z_s = [args.lsZs], gp_deltas_modelOut = args.gpdtsMo, \
    #     phi_deltas_of_modelOut = [args.lsdtsMo], sigma_Zs = args.sigZs, sigma_deltas_of_modelOut = args.sigdtsMo)
    input_folder = 'sampRealData/FPstart2016020612_' + str(args.cntry) + '_numObs_' + str(args.numObs) + '_numMo_' + str(args.numMo) \
    + '/seed' + str(args.SEED) + '/'
    if args.useCluster:
        if args.usecntryFlag:
            if args.crossValFlag:
                input_folder = input_folder + '/fold_' + str(args.idxFold) + '/'
                X_train_in = open(input_folder + 'X_train.pkl', 'rb')
                X_train = pickle.load(X_train_in) 
                y_train_in = open(input_folder + 'y_train.pkl', 'rb')
                y_train = pickle.load(y_train_in) 
                
                X_test_in = open(input_folder + 'X_test.pkl', 'rb')
                X_test = pickle.load(X_test_in)
                y_test_in = open(input_folder + 'y_test.pkl', 'rb')
                y_test = pickle.load(y_test_in) 

                X_tildZs_in = open(input_folder + 'X_tildZs.pkl', 'rb')
                X_tildZs = pickle.load(X_tildZs_in)
                y_tildZs_in = open(input_folder + 'y_tildZs.pkl', 'rb')
                y_tildZs = pickle.load(y_tildZs_in)
            else:
                X_hatZs_in = open(input_folder + 'X_hatZs.pkl', 'rb')
                X_hatZs = pickle.load(X_hatZs_in) 
             
                y_hatZs_in = open(input_folder + 'y_hatZs.pkl', 'rb')
                y_hatZs = pickle.load(y_hatZs_in) 

                X_tildZs_in = open(input_folder + 'X_tildZs.pkl', 'rb')
                X_tildZs = pickle.load(X_tildZs_in) 
              
                y_tildZs_in = open(input_folder + 'y_tildZs.pkl', 'rb')
                y_tildZs = pickle.load(y_tildZs_in)

                areal_hatZs_in = open(input_folder + 'areal_hatZs.pkl', 'rb')
                areal_hatZs = pickle.load(areal_hatZs_in)
    else:
        X_hatZs, y_hatZs, X_tildZs, y_tildZs, areal_hatZs = ldNetCDF.loadNetCdf(SEED = args.SEED)

    # X_hatZs, y_hatZs, X_tildZs, y_tildZs, areal_tildZs = read_Sim_Data()
    # hmc = HMC_estimator(X_hatZs, y_hatZs, X_tildZs, y_tildZs, areal_tildZs)
    # res = hmc.estimate()
    # print 'hmc estimated res is ' + str(res)
    # end = default_timer()

    # print 'running time for HMC sampler is '  + str(end - start) + ' seconds'
    if args.crossValFlag:
        if args.fixMb:
            res = Gibbs_sampler(X_train, y_train, X_tildZs, y_tildZs, areal_hatZs)
            mu, cov = res.sampler(args.withPrior, args.onlyOptimCovPar, args.gpdtsMo, args.useGradsFlag, args.repeat, args.SEED)
        else:
            mu, cov = optimise(X_train, y_train, X_tildZs, y_tildZs, args.withPrior, args.gpdtsMo, args.useGradsFlag, args.repeat, args.SEED)
            # mu = optimise(X_train, y_train, X_tildZs, y_tildZs, args.withPrior, args.gpdtsMo, args.useGradsFlag, args.repeat, args.SEED) # TNC optimization
    else:
        if args.fixMb:
            res = Gibbs_sampler(X_hatZs, y_hatZs, X_tildZs, y_tildZs, areal_hatZs)
            mu, cov = res.sampler(args.withPrior, args.onlyOptimCovPar, args.gpdtsMo, args.useGradsFlag, args.repeat, args.SEED)
        else:
            mu, cov = optimise(X_hatZs, y_hatZs, X_tildZs, y_tildZs, args.withPrior, args.gpdtsMo, args.useGradsFlag, args.repeat, args.SEED)

    end = default_timer()
    print 'running time for optimisation with fixMb is ' + str(args.fixMb) + str(end - start) + ' seconds'

    # computing the 95% confidence intervals  for each parameters

    cov_pars = np.exp(np.array(mu[:-4]))
    bias_pars = np.array(mu[-4:])
    pars = np.concatenate((cov_pars, bias_pars))
    pars = np.round(pars,1)
    print 'estimated pars rounded to one decimal point :' + str(pars)

    tmp = np.diag(np.array(cov))
    variance_log_covPars = tmp[:-4]
    print 'variance_log_covPars is ' + str(variance_log_covPars)
    variance_biasPars = tmp[-4:]
    print 'variance_biasPars is ' + str(variance_biasPars)

    upper_interv_covPars = np.exp(mu[:-4] + 2 * np.sqrt(variance_log_covPars))
    lower_interv_covPars = np.exp(mu[:-4] - 2 * np.sqrt(variance_log_covPars))
    upper_interv_biasPars = bias_pars + 2 * np.sqrt(variance_biasPars)
    lower_interv_biasPars = bias_pars - 2 * np.sqrt(variance_biasPars)
    upper_interval = np.concatenate((upper_interv_covPars, upper_interv_biasPars))
    lower_interval = np.concatenate((lower_interv_covPars, lower_interv_biasPars))
    print 'upper_interval is ' + str(upper_interval)
    print 'lower_interval is ' + str(lower_interval)

    upper_interval_rounded = np.round(upper_interval, 1)
    lower_interval_rounded = np.round(lower_interval, 1)
    print 'rounded upper_interval is ' + str(upper_interval_rounded)
    print 'rounded lower_interval is ' + str(lower_interval_rounded)

    if args.useSimData:
        true_bias_pars = np.array([2., 5., 5., 3.])
        if args.gpdtsMo:
            true_gp_pars = np.array([args.sigZs, args.lsZs, 0.1, args.sigdtsMo, args.lsdtsMo])
        else:
            true_gp_pars = np.array([args.sigZs, args.lsZs, 0.1, args.sigdtsMo])

        true_pars = np.concatenate((true_gp_pars, true_bias_pars))

        flag_in_confiInterv = (true_pars >= lower_interval) & (true_pars <= upper_interval)
        print 'status of within the 95 percent confidence interval is ' + str(flag_in_confiInterv)
        count_in_confiInterv  = np.sum(np.array(map(int, flag_in_confiInterv)))
        print 'number of estimated parameters within the 95 percent confidence interval is ' + str(count_in_confiInterv)

        flag_in_confiInterv_r = (true_pars >= lower_interval_rounded) & (true_pars <= upper_interval_rounded)
        print 'status of within the 95 percent confidence interval with rounding is ' + str(flag_in_confiInterv_r)
        count_in_confiInterv_r  = np.sum(np.array(map(int, flag_in_confiInterv_r)))
        print 'number of estimated parameters within the 95 percent confidence interval with rounding is ' + str(count_in_confiInterv_r)

        res = {'mu':mu, 'cov':cov, 'pars':pars,'upper_interval':upper_interval, 'lower_interval':lower_interval, \
        'upper_interval_rounded':upper_interval_rounded, 'lower_interval_rounded':lower_interval_rounded, \
        'count_in_confiInterv':count_in_confiInterv, 'count_in_confiInterv_rounded':count_in_confiInterv_r}

        res_out = open(output_folder  + 'resOptim.pkl', 'wb')
        pickle.dump(res, res_out)
        res_out.close()
    else:
        res = {'mu':mu, 'cov':cov, 'pars':pars,'upper_interval':upper_interval, 'lower_interval':lower_interval, \
    'upper_interval_rounded':upper_interval_rounded, 'lower_interval_rounded':lower_interval_rounded}
        res_out = open(output_folder  + 'resOptim.pkl', 'wb')
        pickle.dump(res, res_out)
        res_out.close()
        if args.crossValFlag:
            predic_accuracy = gpGaussLikeFuns.predic_gpRegression(mu, X_train, y_train, X_test, y_test, X_tildZs, y_tildZs, args.crossValFlag)
            print 'predic_accuracy for seed ' + str(args.SEED) + ' fold ' + str(args.idxFold) + ' is ' + '{:.1%}'.format(predic_accuracy)
        else:
            X_train = X_hatZs[:-50, :]
            X_test = X_hatZs[-50:, :]
            y_train = y_hatZs[:-50]
            y_test = y_hatZs[-50:]
            predic_accuracy = gpGaussLikeFuns.predic_gpRegression(mu, X_train, y_train, X_test, y_test, X_tildZs, y_tildZs, args.crossValFlag, args.SEED, args.numMo)
            print 'predic_accuracy for seed ' + str(args.SEED)  + ' is ' + '{:.1%}'.format(predic_accuracy)
        
    # start = default_timer()
    # tmp = MH_estimator(X_hatZs, y_hatZs, X_tildZs, y_tildZs)
    # size = 1200
    # # proposal ='fullApproxCov'
    # mod, mu, std = tmp.estimate(size=size)
    # end = default_timer()
    # print 'running time for MH estimation of size ' + str(size) + ' is ' + str(end - start) + ' seconds'
    # print 'estimated mod of parameters is ' + str(mod)
    # print 'estimated mean of parameters is ' + str(mu)
    # print 'estimated std of parameters is ' + str(std)

    # theta = np.array([0., 0., np.log(.1), 1.])
    # log_obsZs_giv_par(theta, X_hatZs, y_hatZs, X_tildZs, y_tildZs)






 
    
    
    

    
