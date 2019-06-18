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

    print((X_hatZs.shape, y_hatZs.shape, X_tildZs.shape, y_tildZs.shape))
    

   
    theta = np.array([0.51390184, -2.30507058, -2.37142105, -0.57473654, -1.76136598,  1.94541264, 5.56365135, 5.26520738, 2.42106564])

    _, grads = log_obsZs_giv_par_with_grad(theta, X_hatZs, y_hatZs, X_tildZs, y_tildZs, gp_deltas_modelOut = True, withPrior= False, \
    a_bias_poly_deg = 2, rbf = True, OMEGA = 1e-6)
    print(grads)

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
    print(res)
    log_pos, _ = log_obsZs_giv_par_with_grad(initial_theta, X_hatZs, y_hatZs, X_tildZs, y_tildZs, gp_deltas_modelOut = True)
    print(log_pos)
    
    exit(-1)
    covAreas, grad_C_tildZs = cov_areal(X_tildZs, np.log(1.5), [np.log(0.1)], 3., np.log(1.0), gp_deltas_modelOut=True, log_phi_deltas_of_modelOut = [np.log(0.2)],  \
     areal_res=20, OMEGA = 1e-6)
    print(len(grad_C_tildZs))
    print(grad_C_tildZs[0].shape)
    avg_pointAreal_lower, avg_pointAreal_upper, grad_C_pointAreal_lower, grad_C_pointAreal_upper = \
    point_areal(X_hatZs, X_tildZs, np.log(1.5), [np.log(0.1)], 2., pointArealFlag = True)
    print(' shape of upper point areal is ' + str (avg_pointAreal_upper.shape))
    print(avg_pointAreal_lower.shape)
    print(len(grad_C_pointAreal_upper))
    print(grad_C_pointAreal_upper[0].shape)

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

def optim_fun(X_hatZs, y_hatZs, X_tildZs, y_tildZs, gpdtsMo, useGradsFlag, withPrior, num_par, method, bounds, repeat, seed, numMo):
    res = []
    count1 = 0
    LBFGSB_status = False
    while count1 != repeat:
        try:#find one intial value for which the optimisation works
            if gpdtsMo:
                initial_theta=np.concatenate((np.log(np.random.gamma(1.2, 3.5, 1)), np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), \
                np.log(np.random.gamma(1.2, 1./0.6, 1)), np.log(np.random.gamma(1.2, 3.5, 1)), \
                np.log(np.random.gamma(1., np.sqrt(num_par), num_par)),  np.zeros(4)), axis=0)     
            else:
                initial_theta=np.concatenate((np.log(np.random.gamma(1.2, 5., 1)), np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), \
                np.log(np.random.gamma(1.2, 1./0.6, 1)), np.log(np.random.gamma(1.2, 5., 1)), np.zeros(4)), axis=0)
            print('initial theta in optim_fun :' + str(initial_theta))
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
            print('The ' + str(count1) + ' repeat of optimisation in optim_fun')
        except:
            continue
        if tmp_res['fun'] is not None: # if log pos at optimisation is not None, record the resutls, else, redo the otpmisation
            temp_res = [tmp_res['x'], np.copy(tmp_res['fun']), np.copy(tmp_res['hess_inv'].todense()), np.copy(tmp_res['success']), \
            np.copy(tmp_res['message']), np.copy(tmp_res['nit'])]
            res.append(temp_res)
            print('theta from the ' + str(count1) + ' repeat of optimisation with LBFGSB in optim_fun is ' + str(tmp_res['x']))
            logPosat_resOptim, grads_at_resOptim = log_obsZs_giv_par_with_grad(tmp_res['x'], X_hatZs, y_hatZs, X_tildZs, y_tildZs, \
                gp_deltas_modelOut = gpdtsMo,  withPrior= withPrior)
            print('grads at theta from the ' + str(count1) + ' repeat of optimisation with LBFGSB in optim_fun is ' + str(grads_at_resOptim))
            flag_grads_equal_zero = np.round(grads_at_resOptim, 2) == 0.
            # if gradients from the first optimisation is zero, break out of the loop
            if np.sum(flag_grads_equal_zero) == len(flag_grads_equal_zero):
                LBFGSB_status = True
                print('LBFGSB optmisation converged successfully at the '+ str(count1) + ' repeat of optimisation in optim_fun.')
                res = np.array(res)
                res = res.T
                # print 'minus_log_like for repeat ' + str(count1)+ ' with LBFGSB in optim_fun is ' + str(res[1, :])
                # print 'parameters after optimisation withPrior is ' + str(withPrior) + \
                # ' & gpdtsMo is ' + str(gpdtsMo) + ' with LBFGSB in optim_fun :'  + \
                # str([np.exp(np.array(tmp_res['x'])[:-4]), np.array(tmp_res['x'])[-4:]])
                # print 'covariance of pars after optimisation withPrior is ' + str(withPrior) + \
                #  ' & gpdtsMo is ' + str(gpdtsMo) + ' with LBFGSB in optim_fun :'  + str(np.array(tmp_res['hess_inv'].todense()))
                # print 'Optim status withPrior is ' + str(withPrior) + \
                # ' & gpdtsMo is ' + str(gpdtsMo) + ' with LBFGSB in optim_fun :'  + str(np.array(tmp_res['success']))
                # print 'Optim message withPrior is ' + str(withPrior) + \
                # ' & gpdtsMo is ' + str(gpdtsMo) + ' with LBFGSB in optim_fun :'  + str(np.array(tmp_res['message']))
                # print 'Optim nit withPrior is ' + str(withPrior) + \
                #  ' & gpdtsMo is ' + str(gpdtsMo) + ' with LBFGSB in optim_fun :'  + str(np.array(tmp_res['nit']))
                break
            else:#if gradients from the LBFGSB optimisation is NOT zero, do another BFGSB optimisation
                if count1 == (repeat -1):#if gradients from the all LBFGSB optimisation is NOT zero, do ONE ROUND OF  NON constraint optimisation AT THE LAST STEP
                    file = 'logs/unsuccessAll' + str(repeat)+ 'repSeed' + str(seed)+ str(method)  + '_numMo' + str(numMo) 
                    f1 = open(file, 'wb')
                    f1.close()

                    print('initial theta from LBFGSB optimisation for BFGS in optim_fun is ' + str(tmp_res['x']))
                    tmp_res = minimize(fun=minus_log_obsZs_giv_par_with_grad, 
                                   x0=tmp_res['x'], method='BFGS',
                                   jac=True,
                                   args=(X_hatZs, y_hatZs, X_tildZs, y_tildZs, gpdtsMo, withPrior),
                                   options={'maxiter': 2000, 'disp': False})
                    print('theta from the ' + str(count1) + ' repeat of optimisation with BFGS in optim_fun is ' + str(tmp_res['x']))
                    logPosat_resOptim, grads_at_resOptim = log_obsZs_giv_par_with_grad(tmp_res['x'], X_hatZs, y_hatZs, X_tildZs, y_tildZs, \
                        gp_deltas_modelOut = gpdtsMo,  withPrior= withPrior)
                    print('grads at theta from the ' + str(count1) + ' repeat of optimisation with BFGS in optim_fun is ' + str(grads_at_resOptim))
                    flag_grads_equal_zero = np.round(grads_at_resOptim, 2) == 0.
                    # if gradients from the BFGS optimisation is zero, break out of the loop

                    tmp = np.diag(np.array(tmp_res['hess_inv']))
                    variance_log_covPars = tmp[:-4]
                    if np.sum(flag_grads_equal_zero) == len(flag_grads_equal_zero):
                        print('BFGS optmisation converged successfully at the '+ str(count1) + ' repeat of optimisation in optim_fun.')
                        print('minus_log_like for repeat ' + str(count1)+ ' with BFGS in optim_fun is ' + str(tmp_res['fun']))
                        print('parameters after optimisation with BFGS in optim_fun :'  + \
                        str([np.exp(np.array(tmp_res['x'])[:-4]), np.array(tmp_res['x'])[-4:]]))

                 #        print 'covariance of pars after optimisation withPrior is ' + str(withPrior) + \
                 # ' & gpdtsMo is ' + str(gpdtsMo) + ' with BFGS in optim_fun :'  + str(np.array(tmp_res['hess_inv']))
                 #        print 'Optim status withPrior is ' + str(withPrior) + \
                 #        ' & gpdtsMo is ' + str(gpdtsMo) + ' with BFGS in optim_fun :'  + str(np.array(tmp_res['success']))
                 #        print 'Optim message withPrior is ' + str(withPrior) + \
                 #        ' & gpdtsMo is ' + str(gpdtsMo) + ' with BFGS in optim_fun :'  + str(np.array(tmp_res['message']))
                 #        print 'Optim nit withPrior is ' + str(withPrior) + \
                 #         ' & gpdtsMo is ' + str(gpdtsMo) + ' with BFGS in optim_fun :'  + str(np.array(tmp_res['nit']))
                        break
                    else:
                        count1 +=1
                else:  
                    count1 += 1        
        else:# if log pos at optimisation is not None, record the resutls, else, redo the otpmisation
            continue 
    if LBFGSB_status:
        return [np.array(tmp_res['x']), tmp_res['fun'], np.array(tmp_res['hess_inv'].todense()), tmp_res['success'], tmp_res['message']]
    else:
        return [np.array(tmp_res['x']), tmp_res['fun'], np.array(tmp_res['hess_inv']), tmp_res['success'], tmp_res['message']]

def optim_RndStart1(X_hatZs, y_hatZs, X_tildZs, y_tildZs, gpdtsMo, useGradsFlag, withPrior, num_par, method, bounds, repeat, seed, numMo, round):
    print('Starting the ' + str(round) + ' round of optimisation in optim_RndStart1')
    count = 0
    max_repeat = 3
    res = []
    while count != repeat:
        try:#find one intial value for which the optimisation works
            if gpdtsMo:
                initial_theta=np.concatenate((np.log(np.random.gamma(1.2, 3.5, 1)), np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), \
                np.log(np.random.gamma(1.2, 1./0.6, 1)), np.log(np.random.gamma(1.2, 3.5, 1)), \
                np.log(np.random.gamma(1., np.sqrt(num_par), num_par)),  np.zeros(4)), axis=0)     
            else:
                initial_theta=np.concatenate((np.log(np.random.gamma(1.2, 5., 1)), np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), \
                np.log(np.random.gamma(1.2, 1./0.6, 1)), np.log(np.random.gamma(1.2, 5., 1)), np.zeros(4)), axis=0)
            print('initial theta in optim_RndStart :' + str(initial_theta))
            if useGradsFlag:
                tmp_res = minimize(fun=minus_log_obsZs_giv_par_with_grad, 
                                   x0=initial_theta, method='BFGS',
                                   jac=True,
                                   args=(X_hatZs, y_hatZs, X_tildZs, y_tildZs, gpdtsMo, withPrior),
                                   options={'maxiter': 2000, 'disp': False})
            else:
                tmp_res = minimize(fun=minus_log_obsZs_giv_par, 
                               x0=initial_theta, method=method,
                               jac=False, bounds = bounds,
                               args=(X_hatZs, y_hatZs, X_tildZs, y_tildZs, gpdtsMo, withPrior),
                               options={'maxiter': 2000, 'disp': False})
            print('The ' + str(count) + ' repeat of optimisation in optim_RndStart')
        except:
            continue
        if tmp_res['fun'] is not None: # if log pos at optimisation is not None, record the resutls, else, redo the otpmisation
            temp_res = [tmp_res['x'], np.copy(tmp_res['fun']), np.copy(tmp_res['hess_inv']), np.copy(tmp_res['success']), \
            np.copy(tmp_res['message']), np.copy(tmp_res['nit'])]
            res.append(temp_res)
            print('theta from the ' + str(count) + ' repeat of optimisation with BFGS is ' + str(tmp_res['x']))
            logPosat_resOptim, grads_at_resOptim = log_obsZs_giv_par_with_grad(tmp_res['x'], X_hatZs, y_hatZs, X_tildZs, y_tildZs, \
                gp_deltas_modelOut = gpdtsMo,  withPrior= withPrior)
            print('grads at theta from the ' + str(count) + ' repeat of optimisation with BFGS is ' + str(grads_at_resOptim))
            flag_grads_equal_zero = np.round(grads_at_resOptim, 2) == 0.
            # if gradients from the BFGS optimisation is zero, break out of the loop
            tmp = np.diag(np.array(tmp_res['hess_inv']))
            variance_log_covPars = tmp[:-4]
            print('np.max(np.abs(variance_log_covPars)) in optim_RndStart -firstPart is ' + str(np.max(np.abs(variance_log_covPars))))
            if np.sum(flag_grads_equal_zero) == len(flag_grads_equal_zero) :
                print('BFGS optmisation converged successfully at the '+ str(count) + ' round of optimisation.')
                print('minus_log_like for repeat ' + str(count)+ ' with BFGS is ' + str(tmp_res['fun']))
                print('parameters after optimisation with BFGS :'  + str([np.exp(np.array(tmp_res['x'])[:-4]), np.array(tmp_res['x'])[-4:]]))
                print('covariance of pars after optimisation with BFGS :'  + str(np.array(tmp_res['hess_inv'])))
                print('Optim status withPrior is ' + str(withPrior) + \
                ' & gpdtsMo is ' + str(gpdtsMo) + ' with BFGS :'  + str(np.array(tmp_res['success'])))
                print('Optim message withPrior with BFGS :'  + str(np.array(tmp_res['message'])))
              
                count += 1        
        else:# if log pos at optimisation is not None, record the resutls, else, redo the otpmisation
            continue 
 
    res = np.array(res)
    res = res.T
    print('minus_log_like for repeat ' + str(repeat) + ' is ' + str(res[1, :]))
    i = np.argmin(res[1,:])
    print('parameters after optimisation with BFGS is ' + str([np.exp(np.array(res[0, :][i])[:-4]), np.array(res[0, :][i])[-4:]]))
    
    return [np.array(res[0, :][i]), np.array(res[1, :][i]), np.array(res[2, :][i]), np.array(res[3, :][i]), np.array(res[4, :][i])]

def optim_RndStart(X_hatZs, y_hatZs, X_tildZs, y_tildZs, gpdtsMo, useGradsFlag, withPrior, num_par, method, bounds, repeat, seed, numMo, round):
    print('Starting the ' + str(round) + ' round of optimisation in optim_RndStart')
    count = 0
    LBFGSB_status = False
    repeat_optim_status = False
    max_repeat = 3
    while count != repeat:
        try:#find one intial value for which the optimisation works
            if gpdtsMo:
                initial_theta=np.concatenate((np.log(np.random.gamma(1.2, 3.5, 1)), np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), \
                np.log(np.random.gamma(1.2, 1./0.6, 1)), np.log(np.random.gamma(1.2, 3.5, 1)), \
                np.log(np.random.gamma(1., np.sqrt(num_par), num_par)),  np.zeros(4)), axis=0)     
            else:
                initial_theta=np.concatenate((np.log(np.random.gamma(1.2, 5., 1)), np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), \
                np.log(np.random.gamma(1.2, 1./0.6, 1)), np.log(np.random.gamma(1.2, 5., 1)), np.zeros(4)), axis=0)
            print('initial theta in optim_RndStart :' + str(initial_theta))
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
            print('The ' + str(count) + ' repeat of optimisation in optim_RndStart')
        except:
            continue
        if tmp_res['fun'] is not None: # if log pos at optimisation is not None, record the resutls, else, redo the otpmisation
            print('theta from the ' + str(count) + ' repeat of optimisation with LBFGSB is ' + str(tmp_res['x']))
            logPosat_resOptim, grads_at_resOptim = log_obsZs_giv_par_with_grad(tmp_res['x'], X_hatZs, y_hatZs, X_tildZs, y_tildZs, \
                gp_deltas_modelOut = gpdtsMo,  withPrior= withPrior)
            print('grads at theta from the ' + str(count) + ' repeat of optimisation with LBFGSB is ' + str(grads_at_resOptim))
            flag_grads_equal_zero = np.round(grads_at_resOptim, 2) == 0.
            # if gradients from the first optimisation is zero, break out of the loop
            if np.sum(flag_grads_equal_zero) == len(flag_grads_equal_zero):
                LBFGSB_status = True
                print('LBFGSB optmisation converged successfully at the '+ str(count) + ' repeat of optimisation.')
        
                print('parameters after optimisation withPrior  with LBFGSB :'  + str([np.exp(np.array(tmp_res['x'])[:-4]), np.array(tmp_res['x'])[-4:]]))
                # print 'covariance of pars after optimisation withPrior is ' + str(withPrior) + \
                #  ' & gpdtsMo is ' + str(gpdtsMo) + ' with LBFGSB :'  + str(np.array(tmp_res['hess_inv'].todense()))
                # print 'Optim status withPrior is ' + str(withPrior) + \
                # ' & gpdtsMo is ' + str(gpdtsMo) + ' with LBFGSB :'  + str(np.array(tmp_res['success']))
                # print 'Optim message withPrior is ' + str(withPrior) + \
                # ' & gpdtsMo is ' + str(gpdtsMo) + ' with LBFGSB :'  + str(np.array(tmp_res['message']))
                # print 'Optim nit withPrior is ' + str(withPrior) + \
                #  ' & gpdtsMo is ' + str(gpdtsMo) + ' with LBFGSB :'  + str(np.array(tmp_res['nit']))
                break
            else:#if gradients from the LBFGSB optimisation is NOT zero, do another BFGSB optimisation
                if count == (repeat -1):#if gradients from the all LBFGSB optimisation is NOT zero, do ONE ROUND OF  NON constraint optimisation AT THE LAST STEP
                    file = 'logs/unsuccessAll' + str(repeat)+ 'Seed' + str(seed)+ str(method)
                    f1 = open(file, 'wb')
                    f1.close()

                    print('initial theta from LBFGSB optimisation for BFGS is ' + str(tmp_res['x']))
                    tmp_res = minimize(fun=minus_log_obsZs_giv_par_with_grad, 
                                   x0=tmp_res['x'], method='BFGS',
                                   jac=True,
                                   args=(X_hatZs, y_hatZs, X_tildZs, y_tildZs, gpdtsMo, withPrior),
                                   options={'maxiter': 2000, 'disp': False})
                    print('theta from the ' + str(count) + ' repeat of optimisation with BFGS is ' + str(tmp_res['x']))
                    logPosat_resOptim, grads_at_resOptim = log_obsZs_giv_par_with_grad(tmp_res['x'], X_hatZs, y_hatZs, X_tildZs, y_tildZs, \
                        gp_deltas_modelOut = gpdtsMo,  withPrior= withPrior)
                    print('grads at theta from the ' + str(count) + ' repeat of optimisation with BFGS is ' + str(grads_at_resOptim))
                    flag_grads_equal_zero = np.round(grads_at_resOptim, 2) == 0.
                    # if gradients from the BFGS optimisation is zero, break out of the loop
                    tmp = np.diag(np.array(tmp_res['hess_inv']))
                    variance_log_covPars = tmp[:-4]
                    print('np.max(np.abs(variance_log_covPars)) in optim_RndStart -firstPart is ' + str(np.max(np.abs(variance_log_covPars))))
                    if np.sum(flag_grads_equal_zero) == len(flag_grads_equal_zero) :
                        print('BFGS optmisation converged successfully at the '+ str(count) + ' round of optimisation.')
                        print('minus_log_like for repeat ' + str(count)+ ' with BFGS is ' + str(tmp_res['fun']))
                        print('parameters after optimisation with BFGS :'  + str([np.exp(np.array(tmp_res['x'])[:-4]), np.array(tmp_res['x'])[-4:]]))
                        print('covariance of pars after optimisation with BFGS :'  + str(np.array(tmp_res['hess_inv'])))
                        print('Optim status withPrior is ' + str(withPrior) + \
                        ' & gpdtsMo is ' + str(gpdtsMo) + ' with BFGS :'  + str(np.array(tmp_res['success'])))
                        print('Optim message withPrior with BFGS :'  + str(np.array(tmp_res['message'])))
                        break
                    else:
                        count += 1
                        repeat_another = repeat
                        print('repeat is ' + str(repeat))
                        print('count here is ' + str(count))
                        if repeat_another == max_repeat:
                            break
                        while repeat_another != max_repeat:
                            repeat_another += 1
                            print('repeat_another now is ' + str(repeat_another))
                            mu, log_like, cov, status, message = optim_fun(X_hatZs, y_hatZs, X_tildZs, y_tildZs, gpdtsMo, useGradsFlag, withPrior, num_par, method, bounds, repeat_another,seed, numMo)
                            ogPosat_resOptim, grads_at_resOptim = log_obsZs_giv_par_with_grad(mu, X_hatZs, y_hatZs, X_tildZs, y_tildZs, \
                        gp_deltas_modelOut = gpdtsMo,  withPrior= withPrior)
                            print('grads at theta from the repeat ' + str(repeat_another) + ' optimisation is ' + str(grads_at_resOptim))
                            flag_grads_equal_zero = np.round(grads_at_resOptim, 2) == 0.
                            tmp = np.diag(cov)
                            variance_log_covPars = tmp[:-4]
                            print('np.max(np.abs(variance_log_covPars)) in optim_RndStart -secondPart is ' + str(np.max(np.abs(variance_log_covPars))))
                            if np.sum(flag_grads_equal_zero) == len(flag_grads_equal_zero):
                                repeat_optim_status = True
                                break
                else:    
                    count += 1        
        else:# if log pos at optimisation is not None, record the resutls, else, redo the otpmisation
            continue 
    if LBFGSB_status:
        return [np.array(tmp_res['x']), tmp_res['fun'], np.array(tmp_res['hess_inv'].todense()), tmp_res['success'], tmp_res['message']]
    elif repeat_optim_status:
        return [mu, log_like, cov, status, message]
    else:
        return [np.array(tmp_res['x']), tmp_res['fun'], np.array(tmp_res['hess_inv']), tmp_res['success'], tmp_res['message']]

def optim_NotRndStart(X_hatZs, y_hatZs, X_tildZs, y_tildZs, gpdtsMo, useGradsFlag, withPrior, num_par, method, bounds, repeat, seed, numMo, round=1):
    print('Starting the ' + str(round) + ' round of optimisation in optim_NotRndStart')
    count = 0
    max_repeat = 2
    input_folder = os.getcwd() + '/dataSimulated/numObs_200_numMo_' + str(numMo - 50) + '/seed' + str(seed) + '/'
    # resOptim_in = open(input_folder + 'resOptim.pkl', 'rb')
    resOptim_in = open(input_folder + 'resOptimSim.pkl', 'rb')
    resOptim = pickle.load(resOptim_in)
    initial_theta = resOptim['mu']
    print('initial theta from previous optimisation  for BFGS is ' + str(initial_theta))
    tmp_res = minimize(fun=minus_log_obsZs_giv_par_with_grad, 
                   x0=initial_theta, method='BFGS',
                   jac=True,
                   args=(X_hatZs, y_hatZs, X_tildZs, y_tildZs, gpdtsMo, withPrior),
                   options={'maxiter': 2000, 'disp': False})
    print('theta from optimisation with BFGS when numMo is ' + str(numMo) + ' is ' + str(tmp_res['x']))
    logPosat_resOptim, grads_at_resOptim = log_obsZs_giv_par_with_grad(tmp_res['x'], X_hatZs, y_hatZs, X_tildZs, y_tildZs, \
        gp_deltas_modelOut = gpdtsMo,  withPrior= withPrior)
    print('grads at theta from optimisation with BFGS when numMo is ' + str(numMo) + ' is ' + str(grads_at_resOptim))
    flag_grads_equal_zero = np.round(grads_at_resOptim, 2) == 0.
    # if gradients from the BFGS optimisation is zero, break out of the loop
    tmp = np.diag(np.array(tmp_res['hess_inv']))
    variance_log_covPars = tmp[:-4]
    print('np.max(np.abs(variance_log_covPars)) in optim_NotRndStart - firstPart is ' + str(np.max(np.abs(variance_log_covPars))))
    if np.sum(flag_grads_equal_zero) == len(flag_grads_equal_zero) :
        print('BFGS optmisation converged successfully when numMo is ' + str(numMo))
        print('minus_log_like with BFGS is when numMo is ' + str(numMo) + ' is ' + str(tmp_res['fun']))
        print('parameters after optimisation withPrior is ' + str(withPrior) + \
        ' & gpdtsMo is ' + str(gpdtsMo) + ' with BFGS :'  + \
        str([np.exp(np.array(tmp_res['x'])[:-4]), np.array(tmp_res['x'])[-4:]]))
        print('covariance of pars after optimisation with BFGS :'  + str(np.array(tmp_res['hess_inv'])))
        print('Optim status with BFGS :'  + str(np.array(tmp_res['success'])))
        print('Optim message with BFGS :'  + str(np.array(tmp_res['message'])))
        return [np.array(tmp_res['x']), tmp_res['fun'], np.array(tmp_res['hess_inv']), tmp_res['success'], tmp_res['message']]
    else:
        file = 'logs/fail_BFGSOptim_Seed' + str(seed) + 'numMo' + str(numMo) 
        f1 = open(file, 'wb')
        f1.close()
        repeat_another = repeat
        print('repeat is ' + str(repeat))
        print('count here is ' + str(count))
        while repeat_another != max_repeat:
            repeat_another += 1
            print('repeat_another now is ' + str(repeat_another))
            mu, log_like, cov, status, message = optim_fun(X_hatZs, y_hatZs, X_tildZs, y_tildZs, gpdtsMo, useGradsFlag, withPrior, num_par, method, bounds, repeat_another,seed, numMo)
            ogPosat_resOptim, grads_at_resOptim = log_obsZs_giv_par_with_grad(mu, X_hatZs, y_hatZs, X_tildZs, y_tildZs, \
        gp_deltas_modelOut = gpdtsMo,  withPrior= withPrior)
            print('grads at theta from the repeat ' + str(repeat_another) + ' optimisation is ' + str(grads_at_resOptim))
            flag_grads_equal_zero = np.round(grads_at_resOptim, 2) == 0.
            tmp = np.diag(cov)
            variance_log_covPars = tmp[:-4]
            print('np.max(np.abs(variance_log_covPars)) in optim_NotRndStart - secondPart is ' + str(np.max(np.abs(variance_log_covPars))))
            if np.sum(flag_grads_equal_zero) == len(flag_grads_equal_zero):
                break
        return [mu, log_like, cov, status, message]

def optimise(X_hatZs, y_hatZs, X_tildZs, y_tildZs, withPrior, gpdtsMo=False, useGradsFlag = False, repeat=3, seed =188, numMo =50, rounds =1, method='L-BFGS-B', rbf=True, OMEGA = 1e-6, \
    bounds = ((-2.3, 4.6), (-2.3, 4.6), (-2.3, 4.6), (-2.3, 4.6), (-2.3, 4.6), (-50, 50), (-50, 50), (-50, 50), (-50, 50))): 
    print('starting optimising when withPrior is ' + str(withPrior) + ' & gpdtsMo is ' + str(gpdtsMo) + \
         '& useGradsFlag is ' + str(useGradsFlag)) 
    if rbf:
        num_par=1
    else:
        num_par=X_hatZs.shape[1]  
    res = []
    if numMo == 50:
        rounds =3
        for round in range(rounds):
            res_tmp = optim_RndStart(X_hatZs, y_hatZs, X_tildZs, y_tildZs, gpdtsMo, useGradsFlag, withPrior, num_par, method, bounds, repeat,seed, numMo, round)
            # print 'res_tmp is ' + str(res_tmp)
            res.append(res_tmp)
        res = np.array(res)
        res = res.T
        print('minus_log_like for ' + str(rounds) + ' rounds is ' + str(res[1, :]))
        i = np.argmin(res[1,:])
        print('log_cov_parameters plus model bias after optimisation  is :'  + str(np.array(res[0, :][i])))
        print('parameters after optimisation withPrior is  :'  + str([np.exp(np.array(res[0, :][i])[:-4]), np.array(res[0, :][i])[-4:]]))
        print('covariance of pars after optimisation  :'  + str(np.array(res[2, :][i])))
        print('Optim status  :'  + str(np.array(res[3, :][i])))
        print('Optim message :'  + str(np.array(res[4, :][i])))
        return [np.array(res[0, :][i]), np.array(res[2, :][i])]
    else:
        for round in range(rounds):
            if round ==0:
                res_tmp = optim_NotRndStart(X_hatZs, y_hatZs, X_tildZs, y_tildZs, gpdtsMo, useGradsFlag, withPrior, num_par, method, bounds, repeat,seed, numMo, round)
                # print 'res_tmp is ' + str(res_tmp)
                res.append(res_tmp)
            else:
                res_tmp = optim_RndStart(X_hatZs, y_hatZs, X_tildZs, y_tildZs, gpdtsMo, useGradsFlag, withPrior, num_par, method, bounds, repeat,seed, numMo, round)
                # print 'res_tmp is ' + str(res_tmp)
                res.append(res_tmp)
        res = np.array(res)
        res = res.T
        print('minus_log_like for ' + str(rounds) + ' rounds is ' + str(res[1, :]))
        i = np.argmin(res[1,:])
        print('log_cov_parameters plus model bias after optimisation  is :'  + str(np.array(res[0, :][i])))
        print('parameters after optimisation withPrior is  :'  + str([np.exp(np.array(res[0, :][i])[:-4]), np.array(res[0, :][i])[-4:]]))
        print('covariance of pars after optimisation  :'  + str(np.array(res[2, :][i])))
        print('Optim status  :'  + str(np.array(res[3, :][i])))
        print('Optim message :'  + str(np.array(res[4, :][i])))
        return [np.array(res[0, :][i]), np.array(res[2, :][i])]
      

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
    input_folder = 'Data/FPstart2016020612_FR_numObs_' + str(128) + '_numMo_' + str(50) \
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
    print('Computed grads are ' + str(grads_computed))
    print('approximated grads are ' + str(grads_approx))
    numerator = np.linalg.norm(grads_approx- grads_computed)                                
    denominator = np.linalg.norm(grads_approx)+np.linalg.norm(grads_computed)               
    difference = numerator/denominator                                          

    if difference > 1e-4:
        print(("\033[93m" + "There is a mistake in computing the gradients! difference = " + str(difference) + "\033[0m"))
    else:
        print(("\033[92m" + "Computing the gradients works perfectly fine! difference = " + str(difference) + "\033[0m"))
    
    return difference      
    
if __name__ == '__main__':
    computeN3Cost.init(0)
    # check_grads()
    # exit(-1)
    p = argparse.ArgumentParser()
    p.add_argument('-SEED', type=int, dest='SEED', default=204, help='The simulation index')
    p.add_argument('-repeat', type=int, dest='repeat', default=2, help='number of repeats in optimisation')
    p.add_argument('-o', type=str, dest='output', default=None, help='Output folder')
    p.add_argument('-withPrior', dest='withPrior', default=False,  type=lambda x: (str(x).lower() == 'true'), help='flag for ML or MAP')
    p.add_argument('-fixMb', dest='fixMb', default=False,  type=lambda x: (str(x).lower() == 'true'), \
        help='flag for fixed model bias parameters from linear regression for intialisation in optimisition')
    p.add_argument('-onlyOptimCovPar', dest='onlyOptimCovPar', default=False,  type=lambda x: (str(x).lower() == 'true'), \
        help='flag for only optimising the cov parameters with fixed model bias from linear regression')
    p.add_argument('-poly_deg', type=int, dest='poly_deg', default=2, help='degree of the polynomial function of the additive model bias')
    p.add_argument('-lsZs', type=float, dest='lsZs', default=0.8, help='lengthscale of the GP covariance for Zs')
    p.add_argument('-lsdtsMo', type=float, dest='lsdtsMo', default=0.2, help='lengthscale of the GP covariance for deltas of model output')
    p.add_argument('-sigZs', type=float, dest='sigZs', default=1.0, help='sigma (marginal variance) of the GP covariance for Zs')
    p.add_argument('-sigdtsMo', type=float, dest='sigdtsMo', default=0.5, help='sigma (marginal variance) of the GP covariance for deltas of model output')
    p.add_argument('-gpdtsMo', dest='gpdtsMo', default=True,  type=lambda x: (str(x).lower() == 'true'), \
        help='flag for whether deltas of model output is a GP')
    p.add_argument('-useGradsFlag', dest='useGradsFlag', default=True,  type=lambda x: (str(x).lower() == 'true'), \
        help='flag for whether to use analytically computed gradients to do optimisation')
    p.add_argument('-useSimData', dest='useSimData', default=True,  type=lambda x: (str(x).lower() == 'true'), \
        help='flag for whether to use simulated data')
    p.add_argument('-useCluster', dest='useCluster', default=True,  type=lambda x: (str(x).lower() == 'true'), \
        help='flag for whether to run code on Uni cluster')
    p.add_argument('-oneRepPerJob', dest='oneRepPerJob', default=False,  type=lambda x: (str(x).lower() == 'true'), \
        help='flag for whether to run one repeat for each job on cluster')
    p.add_argument('-folder', type=int, dest='folder', default=0, help='The folder index')
    p.add_argument('-cntry', type=str, dest='cntry', default='FR', help='Country of the geo data used')
    p.add_argument('-usecntryFlag', dest='usecntryFlag', default=True,  type=lambda x: (str(x).lower() == 'true'), \
        help='flag for whether to use data for a specific country')
    p.add_argument('-numObs', type=int, dest='numObs', default=200, help='Number of observations used in modelling')
    p.add_argument('-numMo', type=int, dest='numMo', default=50, help='Number of model outputs used in modelling')
    p.add_argument('-crossValFlag', dest='crossValFlag', default=False,  type=lambda x: (str(x).lower() == 'true'), \
        help='whether to validate the model using cross validation')
    p.add_argument('-idxFold', type=int, dest='idxFold', default=9, help='the index for the fold for cross validation')
    p.add_argument('-grid', dest='grid', default=False,  type=lambda x: (str(x).lower() == 'true'),  help='flag for whether the predictions are produced for each grid')
    p.add_argument('-predicMo', dest='predicMo', default=False,  type=lambda x: (str(x).lower() == 'true'),  help='flag for whether to predict the value where model outputs are produced')
    args = p.parse_args()
    if args.output is None: args.output = os.getcwd()
    if args.useSimData:  
        output_folder = args.output + '/dataSimulated/numObs_' + str(args.numObs) + '_numMo_' + str(args.numMo) + '/seed' + str(args.SEED) 
    else: 
        if args.usecntryFlag:
            if args.oneRepPerJob:
                output_folder = args.output + '/cntry_' + str(args.cntry) + '_numObs_' + str(args.numObs) + '_numMo_' + str(args.numMo) + \
                '/folder_' + str(args.folder) + '/SEED_' + str(args.SEED) + '_withPrior_' + str(args.withPrior) + '_poly_deg_' + str(args.poly_deg) + \
                '_repeat' + str(args.repeat) 
            else:
                output_folder = args.output + '/Data/FPstart2016020612_FR_numObs_328_numMo_' + str(args.numMo) + '/seed' + str(args.SEED) 
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
    print('Output: ' + output_folder)

    # check_grads()
    # test_fun(args.useCluster, args.SEED)
    # exit(-1)

    start = default_timer()
    np.random.seed(args.SEED)

    if args.useSimData: 
        input_folder = os.getcwd() + '/dataSimulated/numObs_' + str(args.numObs) + '_numMo_' + str(args.numMo) + '/seed' + str(args.SEED) + '/'
    else:
        input_folder = 'Data/FPstart2016020612_' + str(args.cntry) + '_numObs_' + str(args.numObs) + '_numMo_' + str(args.numMo) \
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
                print(X_hatZs.shape)
             
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
        mu, cov = optimise(X_train, y_train, X_tildZs, y_tildZs, args.withPrior, args.gpdtsMo, args.useGradsFlag, args.repeat, args.SEED)
    else:
        mu, cov = optimise(X_hatZs, y_hatZs, X_tildZs, y_tildZs, args.withPrior, args.gpdtsMo, args.useGradsFlag, args.repeat, args.SEED, args.numMo)

    end = default_timer()
    print('running time for optimisation using simData is ' + str(args.useSimData) + ' :' + str(end - start) + ' seconds')

    # computing the 95% confidence intervals  for each parameters

    # cov_pars = np.exp(np.array(mu[:-4]))
    # bias_pars = np.array(mu[-4:])
    # pars = np.concatenate((cov_pars, bias_pars))
    # pars = np.round(pars,1)
    # print 'estimated pars rounded to one decimal point :' + str(pars)

    # tmp = np.diag(np.array(cov))
    # variance_log_covPars = tmp[:-4]
    # print 'variance_log_covPars is ' + str(variance_log_covPars)
    # variance_biasPars = tmp[-4:]
    # print 'variance_biasPars is ' + str(variance_biasPars)

    # upper_interv_covPars = np.exp(mu[:-4] + 2 * np.sqrt(variance_log_covPars))
    # lower_interv_covPars = np.exp(mu[:-4] - 2 * np.sqrt(variance_log_covPars))
    # upper_interv_biasPars = bias_pars + 2 * np.sqrt(variance_biasPars)
    # lower_interv_biasPars = bias_pars - 2 * np.sqrt(variance_biasPars)
    # upper_interval = np.concatenate((upper_interv_covPars, upper_interv_biasPars))
    # lower_interval = np.concatenate((lower_interv_covPars, lower_interv_biasPars))
    # print 'upper_interval is ' + str(upper_interval)
    # print 'lower_interval is ' + str(lower_interval)

    # upper_interval_rounded = np.round(upper_interval, 1)
    # lower_interval_rounded = np.round(lower_interval, 1)
    # print 'rounded upper_interval is ' + str(upper_interval_rounded)
    # print 'rounded lower_interval is ' + str(lower_interval_rounded)

    if args.useSimData:
        # true_bias_pars = np.array([0.6, 2., 3., 15.])
        # if args.gpdtsMo:
        #     true_gp_pars = np.array([args.sigZs, args.lsZs, 0.1, args.sigdtsMo, args.lsdtsMo])
        # else:
        #     true_gp_pars = np.array([args.sigZs, args.lsZs, 0.1, args.sigdtsMo])

        # true_pars = np.concatenate((true_gp_pars, true_bias_pars))

        # flag_in_confiInterv = (true_pars >= lower_interval) & (true_pars <= upper_interval)
        # print 'status of within the 95 percent confidence interval is ' + str(flag_in_confiInterv)
        # count_in_confiInterv  = np.sum(flag_in_confiInterv.astype(int))
        # print 'number of estimated parameters within the 95 percent confidence interval is ' + str(count_in_confiInterv)

        # flag_in_confiInterv_r = (true_pars >= lower_interval_rounded) & (true_pars <= upper_interval_rounded)
        # print 'status of within the 95 percent confidence interval with rounding is ' + str(flag_in_confiInterv_r)
        # count_in_confiInterv_r  = np.sum(flag_in_confiInterv_r.astype(int))
        # print 'number of estimated parameters within the 95 percent confidence interval with rounding is ' + str(count_in_confiInterv_r)

        # res = {'mu':mu, 'cov':cov, 'pars':pars,'upper_interval':upper_interval, 'lower_interval':lower_interval, \
        # 'upper_interval_rounded':upper_interval_rounded, 'lower_interval_rounded':lower_interval_rounded, \
        # 'count_in_confiInterv':count_in_confiInterv, 'count_in_confiInterv_rounded':count_in_confiInterv_r}
        res = {'mu':mu, 'cov':cov}
        res_out = open(output_folder  + 'resOptimSim.pkl', 'wb')
        pickle.dump(res, res_out)
        res_out.close()
        # X_train = X_hatZs[:-50, :]
        # X_test = X_hatZs[-50:, :]
        # y_train = y_hatZs[:-50]
        # y_test = y_hatZs[-50:]
        
        X_train = X_hatZs
        y_train = y_hatZs

        input_folder = os.getcwd() + '/dataSimulated/seed' + str(args.SEED) + '/'
        all_X_Zs_in = open(input_folder + 'all_X_Zs.pickle', 'rb')
        all_X_Zs = pickle.load(all_X_Zs_in) 
        print('shape of all_X_Zs is ' + str(all_X_Zs.shape))

        all_y_Zs_in = open(input_folder + 'all_y_Zs.pickle', 'rb')
        all_y_Zs = pickle.load(all_y_Zs_in) 
        print('shape of all_y_Zs is ' + str(all_y_Zs.shape))

        X_test = all_X_Zs
        y_test = all_y_Zs
        predic_accuracy = gpGaussLikeFuns.predic_gpRegression(mu, X_train, y_train, X_test, y_test, X_tildZs, y_tildZs, args.crossValFlag, \
            args.SEED, args.numMo, args.useSimData, args.grid, args.predicMo)
    else:
    #     res = {'mu':mu, 'cov':cov, 'pars':pars,'upper_interval':upper_interval, 'lower_interval':lower_interval, \
    # 'upper_interval_rounded':upper_interval_rounded, 'lower_interval_rounded':lower_interval_rounded}
        res = {'mu':mu, 'cov':cov}
        res_out = open(output_folder  + 'resOptim.pkl', 'wb')
        pickle.dump(res, res_out)
        res_out.close()
        if args.crossValFlag:
            predic_accuracy = gpGaussLikeFuns.predic_gpRegression(mu, X_train, y_train, X_test, y_test, X_tildZs, y_tildZs, args.crossValFlag)
            print('predic_accuracy for seed ' + str(args.SEED) + ' fold ' + str(args.idxFold) + ' is ' + '{:.1%}'.format(predic_accuracy))
        else:
            X_train = X_hatZs[:-28, :]
            X_test = X_hatZs[-28:, :]
            y_train = y_hatZs[:-28]
            y_test = y_hatZs[-28:]
            predic_accuracy = gpGaussLikeFuns.predic_gpRegression(mu, X_train, y_train, X_test, y_test, X_tildZs, y_tildZs, args.crossValFlag, args.SEED, args.numMo)
            # print 'predic_accuracy for seed ' + str(args.SEED)  + ' is ' + '{:.1%}'.format(predic_accuracy)
        




 
    
    
    

    
