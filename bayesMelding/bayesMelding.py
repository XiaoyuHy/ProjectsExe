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
from sklearn import linear_model

#covariance matrix between areas
def cov_areal(areal_coordinate, sigma, w, b, sigma_deltas_of_modelOut, OMEGA = 1e-6):
    num_areas = len(areal_coordinate)
    num_points_per_area = len(areal_coordinate[0])

    cov_areas = []
    for i in range(num_areas):
        tmp = [avg_cov_two_areal(areal_coordinate[i], areal_coordinate[j], sigma, w, b) for j in range(num_areas)]
        cov_areas.append(tmp)
    covAreas = np.hstack(cov_areas).reshape(num_areas,num_areas) + np.diag(np.repeat(OMEGA + \
        sigma_deltas_of_modelOut * 1./num_points_per_area, num_areas))
    return covAreas

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

def avg_cov_point_areal(x, y, sigma, w, b=2.):
    cov_of_two_vec = cov_mat_xy(x, y, sigma, w)
    avg_cov_point_areal = b * np.mean(cov_of_two_vec, axis=1)
    return avg_cov_point_areal

#average covranice between two areas
def avg_cov_two_areal(x, y, sigma, w, b):
    n_x = x.shape[0]
    n_y = y.shape[0]
    cov_of_two_vec = cov_mat_xy(x, y, sigma, w)
    avg = b**2 * np.float(np.sum(cov_of_two_vec))/(n_x * n_y) 
    return avg

def log_like_normal(w, mu, Sigma):
    l_chol = np.linalg.cholesky(Sigma)
    u = linalg.solve_triangular(l_chol.T, linalg.solve_triangular(l_chol, w, lower=True))
    log_like_normal = -np.sum(np.log(np.diag(l_chol))) - 0.5 * np.dot(w, u) - 0.5 * len(w) * np.log(2*np.pi)
    return log_like_normal

def log_like_gamma(w, shape, rate):
    res = np.sum(shape * w - rate * np.exp(w) + shape * np.log(rate) - np.log(gamma(shape)))
    return res

def fun_a_bias(x, a_bias_coefficients = [5., 5., 0.1]):
    a_bias_coefficients = np.array(a_bias_coefficients)
    a_bias = np.dot(a_bias_coefficients, np.concatenate((x, [1.])))
    return a_bias

def log_obsZs_giv_par(theta, X_hatZs, y_hatZs, X_tildZs, y_tildZs, withPrior= False, gp_deltas_modelOut = False, \
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
        C_tildZs = cov_areal(areal_coordinate = X_tildZs, sigma = np.exp(log_sigma_Zs), w = np.exp(log_phi_Zs), b=b, \
            sigma_deltas_of_modelOut = np.exe(log_sigma_deltas_of_modelOut))
    else:
        C_tildZs = cov_areal(areal_coordinate = X_tildZs, sigma = np.exp(log_sigma_Zs), w = np.exp(log_phi_Zs), b=b, \
            sigma_deltas_of_modelOut = np.exe(log_sigma_deltas_of_modelOut))

    mat[:n_hatZs, :n_hatZs] = C_hatZs
    mat[n_hatZs:n_hatZs + n_tildZs, n_hatZs:n_hatZs + n_tildZs] = C_tildZs
    
    point_areal = np.array([avg_cov_point_areal(X_hatZs, X_tildZs[i], sigma = np.exp(log_sigma_Zs), w = np.exp(log_phi_Zs), b=b) for i in range(n_tildZs)])
    mat[n_hatZs:n_hatZs + n_tildZs, :n_hatZs] = point_areal
    mat[:n_hatZs, n_hatZs:n_hatZs + n_tildZs] = point_areal.T

    mu_hatZs = np.zeros(len(y_hatZs))
    mu_tildZs = np.array([fun_a_bias(np.mean(X_tildZs[i], axis=0), a_bias_coefficients = a_bias_coefficients) for i in range(len(y_tildZs))])
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

def minus_log_obsZs_giv_par(theta, X_hatZs, y_hatZs, X_tildZs, y_tildZs, withPrior= False, gp_deltas_modelOut = False, \
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
        C_tildZs = cov_areal(areal_coordinate = X_tildZs, sigma = np.exp(log_sigma_Zs), w = np.exp(log_phi_Zs), b=b, \
            sigma_deltas_of_modelOut = np.exp(log_sigma_deltas_of_modelOut))
    else:
        C_tildZs = cov_areal(areal_coordinate = X_tildZs, sigma = np.exp(log_sigma_Zs), w = np.exp(log_phi_Zs), b=b, \
            sigma_deltas_of_modelOut = np.exp(log_sigma_deltas_of_modelOut))

    mat[:n_hatZs, :n_hatZs] = C_hatZs
    mat[n_hatZs:n_hatZs + n_tildZs, n_hatZs:n_hatZs + n_tildZs] = C_tildZs
    
    point_areal = np.array([avg_cov_point_areal(X_hatZs, X_tildZs[i], sigma = np.exp(log_sigma_Zs), w = np.exp(log_phi_Zs), b=b) for i in range(n_tildZs)])
    mat[n_hatZs:n_hatZs + n_tildZs, :n_hatZs] = point_areal
    mat[:n_hatZs, n_hatZs:n_hatZs + n_tildZs] = point_areal.T

    mu_hatZs = np.zeros(len(y_hatZs))
    mu_tildZs = np.array([fun_a_bias(np.mean(X_tildZs[i], axis=0), a_bias_coefficients = a_bias_coefficients) for i in range(len(y_tildZs))])
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
        C_tildZs = cov_areal(areal_coordinate = X_tildZs, sigma = np.exp(log_sigma_Zs), w = np.exp(log_phi_Zs), b=b, \
            sigma_deltas_of_modelOut = np.exp(log_sigma_deltas_of_modelOut))
    else:
        C_tildZs = cov_areal(areal_coordinate = X_tildZs, sigma = np.exp(log_sigma_Zs), w = np.exp(log_phi_Zs), b=b, \
            sigma_deltas_of_modelOut = np.exp(log_sigma_deltas_of_modelOut))

    mat[:n_hatZs, :n_hatZs] = C_hatZs
    mat[n_hatZs:n_hatZs + n_tildZs, n_hatZs:n_hatZs + n_tildZs] = C_tildZs
    
    point_areal = np.array([avg_cov_point_areal(X_hatZs, X_tildZs[i], sigma = np.exp(log_sigma_Zs), w = np.exp(log_phi_Zs), b=b) for i in range(n_tildZs)])
    mat[n_hatZs:n_hatZs + n_tildZs, :n_hatZs] = point_areal
    mat[:n_hatZs, n_hatZs:n_hatZs + n_tildZs] = point_areal.T

    mu_hatZs = np.zeros(len(y_hatZs))
    mu_tildZs = np.array([fun_a_bias(np.mean(X_tildZs[i], axis=0), a_bias_coefficients = a_bias_coefficients) for i in range(len(y_tildZs))])
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
        b_Sigma = np.diag([1.]) #choose a 'flat' prior, N(mu = 0, sd = 100)
        a_bias_coefficients_mu = np.zeros(a_bias_poly_deg +1)
        a_bias_coefficients_Sigma = np.diag(np.repeat(1., a_bias_poly_deg + 1))
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

def optimise(X_hatZs, y_hatZs, X_tildZs, y_tildZs, withPrior= False, repeat=5, method='BFGS', rbf=True, OMEGA = 1e-6): 
    print 'starting optimising the parameters when withPrior is ' + str(withPrior)
    if rbf:
        num_par=1
    else:
        num_par=X_hatZs.shape[1]  
    res = []
    count = 0
   
    while count != repeat:
        try:
            initial_theta=np.concatenate((np.log(np.random.gamma(1.2, 5., 1)), np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), \
            np.log(np.random.gamma(1.2, 1./0.6, 1)), np.random.normal(0., 1., 1), np.random.normal(0., 1., 3)), axis=0)
            print 'initial theta when withPrior is ' + str(withPrior) + ' :' + str(initial_theta)
            tmp_res = minimize(fun=minus_log_obsZs_giv_par, 
                           x0=initial_theta, method=method,
                           jac=False,
                           args=(X_hatZs, y_hatZs, X_tildZs, y_tildZs, withPrior),
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
    print 'log_cov_parameters plus model bias after optimisation withPrior is ' + str(withPrior) + ' :'  + str(np.array(res[0, :][i]))
    print 'parameters after optimisation withPrior is ' + str(withPrior) + ' :'  + str([np.exp(np.array(res[0, :][i])[:-4]), np.array(res[0, :][i])[-4:]])
    print 'covariance of pars after optimisation withPrior is ' + str(withPrior) + ' :'  + str(np.array(res[2, :][i]))
    return [np.array(res[0, :][i]), np.array(res[2, :][i])]

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
                    mu, hess_inv= optimize(self.X_hatZs, self.y_hatZs, self.X_tildZs, self.y_tildZs, repeat=5, method='BFGS', rbf=True, OMEGA = 1e-6)
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
    # try to fix the model bias parameters to the ones obtained from linear regression, to check how intialisation can affect the optimisation
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
                    print 'initial theta when withPrior is ' + str(withPrior) + ' & gpdtsMo is ' + str(gpdtsMo) + \
                    ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) + ' :' + str(initial_theta)
                    tmp_res = minimize(fun=minus_log_obsZs_giv_par, 
                                   x0=initial_theta, method=method,
                                   jac=False,
                                   args=(self.X_hatZs, self.y_hatZs, self.X_tildZs, self.y_tildZs, withPrior, gpdtsMo),
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
         
            print 'log_cov_parameters plus model bias after optimisation withPrior is ' + str(withPrior) + \
             ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) + ' & gpdtsMo is ' + str(gpdtsMo) + ' :'  + str(np.array(res[0, :][i]))
            print 'parameters after optimisation withPrior is ' + str(withPrior) + \
            ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) + ' & gpdtsMo is ' + str(gpdtsMo) + ' :'  + \
            str([np.exp(np.array(res[0, :][i])[:-4]), np.array(res[0, :][i])[-4:]])
            print 'covariance of pars after optimisation withPrior is ' + str(withPrior) + \
            ' & onlyOptimCovPar is ' + str(onlyOptimCovPar) + ' & gpdtsMo is ' + str(gpdtsMo) + ' :'  + str(np.array(res[2, :][i]))
          
        return [np.array(res[0, :][i]), np.array(res[2, :][i])]
    def sampler(self, withPrior = False, onlyOptimCovPar = False, gpdtsMo=False, repeat = 5):

        model_bias = self.initial_model_bias() 
        print 'model bias from linear regression is :' + str(model_bias)

        mu, cov = self.optim(withPrior, model_bias, onlyOptimCovPar, gpdtsMo, repeat)
        return [mu, cov] 

def read_Sim_Data():
    #read the samples of hatZs
    X_hatZs = np.array(pd.read_csv('dataSimulated/X_hatZs_res100_a_bias_poly_deg2SEED1_lsZs[0.1]_gpdtsMoFalse_lsdtsMo[0.1].txt', sep=" ", header=None))
    y_hatZs = np.array(pd.read_csv('dataSimulated/y_hatZs_res100_a_bias_poly_deg2SEED1_lsZs[0.1]_gpdtsMoFalse_lsdtsMo[0.1].txt', sep=" ", header=None)).reshape(X_hatZs.shape[0])

    #read the samples of tildZs
    X_tildZs_in = open('dataSimulated/X_tildZs_a_bias_poly_deg2SEED1_lsZs[0.1]_gpdtsMoFalse_lsdtsMo[0.1].pickle', 'rb')
    X_tildZs = pickle.load(X_tildZs_in)

    y_tildZs_in = open('dataSimulated/y_tildZs_a_bias_poly_deg2SEED1_lsZs[0.1]_gpdtsMoFalse_lsdtsMo[0.1].pickle', 'rb')
    y_tildZs = pickle.load(y_tildZs_in)

    areal_tildZs_in = open('dataSimulated/areal_tildZs_a_bias_poly_deg2SEED1_lsZs[0.1]_gpdtsMoFalse_lsdtsMo[0.1].pickle', 'rb')
    areal_tildZs = pickle.load(areal_tildZs_in)

    return[X_hatZs, y_hatZs, X_tildZs, y_tildZs, areal_tildZs]

def test_like_fun():
    X_hatZs, y_hatZs, X_tildZs, y_tildZs, areal_tildZs = read_Sim_Data()
    num_par =1 
    initial_theta=np.concatenate((np.log(np.random.gamma(1.2, 5., 1)), np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), \
                    np.log(np.random.gamma(1.2, 1./0.6, 1)), np.log(np.random.gamma(1.2, 5., 1))), axis=0)
    modelBias = np.array([ 2.00930792,  4.91308723,  5.03209346,  0.12865613])
    test = minus_log_obsZs_giv_par_of_cov(initial_theta, X_hatZs, y_hatZs, X_tildZs, y_tildZs, args.withPrior, modelBias, gp_deltas_modelOut = False, \
    a_bias_poly_deg = 2, rbf = True, OMEGA = 1e-6)
    print test
    return test 


if __name__ == '__main__':
    computeN3Cost.init(0)
    p = argparse.ArgumentParser()
    p.add_argument('-SEED', type=int, dest='SEED', default=0, help='The simulation index')
    p.add_argument('-repeat', type=int, dest='repeat', default=1, help='number of repeats in optimisation')
    p.add_argument('-o', type=str, dest='output', default=None, help='Output folder')
    p.add_argument('-withPrior', dest='withPrior', default=False,  type=lambda x: (str(x).lower() == 'true'), help='flag for ML or MAP')
    p.add_argument('-fixMb', dest='fixMb', default=False,  type=lambda x: (str(x).lower() == 'true'), \
        help='flag for fixed model bias parameters from linear regression for intialisation in optimisition')
    p.add_argument('-onlyOptimCovPar', dest='onlyOptimCovPar', default=False,  type=lambda x: (str(x).lower() == 'true'), \
        help='flag for only optimising the cov parameters with fixed model bias from linear regression')
    p.add_argument('-poly_deg', type=int, dest='poly_deg', default=2, help='degree of the polynomial function of the additive model bias')
    p.add_argument('-lsZs', type=float, dest='lsZs', default=0.1, help='lengthscale of the GP covariance for Zs')
    p.add_argument('-lsdtsMo', type=float, dest='lsdtsMo', default=0.1, help='lengthscale of the GP covariance for deltas of model output')
    p.add_argument('-gpdtsMo', dest='gpdtsMo', default=False,  type=lambda x: (str(x).lower() == 'true'), \
        help='flag for whether deltas of model output is a GP')
    args = p.parse_args()
    if args.output is None: args.output = os.getcwd()
    output_folder = args.output + '/SEED_' + str(args.SEED) + '_withPrior_' + str(args.withPrior) + '_fixMb_' + str(args.fixMb) + '_onlyOptimCovPar_' + str(args.onlyOptimCovPar) + \
    '_poly_deg_' + str(args.poly_deg) + '_lsZs_' + str(args.lsZs) + '_lsdtsMo_' + str(args.lsdtsMo) + '_gpdtsMo_' + str(args.gpdtsMo)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_folder += '/'
    print 'Output: ' + output_folder

    # Test the likelihood function
    # for i in range(10):
    #     test_like_fun()
    # exit(-1)

    start = default_timer()

    X_hatZs, y_hatZs, X_tildZs, y_tildZs, areal_tildZs = simData.sim_hatTildZs_With_Plots(SEED = args.SEED, phi_Z_s = [args.lsZs], gp_deltas_modelOut = args.gpdtsMo, \
        phi_deltas_of_modelOut = [args.lsdtsMo])
    # X_hatZs, y_hatZs, X_tildZs, y_tildZs, areal_tildZs = read_Sim_Data()
    
    if args.fixMb:
        res = Gibbs_sampler(X_hatZs, y_hatZs, X_tildZs, y_tildZs, areal_tildZs)
        mu, cov = res.sampler(args.withPrior, args.onlyOptimCovPar, args.gpdtsMo, args.repeat)
    else:
        mu, cov = optimise(X_hatZs, y_hatZs, X_tildZs, y_tildZs, args.withPrior)

    end = default_timer()
    print 'running time for optimisation with fixMb is ' + str(args.fixMb) + str(end - start) + ' seconds'

    mu_out = open(output_folder  + '.pickle', 'wb')
    pickle.dump(mu, mu_out)
    mu_out.close()
    cov_out = open(output_folder  + '.pickle', 'wb')
    pickle.dump(cov, cov_out)
    cov_out.close()

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






 
    
    
    

    
