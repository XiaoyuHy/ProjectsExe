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

#covariance matrix between areas
def cov_areal(areal_coordinate, sigma, w, b=2., OMEGA = 1e-6):
    num_areas = len(areal_coordinate)
    cov_areas = []
    for i in range(num_areas):
        tmp = [avg_cov_two_areal(areal_coordinate[i], areal_coordinate[j], sigma, w, b) for j in range(num_areas)]
        cov_areas.append(tmp)
    covAreas = np.hstack(cov_areas).reshape(num_areas,num_areas) + np.diag(np.repeat(OMEGA, num_areas))
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
def avg_cov_two_areal(x, y, sigma, w, b=2.):
    n_x = x.shape[0]
    n_y = y.shape[0]
    cov_of_two_vec = cov_mat_xy(x, y, sigma, w)
    avg = b**2 * np.float(np.sum(cov_of_two_vec))/(n_x * n_y)
    return avg

def read_Sim_Data(SEED):
    #read the samples of hatZs
    X_hatZs = np.array(pd.read_csv('simDataFiles/X_hatZs_res100_a_bias_poly_deg2SEED' + str(SEED) + '.txt', sep=" ", header=None))
    y_hatZs = np.array(pd.read_csv('simDataFiles/y_hatZs_res100_a_bias_poly_deg2SEED' + str(SEED) + '.txt', sep=" ", header=None)).reshape(X_hatZs.shape[0])

    #read the samples of tildZs
    X_tildZs_in = open('simDataFiles/X_tildZs_a_bias_poly_deg2SEED' + str(SEED) + '.pickle', 'rb')
    X_tildZs = pickle.load(X_tildZs_in)

    y_tildZs_in = open('simDataFiles/y_tildZs_a_bias_poly_deg2SEED' + str(SEED) + '.pickle', 'rb')
    y_tildZs = pickle.load(y_tildZs_in)

    latLon_tildZs_in = open('simDataFiles/latLon_tildZs_a_bias_poly_deg2SEED' + str(SEED) + '.pickle', 'rb')
    latLon_tildZs = pickle.load(latLon_tildZs_in)

    return[X_hatZs, y_hatZs, X_tildZs, y_tildZs, latLon_tildZs]

def log_like_normal(w, mu, Sigma):
    l_chol = np.linalg.cholesky(Sigma)
    u = linalg.solve_triangular(l_chol.T, linalg.solve_triangular(l_chol, w, lower=True))
    log_like_normal = -np.sum(np.log(np.diag(l_chol))) - 0.5 * np.dot(w, u) - 0.5 * len(w) * np.log(2*np.pi)
    return log_like_normal

def log_like_gamma(w, shape, rate):
    res = np.sum(shape * w - rate * np.exp(w) + shape * np.log(rate) - np.log(gamma(shape)))
    return res

def fun_a_bias(x, a_bias_coefficients = [0.1, 5., 5.]):
    a_bias_coefficients = np.array(a_bias_coefficients)
    a_bias = np.dot(a_bias_coefficients, np.concatenate(([1.], x)))
    return a_bias

def log_obsZs_giv_par(theta, X_hatZs, y_hatZs, X_tildZs, y_tildZs, a_bias_poly_deg = 2, rbf = True, OMEGA = 1e-6):
    theta = np.array(theta)
    if rbf:
        num_len_scal = 1
    else:
        num_len_scal = X_hatZs.shape(1)
    #only one lengthsacle parameter
    # log_w = theta

    #adding parameters of sigma
    # log_sigma = theta[0]
    # log_w = theta[1:]

    #adding parameters of sigma, obs_noi_scale, mulplicative bias b
    log_sigma = theta[0]
    log_w = theta[1:num_len_scal+1]
    log_obs_noi_scale = theta[num_len_scal+1:num_len_scal+2]
    b = theta[num_len_scal+2:num_len_scal+3]
    a_bias_coefficients = theta[len(theta) -(a_bias_poly_deg+1):]

    n_hatZs = X_hatZs.shape[0]
    n_tildZs = X_tildZs.shape[0]  
    n_bothZs = n_hatZs + n_tildZs

    mat = np.zeros(n_bothZs * n_bothZs).reshape(n_bothZs, n_bothZs)

    C_hatZs = gpGaussLikeFuns.cov_matrix_reg(X = X_hatZs, sigma = np.exp(log_sigma), w = np.exp(log_w), obs_noi_scale = np.exp(log_obs_noi_scale))
    C_tildZs = cov_areal(areal_coordinate = X_tildZs, sigma = np.exp(log_sigma), w = np.exp(log_w), b=b)

    mat[:n_hatZs, :n_hatZs] = C_hatZs
    mat[n_hatZs:n_hatZs + n_tildZs, n_hatZs:n_hatZs + n_tildZs] = C_tildZs
    
    point_areal = np.array([avg_cov_point_areal(X_hatZs, X_tildZs[i], sigma = np.exp(log_sigma), w = np.exp(log_w), b=b) for i in range(n_tildZs)])
    mat[n_hatZs:n_hatZs + n_tildZs, :n_hatZs] = point_areal
    mat[:n_hatZs, n_hatZs:n_hatZs + n_tildZs] = point_areal.T

    mu_hatZs = np.zeros(len(y_hatZs))
    mu_tildZs = np.array([fun_a_bias(np.mean(X_tildZs[i], axis=0)) for i in range(len(y_tildZs))])
    mu_hatTildZs = np.concatenate((mu_hatZs, mu_tildZs))

    y = np.concatenate((y_hatZs, y_tildZs))

    l_chol_C = gpGaussLikeFuns.compute_L_chol(mat)
    u = linalg.solve_triangular(l_chol_C.T, linalg.solve_triangular(l_chol_C, y - mu_hatTildZs, lower=True))     
    joint_log_like  = -np.sum(np.log(np.diag(l_chol_C))) - 0.5 * np.dot(y - mu_hatTildZs, u) - 0.5 * n_bothZs * np.log(2*np.pi) 

    #compute the likelihood of the gamma priors
    sigma_shape = 1.2 
    sigma_rate = 0.2 
    len_scal_shape = 1. 
    len_scal_rate = 1./np.sqrt(num_len_scal)
    obs_noi_scale_shape = 1.2
    obs_noi_scale_rate = 0.6
    b_mu =0.
    b_Sigma = np.diag([100.])
    a_bias_coefficients_mu = np.zeros(a_bias_poly_deg +1)
    a_bias_coefficients_Sigma = np.diag(np.repeat(100, a_bias_poly_deg + 1))
    #sigma, length_sacle, obs_noi_scale have to take positive numbers, thus taking gamma priors, whereas the mutiplicative bias b takes a normal prior
    log_prior = log_like_gamma(log_sigma, sigma_rate, sigma_shape) + log_like_gamma(log_w, len_scal_shape, len_scal_rate) + \
    log_like_gamma(log_obs_noi_scale, obs_noi_scale_shape, obs_noi_scale_rate) + log_like_normal(b, b_mu, b_Sigma) + \
    log_like_normal(a_bias_coefficients, a_bias_coefficients_mu, a_bias_coefficients_Sigma)

    #compute the logarithm of the posterior
    log_pos = joint_log_like + log_prior

    return log_pos

def minus_log_obsZs_giv_par(theta, X_hatZs, y_hatZs, X_tildZs, y_tildZs, a_bias_poly_deg = 2, rbf = True, OMEGA = 1e-6):
    theta = np.array(theta)
    if rbf:
        num_len_scal = 1
    else:
        num_len_scal = X_hatZs.shape(1)
    #only one lengthsacle parameter
    # log_w = theta

    #adding parameters of sigma
    # log_sigma = theta[0]
    # log_w = theta[1:]

    #adding parameters of sigma, obs_noi_scale, mulplicative bias b
    log_sigma = theta[0]
    log_w = theta[1:num_len_scal+1]
    log_obs_noi_scale = theta[num_len_scal+1:num_len_scal+2]
    b = theta[num_len_scal+2:num_len_scal+3]
    a_bias_coefficients = theta[len(theta) -(a_bias_poly_deg+1):]

    n_hatZs = X_hatZs.shape[0]
    n_tildZs = X_tildZs.shape[0]  
    n_bothZs = n_hatZs + n_tildZs

    mat = np.zeros(n_bothZs * n_bothZs).reshape(n_bothZs, n_bothZs)

    C_hatZs = gpGaussLikeFuns.cov_matrix_reg(X = X_hatZs, sigma = np.exp(log_sigma), w = np.exp(log_w), obs_noi_scale = np.exp(log_obs_noi_scale))
    C_tildZs = cov_areal(areal_coordinate = X_tildZs, sigma = np.exp(log_sigma), w = np.exp(log_w), b=b)

    mat[:n_hatZs, :n_hatZs] = C_hatZs
    mat[n_hatZs:n_hatZs + n_tildZs, n_hatZs:n_hatZs + n_tildZs] = C_tildZs
    
    point_areal = np.array([avg_cov_point_areal(X_hatZs, X_tildZs[i], sigma = np.exp(log_sigma), w = np.exp(log_w), b=b) for i in range(n_tildZs)])
    mat[n_hatZs:n_hatZs + n_tildZs, :n_hatZs] = point_areal
    mat[:n_hatZs, n_hatZs:n_hatZs + n_tildZs] = point_areal.T

    mu_hatZs = np.zeros(len(y_hatZs))
    mu_tildZs = np.array([fun_a_bias(np.mean(X_tildZs[i], axis=0)) for i in range(len(y_tildZs))])
    mu_hatTildZs = np.concatenate((mu_hatZs, mu_tildZs))

    y = np.concatenate((y_hatZs, y_tildZs))

    l_chol_C = gpGaussLikeFuns.compute_L_chol(mat)
    u = linalg.solve_triangular(l_chol_C.T, linalg.solve_triangular(l_chol_C, y - mu_hatTildZs, lower=True))     
    joint_log_like  = -np.sum(np.log(np.diag(l_chol_C))) - 0.5 * np.dot(y - mu_hatTildZs, u) - 0.5 * n_bothZs * np.log(2*np.pi) 

    #compute the likelihood of the gamma priors
    sigma_shape = 1.2 
    sigma_rate = 0.2 
    len_scal_shape = 1. 
    len_scal_rate = 1./np.sqrt(num_len_scal)
    obs_noi_scale_shape = 1.2
    obs_noi_scale_rate = 0.6
    b_mu =0.
    b_Sigma = np.diag([100.])
    a_bias_coefficients_mu = np.zeros(a_bias_poly_deg +1)
    a_bias_coefficients_Sigma = np.diag(np.repeat(100, a_bias_poly_deg + 1))
    #sigma, length_sacle, obs_noi_scale have to take positive numbers, thus taking gamma priors, whereas the mutiplicative bias b takes a normal prior
    log_prior = log_like_gamma(log_sigma, sigma_rate, sigma_shape) + log_like_gamma(log_w, len_scal_shape, len_scal_rate) + \
    log_like_gamma(log_obs_noi_scale, obs_noi_scale_shape, obs_noi_scale_rate) + log_like_normal(b, b_mu, b_Sigma) + \
    log_like_normal(a_bias_coefficients, a_bias_coefficients_mu, a_bias_coefficients_Sigma)

    #compute the logarithm of the posterior
    log_pos = joint_log_like + log_prior 
    minus_log_pos = - log_pos

    return minus_log_pos

def optimize(X_hatZs, y_hatZs, X_tildZs, y_tildZs, repeat=1, method='BFGS', rbf=True, OMEGA = 1e-6):
    if rbf:
        num_par=1
    else:
        num_par=X_hatZs.shape[1]   
    print 'starting optimising the parameters'
    res = []
    count = 0
    while count != repeat:
        try:
            initial_theta=np.concatenate((np.log(np.random.gamma(1.2, 5., 1)), np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), \
                np.log(np.random.gamma(1.2, 1./0.6, 1)), np.random.normal(0., 1., 1), np.random.normal(0., 1., 3)), axis=0)
            print 'initial theta is ' + str(initial_theta)
            tmp_res = minimize(fun=minus_log_obsZs_giv_par, 
                           x0=initial_theta, method=method,
                           jac=False,
                           args=(X_hatZs, y_hatZs, X_tildZs, y_tildZs),
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
    print 'log_parameters plus mulplicative_bias after optimisation is :'  + str(np.array(res[0, :][i]))
    print 'parameters after optimisation is :'  + str([np.exp(np.array(res[0, :][i])[:-4]), np.array(res[0, :][i])[-4:]])
    print 'covariance of pars after optimisation is :'  + str(np.array(res[2, :][i]))
    return [np.array(res[0, :][i]), np.array(res[2, :][i])]

class MH_estimator():
    def __init__(self, X_hatZs, y_hatZs, X_tildZs, y_tildZs, latLon_tildZs):
        self.X_hatZs = X_hatZs
        self.y_hatZs = y_hatZs
        self.X_tildZs = X_tildZs
        self.y_tildZs = y_tildZs
        self.latLon_tildZs = latLon_tildZs
    
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
                    mu, hess_inv= optimize(self.X_hatZs, self.y_hatZs, self.X_tildZs, self.y_tildZs, self.latLon_tildZs, repeat=5, method='BFGS', rbf=True, OMEGA = 1e-6)
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

if __name__ == '__main__':
    computeN3Cost.init(0)
    p = argparse.ArgumentParser()
    p.add_argument('-SEED', type=int, dest='SEED', default=0, help='The simulation index')
    p.add_argument('-o', type=str, dest='output', default=None, help='Output folder')
    p.add_argument('-REPEAT', type=int, dest='repeat', default=1, help='number of repeats in optimisation')
    args = p.parse_args()
    if args.output is None: args.output = os.getcwd()
    output_folder = args.output + '/SEED_' + str(args.SEED) 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_folder += '/'
    print 'Output: ' + output_folder
    #sim_hatTildZs_With_Plots()
    X_hatZs, y_hatZs, X_tildZs, y_tildZs = simData.sim_hatTildZs_With_Plots(SEED = args.SEED)
    start = default_timer()
    mu, cov = optimize(X_hatZs, y_hatZs, X_tildZs, y_tildZs, repeat = args.repeat)
    mu_out = open(output_folder + 'mu.pickle', 'wb')
    pickle.dump(mu, mu_out)
    mu_out.close()
    cov_out = open(output_folder + 'cov.pickle', 'wb')
    pickle.dump(cov, cov_out)
    cov_out.close()
    end = default_timer()
    print 'running time for optimisation is ' + str(end - start) + ' seconds'
    

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






 
    
    
    

    
