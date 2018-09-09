
 #!/usr/bin/python -tt   #This line is to solve any difference between spaces and tabs
import numpy as np
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
import statsmodels.api as sm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
plt.switch_backend('agg') # This line is for running code on cluster to make pyplot working on cluster

# cov_matrix returns the covariance matix, where X is assumed to be stroed row-wise in a matrix, sigma and w are all positive
def cov_matrix(X, sigma, w, OMEGA = 1e-6):
    d = X.shape[1]
    n = X.shape[0]
    W = np.eye(d)
    if len(w) ==1:
        w = np.repeat(w, d)
    W[np.diag_indices(d)] = 1./w**2
    # temp0: multiply every feature(i) of a X point with 1./wi**2
    temp0 = np.dot(X, W)
    #say we have data point x1 with two dimensions, temp 1 constrcut a vector [x11**2/w1**2+ x12**2/w2**2 , x21**2/w1**2 + x22**2/w2**2]
    temp1 = np.sum(temp0*X, axis=1)
    # temp3 constructs a matrix with first row being a vector [x11**2/w1**2 + x12**2/w2**2 , x11*x21/w1**2 + x12*x22/w2**2]
    #second row being a vector [x11*x21/w1**2 + x12*x22/w2**2, x21**2/w1**2 + x22**2/w2**2]
    temp3= np.dot(temp0,X.T)
    # broadcast:temp4 = temp1.reshape(n,1) + temp1 is to construct the sum of the square part of each entry of the convariance matrix 
    temp4 = temp1.reshape(n,1) + temp1
    square_dist = temp4 - 2* temp3
    square_dist[square_dist<0] = 0 
    K = sigma * np.exp(-0.5 * square_dist) + np.diag(np.repeat(OMEGA, n))
    return K

# covrance matrix for GP regression, adding the nosie of the observations
def cov_matrix_reg(X, w, sigma=1., obs_noi_scale=0.1, OMEGA = 1e-6):
    K = cov_matrix(X, sigma, w)
    n = X.shape[0]
    C = K + np.diag(np.repeat(obs_noi_scale**2, n))
    return C

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

def compute_L_chol(cov):
    l_chol = np.linalg.cholesky(cov)
    return l_chol
def log_obsZs_giv_par_with_grad(theta, X_hatZs, y_hatZs, withPrior= False, zeroMeanHatZs = False, rbf = True, OMEGA = 1e-6):
    theta = np.array(theta)
    if rbf:
        num_len_scal = 1
    else:
        num_len_scal = X_hatZs.shape(1)
    
    log_sigma_Zs = theta[0] #sigma of GP function for Zs
    log_phi_Zs = theta[1:num_len_scal+1]  # length scale of GP function for Zs
    log_obs_noi_scale = theta[num_len_scal+1:num_len_scal+2]
    if not zeroMeanHatZs:
        mu_hatZs_coeffis = theta[num_len_scal+2:]

    n_hatZs = X_hatZs.shape[0]
    C_hatZs = cov_matrix_reg(X = X_hatZs, sigma = np.exp(log_sigma_Zs), w = np.exp(log_phi_Zs), obs_noi_scale = np.exp(log_obs_noi_scale))

    print zeroMeanHatZs
    
    if zeroMeanHatZs:# treating y_hatZs with mean ZERO (removing mean from y_hatZs)
        mu_hatZs = np.zeros(len(y_hatZs))
    else:#14/08/2018 adding polynomial term to mu_hatZs and NOT remove mean from y_hatZs
        n_row = X_hatZs.shape[0]
        tmp0 = np.repeat(1.,n_row).reshape(n_row,1)
        X_hatZs_extend = np.hstack((X_hatZs, tmp0))
        mu_hatZs = np.dot(X_hatZs_extend, mu_hatZs_coeffis)

    y = y_hatZs
    l_chol_C = compute_L_chol(C_hatZs)
    u = linalg.solve_triangular(l_chol_C.T, linalg.solve_triangular(l_chol_C, y - mu_hatZs, lower=True))     
    joint_log_like  = -np.sum(np.log(np.diag(l_chol_C))) - 0.5 * np.dot(y - mu_hatZs, u) - 0.5 * n_hatZs * np.log(2*np.pi) 

    if withPrior:
        #compute the likelihood of the gamma priors
        sigma_shape = 1.2 
        sigma_rate = 0.2 
        len_scal_shape = 1. 
        len_scal_rate = 1./np.sqrt(num_len_scal)
        obs_noi_scale_shape = 1.2
        obs_noi_scale_rate = 0.6

        #sigma, length_sacle, obs_noi_scale have to take positive numbers, thus taking gamma priors, whereas the mutiplicative bias b takes a normal prior
        log_prior = log_like_gamma(log_sigma_Zs, sigma_rate, sigma_shape) + log_like_gamma(log_phi_Zs, len_scal_shape, len_scal_rate) + \
        log_like_gamma(log_obs_noi_scale, obs_noi_scale_shape, obs_noi_scale_rate) 
        #compute the logarithm of the posterior
        log_pos = joint_log_like + log_prior
    else:
        log_pos = joint_log_like

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

    num_covPars = len(grad_C_hatZs)
    grads_par_covPars = np.zeros(num_covPars)    
    inver_C = linalg.solve_triangular(l_chol_C.T, linalg.solve_triangular(l_chol_C, np.eye(n_hatZs), lower=True))   

    for i in range(num_covPars):       
        temp = np.dot(grad_C_hatZs[i], u)
        grads_par_covPars[i] = -0.5 * np.sum(inver_C * grad_C_hatZs[i]) + 0.5 * np.dot(y - mu_hatZs, linalg.solve_triangular(l_chol_C.T, \
            linalg.solve_triangular(l_chol_C, temp, lower=True)))

    if zeroMeanHatZs:
       grads_all_pars = grads_par_covPars
    else: 
        grads_mu_hatZs_coeffs = np.dot(X_hatZs_extend.T, u)
        grads_all_pars = np.concatenate((grads_par_covPars, grads_mu_hatZs_coeffs))

    return [log_pos, grads_all_pars]

def minus_log_obsZs_giv_par_with_grad(theta, X_hatZs, y_hatZs, withPrior= False, zeroMeanHatZs = False, rbf = True, OMEGA = 1e-6):
    theta = np.array(theta)
    if rbf:
        num_len_scal = 1
    else:
        num_len_scal = X_hatZs.shape(1)
    
    log_sigma_Zs = theta[0] #sigma of GP function for Zs
    log_phi_Zs = theta[1:num_len_scal+1]  # length scale of GP function for Zs
    log_obs_noi_scale = theta[num_len_scal+1:num_len_scal+2]
    if not zeroMeanHatZs:
        mu_hatZs_coeffis = theta[num_len_scal+2:]

    n_hatZs = X_hatZs.shape[0]
    C_hatZs = cov_matrix_reg(X = X_hatZs, sigma = np.exp(log_sigma_Zs), w = np.exp(log_phi_Zs), obs_noi_scale = np.exp(log_obs_noi_scale))
    
    if zeroMeanHatZs:
        mu_hatZs = np.zeros(len(y_hatZs)) # treating y_hatZs with mean ZERO (removing mean from y_hatZs)
    else:#14/08/2018 adding polynomial term to mu_hatZs and NOT remove mean from y_hatZs
        n_row = X_hatZs.shape[0]
        tmp0 = np.repeat(1.,n_row).reshape(n_row,1)
        X_hatZs_extend = np.hstack((X_hatZs, tmp0))
        mu_hatZs = np.dot(X_hatZs_extend, mu_hatZs_coeffis)

    y = y_hatZs
    l_chol_C = compute_L_chol(C_hatZs)
    u = linalg.solve_triangular(l_chol_C.T, linalg.solve_triangular(l_chol_C, y - mu_hatZs, lower=True))     
    joint_log_like  = -np.sum(np.log(np.diag(l_chol_C))) - 0.5 * np.dot(y - mu_hatZs, u) - 0.5 * n_hatZs * np.log(2*np.pi) 

    if withPrior:
        #compute the likelihood of the gamma priors
        sigma_shape = 1.2 
        sigma_rate = 0.2 
        len_scal_shape = 1. 
        len_scal_rate = 1./np.sqrt(num_len_scal)
        obs_noi_scale_shape = 1.2
        obs_noi_scale_rate = 0.6

        #sigma, length_sacle, obs_noi_scale have to take positive numbers, thus taking gamma priors, whereas the mutiplicative bias b takes a normal prior
        log_prior = log_like_gamma(log_sigma_Zs, sigma_rate, sigma_shape) + log_like_gamma(log_phi_Zs, len_scal_shape, len_scal_rate) + \
        log_like_gamma(log_obs_noi_scale, obs_noi_scale_shape, obs_noi_scale_rate) 
        #compute the logarithm of the posterior
        log_pos = joint_log_like + log_prior
    else:
        log_pos = joint_log_like

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

    num_covPars = len(grad_C_hatZs)
    grads_par_covPars = np.zeros(num_covPars)    
    inver_C = linalg.solve_triangular(l_chol_C.T, linalg.solve_triangular(l_chol_C, np.eye(n_hatZs), lower=True))   

    for i in range(num_covPars):       
        temp = np.dot(grad_C_hatZs[i], u)
        grads_par_covPars[i] = -0.5 * np.sum(inver_C * grad_C_hatZs[i]) + 0.5 * np.dot(y - mu_hatZs, linalg.solve_triangular(l_chol_C.T, \
            linalg.solve_triangular(l_chol_C, temp, lower=True)))

    if zeroMeanHatZs:
       grads_all_pars = grads_par_covPars
    else: 
        grads_mu_hatZs_coeffs = np.dot(X_hatZs_extend.T, u)
        grads_all_pars = np.concatenate((grads_par_covPars, grads_mu_hatZs_coeffs))

    minus_log_pos = -log_pos
    minus_grads_all_pars = - grads_all_pars

    return [minus_log_pos, minus_grads_all_pars]

def gradsApprox(theta, X_hatZs, y_hatZs,  withPrior= False,  rbf = True, OMEGA = 1e-6, epsilon = 1e-7):
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
        J_plus[i], _ = log_obsZs_giv_par_with_grad(thetaplus, X_hatZs, y_hatZs) # Step 3

        
        # Compute J_minus[i]. Inputs: "thetaminus, epsilon". Output = "J_minus[i]".
        thetaminus = np.copy(theta)    # Step 1
        thetaminus[i] = thetaminus[i]-epsilon   # Step 2   
        J_minus[i], _= log_obsZs_giv_par_with_grad(thetaminus, X_hatZs, y_hatZs)# Step 3
        
        # Compute gradapprox[i]
        gradapprox[i] = (J_plus[i]-J_minus[i])/(2.*epsilon)
    gradapprox = gradapprox.reshape(num_parameters)
   
    return gradapprox
def check_grads():

    input_folder = 'sampRealData/FPstart2016020612_FR_numObs_' + str(328) + '_numMo_' + str(300) \
    + '/seed' + str(123) + '/'
    X_hatZs_in = open(input_folder + 'X_hatZs.pkl', 'rb')
    X_hatZs = pickle.load(X_hatZs_in) 
    y_hatZs_in = open(input_folder + 'y_hatZs.pkl', 'rb')
    y_hatZs = pickle.load(y_hatZs_in) 
         
    modelBias = np.array([5.47022754 , 5.22854712, 2.5])
    initial_theta=np.concatenate((np.array([np.log(1.5), np.log(0.1), np.log(0.1)]),modelBias), axis=0)
    _, grads_computed = log_obsZs_giv_par_with_grad(initial_theta, X_hatZs, y_hatZs)
    grads_approx = gradsApprox(initial_theta, X_hatZs, y_hatZs)
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

def optimise(X_hatZs, y_hatZs, withPrior, useGradsFlag = False, repeat=3, seed =188, zeroMeanHatZs=False, method='L-BFGS-B', rbf=True, OMEGA = 1e-6, \
    bounds = ((-5, 5), (-5, 5), (-5, 5))): 
    print 'starting optimising when withPrior is ' + str(withPrior)  + '& useGradsFlag is ' + str(useGradsFlag) 
    if rbf:
        num_par=1
    else:
        num_par=X_hatZs.shape[1]  
    res = []
    count = 0
    LBFGSB_status = False
   
    while count != repeat:
        try:#find one intial value for which the optimisation works
            if zeroMeanHatZs:
                initial_theta=np.concatenate((np.log(np.random.gamma(1.2, 5., 1)), np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), \
                np.log(np.random.gamma(1.2, 1./0.6, 1))), axis=0)
                bounds = bounds
            else:
                initial_theta=np.concatenate((np.log(np.random.gamma(1.2, 5., 1)), np.log(np.random.gamma(1., np.sqrt(num_par), num_par)), \
                np.log(np.random.gamma(1.2, 1./0.6, 1)), np.zeros(3)), axis=0)
                bounds = ((-5, 5), (-5, 5), (-5, 5), (-100, 100), (-100, 100), (-100, 100))
            print 'bouds is ' + str(bounds)
            print 'initial theta when withPrior is ' + str(withPrior) + '& useGradsFlag is ' + str(useGradsFlag) +  ' :' + str(initial_theta)
            if useGradsFlag:
                tmp_res = minimize(fun=minus_log_obsZs_giv_par_with_grad, 
                               x0=initial_theta, method=method,
                               jac=True, bounds = bounds,
                               args=(X_hatZs, y_hatZs, withPrior, zeroMeanHatZs),
                               options={'maxiter': 2000, 'disp': False})
            else:
                tmp_res = minimize(fun=minus_log_obsZs_giv_par, 
                               x0=initial_theta, method=method,
                               jac=False, bounds = bounds,
                               args=(X_hatZs, y_hatZs, withPrior, zeroMeanHatZs),
                               options={'maxiter': 2000, 'disp': False})
            print 'The ' + str(count) + ' round of optimisation'
        except:
            continue
        if tmp_res['fun'] is not None: # if log pos at optimisation is not None, record the resutls, else, redo the otpmisation
            temp_res = [tmp_res['x'], np.copy(tmp_res['fun']), np.copy(tmp_res['hess_inv'].todense()), np.copy(tmp_res['success']), \
            np.copy(tmp_res['message']), np.copy(tmp_res['nit'])]
            res.append(temp_res)
            print 'theta from the ' + str(count) + ' round of optimisation with LBFGSB is ' + str(tmp_res['x'])
            if zeroMeanHatZs:
                parameters = np.exp(np.array(tmp_res['x']))
            else:
                parameters = (np.exp(np.array(tmp_res['x'])[:3]), np.array(tmp_res['x'])[3:])
            logPosat_resOptim, grads_at_resOptim = log_obsZs_giv_par_with_grad(tmp_res['x'], X_hatZs, y_hatZs, withPrior, zeroMeanHatZs)
            print 'grads at theta from the ' + str(count) + ' round of optimisation with LBFGSB is ' + str(grads_at_resOptim)
            flag_grads_equal_zero = np.round(grads_at_resOptim, 3) == 0.
            # if gradients from the first optimisation is zero, break out of the loop
            if np.sum(flag_grads_equal_zero) == len(flag_grads_equal_zero):
                LBFGSB_status = True
                print 'LBFGSB optmisation converged successfully at the '+ str(count) + ' round of optimisation.'
                res = np.array(res)
                res = res.T
                print 'minus_log_like for repeat ' + str(count)+ ' with LBFGSB is ' + str(res[1, :])
                print 'parameters after optimisation withPrior is ' + str(withPrior) + \
                  ' with LBFGSB :'  + str(parameters)
                print 'covariance of pars after optimisation withPrior is ' + str(withPrior) + \
                  ' with LBFGSB :'  + str(np.array(tmp_res['hess_inv'].todense()))
                print 'Optim status withPrior is ' + str(withPrior) + \
                  ' with LBFGSB :'  + str(np.array(tmp_res['success']))
                print 'Optim message withPrior is ' + str(withPrior) + \
                 ' with LBFGSB :'  + str(np.array(tmp_res['message']))
                print 'Optim nit withPrior is ' + str(withPrior) + \
                  ' with LBFGSB :'  + str(np.array(tmp_res['nit']))
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
                                   args=(X_hatZs, y_hatZs, withPrior, zeroMeanHatZs),
                                   options={'maxiter': 2000, 'disp': False})
                    print 'theta from the ' + str(count) + ' round of optimisation with BFGS is ' + str(tmp_res['x'])
                    if zeroMeanHatZs:
                        parameters = np.exp(np.array(tmp_res['x']))
                    else:
                        parameters = (np.exp(np.array(tmp_res['x'])[:3]), np.array(tmp_res['x'])[3:])
                    logPosat_resOptim, grads_at_resOptim = log_obsZs_giv_par_with_grad(tmp_res['x'], X_hatZs, y_hatZs, withPrior, zeroMeanHatZs)
                    print 'grads at theta from the ' + str(count) + ' round of optimisation with BFGS is ' + str(grads_at_resOptim)
                    flag_grads_equal_zero = np.round(grads_at_resOptim, 2) == 0.
                    # if gradients from the BFGS optimisation is zero, break out of the loop
                    if np.sum(flag_grads_equal_zero) == len(flag_grads_equal_zero):
                        print 'BFGS optmisation converged successfully at the '+ str(count) + ' round of optimisation.'
                        print 'minus_log_like for repeat ' + str(count)+ ' with BFGS is ' + str(tmp_res['fun'])
                        print 'parameters after optimisation withPrior is ' + str(withPrior) + \
                        ' with BFGS :'  + str(parameters)
                        print 'covariance of pars after optimisation withPrior is ' + str(withPrior) + \
                 ' with BFGS :'  + str(np.array(tmp_res['hess_inv']))
                        print 'Optim status withPrior is ' + str(withPrior) + \
                         ' with BFGS :'  + str(np.array(tmp_res['success']))
                        print 'Optim message withPrior is ' + str(withPrior) + \
                         ' with BFGS :'  + str(np.array(tmp_res['message']))
                        print 'Optim nit withPrior is ' + str(withPrior) + \
                          ' with BFGS :'  + str(np.array(tmp_res['nit']))
                    
                count += 1        
        else:# if log pos at optimisation is not None, record the resutls, else, redo the otpmisation
            continue 
    if LBFGSB_status:
        return [np.array(tmp_res['x']), np.array(tmp_res['hess_inv'].todense())]
    else:
        return [np.array(tmp_res['x']), np.array(tmp_res['hess_inv'])]

def predic_gpRegression(theta, X_train, y_train, X_test, y_test, crossValFlag = True, SEED=None, zeroMeanHatZs = False,  \
 withPrior= False, rbf = True, OMEGA = 1e-6):
    theta = np.array(theta)
    if rbf:
        num_len_scal = 1
    else:
        num_len_scal = X_hatZs.shape(1)
    
    log_sigma_Zs = theta[0] #sigma of GP function for Zs
    log_phi_Zs = theta[1:num_len_scal+1]  # length scale of GP function for Zs
    log_obs_noi_scale = theta[num_len_scal+1:num_len_scal+2]
    if not zeroMeanHatZs:
        mu_hatZs_coeffis = theta[num_len_scal+2:]

    n_hatZs = X_train.shape[0]
    C_hatZs = cov_matrix_reg(X = X_train, sigma = np.exp(log_sigma_Zs), w = np.exp(log_phi_Zs), obs_noi_scale = np.exp(log_obs_noi_scale))

    if zeroMeanHatZs:
        mu_train = np.zeros(len(y_train))
    else:
        n_row = X_train.shape[0]
        tmp0 = np.repeat(1.,n_row).reshape(n_row,1)
        X_train_extend = np.hstack((X_train, tmp0))
        mu_train = np.dot(X_train_extend, mu_hatZs_coeffis)

    y = y_train
    l_chol_C = compute_L_chol(C_hatZs)
    u = linalg.solve_triangular(l_chol_C.T, linalg.solve_triangular(l_chol_C, y - mu_train, lower=True))

    output_folder = 'Kriging/seed' + str(SEED) + '/'

    #*******************************comupute the prediction part for out sample ntest test data points under each theta **********************************************************
    ntest = X_test.shape[0]
    K_star_star = np.zeros((ntest,1))
    K_star_hatZs = cov_mat_xy(X_train, X_test, np.exp(log_sigma_Zs), np.exp(log_phi_Zs)) # is a matrix of size (n_train, n_test)
    K_star_hatZs = K_star_hatZs.T
    K_star = K_star_hatZs
    
    if zeroMeanHatZs:
        mu_test = np.zeros(len(y_test))
    else:
        n_row = X_test.shape[0]
        tmp0 = np.repeat(1.,n_row).reshape(n_row,1)
        X_test_extend = np.hstack((X_test, tmp0))
        mu_test = np.dot(X_test_extend, mu_hatZs_coeffis)


    mu_star = mu_test + np.dot(K_star, u)
    # print 'estimated mean is ' + str(mu_star)
    # print 'y_test is ' + str(y_test)

    rmse = np.sqrt(np.mean((y_test - mu_star)**2))

    print 'Out-of-sample RMSE for seed' + str(SEED) + ' is :' + str(rmse)

    rmse_out = open(output_folder + 'rmse_krig_outSample.pkl', 'wb')
    pickle.dump(rmse, rmse_out) 
    rmse_out.close()
 
    if not crossValFlag:
        index = np.arange(len(y_test))
        standardised_y_estimate = (mu_star - mu_test)/np.exp(log_obs_noi_scale)
        plt.figure()
        plt.scatter(index, standardised_y_estimate, facecolors='none',  edgecolors='k', linewidths=1.2)
        # plt.scatter(index, standardised_y_etstimate, c='k')
        plt.axhline(0, color='black', lw=1.2, ls ='-')
        plt.axhline(2, color='black', lw=1.2, ls =':')
        plt.axhline(-2, color='black', lw=1.2, ls =':')
        plt.xlabel('Index')
        plt.ylabel('Standardised residual')
        plt.savefig(output_folder + 'SEED'+ str(SEED) +'stdPredicErr_krig_outSample.png')
        plt.show()
        plt.close()

        sm.qqplot(standardised_y_estimate, line='45')
        plt.savefig(output_folder + 'SEED' + str(SEED) + 'normalQQ_krig_outSample.png')
        plt.show()
        plt.close()
    
    LKstar = linalg.solve_triangular(l_chol_C, K_star.T, lower = True)
    for i in range(ntest):
        K_star_star[i] = cov_matrix(X_test[i].reshape(1, 2), np.exp(log_sigma_Zs), np.exp(log_phi_Zs))
    
    vstar = K_star_star - np.sum(LKstar**2, axis=0).reshape(ntest,1) 
    vstar[vstar < 0] = 1e-9
    vstar = vstar.reshape(ntest, )
    print 'Out of sample estimated variance is ' + str(vstar)

    avg_width_of_predic_var = np.mean(np.sqrt(vstar + np.exp(log_obs_noi_scale)**2))

    print 'Out of sample average width of the prediction variance for seed ' + str(SEED) + ' is ' + str(avg_width_of_predic_var) 

    avgVar_out = open(output_folder + 'avgVar_krig_outSample.pkl', 'wb')
    pickle.dump(avg_width_of_predic_var, avgVar_out) 
    avgVar_out.close()

    upper_interv_predic = mu_star + 2 * np.sqrt(vstar + np.exp(log_obs_noi_scale)**2)
    lower_interv_predic = mu_star - 2 * np.sqrt(vstar + np.exp(log_obs_noi_scale)**2)

    upper_interval_rounded = np.round(upper_interv_predic, 1)
    lower_interval_rounded = np.round(lower_interv_predic, 1)
    # print 'rounded upper_interval is ' + str(upper_interval_rounded)
    # print 'rounded lower_interval is ' + str(lower_interval_rounded)

    flag_in_confiInterv_r = (y_test >= lower_interval_rounded) & (y_test <= upper_interval_rounded)
    count_in_confiInterv_r  = np.sum(np.array(map(int, flag_in_confiInterv_r)))
    # print 'number of estimated parameters within the 95 percent confidence interval with rounding is ' + str(count_in_confiInterv_r)
    succRate = count_in_confiInterv_r/np.float(len(y_test))
    print 'Out of sample prediction accuracy is ' + '{:.1%}'.format(succRate)

    accuracy_out = open(output_folder + 'predicAccuracy_krig_outSample.pkl', 'wb')
    pickle.dump(succRate, accuracy_out) 
    accuracy_out.close()

    lower_bound = np.array([-10, -6])
    upper_bound = np.array([-4, 2])
    point_res = 20
    x1, x2 = np.meshgrid(np.linspace(lower_bound[0], upper_bound[0], point_res),  
                         np.linspace(lower_bound[1], upper_bound[1], point_res))
    x1_vec = x1.ravel()
    x2_vec = x2.ravel()
    X_plot = np.vstack((x1_vec, x2_vec)).T
    #*******************************comupute the prediction part for ntest test data points under each theta **********************************************************
    ntest = X_plot.shape[0]
    K_star_star = np.zeros((ntest,1))
    K_star_hatZs = cov_mat_xy(X_train, X_plot, np.exp(log_sigma_Zs), np.exp(log_phi_Zs)) # is a matrix of size (n_train, n_test)
    K_star_hatZs = K_star_hatZs.T
    K_star = K_star_hatZs
    
    if zeroMeanHatZs:
        mu_test = np.zeros(ntest)
    else:
        n_row = X_test.shape[0]
        tmp0 = np.repeat(1.,n_row).reshape(n_row,1)
        X_plot_extend = np.hstack((X_plot, tmp0))
        mu_test = np.dot(X_plot_extend, mu_hatZs_coeffis)

    mu_plot = mu_test + np.dot(K_star, u)

    fig = plt.figure()
    ax = Axes3D(fig)
    scat = ax.scatter(x1_vec, x2_vec, mu_plot, c=mu_plot, cmap='viridis', linewidth=0.5)
    ax.set_xlabel('$lon$')
    ax.set_ylabel('$lat$')
    ax.set_zlabel('$Z(s)$')
    fig.colorbar(scat, shrink=0.85)
    plt.savefig(output_folder + 'SEED'+ str(SEED) + 'Krig_predic_scat.png')
    plt.close()

    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(x1, x2, mu_plot.reshape(point_res, point_res), rstride=1, cstride=1, cmap='viridis')
    ax.set_xlabel('$lon$')
    ax.set_ylabel('$lat$')
    ax.set_zlabel('$Z(s)$')
    fig.colorbar(surf, shrink=0.85)
    plt.savefig(output_folder + 'SEED'+ str(SEED) + 'Krig_predic_surf.png')
    plt.close()
    #*******************************comupute the prediction part for in sample ntrain test data points under each theta **********************************************************
    X_test = X_train
    y_test = y_train
    ntest = X_test.shape[0]
    K_star_star = np.zeros((ntest,1))
    K_star_hatZs = cov_mat_xy(X_train, X_test, np.exp(log_sigma_Zs), np.exp(log_phi_Zs)) # is a matrix of size (n_train, n_test)
    K_star_hatZs = K_star_hatZs.T
    K_star = K_star_hatZs
    
    if zeroMeanHatZs:
        mu_test = np.zeros(len(y_test))
    else:
        n_row = X_test.shape[0]
        tmp0 = np.repeat(1.,n_row).reshape(n_row,1)
        X_test_extend = np.hstack((X_test, tmp0))
        mu_test = np.dot(X_test_extend, mu_hatZs_coeffis)


    mu_star = mu_test + np.dot(K_star, u)

    rmse = np.sqrt(np.mean((y_test - mu_star)**2))

    print 'In-sample RMSE for seed' + str(SEED) + ' is :' + str(rmse)

    rmse_out = open(output_folder + 'rmse_krig_inSample.pkl', 'wb')
    pickle.dump(rmse, rmse_out) 
    rmse_out.close()
 
    if not crossValFlag:
        index = np.arange(len(y_test))
        standardised_y_estimate = (mu_star - mu_test)/np.exp(log_obs_noi_scale)
        plt.figure()
        plt.scatter(index, standardised_y_estimate, facecolors='none',  edgecolors='k', linewidths=1.2)
        # plt.scatter(index, standardised_y_etstimate, c='k')
        plt.axhline(0, color='black', lw=1.2, ls ='-')
        plt.axhline(2, color='black', lw=1.2, ls =':')
        plt.axhline(-2, color='black', lw=1.2, ls =':')
        plt.xlabel('Index')
        plt.ylabel('Standardised residual')
        plt.savefig(output_folder + 'SEED'+ str(SEED) +'stdPredicErr_krig_inSample.png')
        plt.show()
        plt.close()

        sm.qqplot(standardised_y_estimate, line='45')
        plt.savefig(output_folder + 'SEED' + str(SEED) + 'normalQQ_krig_inSample.png')
        plt.show()
        plt.close()
    
    LKstar = linalg.solve_triangular(l_chol_C, K_star.T, lower = True)
    for i in range(ntest):
        K_star_star[i] = cov_matrix(X_test[i].reshape(1, 2), np.exp(log_sigma_Zs), np.exp(log_phi_Zs))
    
    vstar = K_star_star - np.sum(LKstar**2, axis=0).reshape(ntest,1) 
    vstar[vstar < 0] = 1e-9
    vstar = vstar.reshape(ntest, )
    print 'In sample estimated variance is ' + str(vstar)

    avg_width_of_predic_var = np.mean(np.sqrt(vstar + np.exp(log_obs_noi_scale)**2))

    print 'In sample average width of the prediction variance for seed ' + str(SEED) + ' is ' + str(avg_width_of_predic_var) 

    avgVar_out = open(output_folder + 'avgVar_krig_inSample.pkl', 'wb')
    pickle.dump(avg_width_of_predic_var, avgVar_out) 
    avgVar_out.close()

    upper_interv_predic = mu_star + 2 * np.sqrt(vstar + np.exp(log_obs_noi_scale)**2)
    lower_interv_predic = mu_star - 2 * np.sqrt(vstar + np.exp(log_obs_noi_scale)**2)

    upper_interval_rounded = np.round(upper_interv_predic, 1)
    lower_interval_rounded = np.round(lower_interv_predic, 1)
    # print 'rounded upper_interval is ' + str(upper_interval_rounded)
    # print 'rounded lower_interval is ' + str(lower_interval_rounded)

    flag_in_confiInterv_r = (y_test >= lower_interval_rounded) & (y_test <= upper_interval_rounded)
    count_in_confiInterv_r  = np.sum(np.array(map(int, flag_in_confiInterv_r)))
    # print 'number of estimated parameters within the 95 percent confidence interval with rounding is ' + str(count_in_confiInterv_r)
    succRate = count_in_confiInterv_r/np.float(len(y_test))
    print 'In sample prediction accuracy is ' + '{:.1%}'.format(succRate)

    accuracy_out = open(output_folder + 'predicAccuracy_krig_inSample.pkl', 'wb')
    pickle.dump(succRate, accuracy_out) 
    accuracy_out.close()

    return succRate
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-SEED', type=int, dest='SEED', default=120, help='The simulation index')
    p.add_argument('-repeat', type=int, dest='repeat', default=1, help='number of repeats in optimisation')
    p.add_argument('-o', type=str, dest='output', default=None, help='Output folder')
    p.add_argument('-withPrior', dest='withPrior', default=False,  type=lambda x: (str(x).lower() == 'true'), help='flag for ML or MAP')
    p.add_argument('-lsZs', type=float, dest='lsZs', default=0.1, help='lengthscale of the GP covariance for Zs')
    p.add_argument('-sigZs', type=float, dest='sigZs', default=1.5, help='sigma (marginal variance) of the GP covariance for Zs')
    p.add_argument('-useGradsFlag', dest='useGradsFlag', default=True,  type=lambda x: (str(x).lower() == 'true'), \
        help='flag for whether to use analytically computed gradients to do optimisation')
    p.add_argument('-cntry', type=str, dest='cntry', default='FR', help='Country of the geo data used')
    p.add_argument('-usecntryFlag', dest='usecntryFlag', default=True,  type=lambda x: (str(x).lower() == 'true'), \
        help='flag for whether to use data for a specific country')
    p.add_argument('-numObs', type=int, dest='numObs', default=328, help='Number of observations used in modelling')
    p.add_argument('-numMo', type=int, dest='numMo', default=250, help='Number of model outputs used in modelling')
    p.add_argument('-crossValFlag', dest='crossValFlag', default=False,  type=lambda x: (str(x).lower() == 'true'), \
        help='whether to validate the model using cross validation')
    p.add_argument('-idxFold', type=int, dest='idxFold', default=9, help='the index for the fold for cross validation')
    p.add_argument('-zeroMeanHatZs', dest='zeroMeanHatZs', default=True,  type=lambda x: (str(x).lower() == 'true'), \
        help='whether to zero mean for y_hatZs')
    args = p.parse_args()
    if args.output is None: args.output = os.getcwd()
    if args.usecntryFlag:
        output_folder = args.output + '/Kriging/seed' + str(args.SEED) 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_folder += '/'
    print 'Output: ' + output_folder

    start = default_timer()
    np.random.seed(args.SEED)
    input_folder = 'Data/FPstart2016020612_' + str(args.cntry) + '_numObs_' + str(args.numObs) + '_numMo_' + str(args.numMo) \
    + '/seed' + str(args.SEED) + '/'
    
    if args.usecntryFlag:
        if args.crossValFlag:
            input_folder = input_folder + '/fold_' + str(args.idxFold) + '/'
            X_train_in = open(input_folder + 'X_train.pkl', 'rb')
            X_train = pickle.load(X_train_in) 
            if args.zeroMeanHatZs:
                y_train_in = open(input_folder + 'y_train.pkl', 'rb')
            else:
                y_train_in = open(input_folder + 'y_train_withMean.pkl', 'rb')
            y_train = pickle.load(y_train_in) 
            
            X_test_in = open(input_folder + 'X_test.pkl', 'rb')
            X_test = pickle.load(X_test_in)
            y_test_in = open(input_folder + 'y_test.pkl', 'rb')
            y_test = pickle.load(y_test_in) 
        else:
            X_hatZs_in = open(input_folder + 'X_hatZs.pkl', 'rb')
            X_hatZs = pickle.load(X_hatZs_in) 
            if args.zeroMeanHatZs:
                y_hatZs_in = open(input_folder + 'y_hatZs.pkl', 'rb')
            else:
                y_hatZs_in = open(input_folder + 'y_hatZs_withMean.pkl', 'rb')
            print 'y_hatZs_in is ' + str(y_hatZs_in)
            y_hatZs = pickle.load(y_hatZs_in) 

            areal_hatZs_in = open(input_folder + 'areal_hatZs.pkl', 'rb')
            areal_hatZs = pickle.load(areal_hatZs_in)

    print args.zeroMeanHatZs
    if args.crossValFlag:   
        mu, cov = optimise(X_train, y_train, args.withPrior, args.useGradsFlag, args.repeat, args.SEED, args.zeroMeanHatZs)
    else:   
        mu, cov = optimise(X_hatZs, y_hatZs, args.withPrior, args.useGradsFlag, args.repeat, args.SEED, args.zeroMeanHatZs)

    end = default_timer()
    print 'running time for optimisation is ' + str(end - start) + ' seconds'

    # computing the 95% confidence intervals  for each parameters

    if args.zeroMeanHatZs:
        cov_pars = np.exp(np.array(mu))
        pars = cov_pars
    else:
        cov_pars = np.exp(np.array(mu[:3]))
        bias_pars = np.array(mu[3:])
        pars = np.concatenate((cov_pars, bias_pars))
    pars = np.round(pars,1)
    print 'estimated pars rounded to one decimal point :' + str(pars)

    tmp = np.diag(np.array(cov))
    if args.zeroMeanHatZs:
        variance_log_covPars = tmp
        print 'variance_log_covPars is ' + str(variance_log_covPars)
        upper_interv_covPars = np.exp(mu + 2 * np.sqrt(variance_log_covPars))
        lower_interv_covPars = np.exp(mu - 2 * np.sqrt(variance_log_covPars))
        upper_interval = upper_interv_covPars
        lower_interval = lower_interv_covPars
        print 'upper_interval is ' + str(upper_interval)
        print 'lower_interval is ' + str(lower_interval)
    else:
        variance_log_covPars = tmp[:3]
        print 'variance_log_covPars is ' + str(variance_log_covPars)
        variance_biasPars = tmp[3:]
        print 'variance_biasPars is ' + str(variance_biasPars)
        upper_interv_covPars = np.exp(mu[:3] + 2 * np.sqrt(variance_log_covPars))
        lower_interv_covPars = np.exp(mu[:3] - 2 * np.sqrt(variance_log_covPars))
        upper_interv_biasPars = bias_pars + 2 * np.sqrt(variance_biasPars)
        lower_interv_biasPars = bias_pars - 2 * np.sqrt(variance_biasPars)
        upper_interval = np.concatenate((upper_interv_covPars, upper_interv_biasPars))
        lower_interval = np.concatenate((lower_interv_covPars, lower_interv_biasPars))

    upper_interval_rounded = np.round(upper_interval, 1)
    lower_interval_rounded = np.round(lower_interval, 1)
    print 'rounded upper_interval is ' + str(upper_interval_rounded)
    print 'rounded lower_interval is ' + str(lower_interval_rounded)

    res = {'mu':mu, 'cov':cov, 'pars':pars,'upper_interval':upper_interval, 'lower_interval':lower_interval, \
'upper_interval_rounded':upper_interval_rounded, 'lower_interval_rounded':lower_interval_rounded}
    res_out = open(output_folder  + 'resOptim_krig.pkl', 'wb')
    pickle.dump(res, res_out)
    res_out.close()
    if args.crossValFlag:
        predic_accuracy = predic_gpRegression(mu, X_train, y_train, X_test, y_test, args.crossValFlag, args.SEED, args.zeroMeanHatZs)
        print 'predic_accuracy for seed ' + str(args.SEED) + ' fold ' + str(args.idxFold) + ' is ' + '{:.1%}'.format(predic_accuracy)
    else:
        X_train = X_hatZs[:-50, :]
        X_test = X_hatZs[-50:, :]
        y_train = y_hatZs[:-50]
        y_test = y_hatZs[-50:]
        predic_accuracy = predic_gpRegression(mu, X_train, y_train, X_test, y_test, args.crossValFlag, args.SEED, args.zeroMeanHatZs)
        # print 'predic_accuracy for seed ' + str(args.SEED)  + ' is ' + '{:.1%}'.format(predic_accuracy)



