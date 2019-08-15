import numpy as np
from scipy import linalg
import computeN3Cost
import numbers
import pickle
import os
import argparse
from itertools import chain
# import statsmodels.api as sm
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import matplotlib.colors
import matplotlib as mpl
# plt.switch_backend('agg') # This line is for running code on cluster to make pyplot working on cluster
from rpy2.robjects.packages import importr
from rpy2.robjects import r
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from matplotlib.lines import Line2D
import scipy.stats as stats

np.seterr(over='raise', divide='raise')

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
    temp3= np.dot(temp0, X.T)
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
# Compute the cholesky decomposition
def compute_L_chol(cov):
    l_chol = np.linalg.cholesky(cov)
    computeN3Cost.num_calls_n3 += 1
    return l_chol
# Compute the pivoted cholesky decomposition
def compute_chol_pivoted(cov):
    numpy2ri.activate()
    Q = r.chol(cov, pivot= True)
    pivot = r.attr(Q, "pivot")
    tmp  = Q.rx(True, r.order(pivot))
    numpy2ri.deactivate() 
    tmp = np.array(tmp)
    G = tmp.T

    pivot = np.array(pivot) # the index in Python starting from 0 compared to that of 1 in R
    pivot = pivot -1
    return [G, pivot]

# compute minus the log of likelihood (-0.5*log(|C|)- 0.5*y^T*C^-1*y - -0.5 * n * log(2pi))and the gradients with respect to parameters for GP regression: 

def minus_log_py_giv_par_with_grad(theta, X, y, OMEGA = 1e-6):
    theta = np.array(theta)
    log_sigma = theta[0]
    log_w = theta[1:-1]
    log_obs_noi = theta[-1]
    n = X.shape[0]
    # d is the number of kernel parameters    
    # if isinstance(log_w, numbers.Number): # for the RBF kernel case where log_w is a scalar
    #     d=1
    #else:
    d = len(log_w)  # d==1 is for the RBF case, else is for the ARD kernel case 
    
    C = cov_matrix_reg(X, np.exp(log_sigma), np.exp(log_w), np.exp(log_obs_noi), OMEGA)
    l_chol_C = compute_L_chol(C)
    u = linalg.solve_triangular(l_chol_C.T, linalg.solve_triangular(l_chol_C, y, lower=True)) 

    # for i in range(len(u)):
    #     assert isinstance(u[i], numbers.Number)==True 
    
    log_like  = -np.sum(np.log(np.diag(l_chol_C))) - 0.5 * np.dot(y,u) - 0.5 * n * np.log(2*np.pi)

    grad_C = []
    grad_C_log_sigma = C - np.diag(np.repeat(OMEGA + np.exp(log_obs_noi), n))
    grad_C.append(grad_C_log_sigma)
    if d==1:
        tmp1 = np.sum(X*X, axis=1)
        tmp2 = np.dot(X,X.T)
        tmp3 = tmp1.reshape(n,1) + tmp1
        tmp_norm = tmp3 - 2 * tmp2
        grad_C_log_w = grad_C_log_sigma *  (1./np.exp(log_w)**2) * tmp_norm
        grad_C.append(grad_C_log_w)
    else:
        for i in range(d):
            temp0= (X[:,i].reshape(n,1) - X[:,i])**2
            temp1 =  (1./np.exp(log_w[i])**2) * temp0
            grad_C_log_wi = grad_C_log_sigma * temp1
            grad_C.append(grad_C_log_wi)
    grad_C_log_obs_noi = np.diag(np.repeat(np.exp(log_obs_noi),n))
    grad_C.append(grad_C_log_obs_noi)


    num_par = d + 2
    grad_par = np.zeros(num_par)    
    inver_C = linalg.solve_triangular(l_chol_C.T, linalg.solve_triangular(l_chol_C, np.eye(n), lower=True))   

    for i in range(num_par):
        
        temp = np.dot(grad_C[i],u)
        grad_par[i] = -0.5 * np.sum(inver_C * grad_C[i]) + 0.5 * np.dot(y, linalg.solve_triangular(l_chol_C.T, linalg.solve_triangular(l_chol_C, temp, lower=True)))

    return [-log_like, -grad_par]

def log_py_giv_par_with_grad(theta, X, y, OMEGA = 1e-6):
    theta = np.array(theta)
    log_sigma = theta[0]
    log_w = theta[1:-1]
    log_obs_noi = theta[-1]
    n = X.shape[0]
    # d is the number of kernel parameters    
    # if isinstance(log_w, numbers.Number): # for the RBF kernel case where log_w is a scalar
    #     d=1
    #else:
    d = len(log_w)  # d==1 is for the RBF case, else is for the ARD kernel case 
    
    C = cov_matrix_reg(X, np.exp(log_sigma), np.exp(log_w), np.exp(log_obs_noi), OMEGA)
    l_chol_C = compute_L_chol(C)
    u = linalg.solve_triangular(l_chol_C.T, linalg.solve_triangular(l_chol_C, y, lower=True)) 

    # for i in range(len(u)):
    #     assert isinstance(u[i], numbers.Number)==True 
    
    log_like  = -np.sum(np.log(np.diag(l_chol_C))) - 0.5 * np.dot(y,u) - 0.5 * n * np.log(2*np.pi)

    grad_C = []
    grad_C_log_sigma = C - np.diag(np.repeat(OMEGA + np.exp(log_obs_noi), n))
    grad_C.append(grad_C_log_sigma)
    if d==1:
        tmp1 = np.sum(X*X, axis=1)
        tmp2 = np.dot(X,X.T)
        tmp3 = tmp1.reshape(n,1) + tmp1
        tmp_norm = tmp3 - 2 * tmp2
        grad_C_log_w = grad_C_log_sigma *  (1./np.exp(log_w)**2) * tmp_norm
        grad_C.append(grad_C_log_w)
    else:
        for i in range(d):
            temp0= (X[:,i].reshape(n,1) - X[:,i])**2
            temp1 =  (1./np.exp(log_w[i])**2) * temp0
            grad_C_log_wi = grad_C_log_sigma * temp1
            grad_C.append(grad_C_log_wi)
    grad_C_log_obs_noi = np.diag(np.repeat(np.exp(log_obs_noi),n))
    grad_C.append(grad_C_log_obs_noi)


    num_par = d + 2
    grad_par = np.zeros(num_par)    
    inver_C = linalg.solve_triangular(l_chol_C.T, linalg.solve_triangular(l_chol_C, np.eye(n), lower=True))
    computeN3Cost.num_calls_n3 += 2   

    for i in range(num_par):
        
        temp = np.dot(grad_C[i],u)
        grad_par[i] = -0.5 * np.sum(inver_C * grad_C[i]) + 0.5 * np.dot(y, linalg.solve_triangular(l_chol_C.T, linalg.solve_triangular(l_chol_C, temp, lower=True)))

    return [log_like, grad_par]

def log_py_giv_par(theta, X, y, OMEGA = 1e-6):
    theta = np.array(theta)
    log_sigma = theta[0]
    log_w = theta[1:-1]
    log_obs_noi = theta[-1]
    n = X.shape[0]
    d = len(log_w)  # d==1 is for the RBF case, else is for the ARD kernel case     
    C = cov_matrix_reg(X, np.exp(log_sigma), np.exp(log_w), np.exp(log_obs_noi), OMEGA)
    l_chol_C = compute_L_chol(C)
    u = linalg.solve_triangular(l_chol_C.T, linalg.solve_triangular(l_chol_C, y, lower=True))     
    log_like  = -np.sum(np.log(np.diag(l_chol_C))) - 0.5 * np.dot(y,u) - 0.5 * n * np.log(2*np.pi)    

    return log_like

def predic_gpRegression(theta, X_train, y_train, X_test, y_test, X_tildZs, y_tildZs, crossValFlag = False,  SEED=None, numMo = None, useSimData =False, grid= False, \
    predicMo = False, a_bias_poly_deg = 5, indivError = True, index_Xaxis =True, conditionZhat = True,  marginZtilde=False, conZhatZtilde= False, marginZhat=False, \
    gp_deltas_modelOut = True, withPrior= False, rbf = True, OMEGA = 1e-6):
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


    n_hatZs = X_train.shape[0]
    n_tildZs = X_tildZs.shape[0]  
    n_bothZs = n_hatZs + n_tildZs

    mat = np.zeros(n_bothZs * n_bothZs).reshape(n_bothZs, n_bothZs)

    C_hatZs = cov_matrix_reg(X = X_train, sigma = np.exp(log_sigma_Zs), w = np.exp(log_phi_Zs), \
     obs_noi_scale = np.exp(log_obs_noi_scale))

    if gp_deltas_modelOut:
        C_tildZs, _= cov_areal(X_tildZs, log_sigma_Zs, log_phi_Zs, b, log_sigma_deltas_of_modelOut, \
            gp_deltas_modelOut, log_phi_deltas_of_modelOut)

    else:
        C_tildZs, _ = cov_areal(X_tildZs, log_sigma_Zs, log_phi_Zs, b, log_sigma_deltas_of_modelOut, \
            gp_deltas_modelOut)

    mat[:n_hatZs, :n_hatZs] = C_hatZs
    mat[n_hatZs:n_hatZs + n_tildZs, n_hatZs:n_hatZs + n_tildZs] = C_tildZs
    
    avg_pointAreal_lower, avg_pointAreal_upper, _, _ = \
    point_areal(X_train, X_tildZs, log_sigma_Zs, log_phi_Zs, b, pointArealFlag = True)

    mat[n_hatZs:n_hatZs + n_tildZs, :n_hatZs] = avg_pointAreal_lower
    mat[:n_hatZs, n_hatZs:n_hatZs + n_tildZs] = avg_pointAreal_upper

    mu_train = np.zeros(len(y_train))

    if a_bias_poly_deg ==2:
        X_tildZs_mean = np.array([np.mean(X_tildZs[i], axis=0) for i in range(len(y_tildZs))])
        n_row = X_tildZs_mean.shape[0]
        tmp0 = np.repeat(1.,n_row).reshape(n_row,1)
        X_tildZs_mean_extend = np.hstack((X_tildZs_mean, tmp0))
        mu_tildZs = np.dot(X_tildZs_mean_extend, a_bias_coefficients)
    else:
        X_tildZs_mean = np.array([np.mean(X_tildZs[i], axis=0) for i in range(len(y_tildZs))])
        n_row = X_tildZs_mean.shape[0]
        tmp0 = np.repeat(1.,n_row).reshape(n_row,1)
        X_tildZs_mean_extend0 = np.hstack((X_tildZs_mean, tmp0))
        tmp1 = np.array([X_tildZs[i]**2 for i in range(len(y_tildZs))]) # construct lon**2, lat**2
        tmp1 = np.array([np.mean(tmp1[i], axis =0) for i in range(len(y_tildZs))])
        tmp2 = np.array([X_tildZs[i][:,0] * X_tildZs[i][:, 1] for i in range(len(y_tildZs))]) # construct lon*lat  
        tmp2 = np.array([np.mean(tmp2[i]) for i in range(len(y_tildZs))])
        tmp2 = tmp2.reshape(n_row,1)
        X_tildZs_mean_extend = np.hstack((tmp1, tmp2, X_tildZs_mean_extend0))
        mu_tildZs = np.dot(X_tildZs_mean_extend, a_bias_coefficients)

    mu_hatTildZs = np.concatenate((mu_train, mu_tildZs))

    y = np.concatenate((y_train, y_tildZs))

    l_chol_C = compute_L_chol(mat)
    u = linalg.solve_triangular(l_chol_C.T, linalg.solve_triangular(l_chol_C, y - mu_hatTildZs, lower=True))
    if useSimData:
        if grid:
            output_folder = os.getcwd() + '/bmVsKrigGridScale/numObs_200_numMo_' + str(numMo) + '/seed' + str(SEED) 
        else:
            output_folder = os.getcwd() + '/dataSimulated/numObs_200_numMo_' + str(numMo) + '/seed' + str(SEED) 
    else:
        if predicMo:
            output_folder = ' DataImogenFrNoGridMo/FPstart2016020612_FR_numObs_128_numMo_' + str(numMo) + '/seed' + str(SEED) + '/predicMo'
        else:
            output_folder = 'DataImogenFrGridMoNotCentre/FPstart2016020612_FR_numObs_128_numMo_' + str(numMo) + '/seed' + str(SEED)
            
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_folder += '/'
    print('output_folder in gpGaussLikeFuns is ' + str(output_folder))
    #*******************************comupute the prediction part for out-of-sample ntest test data points under each theta **********************************************************
    # idx = np.argsort(y_test)
    # y_test = y_test[idx]
    # X_test = X_test[idx, :]
    ntest = X_test.shape[0]
    K_star_star = np.zeros((ntest,1))
    K_star_hatZs = cov_mat_xy(X_train, X_test, np.exp(log_sigma_Zs), np.exp(log_phi_Zs)) # is a matrix of size (n_train, n_test)
    K_star_hatZs = K_star_hatZs.T
    _, avg_pointAreal_upper, _, _ = point_areal(X_test, X_tildZs, log_sigma_Zs, log_phi_Zs, b)
    K_star_tildZs =avg_pointAreal_upper

    K_star = np.hstack((K_star_hatZs, K_star_tildZs))
    mu_star = np.dot(K_star, u)
    

    # print 'estimated mean is ' + str(mu_star)
    # print 'y_test is ' + str(y_test)

    if grid:
        rmse = y_test - mu_star
        print('Out-of-sample RMSE for seed' + str(SEED) + ' is :' + str(rmse))
    else:
        rmse = np.sqrt(np.mean((y_test - mu_star)**2))
        print('Out-of-sample RMSE for seed' + str(SEED) + ' is :' + str(rmse))
 
    rmse_out = open(output_folder + 'rmse_outSample.pkl', 'wb')
    pickle.dump(rmse, rmse_out) 
    rmse_out.close()

    
    LKstar = linalg.solve_triangular(l_chol_C, K_star.T, lower = True)
    for i in range(ntest):
        K_star_star[i] = cov_matrix(X_test[i].reshape(1, 2), np.exp(log_sigma_Zs), np.exp(log_phi_Zs))
    
    vstar = K_star_star - np.sum(LKstar**2, axis=0).reshape(ntest,1) 
    vstar[vstar < 0] = 1e-9
    vstar = vstar.reshape(ntest, )
    # print('Out of sample estimated variance is ' + str(vstar))

    if useSimData:
        if grid:
            avg_width_of_predic_var = np.sqrt(vstar)
        else:
            avg_width_of_predic_var = np.mean(np.sqrt(vstar))    
    else:
        avg_width_of_predic_var = np.mean(np.sqrt(vstar + np.exp(log_obs_noi_scale)**2))
    print('Out of sample average width of the prediction variance for seed ' + str(SEED) + ' is ' + str(avg_width_of_predic_var)) 
    # print('Out of sample width of the prediction variance for seed ' + str(SEED) + ' is ' + str(avg_width_of_predic_var)) 
    avgVar_out = open(output_folder + 'avgVar_outSample.pkl', 'wb')
    pickle.dump(avg_width_of_predic_var, avgVar_out) 
    # avgVar_out.close()

    if not useSimData:
        input_folder = os.getcwd() + '/DataImogenFrGridMoNotCentre/FPstart2016020612_FR_numObs_128/seed' + str(SEED) + '/'
        # input_folder = os.getcwd() + '/DataImogenFrNoGridMo/FPstart2016020612_FR_numObs_128/seed' + str(SEED) + '/'
        mean_y_hatZs_in = open(input_folder + 'mean.pickle', 'rb')
        mean_y_hatZs = pickle.load(mean_y_hatZs_in) 

        u0 = linalg.solve_triangular(l_chol_C.T, linalg.solve_triangular(l_chol_C, K_star.T, lower=True))
        cov_of_predic = cov_matrix(X_test, np.exp(log_sigma_Zs), np.exp(log_phi_Zs)) -  np.dot(K_star, u0) +  np.diag(np.repeat(np.exp(log_obs_noi_scale)**2, X_test.shape[0]))
        margin_var = np.diag(cov_of_predic) # This is equal to vstar + np.exp(log_obs_noi_scale)**2
        print ('Shape of X_test is ' + str(X_test.shape))
        print( 'Shape of cov_of_predic is ' + str(cov_of_predic.shape))
        G, pivot = compute_chol_pivoted(cov_of_predic)
        # Gtmp = G + np.diag(np.repeat(OMEGA, G.shape[0])) #even Gtmp is NOT positive definite
        # l_chol_G = compute_L_chol(Gtmp)
        # inv_G = linalg.solve_triangular(l_chol_G.T, linalg.solve_triangular(l_chol_G,np.eye(l_chol_G.shape[0])))
        inv_G = linalg.inv(G)
        print('margin_var is ' + str(margin_var))
        print('pivot is ' + str(pivot))

    index = np.arange(len(y_test))
    if useSimData:
        standardised_y_estimate = (y_test - mu_star)
    else:
        if indivError:
            standardised_y_estimate = (y_test - mu_star)/np.sqrt(margin_var)
            print(standardised_y_estimate)
        else:
            standardised_y_estimate = np.dot(inv_G, y_test - mu_star)
            std_yEst_Zs = standardised_y_estimate
      
        std_yEst_out = open(output_folder + 'std_yEst_outSample.pkl', 'wb')
        pickle.dump(standardised_y_estimate, std_yEst_out)
        
        plt.figure()
        # plt.scatter(index, standardised_y_estimate, facecolors='none', edgecolors='k', linewidths=1.2)
       
        if index_Xaxis:      
            plt.scatter(index, standardised_y_estimate, facecolors='none', edgecolors='k', linewidths=1.2)
            if indivError:
                plt.xlabel('Index')
            else:
                plt.xlabel('Pivoting order')
        else:
            plt.scatter(mu_star + mean_y_hatZs, standardised_y_estimate, facecolors='none', edgecolors='k', linewidths=1.2)
            plt.xlabel('Predicted mean')

        plt.axhline(0, color='black', lw=1.2, ls ='-')
        plt.axhline(2, color='black', lw=1.2, ls =':')
        plt.axhline(-2, color='black', lw=1.2, ls =':')
        
        plt.ylabel('Standardised residual')
        plt.savefig(output_folder + 'SEED'+ str(SEED) +'OutSamp_indivErr' + str(indivError) + 'Idx' + str(index_Xaxis) + '.eps')
        # plt.show()
        plt.close()

        # lower_chol =  np.linalg.cholesky(cov_of_predic)
        # num_Outputs =  cov_of_predic.shape[0]
        # tmp = np.dot(lower_chol, np.random.normal(0., 1., num_Outputs))
        # tmp = np.dot(inv_G, tmp)
        # tmp = np.sort(tmp)
        # samples_Zs = tmp
        # print('shape of samples_Zs is ' + str(samples_Zs.shape))
        
    print(np.abs(standardised_y_estimate))
    idx_largeResiduls = np.argsort(np.abs(standardised_y_estimate))[-3:]
    print(idx_largeResiduls)
    predicMean = mu_star + mean_y_hatZs
    if indivError:
        print(X_test[idx_largeResiduls, :])
        print(predicMean[idx_largeResiduls])
    else:
        idx_pivot = pivot[idx_largeResiduls]
        print('idx_pivot is ' + str(idx_pivot))
        print(X_test[idx_pivot, :])
      
    if useSimData:
        upper_interv_predic = mu_star + 2 * np.sqrt(vstar)
        lower_interv_predic = mu_star - 2 * np.sqrt(vstar)
    else:
        upper_interv_predic = mu_star + 2 * np.sqrt(vstar + np.exp(log_obs_noi_scale)**2)
        lower_interv_predic = mu_star - 2 * np.sqrt(vstar + np.exp(log_obs_noi_scale)**2)
        std = np.sqrt(vstar + np.exp(log_obs_noi_scale)**2)
    
    if not useSimData:
        # input_folder = os.getcwd() + '/DataImogenFrGridMoNotCentre/FPstart2016020612_FR_numObs_128/seed' + str(SEED) + '/'
        # mean_y_hatZs_in = open(input_folder + 'mean.pickle', 'rb')
        # mean_y_hatZs = pickle.load(mean_y_hatZs_in) 

        y_test_withMean = y_test + mean_y_hatZs
        y_train_withMean = y_train + mean_y_hatZs
        y_tildZs_withMean = y_tildZs + mean_y_hatZs
        mu_star_withMean = mu_star + mean_y_hatZs
        # upper_interv_predic = upper_interv_predic + mean_y_hatZs
        # lower_interv_predic = lower_interv_predic + mean_y_hatZs

        # print('maximum of y_test is ' + str(y_test.max()))
        # print('maximum of y_tildZs is ' + str(y_tildZs.max()))
        # print('minimum of y_test is ' + str(y_test.min()))
        # print('minimum of y_tildZs is ' + str(y_tildZs.min()))

        max_all = np.array([y_test_withMean.max(), y_train_withMean.max(), y_tildZs_withMean.max()]).max()
        print('max_all is ' + str(max_all))
        min_all = np.array([y_test_withMean.min(),y_train_withMean.min(), y_tildZs_withMean.min()]).min()
        print('min_all is ' + str(min_all))

        # exit(-1)
        # X_mo = np.array(list(chain.from_iterable(X_tildZs)))
        # plt.figure()
        # plt.scatter(X_mo[:, 0], X_mo[:, 1],  c = y_tildZs, cmap=plt.cm.jet, vmin=min_all, vmax=max_all, marker ='s', s=2)
        # plt.show()
        # exit(-1)
        plt.figure()
        plt.plot(y_test_withMean, y_test_withMean, ls='-', color = 'k')
        eb=plt.errorbar(y_test_withMean, mu_star_withMean,  yerr=2*std, linestyle='', capsize=5., fmt='o', elinewidth=1.5, markeredgewidth=1.5, ecolor='k', \
            markerfacecolor='w', markeredgecolor='k')
        eb[-1][0].set_linestyle(':')
        plt.xlabel('Observations')
        plt.ylabel('Predictions')
        plt.savefig(output_folder + 'BM_predic_scatter_seed' + str(SEED)+ 'numMo' + str(numMo) + 'mean.eps')
        plt.show()
        plt.close()
      
        
        # plt.figure()
        # plt.plot(y_test, y_test, ls='-', color = 'r')
        # plt.scatter(y_test, mu_star, color='black', label='predic_mean')
        # plt.scatter(y_test, upper_interv_predic, color = 'blue', label = 'predic_upper_CI', marker = '^')
        # plt.scatter(y_test, lower_interv_predic, color ='green', label = 'predic_lower_CI', marker = 'v')
        # plt.xlabel('Observations')
        # plt.ylabel('Predictions')
        # plt.legend(loc='best')
        # plt.title('Out of sample prediction')
        # plt.savefig(output_folder + 'BM_predic_scatter_seed' + str(SEED)+ 'numMo' + str(numMo) + 'mean.png')
        # plt.show()
        # plt.close()

        # residuals = y_test - mu_star
        r.load('france_rcoords.RData')
        france_rcoords = r['france_rcoords']
        france_rcoords=np.array(france_rcoords)

        residuals = standardised_y_estimate
        print(residuals)
        if indivError:
            max_coords = X_test[np.argsort(np.abs(residuals))[-3:], :]
        else:
            X_testTmp = X_test[pivot, :]
            max_coords = X_testTmp[np.argsort(np.abs(residuals))[-3:], :]
        print('max_coords of residuals is ' + str(max_coords))

        maxAbs = np.array([np.abs(residuals.min()), np.abs(residuals.max())]).max()

        fig, ax = plt.subplots()
        # cmap = mpl.colors.ListedColormap(['red', 'green', 'orange' ,'blue',  'cyan', 'white'])
        cmap = mpl.colors.ListedColormap(["#00007F", "blue",'cyan', 'white', 'green', "red", "#7F0000"])
        # cmap.set_over('0.25')
        # cmap.set_under('0.75')
        cmap.set_under("crimson")
        cmap.set_over('black')

        if maxAbs >4:
            bounds = np.array([-np.ceil(maxAbs), -4, -2, -1, 1, 2, 4, np.ceil(maxAbs)])
        else:
            bounds = np.array([-np.ceil(maxAbs), -2, -1, 1, 2,  np.ceil(maxAbs)])
            cmap = mpl.colors.ListedColormap(["blue", 'cyan', 'white', 'green',  "red"])
        norm0 = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        # residualsPlot = ax.scatter(X_test[:, 0], X_test[:, 1], c= y_test - mu_star, cmap=cmap, vmin = -maxAbs, vmax = maxAbs)
        if indivError:
            # residualsPlot = ax.scatter(X_test[:, 0], X_test[:, 1], c= standardised_y_estimate, cmap=cmap, vmin = -maxAbs, vmax = maxAbs)
            residualsPlot = ax.scatter(X_test[:, 0], X_test[:, 1], c= standardised_y_estimate, cmap=cmap, norm= norm0)
        else:
            # residualsPlot = ax.scatter(X_test[pivot, 0], X_test[pivot, 1], c= standardised_y_estimate, cmap=cmap, vmin = -maxAbs, vmax = maxAbs)
            residualsPlot = ax.scatter(X_test[pivot, 0], X_test[pivot, 1], c= standardised_y_estimate, \
                cmap=cmap, norm = norm0, edgecolors='k')

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        # plt.title('Residuals of out-of-sample prediction')
        plt.colorbar(residualsPlot, ax=ax)
        plt.plot(france_rcoords[:,0], france_rcoords[:,1], '-', color='k', lw=0.5)
        plt.savefig(output_folder + 'Residuals_seed' + str(SEED) + 'numMo' + str(numMo) + 'IndivErr' + str(indivError) +  '_outSampleStd.eps')
        plt.show()
        plt.close()
     
  
        cmap = plt.cm.jet
        bounds = np.round(np.linspace(min_all, max_all, 20), 0)
        norm0 = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

        #10/01/2019: creat cell coordinated of resoluton 10 by 10
        cell_res = 10
        cell_coords = np.zeros(2*cell_res**2).reshape(cell_res**2,2)
        cell_len = 0.04
        cell_width = 0.04
        x_coords = np.zeros(cell_res)
        y_coords = np.zeros(cell_res)
        for i in range(cell_res):
            x_coords[i] = cell_width/(2*cell_res) + i * cell_width/cell_res
            y_coords[i] = cell_len/(2*cell_res) + i * cell_width/cell_res

        for i in range(cell_res):
            cell_coords[i*10:(i*10+ cell_res), 0] = np.repeat(x_coords[i],cell_res)
            cell_coords[i*10:(i*10+ cell_res), 1] = y_coords
        # cell_coords = cell_coords - np.array([cell_width/2., cell_len/2.]) # this line center the cooridnated at the center of the cell
        cell_coords = cell_coords 
            
        tmp = X_tildZs - cell_coords 
        
        # get the center coordinates of X_tildZs
        centre_MoCoords = np.array([tmp[i][0] for i in range(len(X_tildZs))])
        X_mo  = centre_MoCoords

        Nr = 1
        Nc = 3
        fig, (ax1, ax2,cax) = plt.subplots(Nr, Nc, figsize=(12,5), gridspec_kw={"width_ratios":[1, 1, 0.06]})
        fig.set_rasterized(True)
        ax1.set_rasterized(True)
        ax2.set_rasterized(True)

        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Observations', markerfacecolor=None, markeredgecolor='k', markersize=8),\
            Line2D([0], [0], marker='s', color='w', markerfacecolor=None, markeredgecolor='k', markersize=8, label='Test points')] 

        if indivError:
            maxYtest = y_test_withMean[np.argsort(np.abs(residuals))[-3:]]
        else:
            y_testTmp = y_test_withMean[pivot]
            maxYtest = y_testTmp[np.argsort(np.abs(residuals))[-3:]]
        maxPlot = ax1.scatter(max_coords[:, 0], max_coords[:, 1], c= maxYtest, cmap=cmap, norm =norm0, marker = 's')
        trainObs = ax1.scatter(X_train[:, 0], X_train[:, 1], c= y_train_withMean, cmap=cmap, norm =norm0)
        ax1.plot(france_rcoords[:,0], france_rcoords[:,1], '-', color='k', lw=0.5)
       
        ax1.set_xlabel('$Longitude$')
        ax1.set_ylabel('$Latitude$')
        ax1.set_title('(a)')
        ax1.legend(handles=legend_elements, loc='best')

        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Model outputs', markerfacecolor=None, markeredgecolor='k', markersize=8), \
            Line2D([0], [0], marker='s', color='w', markerfacecolor=None, markeredgecolor='k', markersize=8, label='Test points')] 

        if indivError:
            maxYtest = y_test_withMean[np.argsort(np.abs(residuals))[-3:]]
        else:
            y_testTmp = y_test_withMean[pivot]
            maxYtest = y_testTmp[np.argsort(np.abs(residuals))[-3:]]
        maxPlot = ax2.scatter(max_coords[:, 0], max_coords[:, 1], c= maxYtest, cmap=cmap, norm =norm0, marker = 's')
        modelOutputs = ax2.scatter(X_mo[:, 0], X_mo[:, 1],  c = y_tildZs_withMean, cmap=cmap, norm = norm0)
        ax2.plot(france_rcoords[:,0], france_rcoords[:,1], '-', color='k', lw=0.5)
        ax2.set_xlabel('$Longitude$')
        ax2.set_ylabel('$Latitude$')
        ax2.set_title('(b)')
        ax2.legend(handles=legend_elements, loc='best')

        plt.colorbar(modelOutputs, cax= cax, ax=[ax1,ax2], use_gridspec = True)
        plt.savefig(output_folder + 'SEED'+ str(SEED) + 'ObsMo' + str(numMo) + '.eps', rasterized=True)
        plt.show()
        plt.close()

        # Nr = 1
        # Nc = 2
        # fig, ax = plt.subplots(Nr, Nc, constrained_layout= True, sharex=True, sharey=True, figsize=(10,5))
        # axs = ax.flat
        
        # # X_mo = np.array(list(chain.from_iterable(X_tildZs))) # This line of code is for the case where only one point for X_tildZs
        # legend_elements = [Line2D([0], [0], marker='o', color='w', label='Observations', markerfacecolor=None, markeredgecolor='k', markersize=8),\
        #     Line2D([0], [0], marker='s', color='w', markerfacecolor=None, markeredgecolor='k', markersize=8, label='Test points')] 

        # if indivError:
        #     maxYtest = y_test_withMean[np.argsort(np.abs(residuals))[-3:]]
        # else:
        #     y_testTmp = y_test_withMean[pivot]
        #     maxYtest = y_testTmp[np.argsort(np.abs(residuals))[-3:]]
        # maxPlot = ax[0].scatter(max_coords[:, 0], max_coords[:, 1], c= maxYtest, cmap=cmap, norm =norm0, marker = 's')
        # # for i, txt in enumerate(np.round(maxYtest, 1)):
        # #     ax.annotate(txt, (max_coords[:, 0][i], max_coords[:, 1][i]))
        # trainObs = ax[0].scatter(X_train[:, 0], X_train[:, 1], c= y_train_withMean, cmap=cmap, norm =norm0)
        # ax[0].plot(france_rcoords[:,0], france_rcoords[:,1], '-', color='k', lw=0.5)
        # # for i, txt in enumerate(np.round(y_train, 1)):
        # #     ax.annotate(txt, (X_train[:, 0][i], X_train[:, 1][i]))
        # # plt.colorbar(maxPlot, ax = ax)
        # # plt.savefig('SEED'+ str(SEED) + 'TrainObsAndTestMaxObs.png')
        # # plt.colorbar(trainObs, ax = ax)
        # # plt.title('Observations & test points')
        # axs[0].set_xlabel('$Longitude$')
        # axs[0].set_ylabel('$Latitude$')
        # axs[0].set_title('(a)')
        # ax[0].legend(handles=legend_elements, loc='best')

        # legend_elements = [Line2D([0], [0], marker='o', color='w', label='Model outputs', markerfacecolor=None, markeredgecolor='k', markersize=8), \
        #     Line2D([0], [0], marker='s', color='w', markerfacecolor=None, markeredgecolor='k', markersize=8, label='Test points')] 

        # if indivError:
        #     maxYtest = y_test_withMean[np.argsort(np.abs(residuals))[-3:]]
        # else:
        #     y_testTmp = y_test_withMean[pivot]
        #     maxYtest = y_testTmp[np.argsort(np.abs(residuals))[-3:]]
        # maxPlot = ax[1].scatter(max_coords[:, 0], max_coords[:, 1], c= maxYtest, cmap=cmap, norm =norm0, marker = 's')
        # # for i, txt in enumerate(np.round(maxYtest, 1)):
        # #     ax.annotate(txt, (max_coords[:, 0][i], max_coords[:, 1][i]))

        # modelOutputs = ax[1].scatter(X_mo[:, 0], X_mo[:, 1],  c = y_tildZs_withMean, cmap=cmap, norm = norm0)
        # ax[1].plot(france_rcoords[:,0], france_rcoords[:,1], '-', color='k', lw=0.5)
        # axs[1].set_xlabel('$Longitude$')
        # axs[1].set_ylabel('$Latitude$')
        # axs[1].set_title('(b)')
        # ax[1].legend(handles=legend_elements, loc='best')
        # plt.colorbar(modelOutputs, ax=ax.ravel().tolist())
        # plt.savefig(output_folder + 'SEED'+ str(SEED) + 'ObsMo' + str(numMo) + '.eps')
        # plt.show()
        # plt.close()

    upper_interval_rounded = np.round(upper_interv_predic, 1)
    lower_interval_rounded = np.round(lower_interv_predic, 1)
    # print 'rounded upper_interval is ' + str(upper_interval_rounded)
    # print 'rounded lower_interval is ' + str(lower_interval_rounded)

    flag_in_confiInterv_r = (y_test >= lower_interval_rounded) & (y_test <= upper_interval_rounded)
    flag_in_confiInterv_r = flag_in_confiInterv_r.astype(int)
    count_in_confiInterv_r  = np.sum(flag_in_confiInterv_r.astype(int))
    
    succRate = count_in_confiInterv_r/np.float(len(y_test))
    print('Out of sample prediction accuracy is ' + '{:.1%}'.format(succRate))

    return succRate

if __name__ == '__main__':
    computeN3Cost.init(0)
    p = argparse.ArgumentParser()
    p.add_argument('-SEED', type=int, dest='SEED', default=120, help='The simulation index')
    p.add_argument('-numObs', type=int, dest='numObs', default=128, help='Number of observations used in modelling')
    p.add_argument('-numMo', type=int, dest='numMo', default=500, help='Number of model outputs used in modelling')
    p.add_argument('-crossValFlag', dest='crossValFlag', default=False,  type=lambda x: (str(x).lower() == 'true'), \
        help='whether to validate the model using cross validation')
    p.add_argument('-useSimData', dest='useSimData', default=False,  type=lambda x: (str(x).lower() == 'true'), \
        help='flag for whether to use simulated data')
    p.add_argument('-grid', dest='grid', default=False,  type=lambda x: (str(x).lower() == 'true'),  help='flag for whether the predictions are produced for each grid')
    p.add_argument('-predicMo', dest='predicMo', default=False,  type=lambda x: (str(x).lower() == 'true'),  help='flag for whether to predict the value where model outputs are produced')
    p.add_argument('-poly_deg', type=int, dest='poly_deg', default=2, help='degree of the polynomial function of the additive model bias')
    p.add_argument('-indivError', dest='indivError', default=False,  type=lambda x: (str(x).lower() == 'true'),  help='flag for whether to individual std errors for GP diagnosis')
    p.add_argument('-index_Xaxis', dest='index_Xaxis', default=True,  type=lambda x: (str(x).lower() == 'true'),  help='flag for whether X-axis is index for GP diagnosis')

    args = p.parse_args()
    if args.useSimData: 
        # input_folder = os.getcwd() + '/dataRsimGammaTransformErrorInZtilde/numObs_200_numMo_' + str(args.numMo) + '/seed' + str(args.SEED) + '/'
        # input_folder = os.getcwd() + '/dataRsimNoFrGammaTransformArealRes25Cods100butArealZs100/numObs_200_numMo_' + str(args.numMo) + '/seed' + str(args.SEED) + '/'
        input_folder = os.getcwd() + '/dataSimulated/numObs_200_numMo_' + str(args.numMo) + '/seed' + str(args.SEED) + '/'
    else:
        input_folder = os.getcwd() + '/DataImogenFrGridMoNotCentre/FPstart2016020612_FR_numObs_128_numMo_' + str(args.numMo) + '/seed' + str(args.SEED) + '/'
        

    X_hatZs_in = open(input_folder + 'X_hatZs.pkl', 'rb')
    X_hatZs = pickle.load(X_hatZs_in) 
 
    y_hatZs_in = open(input_folder + 'y_hatZs.pkl', 'rb')
    y_hatZs = pickle.load(y_hatZs_in) 

    X_tildZs_in = open(input_folder + 'X_tildZs.pkl', 'rb')
    X_tildZs = pickle.load(X_tildZs_in) 
    print(X_tildZs.shape)
 

    y_tildZs_in = open(input_folder + 'y_tildZs.pkl', 'rb')
    y_tildZs = pickle.load(y_tildZs_in)

    if args.useSimData:
        resOptim_in = open(input_folder + 'resOptimSim.pkl', 'rb')
    else:
        resOptim_in = open(input_folder + 'resOptim.pkl', 'rb')
    
    resOptim = pickle.load(resOptim_in)

    mu = resOptim['mu']

    print('theta from optimisation is ' + str(mu)) 
    print(np.exp(mu[:-(4 + args.poly_deg -2)]))

    # when looking at the coefficients of the betas of a_bias, need to check its relative importance: beta/std(beta), not only Beta itself.
    cov = resOptim['cov']
    print('var of theta is' + str(np.diag(cov)))

    relative_importance_of_betas = mu[-(4 + args.poly_deg -2 -1):]/np.sqrt(np.diag(cov)[-(4 + args.poly_deg -2 -1):])
    print('relative_importance_of_betas is ' + str(relative_importance_of_betas))
   
    # tmp = list(np.log(np.array([20.20773792,  1.28349713  ,3.83773256, 1.47482164,  0.10713728]))) + [0.73782754, -0.44369921,  0.64953645, -2.61332868]
    # # tmp = list(np.log(np.array([22.14058558,  1.47980593,  3.33445096,  8.72246743,  0.07932848]))) + [0.87132649, -0.81637815,  0.17456605, -5.54637298]
    # mu = np.array(tmp)
   
    if args.useSimData:
        X_train = X_hatZs
        y_train = y_hatZs

        # input_folder = os.getcwd() + '/dataRsimGammaTransformErrorInZtilde/seed' + str(args.SEED) + '/'
        # input_folder = os.getcwd() + '/dataRsimNoFrGammaTransformArealRes25Cods100butArealZs100/seed' + str(args.SEED) + '/'
        input_folder = os.getcwd() + '/dataSimulated/seed' + str(args.SEED) + '/'
        all_X_Zs_in = open(input_folder + 'all_X_Zs.pickle', 'rb')
        all_X_Zs = pickle.load(all_X_Zs_in) 

        all_y_Zs_in = open(input_folder + 'all_y_Zs.pickle', 'rb')
        all_y_Zs = pickle.load(all_y_Zs_in) 

        X_test = all_X_Zs
        y_test = all_y_Zs
    else:
        if args.predicMo:
            # X_train = X_hatZs
            # y_train = y_hatZs
            X_train = X_hatZs[:-28, :]
            y_train = y_hatZs[:-28]

            # X_test = np.array(list(chain.from_iterable(X_tildZs)))
            # y_test = y_tildZs

            # input_folder = os.getcwd() + '/DataImogenFrGridMoNotCentre/FPstart2016020612_FR_numObs_128/seed' + str(args.SEED) + '/'
            input_folder = os.getcwd() + '/DataImogenFrNoGridMo/FPstart2016020612_FR_numObs_128/seed' + str(args.SEED) + '/'
            all_X_Zs_in = open(input_folder + 'all_X_Zs.pickle', 'rb')
            all_X_Zs = pickle.load(all_X_Zs_in) 

            all_y_Zs_in = open(input_folder + 'all_y_Zs.pickle', 'rb')
            all_y_Zs = pickle.load(all_y_Zs_in) 

            X_test = all_X_Zs
            y_test = all_y_Zs
            print('shape of X_test, y_test when predicMo is true ' + str((X_test.shape,y_test.shape)))

        else:
            X_train = X_hatZs[:-28, :]
            X_test = X_hatZs[-28:, :]
            y_train = y_hatZs[:-28]
            y_test = y_hatZs[-28:]
 
        print('shape of X_test, y_test'+ str((X_test.shape, y_test.shape)))

    predic_accuracy = predic_gpRegression(mu, X_train, y_train, X_test, y_test, X_tildZs, y_tildZs, args.crossValFlag, args.SEED, args.numMo, \
        args.useSimData, args.grid, args.predicMo, args.poly_deg, args.indivError, args.index_Xaxis)
         
