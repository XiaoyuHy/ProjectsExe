import numpy as np
from scipy import linalg
import computeN3Cost
import numbers
import pickle
import os
import argparse
from itertools import chain
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import matplotlib.colors
import matplotlib as mpl
# plt.switch_backend('agg') # This line is for running code on cluster to make pyplot working on cluster
from rpy2.robjects.packages import importr
from rpy2.robjects import r
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from matplotlib.lines import Line2D

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
    return G

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
    predicMo = False, a_bias_poly_deg = 2, gp_deltas_modelOut = True, withPrior= False, rbf = True, OMEGA = 1e-6):
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
            output_folder = 'Data/FPstart2016020612_FR_numObs_128_numMo_' + str(numMo) + '/seed' + str(SEED) + '/predicMo'
        else:
            # output_folder = 'DataImogenFrGridMoNotCentre/FPstart2016020612_FR_numObs_128_numMo_' + str(numMo) + '/seed' + str(SEED) 
            output_folder = 'Data/FPstart2016020612_FR_numObs_128_numMo_' + str(numMo) + '/seed' + str(SEED) 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_folder += '/'
    print('output_folder in gpGaussLikeFuns is ' + str(output_folder))
    #*******************************comupute the prediction part for out-of-sample ntest test data points under each theta **********************************************************
    ntest = X_test.shape[0]
    K_star_star = np.zeros((ntest,1))
    K_star_hatZs = cov_mat_xy(X_train, X_test, np.exp(log_sigma_Zs), np.exp(log_phi_Zs)) # is a matrix of size (n_train, n_test)
    K_star_hatZs = K_star_hatZs.T
    _, avg_pointAreal_upper, _, _ = point_areal(X_test, X_tildZs, log_sigma_Zs, log_phi_Zs, b)
    K_star_tildZs =avg_pointAreal_upper

    K_star = np.hstack((K_star_hatZs, K_star_tildZs))
    mu_star = np.dot(K_star, u)
    
    if predicMo:
        meanPredic_out = open(output_folder + 'meanPredic_outSample.pkl', 'wb')
        pickle.dump(mu_star, meanPredic_out) 
        meanPredic_out.close()
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
    avgVar_out.close()

    if not crossValFlag:
        index = np.arange(len(y_test))
        if useSimData:
            standardised_y_estimate = (mu_star - y_test)
        else:
            
            standardised_y_estimate = (mu_star - y_test)/np.sqrt(vstar + np.exp(log_obs_noi_scale)**2)
            std_yEst_out = open(output_folder + 'std_yEst_outSample.pkl', 'wb')
            pickle.dump(standardised_y_estimate, std_yEst_out)
         
    if useSimData:
        upper_interv_predic = mu_star + 2 * np.sqrt(vstar)
        lower_interv_predic = mu_star - 2 * np.sqrt(vstar)
    else:
        upper_interv_predic = mu_star + 2 * np.sqrt(vstar + np.exp(log_obs_noi_scale)**2)
        lower_interv_predic = mu_star - 2 * np.sqrt(vstar + np.exp(log_obs_noi_scale)**2)
    
 
    upper_interval_rounded = np.round(upper_interv_predic, 1)
    lower_interval_rounded = np.round(lower_interv_predic, 1)
    # print 'rounded upper_interval is ' + str(upper_interval_rounded)
    # print 'rounded lower_interval is ' + str(lower_interval_rounded)

    flag_in_confiInterv_r = (y_test >= lower_interval_rounded) & (y_test <= upper_interval_rounded)
    flag_in_confiInterv_r = flag_in_confiInterv_r.astype(int)
    count_in_confiInterv_r  = np.sum(flag_in_confiInterv_r.astype(int))
    
    succRate = np.round(count_in_confiInterv_r/np.float(len(y_test)), 3)
    print('Out of sample prediction accuracy is ' + '{:.1%}'.format(succRate))

    accuracy_out = open(output_folder + 'predicAccuracy_outSample.pkl', 'wb')
    if grid:
        print('flag_in_confiInterv_r is ' + str(flag_in_confiInterv_r))
        pickle.dump(flag_in_confiInterv_r, accuracy_out)
    else:
        pickle.dump(succRate, accuracy_out) 
     
    accuracy_out.close()

    #*******************************comupute the prediction part for in-sample ntrain test data points under each theta **********************************************************
    ########## The following codes are for in-sample observations #################
    X_test = X_train
    y_test = y_train

    # idx = np.argsort(y_test)
    # y_test = y_test[idx]
    # X_test = X_test[idx, :]
  # The following code is for Zhat | Zhat, Ztilde
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
    print('length of y_test, mu_star' + str((len(y_test), len(mu_star))))
    rmse = np.sqrt(np.mean((y_test - mu_star)**2))

    print('In-sample RMSE for seed' + str(SEED) + ' is :' + str(rmse))

    rmse_out = open(output_folder + 'rmse_inSample.pkl', 'wb')
    pickle.dump(rmse, rmse_out) 
    rmse_out.close()

# The following code is for Zhat | Zhat, Ztilde
    LKstar = linalg.solve_triangular(l_chol_C, K_star.T, lower = True)
    for i in range(ntest):
        K_star_star[i] = cov_matrix(X_test[i].reshape(1, 2), np.exp(log_sigma_Zs), np.exp(log_phi_Zs))
    
    vstar = K_star_star - np.sum(LKstar**2, axis=0).reshape(ntest,1) 
    vstar[vstar < 0] = 1e-9
    vstar = vstar.reshape(ntest, )
    # print('In-sample estimated variance is ' + str(vstar))

    if not crossValFlag:
        standardised_y_estimate = (mu_star - y_test)/np.sqrt(vstar + np.exp(log_obs_noi_scale)**2)
        std_yEst_out = open(output_folder + 'std_yEst_inSample.pkl', 'wb')
        pickle.dump(standardised_y_estimate, std_yEst_out)  

    avg_width_of_predic_var = np.mean(np.sqrt(vstar + np.exp(log_obs_noi_scale)**2))

    print('In-sample average width of the prediction variance for seed ' + str(SEED) + ' is ' + str(avg_width_of_predic_var)) 
    avgVar_out = open(output_folder + 'avgVar_inSample.pkl', 'wb')
    pickle.dump(avg_width_of_predic_var, avgVar_out) 
    avgVar_out.close()

    upper_interv_predic = mu_star + 2 * np.sqrt(vstar + np.exp(log_obs_noi_scale)**2)
    lower_interv_predic = mu_star - 2 * np.sqrt(vstar + np.exp(log_obs_noi_scale)**2)

    upper_interval_rounded = np.round(upper_interv_predic, 1)
    lower_interval_rounded = np.round(lower_interv_predic, 1)
    # print 'rounded upper_interval is ' + str(upper_interval_rounded)
    # print 'rounded lower_interval is ' + str(lower_interval_rounded)

    flag_in_confiInterv_r = (y_test >= lower_interval_rounded) & (y_test <= upper_interval_rounded)
    count_in_confiInterv_r  = np.sum(flag_in_confiInterv_r.astype(int))
    # print 'number of estimated parameters within the 95 percent confidence interval with rounding is ' + str(count_in_confiInterv_r)
    succRate = count_in_confiInterv_r/np.float(len(y_test))
    print('In-sample prediction accuracy is ' + '{:.1%}'.format(succRate))

    accuracy_out = open(output_folder + 'predicAccuracy_inSample.pkl', 'wb')
    pickle.dump(succRate, accuracy_out) 
    accuracy_out.close()
    # exit(-1)
    ########## The following codes are for in-sample model outputs #################
    X_test = X_tildZs
    y_test = y_tildZs
    index = np.arange(len(y_test))

    if a_bias_poly_deg ==2:
        X_test_mean = np.array([np.mean(X_test[i], axis=0) for i in range(len(y_tildZs))])
        n_row = X_test_mean.shape[0]
        tmp0 = np.repeat(1.,n_row).reshape(n_row,1)
        X_test_mean_extend = np.hstack((X_test_mean, tmp0))
        mu_test = np.dot(X_test_mean_extend, a_bias_coefficients)
    else:
        X_test_mean = np.array([np.mean(X_test[i], axis=0) for i in range(len(y_tildZs))])
        n_row = X_tildZs_mean.shape[0]
        tmp0 = np.repeat(1.,n_row).reshape(n_row,1)
        X_test_mean_extend0 = np.hstack((X_test_mean, tmp0))
        tmp1 = np.array([X_test[i]**2 for i in range(len(y_tildZs))]) # construct lon**2, lat**2
        tmp1 = np.array([np.mean(tmp1[i], axis =0) for i in range(len(y_tildZs))])
        tmp2 = np.array([X_test[i][:,0] * X_test[i][:, 1] for i in range(len(y_tildZs))]) # construct lon*lat  
        tmp2 = np.array([np.mean(tmp2[i]) for i in range(len(y_tildZs))])
        tmp2 = tmp2.reshape(n_row,1)
        X_test_mean_extend = np.hstack((tmp1, tmp2, X_test_mean_extend0))
        mu_test = np.dot(X_test_mean_extend, a_bias_coefficients)

    # The following is  for Ztilde|Zhat ~ MVN(mu, COV`)
    l_chol_ChatZs = compute_L_chol(C_hatZs)
    u3=linalg.solve_triangular(l_chol_ChatZs.T, linalg.solve_triangular(l_chol_ChatZs, y_train, lower=True))
    mu_star =  mu_test + np.dot(avg_pointAreal_lower, u3)
   
    print('length of y_test, mu_star' + str((len(y_test), len(mu_star))))
    rmse = np.sqrt(np.mean((y_test - mu_star)**2))
    print('In-sample RMSE of y_tildZs for seed' + str(SEED) + ' is :' + str(rmse))

    rmse_out = open(output_folder + 'rmse_inSample_ytildZsCon.pkl', 'wb')
    pickle.dump(rmse, rmse_out) 
    rmse_out.close()

    u4 = linalg.solve_triangular(l_chol_ChatZs.T, linalg.solve_triangular(l_chol_ChatZs, avg_pointAreal_upper, lower=True))
    cov_of_predic = C_tildZs - np.dot(avg_pointAreal_lower, u4)
    print ('Shape of X_test is ' + str(X_test.shape))
    print( 'Shape of cov_of_predic is ' + str(cov_of_predic.shape))

    standardised_y_estimate = (mu_star - y_test)/np.sqrt(np.diag(cov_of_predic))
    std_yEst_out = open(output_folder + 'std_yEst_inSampleCon.pkl', 'wb')
    pickle.dump(standardised_y_estimate, std_yEst_out)

    return succRate

if __name__ == '__main__':
    computeN3Cost.init(0)
    p = argparse.ArgumentParser()
    p.add_argument('-SEED', type=int, dest='SEED', default=120, help='The simulation index')
    p.add_argument('-numObs', type=int, dest='numObs', default=128, help='Number of observations used in modelling')
    p.add_argument('-numMo', type=int, dest='numMo', default=50, help='Number of model outputs used in modelling')
    p.add_argument('-crossValFlag', dest='crossValFlag', default=False,  type=lambda x: (str(x).lower() == 'true'), \
        help='whether to validate the model using cross validation')
    p.add_argument('-useSimData', dest='useSimData', default=False,  type=lambda x: (str(x).lower() == 'true'), \
        help='flag for whether to use simulated data')
    p.add_argument('-grid', dest='grid', default=False,  type=lambda x: (str(x).lower() == 'true'),  help='flag for whether the predictions are produced for each grid')
    p.add_argument('-predicMo', dest='predicMo', default=False,  type=lambda x: (str(x).lower() == 'true'),  help='flag for whether to predict the value where model outputs are produced')

    args = p.parse_args()
    if args.useSimData: 
        # input_folder = os.getcwd() + '/dataRsimGammaTransformErrorInZtilde/numObs_200_numMo_' + str(args.numMo) + '/seed' + str(args.SEED) + '/'
        # input_folder = os.getcwd() + '/dataRsimNoFrGammaTransformArealRes25Cods100butArealZs100/numObs_200_numMo_' + str(args.numMo) + '/seed' + str(args.SEED) + '/'
        input_folder = os.getcwd() + '/dataSimulated/numObs_200_numMo_' + str(args.numMo) + '/seed' + str(args.SEED) + '/'
    else:
        # input_folder = os.getcwd() + '/DataImogenFrGridMoNotCentre/FPstart2016020612_FR_numObs_128_numMo_' + str(args.numMo) + '/seed' + str(args.SEED) + '/'
        input_folder = os.getcwd() + '/Data/FPstart2016020612_FR_numObs_128_numMo_' + str(args.numMo) + '/seed' + str(args.SEED) + '/'

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
    print(np.exp(mu[:-4]))
   
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
            input_folder = os.getcwd() + '/Data/FPstart2016020612_FR_numObs_128/seed' + str(args.SEED) + '/'
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
        args.useSimData, args.grid, args.predicMo)
         
