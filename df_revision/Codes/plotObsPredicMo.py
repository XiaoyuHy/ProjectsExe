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
# import matplotlib as mpl
# plt.switch_backend('agg') # This line is for running code on cluster to make pyplot working on cluster
from rpy2.robjects.packages import importr
from rpy2.robjects import r
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
# from matplotlib.lines import Line2D
import scipy.stats as stats
from scipy.interpolate import interp2d
from scipy.interpolate import fitpack
from scipy.interpolate import RectBivariateSpline

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

def predic_gpRegression(theta, X_train, y_train, X_test, y_test, X_tildZs, y_tildZs, elev_fp, crossValFlag = False,  SEED=None, numMo = None, useSimData =False, grid= False, \
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
        a_bias_coefficients = theta[len(theta) - (a_bias_poly_deg+2):]
        
    else:
        log_sigma_Zs = theta[0] #sigma of GP function for Zs
        log_phi_Zs = theta[1:num_len_scal+1]  # length scale of GP function for Zs
        log_obs_noi_scale = theta[num_len_scal+1:num_len_scal+2]
        log_sigma_deltas_of_modelOut = theta[num_len_scal+2:num_len_scal+3] # sigma of Normal for deltas of model output
        b = theta[num_len_scal+3:num_len_scal+4]
        a_bias_coefficients = theta[len(theta) - (a_bias_poly_deg+2):]


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
        X_tildZs_mean_extend = np.hstack((elev_fp.reshape(len(elev_fp),1), X_tildZs_mean, tmp0))
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
        X_tildZs_mean_extend = np.hstack((elev_fp.reshape(len(elev_fp),1), tmp1, tmp2, X_tildZs_mean_extend0))
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
            output_folder = 'Data/FPstart2016020612_FR_numObs_128_numMo_' + str(numMo) + '/seed' + str(SEED) 
            # output_folder = 'DataImogenFrGridMoNotCentre/FPstart2016020612_FR_numObs_128_numMo_' + str(numMo) + '/seed' + str(SEED) + '/theta' + str(theta_idx)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_folder += '/'
    print('output_folder in gpGaussLikeFuns is ' + str(output_folder))
    # #*******************************comupute the prediction part for out-of-sample ntest test data points under each theta **********************************************************
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

    if not useSimData:
        input_folder = os.getcwd() + '/Data/FPstart2016020612_FR_numObs_128/seed' + str(SEED) + '/'
        mean_y_hatZs_in = open(input_folder + 'mean.pickle', 'rb')
        mean_y_hatZs = pickle.load(mean_y_hatZs_in) 

        # input_folder = 'Data/FPstart2016020612_FR_numObs_128_numMo_' + str(numMo) + '/seed' + str(SEED) + '/predicMo/' 
        # mu_star_in = open(input_folder + 'meanPredic_outSample.pkl', 'rb')
        # mu_star = pickle.load(mu_star_in)
    
    if not useSimData:
        y_test = y_test + mean_y_hatZs
        y_train_withMean = y_train + mean_y_hatZs
        y_tildZs_withMean = y_tildZs + mean_y_hatZs
        mu_star = mu_star + mean_y_hatZs

        max_all = np.array([y_test.max(), y_train_withMean.max(), y_tildZs_withMean.max()]).max()
        print('max_all is ' + str(max_all))
        min_all = np.array([y_test.min(),y_train_withMean.min(), y_tildZs_withMean.min()]).min()
        print('min_all is ' + str(min_all))

        if predicMo:

            max_all = np.array([y_test.max(), y_train_withMean.max(),  mu_star.max()]).max()
            print('max_all with prediction is ' + str(max_all))
            min_all = np.array([y_test.min(),y_train_withMean.min(), mu_star.min()]).min()
            print('min_all with prediction is ' + str(min_all))

            # f = interp2d(X_test[:, 0], X_test[:, 1], y_test,kind="linear")
            # x_coords = np.linspace(-11.7, -3.21, 500)
            # y_coords = np.linspace(-6.2, 3.0, 500)
            # Z = f(x_coords,y_coords)

            # fig = plt.imshow(Z,
            #            extent=[min(x_coords),max(x_coords),min(y_coords),max(y_coords)],
            #            origin="lower")
            # exit(-1)


            ama = importr('akima')
            flds = importr('fields')
            sp = importr('sp')
            numpy2ri.activate() 
            # Plot the the interpolated climate model outputs
            r.png('plotObsMo.png', width = 960,  height= 560)
            r.par(mfrow=r.c(1,2))
            # r.png(output_folder +'SEED'+ str(SEED) + 'TrainObsAndAllTrainMo' + str(numMo) + '.png')
            x_plot =  np.linspace(-11.7, -3.21, 500)
            y_plot = np.linspace(-6.2, 3.0, 500)
            interpolated = ama.interp(X_test[:, 0], X_test[:, 1], y_test, xo=x_plot, yo=y_plot)
            
            
            as_vector = r['as.vector']

            z = np.array(as_vector(interpolated.rx2('z')))

            d = {'a':interpolated.rx2('x'), 'b':interpolated.rx2('y')}
            dataf = ro.DataFrame(d)

            as_matrix = r['as.matrix']
            expand_grid = r['expand.grid']

            xy = as_matrix(expand_grid(dataf))
           
            r.load('france_rcoords.RData')
            france_rcoords = r['france_rcoords']
      
            as_logical = r['as.logical']
          
            is_france = as_logical(sp.point_in_polygon(xy.rx(True,1), xy.rx(True,2), \
                france_rcoords[:,0], france_rcoords[:,1]))

            is_france = np.array(is_france).astype(bool)

            z[~is_france] = np.nan
            z_mos = z
            z1= r.matrix(z, 500, 500)

            interpo_data = np.zeros(500000).reshape(250000, 2)
            interpo_data[:, 0] = np.array(xy.rx(True,1))
            interpo_data[:, 1] = np.array(xy.rx(True,2))

            # moNearObs = []
            # for i in range(len(y_train_withMean)):
            #     idx_min_dist = np.argmin(np.array([np.linalg.norm(X_train[i, :] - interpo_data[j, :]) for j in range(len(z_mos))]))
            #     moNearObs.append(z_mos[idx_min_dist])
            # moNearObs = np.array(moNearObs)

            # plt.figure()
            # plt.scatter(y_train_withMean, moNearObs, color='k')
            # plt.plot(y_train_withMean, y_train_withMean, color='k')
            # plt.xlabel('Observations')
            # plt.ylabel('Model outputs')
            # plt.savefig("obsVsMos.eps")
            # plt.savefig("obsVsMos.png")
            # plt.show()
            # plt.close()


            # moNearObs1 = []
            # for i in range(len(y_train_withMean)):
            #     idx_min_dist = np.argmin(np.array([np.linalg.norm(X_train[i, :] - X_test[j, :]) for j in range(len(y_test))]))
            #     moNearObs1.append(y_test[idx_min_dist])
            # moNearObs1 = np.array(moNearObs1)

            # plt.figure()
            # plt.scatter(y_train_withMean, moNearObs1, color='k')
            # plt.plot(y_train_withMean, y_train_withMean, color='k')
            # plt.xlabel('Observations')
            # plt.ylabel('Model outputs')
            # plt.savefig("obsVsMos1.eps")
            # plt.savefig("obsVsMos1.png") 
            # plt.show()
            # plt.close()
                     
            d1 = {'x':interpolated.rx2('x'), 'y':interpolated.rx2('y'), 'z':z1}
            d2 = ro.ListVector(d1)
           
            minimum = np.array([y_test.min(), y_train_withMean.min(), mu_star.min()]).min()
            print('minimum is ' + str(minimum))
            maximum = np.array([y_test.max(), y_train_withMean.max(), mu_star.max()]).max()
            print('maximum is ' + str(maximum))

            plot_seq = r.pretty(np.arange(6,42), 20)
            jet_colors = r.colorRampPalette(r.c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))
            pal = jet_colors(len(plot_seq) - 1)
            r.image(interpolated.rx2('x'), interpolated.rx2('y'), z1, breaks=plot_seq, col=pal, xlab="Longitude", ylab="Latitude", main='(a)')
        
            as_numeric = r['as.numeric']
            col_pts = pal.rx(as_numeric(r.cut(y_train_withMean, plot_seq)))
            r.points(X_train[:, 0], X_train[:, 1], bg=col_pts, pch=21)
            r.lines(france_rcoords[:,0], france_rcoords[:,1])
            # r.legend('topright', legend=r.c("observations"), pch =21)
           
            # Plot the the interpolated predicted values of model outputs
            # r.png(output_folder +'SEED'+ str(SEED) + 'TrainObsAndAllPredicMo' + str(numMo) + '.png')
            x_plot =  np.linspace(-11.7, -3.21, 500)
            y_plot = np.linspace(-6.2, 3.0, 500)
            interpolated = ama.interp(X_test[:, 0], X_test[:, 1], mu_star, xo=x_plot, yo=y_plot)
            # print(interpolated.rx2('x'))

            z = np.array(as_vector(interpolated.rx2('z')))

            d = {'a':interpolated.rx2('x'), 'b':interpolated.rx2('y')}
            dataf = ro.DataFrame(d)
            xy = as_matrix(expand_grid(dataf))
            print(r.dim(xy))
          
            is_france = as_logical(sp.point_in_polygon(xy.rx(True,1), xy.rx(True,2), \
                france_rcoords[:,0], france_rcoords[:,1]))

            is_france = np.array(is_france).astype(bool)

            z[~is_france] = np.nan
            z_predicMos = z
            z1= r.matrix(z, 500, 500)
            print(r.dim(interpolated.rx2('z')))

            # interpo_xy = np.array(xy)
            # moNearObs2 = []
            # for i in range(len(y_train_withMean)):
            #     idx_min_dist = np.argmin(np.array([np.linalg.norm(X_train[i, :] - interpo_xy[j, :]) for j in range(len(z_predicMos))]))
            #     moNearObs2.append(z_predicMos[idx_min_dist])
            # moNearObs2 = np.array(moNearObs2)

            # plt.figure()
            # plt.scatter(y_train_withMean, moNearObs2, color='k')
            # plt.plot(y_train_withMean, y_train_withMean, color='k')
            # plt.xlabel('Observations')
            # plt.ylabel('Predicted model outputs')
            # plt.savefig("obsVsPreMos.eps")
            # plt.savefig("obsVsPreMos.png") 
            # # plt.show()
            # plt.close()

            d1 = {'x':interpolated.rx2('x'), 'y':interpolated.rx2('y'), 'z':z1}
            d2 = ro.ListVector(d1)
            
            r.par(mar=r.c(5.1, 5.1, 5.1, 3.1))
            jet_colors = r.colorRampPalette(r.c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))
            pal = jet_colors(len(plot_seq) - 1)
            flds.image_plot(d2, breaks=plot_seq, col=pal, xlab="Longitude", ylab="Latitude", main='(b)')
           
            col_pts = pal.rx(as_numeric(r.cut(y_train_withMean, plot_seq)))
            r.points(X_train[:, 0], X_train[:, 1], bg=col_pts, pch=21)
            r.lines(france_rcoords[:,0], france_rcoords[:,1])
            # r.legend('topright', legend=r.c("observations"), pch =21)
            numpy2ri.deactivate()

            r.load('france_rcoords.RData')
            france_rcoords = r['france_rcoords']
            france_rcoords=np.array(france_rcoords)

            cmap = plt.cm.jet
            # define the bins and normalize
            # bounds = np.round(np.linspace(6, 42, 20),0)
            bounds = np.arange(6,43)
            norm0 = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

            # fig = plt.figure()
            # fig.set_rasterized(True)
            # ax = fig.add_subplot(111)
            # ax.set_rasterized(True)
            # im=ax.imshow(np.flipud(np.array(z_mos).reshape((500, 500))), extent=(-11.7, -3.21, -6.2, 3.0), cmap  =cmap, norm = norm0)
            # plt.scatter(X_train[:, 0], X_train[:, 1], c= y_train_withMean, cmap=cmap, norm= norm0, edgecolors='k')
            # plt.plot(france_rcoords[:,0], france_rcoords[:,1], '-', color='k', lw=0.5)
            # plt.xlabel('$Longitude$')
            # plt.ylabel('$Latitude$')
            # plt.colorbar(im)
            # plt.savefig("SEED120TrainObsAndAllTrainMo500_v2.eps",rasterized=True)
            # # plt.show()
            # plt.close()
          


            # Nr = 1
            # Nc = 2
            # fig, ax = plt.subplots(Nr, Nc, constrained_layout= True, sharex=True, sharey=True, figsize=(10,5))
            # fig.set_rasterized(True)
            # axs = ax.flat
            # axs[0].set_rasterized(True)
            # axs[0].imshow(np.flipud(np.array(z_mos).reshape((500, 500))), extent=(-11.7, -3.21, -6.2, 3.0), cmap  =cmap, norm = norm0)
            # ax[0].scatter(X_train[:, 0], X_train[:, 1], c= y_train_withMean, cmap=cmap, norm= norm0, edgecolors='k')
            # ax[0].plot(france_rcoords[:,0], france_rcoords[:,1], '-', color='k', lw=0.5)
            # axs[0].set_xlabel('$Longitude$')
            # axs[0].set_ylabel('$Latitude$')
            # axs[0].set_title('(a)')
            # axs[1].set_rasterized(True)
            # figPredicMos = axs[1].imshow(np.flipud(np.array(z_predicMos).reshape((500, 500))), extent=(-11.7, -3.21, -6.2, 3.0), cmap  =cmap, norm = norm0)
            # ax[1].scatter(X_train[:, 0], X_train[:, 1], c= y_train_withMean, cmap=cmap, norm= norm0, edgecolors='k')
            # ax[1].plot(france_rcoords[:,0], france_rcoords[:,1], '-', color='k', lw=0.5)
            # axs[1].set_xlabel('$Longitude$')
            # axs[1].set_ylabel('$Latitude$')
            # axs[1].set_title('(b)')
            # plt.colorbar(figPredicMos, ax=ax.ravel().tolist(), shrink=0.80)
            # plt.savefig('figObsMos1.eps',rasterized=True)
            # plt.show()
            # plt.close()


            Nr = 1
            Nc = 3
            fig, (ax1, ax2,cax) = plt.subplots(Nr, Nc, figsize=(12,5), gridspec_kw={"width_ratios":[1, 1, 0.06]})
            fig.set_rasterized(True)
            ax1.set_rasterized(True)
            ax2.set_rasterized(True)

            im1=ax1.imshow(np.flipud(np.array(z_mos).reshape((500, 500))), extent=(-11.7, -3.21, -6.2, 3.0), cmap  =cmap, norm = norm0, aspect = "auto")
            ax1.scatter(X_train[:, 0], X_train[:, 1], c= y_train_withMean, cmap=cmap, norm= norm0, edgecolors='k')
            ax1.plot(france_rcoords[:,0], france_rcoords[:,1], '-', color='k', lw=0.5)
            ax1.set_xlabel('$Longitude$')
            ax1.set_ylabel('$Latitude$')
            # ax1.axis('off')
            ax1.set_title('(a)')

            figPredicMos = ax2.imshow(np.flipud(np.array(z_predicMos).reshape((500, 500))), extent=(-11.7, -3.21, -6.2, 3.0), cmap  =cmap, norm = norm0, aspect = "auto")
            ax2.scatter(X_train[:, 0], X_train[:, 1], c= y_train_withMean, cmap=cmap, norm= norm0, edgecolors='k')
            ax2.plot(france_rcoords[:,0], france_rcoords[:,1], '-', color='k', lw=0.5)
            ax2.set_xlabel('$Longitude$')
            ax2.set_ylabel('$Latitude$')
            # ax2.axis('off')
            ax2.set_title('(b)')

            # plt.colorbar(im1, ax=axs, orientation = 'horizontal')
            plt.colorbar(figPredicMos, cax= cax, ax=[ax1,ax2], use_gridspec = True)
            plt.subplots_adjust(wspace=0.2, hspace=0)
            plt.savefig('figObsMos.eps', rasterized=True)
            # plt.show()
            plt.close()




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
    p.add_argument('-predicMo', dest='predicMo', default=True,  type=lambda x: (str(x).lower() == 'true'),  help='flag for whether to predict the value where model outputs are produced')
    p.add_argument('-poly_deg', type=int, dest='poly_deg', default=2, help='degree of the polynomial function of the additive model bias')
    p.add_argument('-indivError', dest='indivError', default=True,  type=lambda x: (str(x).lower() == 'true'),  help='flag for whether to individual std errors for GP diagnosis')
    p.add_argument('-index_Xaxis', dest='index_Xaxis', default=False,  type=lambda x: (str(x).lower() == 'true'),  help='flag for whether X-axis is index for GP diagnosis')

    args = p.parse_args()
    if args.useSimData: 
        # input_folder = os.getcwd() + '/dataRsimGammaTransformErrorInZtilde/numObs_200_numMo_' + str(args.numMo) + '/seed' + str(args.SEED) + '/'
        # input_folder = os.getcwd() + '/dataRsimNoFrGammaTransformArealRes25Cods100butArealZs100/numObs_200_numMo_' + str(args.numMo) + '/seed' + str(args.SEED) + '/'
        input_folder = os.getcwd() + '/dataSimulated/numObs_200_numMo_' + str(args.numMo) + '/seed' + str(args.SEED) + '/'
    else:
        input_folder = os.getcwd() + '/Data/FPstart2016020612_FR_numObs_128_numMo_' + str(args.numMo) + '/seed' + str(args.SEED) + '/'

    X_hatZs_in = open(input_folder + 'X_hatZs.pkl', 'rb')
    X_hatZs = pickle.load(X_hatZs_in) 

    elev_fp_in = open(input_folder + 'elev_fp.pkl', 'rb')
    elev_fp = pickle.load(elev_fp_in) 
 
    y_hatZs_in = open(input_folder + 'y_hatZs.pkl', 'rb')
    y_hatZs = pickle.load(y_hatZs_in) 

    X_tildZs_in = open(input_folder + 'X_tildZs.pkl', 'rb')
    X_tildZs = pickle.load(X_tildZs_in) 
    print(X_tildZs.shape)
 

    y_tildZs_in = open(input_folder + 'y_tildZs.pkl', 'rb')
    y_tildZs = pickle.load(y_tildZs_in)

    input_folder = input_folder + 'poly_deg' + str(args.poly_deg) + '/'

    if args.useSimData:
        resOptim_in = open(input_folder + 'resOptimSim.pkl', 'rb')
    else:
        resOptim_in = open(input_folder + 'resOptim.pkl', 'rb')
    
    resOptim = pickle.load(resOptim_in)

    mu = resOptim['mu']

    print('theta from optimisation is ' + str(mu)) 
    print(np.exp(mu[:-(4 + args.poly_deg -1)]))

    # when looking at the coefficients of the betas of a_bias, need to check its relative importance: beta/std(beta), not only Beta itself.
    cov = resOptim['cov']
    print('var of theta is' + str(np.diag(cov)))

    relative_importance_of_betas = mu[-(4 + args.poly_deg -1 -1):]/np.sqrt(np.diag(cov)[-(4 + args.poly_deg -1 -1):])
    print('relative_importance_of_betas is ' + str(relative_importance_of_betas))
   
   
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
            all_X_Zs_temp_in = open(input_folder + 'all_X_Zs.pickle', 'rb')
            all_X_Zs_temp = pickle.load(all_X_Zs_temp_in) 

            all_X_Zs = all_X_Zs_temp[:, :2]
            all_elev_fp  = all_X_Zs_temp[:, 2]

            all_y_Zs_in = open(input_folder + 'all_y_Zs.pickle', 'rb')
            all_y_Zs = pickle.load(all_y_Zs_in) 

            X_test = all_X_Zs
            y_test = all_y_Zs
            # elev_fp = all_elev_fp
            print('shape of X_test, y_test when predicMo is true ' + str((X_test.shape,y_test.shape)))
            exit(-1)

        else:
            X_train = X_hatZs[:-28, :]
            X_test = X_hatZs[-28:, :]
            y_train = y_hatZs[:-28]
            y_test = y_hatZs[-28:]
 
        print('shape of X_test, y_test'+ str((X_test.shape, y_test.shape)))

    predic_accuracy = predic_gpRegression(mu, X_train, y_train, X_test, y_test, X_tildZs, y_tildZs, elev_fp, args.crossValFlag, args.SEED, args.numMo, \
        args.useSimData, args.grid, args.predicMo, args.poly_deg, args.indivError, args.index_Xaxis)
         
