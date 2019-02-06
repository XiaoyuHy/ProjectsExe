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
    K = sigma * np.exp(- 0.5 * square_dist) + np.diag(np.repeat(OMEGA, n))
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

def compute_L_chol(cov):
    l_chol = np.linalg.cholesky(cov)
    computeN3Cost.num_calls_n3 += 1
    return l_chol

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

def predic_gpRegression(theta, X_train, y_train, X_test, y_test, X_tildZs, y_tildZs, crossValFlag = False,  SEED=None, numMo = None, useSimData =False, grid= False, predicMo = False, \
    gp_deltas_modelOut = True, withPrior= False, a_bias_poly_deg = 2, rbf = True, OMEGA = 1e-6):
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


    # print(np.exp(log_sigma_Zs), np.exp(log_phi_Zs),  np.exp(log_obs_noi_scale))

    # print(np.round(np.exp(log_sigma_Zs), 1), np.round(np.exp(log_phi_Zs), 1),  np.round(np.exp(log_obs_noi_scale),1))

    C_hatZs = cov_matrix_reg(X = X_train, sigma = np.exp(log_sigma_Zs), w = np.exp(log_phi_Zs), \
     obs_noi_scale = np.exp(log_obs_noi_scale))

    # C_hatZs = cov_matrix_reg(X = X_train, sigma = 1.1143612, w = np.array( [0.82821735]), \
    #  obs_noi_scale = np.array([0.1067566]))

    # C_hatZs = cov_matrix_reg(X = X_train, sigma = 1.0834896224616308, w = np.array( [0.80397927]), \
    #  obs_noi_scale = np.array([0.10559963]))

    # C_hatZs = cov_matrix_reg(X = X_train, sigma = 1.1, w = np.array( [0.8]), \
    #  obs_noi_scale = np.array([0.1]))

    # C_hatZs = cov_matrix_reg(X = X_train, sigma = np.round(np.exp(log_sigma_Zs), 1), w = np.round(np.exp(log_phi_Zs), 1), \
    #  obs_noi_scale = np.round(np.exp(log_obs_noi_scale),1))
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

    X_tildZs_mean = np.array([np.mean(X_tildZs[i], axis=0) for i in range(len(y_tildZs))])
    n_row = X_tildZs_mean.shape[0]
    tmp0 = np.repeat(1.,n_row).reshape(n_row,1)
    X_tildZs_mean_extend = np.hstack((X_tildZs_mean, tmp0))
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
    
    if predicMo:
        meanPredic_out = open(output_folder + 'meanPredic_outSample.pkl', 'wb')
        pickle.dump(mu_star, meanPredic_out) 
        meanPredic_out.close()


        # exit(-1)

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

    if not useSimData:
        input_folder = os.getcwd() + '/DataImogenFrGridMoNotCentre/FPstart2016020612_FR_numObs_128/seed' + str(SEED) + '/'
        mean_y_hatZs_in = open(input_folder + 'mean.pickle', 'rb')
        mean_y_hatZs = pickle.load(mean_y_hatZs_in) 

    if not crossValFlag:
        index = np.arange(len(y_test))
        if useSimData:
            standardised_y_estimate = (mu_star - y_test)
        else:
            standardised_y_estimate = (mu_star - y_test)/np.exp(log_obs_noi_scale)
        std_yEst_out = open(output_folder + 'std_yEst_outSample.pkl', 'wb')
        pickle.dump(standardised_y_estimate, std_yEst_out)
        plt.figure()
        # plt.scatter(index, standardised_y_estimate, facecolors='none', edgecolors='k', linewidths=1.2)
        plt.scatter(mu_star + mean_y_hatZs, standardised_y_estimate, facecolors='none', edgecolors='k', linewidths=1.2)
        # plt.scatter(index, standardised_y_etstimate, c='k')
        plt.axhline(0, color='black', lw=1.2, ls ='-')
        plt.axhline(2, color='black', lw=1.2, ls =':')
        plt.axhline(-2, color='black', lw=1.2, ls =':')
        plt.xlabel('Predictions')
        plt.ylabel('Standardised residual')
        plt.savefig(output_folder + 'SEED'+ str(SEED) +'stdPredicErr_outSample.png')
        plt.show()
        plt.close()

        sm.qqplot(standardised_y_estimate, line='45')
        plt.savefig(output_folder + 'SEED'+ str(SEED) + 'normalQQ_outSample.png')
        plt.show()
        plt.close()

    
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

    if useSimData:
        upper_interv_predic = mu_star + 2 * np.sqrt(vstar)
        lower_interv_predic = mu_star - 2 * np.sqrt(vstar)
    else:
        upper_interv_predic = mu_star + 2 * np.sqrt(vstar + np.exp(log_obs_noi_scale)**2)
        lower_interv_predic = mu_star - 2 * np.sqrt(vstar + np.exp(log_obs_noi_scale)**2)
    
    if not useSimData:
        # input_folder = os.getcwd() + '/DataImogenFrGridMoNotCentre/FPstart2016020612_FR_numObs_128/seed' + str(SEED) + '/'
        # mean_y_hatZs_in = open(input_folder + 'mean.pickle', 'rb')
        # mean_y_hatZs = pickle.load(mean_y_hatZs_in) 

        y_test = y_test + mean_y_hatZs
        y_train_withMean = y_train + mean_y_hatZs
        y_tildZs_withMean = y_tildZs + mean_y_hatZs
        mu_star = mu_star + mean_y_hatZs
        upper_interv_predic = upper_interv_predic + mean_y_hatZs
        lower_interv_predic = lower_interv_predic + mean_y_hatZs

        # print('maximum of y_test is ' + str(y_test.max()))
        # print('maximum of y_tildZs is ' + str(y_tildZs.max()))
        # print('minimum of y_test is ' + str(y_test.min()))
        # print('minimum of y_tildZs is ' + str(y_tildZs.min()))

        max_all = np.array([y_test.max(), y_train_withMean.max(), y_tildZs_withMean.max()]).max()
        print('max_all is ' + str(max_all))
        min_all = np.array([y_test.min(),y_train_withMean.min(), y_tildZs_withMean.min()]).min()
        print('min_all is ' + str(min_all))

        if predicMo:

            max_all = np.array([y_test.max(), y_train_withMean.max(),  mu_star.max()]).max()
            print('max_all with prediction is ' + str(max_all))
            min_all = np.array([y_test.min(),y_train_withMean.min(), mu_star.min()]).min()
            print('min_all with prediction is ' + str(min_all))
            # X_mo = np.array(list(chain.from_iterable(X_tildZs)))

            # plt.figure()
            # plt.scatter(X_mo[:, 0], X_mo[:, 1], c= y_test, cmap=plt.cm.jet, vmin=min_all, vmax=max_all, marker = '^', s = 300, label = "Mo")
            # plt.scatter(X_train[:, 0], X_train[:, 1], c= y_train_withMean, cmap=plt.cm.jet, vmin=min_all, vmax=max_all, label = "Obs", edgecolors ='black')
            # plt.colorbar()
            # plt.legend(loc='best')
            # plt.savefig(output_folder + 'SEED'+ str(SEED) + 'TrainObsAndAllTrainMo' + str(numMo) + '.png')
            # plt.show()
            # plt.close()

            # plt.figure()
            # plt.scatter(X_mo[:, 0], X_mo[:, 1], c= mu_star, cmap=plt.cm.jet, vmin=min_all, vmax=max_all,  marker = '^', s = 300, label = "Mo")
            # plt.scatter(X_train[:, 0], X_train[:, 1], c= y_train_withMean, cmap=plt.cm.jet, vmin=min_all, vmax=max_all, label = "Obs", edgecolors = 'black')
            # plt.colorbar()
            # plt.legend(loc='best')
            # plt.savefig(output_folder + 'SEED'+ str(SEED) + 'TrainObsAndAllPredicMo' + str(numMo) + '.png')
            # plt.show()
            # plt.close()

            # im = plt.imshow(np.flipud(mu_star.reshape((point_res,point_res))), extent=(lower_bound[0], upper_bound[0],lower_bound[1], upper_bound[1]), cmap =plt.matplotlib.cm.jet)
            # plt.scatter(X_hatZs[:,0], X_hatZs[:,1], s=12, c='k', marker = 'o')
            ama = importr('akima')
            flds = importr('fields')
            sp = importr('sp')
            numpy2ri.activate() 
            # Plot the the interpolated climate model outputs
            r.png(output_folder +'SEED'+ str(SEED) + 'TrainObsAndAllTrainMo' + str(numMo) + '.png')
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
            numpy2ri.activate() 
            z1= r.matrix(z, 500, 500)
          

            d1 = {'x':interpolated.rx2('x'), 'y':interpolated.rx2('y'), 'z':z1}
            d2 = ro.ListVector(d1)
           
            minimum = np.array([y_test.min(), y_train_withMean.min(), mu_star.min()]).min()
            print('minimum is ' + str(minimum))
            maximum = np.array([y_test.max(), y_train_withMean.max(), mu_star.max()]).max()
            print('maximum is ' + str(maximum))

            plot_seq = r.pretty(np.arange(6,42), 20)
            jet_colors = r.colorRampPalette(r.c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))
            pal = jet_colors(len(plot_seq) - 1)
            flds.image_plot(d2, breaks=plot_seq, col=pal, xlab="Longitude", ylab="Latitude", main='Observations & training model outputs')
           
            as_numeric = r['as.numeric']
            col_pts = pal.rx(as_numeric(r.cut(y_train_withMean, plot_seq)))
            r.points(X_train[:, 0], X_train[:, 1], bg=col_pts, pch=21)
            r.legend('topright', legend=r.c("observations"), pch =21)
           
            # Plot the the interpolated predicted values of model outputs
            r.png(output_folder +'SEED'+ str(SEED) + 'TrainObsAndAllPredicMo' + str(numMo) + '.png')
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
            numpy2ri.activate() 
            z1= r.matrix(z, 500, 500)
            print(r.dim(interpolated.rx2('z')))

            d1 = {'x':interpolated.rx2('x'), 'y':interpolated.rx2('y'), 'z':z1}
            d2 = ro.ListVector(d1)
            

            jet_colors = r.colorRampPalette(r.c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))
            pal = jet_colors(len(plot_seq) - 1)
            flds.image_plot(d2, breaks=plot_seq, col=pal, xlab="Longitude", ylab="Latitude", main='Observations & predicted model outputs')
           
            col_pts = pal.rx(as_numeric(r.cut(y_train_withMean, plot_seq)))
            r.points(X_train[:, 0], X_train[:, 1], bg=col_pts, pch=21)
            r.legend('topright', legend=r.c("observations"), pch =21)

            numpy2ri.deactivate()
            exit(-1)

            # plt.figure()
            # plt.scatter(X_test[:, 0], X_test[:, 1], c= y_test, cmap=plt.cm.jet, vmin=min_all, vmax=max_all, marker ='s', s=2)
            # plt.scatter(X_train[:, 0], X_train[:, 1], c= y_train_withMean, cmap=plt.cm.jet, vmin=min_all, vmax=max_all, label = "Obs", edgecolors ='black')
            # plt.colorbar()
            # plt.legend(loc='best')
            # plt.savefig(output_folder + 'SEED'+ str(SEED) + 'TrainObsAndAllTrainMo' + str(numMo) + '.png')
            # plt.show()
            # plt.close()

            # plt.figure()
            # plt.scatter(X_test[:, 0], X_test[:, 1], c= mu_star, cmap=plt.cm.jet, vmin=min_all, vmax=max_all, marker ='s', s=2)
            # plt.scatter(X_train[:, 0], X_train[:, 1], c= y_train_withMean, cmap=plt.cm.jet, vmin=min_all, vmax=max_all, label = "Obs", edgecolors = 'black')
            # plt.colorbar()
            # plt.legend(loc='best')
            # plt.savefig(output_folder + 'SEED'+ str(SEED) + 'TrainObsAndAllPredicMo' + str(numMo) + '.png')
            # plt.show()
            # plt.close()
            # exit(-1)
        # exit(-1)
        # X_mo = np.array(list(chain.from_iterable(X_tildZs)))
        # plt.figure()
        # plt.scatter(X_mo[:, 0], X_mo[:, 1],  c = y_tildZs, cmap=plt.cm.jet, vmin=min_all, vmax=max_all, marker ='s', s=2)
        # plt.show()
        # exit(-1)

        plt.figure()
        plt.plot(y_test, y_test, ls='-', color = 'r')
        plt.scatter(y_test, mu_star, color='black', label='predic_mean')
        plt.scatter(y_test, upper_interv_predic, color = 'blue', label = 'predic_upper_CI', marker = '^')
        plt.scatter(y_test, lower_interv_predic, color ='green', label = 'predic_lower_CI', marker = 'v')
        plt.xlabel('Observations')
        plt.ylabel('Predictions')
        plt.legend(loc='best')
        plt.title('Out of sample prediction')
        plt.savefig(output_folder + 'BM_predic_scatter_seed' + str(SEED)+ 'numMo' + str(numMo) + 'mean.png')
        plt.show()
        plt.close()


        residuals = y_test - mu_star

        max_coords = X_test[np.argsort(np.abs(residuals))[-3:], :]
        print('max_coords of residuals is ' + str(max_coords))

        maxAbs = np.array([np.abs(residuals.min()), np.abs(residuals.max())]).max()

        fig, ax = plt.subplots()
        # cmap = mpl.colors.ListedColormap(['red', 'green', 'orange' ,'blue',  'cyan', 'white'])
        cmap = mpl.colors.ListedColormap(["#00007F", "blue",'cyan', 'white', 'green', "red", "#7F0000"])
        # cmap.set_over('0.25')
        # cmap.set_under('0.75')
        cmap.set_under("crimson")
        cmap.set_over('black')

        residualsPlot = ax.scatter(X_test[:, 0], X_test[:, 1], c= y_test - mu_star, cmap=cmap, vmin = -maxAbs, vmax = maxAbs)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Residuals of out-of-sample prediction')
        plt.colorbar(residualsPlot, ax=ax)
        plt.savefig(output_folder + 'Residuals_seed' + str(SEED) + 'numMo' + str(numMo) + 'outSample.png')
        plt.show()
        plt.close()
       
        numpy2ri.activate() 
        plot_seq = r.pretty(np.arange(6,42), 20)
        jet_colors = r.colorRampPalette(r.c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))
        pal = jet_colors(len(plot_seq) - 1)
        pal = np.array(pal)
        numpy2ri.deactivate()
        cmap = mpl.colors.ListedColormap(list(pal))
        
        fig, ax = plt.subplots()
        # testObs = ax.scatter(X_test[:, 0], X_test[:, 1], c= y_test, cmap=plt.cm.jet, vmin=min_all, vmax=max_all)
        testObs = ax.scatter(X_test[:, 0], X_test[:, 1], c= y_test, cmap=cmap, vmin=min_all, vmax=max_all)
        # plt.colorbar()
        for i, txt in enumerate(np.round(y_test, 1)):
            ax.annotate(txt, (X_test[:, 0][i], X_test[:, 1][i]))
        plt.colorbar(testObs, ax = ax)
        plt.title('Out-of-sample test observations')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.savefig(output_folder + 'SEED'+ str(SEED) + 'TestObs.png')
        plt.show()
        plt.close()

        #10/01/2019: creat cell coordinated of resoluton 10 by 10
        # cell_res = 10
        # cell_coords = np.zeros(2*cell_res**2).reshape(cell_res**2,2)
        # cell_len = 0.04
        # cell_width = 0.04
        # x_coords = np.zeros(cell_res)
        # y_coords = np.zeros(cell_res)
        # for i in range(cell_res):
        #     x_coords[i] = cell_width/(2*cell_res) + i * cell_width/cell_res
        #     y_coords[i] = cell_len/(2*cell_res) + i * cell_width/cell_res

        # for i in range(cell_res):
        #     cell_coords[i*10:(i*10+ cell_res), 0] = np.repeat(x_coords[i],cell_res)
        #     cell_coords[i*10:(i*10+ cell_res), 1] = y_coords
            
        # tmp = X_tildZs - cell_coords 
        
        # # get the center coordinates of X_tildZs
        # centre_MoCoords = np.array([tmp[i][0] for i in range(len(X_tildZs))])
        # X_mo  = centre_MoCoords
        
        # # X_mo = np.array(list(chain.from_iterable(X_tildZs))) # This line of code is for the case where only one point for X_tildZs

        # legend_elements = [Line2D([0], [0], marker='o', color='w', label='Observations',markerfacecolor=None,markeredgecolor='k', markersize=8),\
        #     Line2D([0], [0], marker='s',color='w', markerfacecolor=None,markeredgecolor='k', markersize=8, label='Test points')] 

        # fig, ax = plt.subplots()
        # maxYtest = y_test[np.argsort(np.abs(residuals))[-3:]]
        # maxPlot = ax.scatter(max_coords[:, 0], max_coords[:, 1], c= maxYtest, cmap=cmap, vmin=min_all, vmax=max_all, marker = 's')
        # # for i, txt in enumerate(np.round(maxYtest, 1)):
        # #     ax.annotate(txt, (max_coords[:, 0][i], max_coords[:, 1][i]))
        # trainObs = ax.scatter(X_train[:, 0], X_train[:, 1], c= y_train, cmap=cmap, vmin=min_all, vmax=max_all)
        # # for i, txt in enumerate(np.round(y_train, 1)):
        # #     ax.annotate(txt, (X_train[:, 0][i], X_train[:, 1][i]))
        # # plt.colorbar(maxPlot, ax = ax)
        # # plt.savefig('SEED'+ str(SEED) + 'TrainObsAndTestMaxObs.png')
        # plt.colorbar(trainObs, ax = ax)
        # # plt.title('Observations & test points')
        # plt.xlabel('Longitude')
        # plt.ylabel('Latitude')
        # # plt.legend(loc='best')
        # ax.legend(handles=legend_elements, loc='best')

        # plt.savefig(output_folder + 'SEED'+ str(SEED) + 'TrainObs.png')
        # plt.show()
        # plt.close()

        # legend_elements = [Line2D([0], [0], marker='o', color='w', label='Model outputs',markerfacecolor=None,markeredgecolor='k', markersize=8), \
        #     Line2D([0], [0], marker='s',color='w', markerfacecolor=None,markeredgecolor='k', markersize=8, label='Test points')] 

        # fig, ax = plt.subplots()
        # maxYtest = y_test[np.argsort(np.abs(residuals))[-3:]]
        # maxPlot = ax.scatter(max_coords[:, 0], max_coords[:, 1], c= maxYtest, cmap=cmap, vmin=min_all, vmax=max_all, marker = 's')
        # # for i, txt in enumerate(np.round(maxYtest, 1)):
        # #     ax.annotate(txt, (max_coords[:, 0][i], max_coords[:, 1][i]))

        # modelOutputs = ax.scatter(X_mo[:, 0], X_mo[:, 1],  c = y_tildZs, cmap=cmap, vmin=min_all, vmax=max_all)
        # # for i, txt in enumerate(np.round(y_tildZs, 1)):
        # #     ax.annotate(txt, (X_mo[:, 0][i], X_mo[:, 1][i]))

        # # plt.colorbar(maxPlot, ax = ax)
        # # plt.savefig('SEED'+ str(SEED) + 'Mo' + str(numMo) + 'andTestObsMax.png')
        # plt.colorbar(modelOutputs, ax = ax)
        # # plt.title('Model outputs & test points')
        # plt.xlabel('Longitude')  
        # plt.ylabel('Latitude')
        # ax.legend(handles=legend_elements, loc='best')
        # plt.savefig(output_folder + 'SEED'+ str(SEED) + 'Mo' + str(numMo) + '.png')
        # plt.show()
        # plt.close()

        # fig, ax = plt.subplots()
        # maxPredic = mu_star[np.argsort(y_test)[-3:]]
        # maxPlot = ax.scatter(max_coords[:, 0], max_coords[:, 1], c = maxPredic, cmap=plt.cm.jet, vmin=min_all, vmax=max_all, marker = 'x', s =150)
        # for i, txt in enumerate(np.round(maxPredic, 1)):
        #     ax.annotate(txt, (max_coords[:, 0][i], max_coords[:, 1][i]))

        # ax.scatter(X_mo[:, 0], X_mo[:, 1],  c = y_tildZs, cmap=plt.cm.jet, vmin=min_all, vmax=max_all)
        # for i, txt in enumerate(np.round(y_tildZs, 1)):
        #     ax.annotate(txt, (X_mo[:, 0][i], X_mo[:, 1][i]))
        # plt.colorbar(maxPlot, ax = ax)
        # plt.savefig(output_folder + 'SEED'+ str(SEED) + 'Mo' + str(numMo) + 'andPredicMax.png')
        # plt.show()
        # plt.close()


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


    # lower_bound = np.array([-10, -6])
    # upper_bound = np.array([-4, 2])
    lower_bound = np.array([-12., -6.5])
    upper_bound = np.array([-3., 3.])
    point_res = 100
    # print 'len of mu_star is ' + str(len(mu_star))
    # x1, x2 = np.meshgrid(np.linspace(lower_bound[0], upper_bound[0], point_res),  
    #                      np.linspace(lower_bound[1], upper_bound[1], point_res))
    # x1_vec = x1.ravel()
    # x2_vec = x2.ravel()
    # X_plot = np.vstack((x1_vec, x2_vec)).T

    # nplot = X_plot.shape[0]
    # K_star_star = np.zeros((nplot,1))
    # K_star_hatZs = cov_mat_xy(X_train, X_plot, np.exp(log_sigma_Zs), np.exp(log_phi_Zs)) # is a matrix of size (n_train, n_plot)
    # K_star_hatZs = K_star_hatZs.T
    # _, avg_pointAreal_upper, _, _ = point_areal(X_plot, X_tildZs, log_sigma_Zs, log_phi_Zs, b)
    # K_star_tildZs =avg_pointAreal_upper

    # K_star = np.hstack((K_star_hatZs, K_star_tildZs))
    # mu_plot = np.dot(K_star, u)

    # fig = plt.figure()
    # ax = Axes3D(fig)
    # scat = ax.scatter(x1_vec, x2_vec, mu_plot, c=mu_plot, cmap='viridis', linewidth=0.5)
    # ax.set_xlabel('$lon$')
    # ax.set_ylabel('$lat$')
    # ax.set_zlabel('$Z(s)$')
    # fig.colorbar(scat, shrink=0.85)
    # plt.savefig(output_folder + 'SEED'+ str(SEED) + 'BM_predic_scat.png')
    # plt.close()

    # plt.figure()
    # im = plt.imshow(np.flipud(mu_star.reshape((point_res,point_res))), extent=(lower_bound[0], upper_bound[0],lower_bound[1], upper_bound[1]), cmap =plt.matplotlib.cm.jet)
    # # plt.scatter(X_hatZs[:,0], X_hatZs[:,1], s=12, c='k', marker = 'o')
    # print (mu_star.min(), mu_star.max())
    # cb=plt.colorbar(im)
    # cb.set_label('${Z(s)}$')
    # # plt.title('min = %.2f , max = %.2f , avg = %.2f' % (mu_plot.min(), mu_plot.max(), mu_plot.mean()))
    # plt.xlabel('$lon$')
    # plt.ylabel('$lat$')
    # plt.title('Prediction of BM')
    # plt.grid()
    # plt.savefig(output_folder + 'SEED'+ str(SEED) + 'BM_predic_2D.png')
    # plt.show()
    # plt.close()


    # fig = plt.figure()
    # ax = Axes3D(fig)
    # surf = ax.plot_surface(x1, x2, mu_plot.reshape(point_res, point_res), rstride=1, cstride=1, cmap='viridis')
    # ax.set_xlabel('$lon$')
    # ax.set_ylabel('$lat$')
    # ax.set_zlabel('$Z(s)$')
    # fig.colorbar(surf, shrink=0.85)
    # ax = plt.gca()
    # print('ylim is ' + str(ax.get_ylim()))
    # print('xlim is ' + str(ax.get_xlim()))
    # print('zlim is ' + str(ax.get_zlim()))
    # plt.savefig(output_folder + 'SEED'+ str(SEED) + 'BM_predic_surf.png')
    # plt.close()

    #*******************************comupute the prediction part for in-sample ntrain test data points under each theta **********************************************************
    X_test = X_train
    y_test = y_train

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
    print('length of y_test, mu_star' + str((len(y_test), len(mu_star))))


    rmse = np.sqrt(np.mean((y_test - mu_star)**2))

    print('In-sample RMSE for seed' + str(SEED) + ' is :' + str(rmse))

    rmse_out = open(output_folder + 'rmse_inSample.pkl', 'wb')
    pickle.dump(rmse, rmse_out) 
    rmse_out.close()

    if not crossValFlag:
        index = np.arange(len(y_test))
        standardised_y_estimate = (mu_star - y_test)/(np.exp(log_obs_noi_scale) + 1e-8)
        std_yEst_out = open(output_folder + 'std_yEst_inSample.pkl', 'wb')
        pickle.dump(standardised_y_estimate, std_yEst_out)
        plt.figure()
        # plt.scatter(index, standardised_y_estimate, facecolors='none', edgecolors='k', linewidths=1.2)
        plt.scatter(mu_star + mean_y_hatZs, standardised_y_estimate, facecolors='none', edgecolors='k', linewidths=1.2)
        # plt.scatter(index, standardised_y_etstimate, c='k')
        plt.axhline(0, color='black', lw=1.2, ls ='-')
        plt.axhline(2, color='black', lw=1.2, ls =':')
        plt.axhline(-2, color='black', lw=1.2, ls =':')
        plt.xlabel('Predictions')
        plt.ylabel('Standardised residual')
        plt.savefig(output_folder + 'SEED'+ str(SEED) +'stdPredicErr_inSample.png')
        plt.show()
        plt.close()

        sm.qqplot(standardised_y_estimate, line='45')
        plt.savefig(output_folder + 'SEED'+ str(SEED) + 'normalQQ_inSample.png')
        plt.show()
        plt.close()

        residuals = y_test - mu_star

        max_coords = X_test[np.argsort(np.abs(residuals))[-3:], :]
        print('max_coords of residuals is ' + str(max_coords))

        maxAbs = np.array([np.abs(residuals.min()), np.abs(residuals.max())]).max()

        fig, ax = plt.subplots()
        cmap = mpl.colors.ListedColormap(["#00007F", "blue",'cyan', 'white', 'green', "red", "#7F0000"])
        cmap.set_under("crimson")
        cmap.set_over('black')

        residualsPlot = ax.scatter(X_test[:, 0], X_test[:, 1], c= y_test - mu_star, cmap=cmap, vmin = -maxAbs, vmax = maxAbs)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Residuals of in-sample prediction')
        plt.colorbar(residualsPlot, ax=ax)
        plt.savefig(output_folder + 'Residuals_seed' + str(SEED) + 'numMo' + str(numMo) + 'insample.png')
        plt.show()
        plt.close()
    
    LKstar = linalg.solve_triangular(l_chol_C, K_star.T, lower = True)
    for i in range(ntest):
        K_star_star[i] = cov_matrix(X_test[i].reshape(1, 2), np.exp(log_sigma_Zs), np.exp(log_phi_Zs))
    
    vstar = K_star_star - np.sum(LKstar**2, axis=0).reshape(ntest,1) 
    vstar[vstar < 0] = 1e-9
    vstar = vstar.reshape(ntest, )
    # print('In-sample estimated variance is ' + str(vstar))

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
    
    args = p.parse_args()
    if args.useSimData: 
        # input_folder = os.getcwd() + '/dataRsimGammaTransformErrorInZtilde/numObs_200_numMo_' + str(args.numMo) + '/seed' + str(args.SEED) + '/'
        input_folder = os.getcwd() + '/dataRsimNoFrGammaTransformArealRes25Cods100butArealZs100/numObs_200_numMo_' + str(args.numMo) + '/seed' + str(args.SEED) + '/'
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
    print(np.exp(mu[:-4]))
   
    # tmp = list(np.log(np.array([20.20773792,  1.28349713  ,3.83773256, 1.47482164,  0.10713728]))) + [0.73782754, -0.44369921,  0.64953645, -2.61332868]
    # # tmp = list(np.log(np.array([22.14058558,  1.47980593,  3.33445096,  8.72246743,  0.07932848]))) + [0.87132649, -0.81637815,  0.17456605, -5.54637298]
    # mu = np.array(tmp)
   
    if args.useSimData:
        X_train = X_hatZs
        y_train = y_hatZs

        # input_folder = os.getcwd() + '/dataRsimGammaTransformErrorInZtilde/seed' + str(args.SEED) + '/'
        input_folder = os.getcwd() + '/dataRsimNoFrGammaTransformArealRes25Cods100butArealZs100/seed' + str(args.SEED) + '/'
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

            input_folder = os.getcwd() + '/DataImogenFrGridMoNotCentre/FPstart2016020612_FR_numObs_128/seed' + str(args.SEED) + '/'
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
 
        print(X_test.shape, y_test.shape)

    predic_accuracy = predic_gpRegression(mu, X_train, y_train, X_test, y_test, X_tildZs, y_tildZs, args.crossValFlag, args.SEED, args.numMo, args.useSimData, args.grid, args.predicMo)
         
