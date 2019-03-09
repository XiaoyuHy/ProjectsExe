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
import scipy.stats as stats
import gpValid_parUncerty
import gpValid_parUncertyOverSeeds


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
			output_folder = 'Data/FPstart2016020612_FR_numObs_128_numMo_' + str(numMo) + '/seed' + str(SEED) + '/predicMo'
		else:
			output_folder = 'DataImogenFrGridMoNotCentre/FPstart2016020612_FR_numObs_128_numMo_' + str(numMo) + '/seed' + str(SEED) 
			# output_folder = 'Data/FPstart2016020612_FR_numObs_128_numMo_' + str(numMo) + '/seed' + str(SEED) 
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

	if not useSimData:
		# input_folder = os.getcwd() + '/DataImogenFrGridMoNotCentre/FPstart2016020612_FR_numObs_128/seed' + str(SEED) + '/'
		input_folder = os.getcwd() + '/Data/FPstart2016020612_FR_numObs_128/seed' + str(SEED) + '/'
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
		plt.savefig(output_folder + 'SEED'+ str(SEED) +'OutSamp_indivErr' + str(indivError) + 'Idx' + str(index_Xaxis) + '.png')
		plt.show()
		plt.close()

		lower_chol =  np.linalg.cholesky(cov_of_predic)
		num_Outputs =  cov_of_predic.shape[0]
		samples_Zs = []
		for i in range(1000):
			tmp = np.dot(lower_chol, np.random.normal(0., 1., num_Outputs))
			tmp = np.dot(inv_G, tmp)
			tmp = np.sort(tmp)
			samples_Zs.append(tmp)
		samples_Zs = np.array(samples_Zs)
		print('shape of samples_Zs is ' + str(samples_Zs.shape))
		Lqunatile = np.quantile(samples_Zs, 0.025, axis=0)
		Uqunatile = np.quantile(samples_Zs, 0.975, axis=0)
		print(Lqunatile, Uqunatile)

		std_norm_quantile = np.array([stats.norm.ppf((i-0.5)/num_Outputs) for i in range(1, num_Outputs+1)])

		plt.figure
		sm.qqplot(standardised_y_estimate, line='45')
		plt.savefig(output_folder + 'SEED'+ str(SEED) +'OutSampQQ_indivErr' + str(indivError) + 'Idx' + str(index_Xaxis) + 'NoCI.png')
		plt.show()
		plt.close()

		plt.figure()
		# sm.qqplot(standardised_y_estimate, line='45')
		plt.scatter(std_norm_quantile, np.sort(standardised_y_estimate), marker = '.', color ='b', label='Truth')
		plt.scatter(std_norm_quantile, Uqunatile, color = 'k', marker = '_', label='Upper_CI') 
		plt.scatter(std_norm_quantile, Lqunatile, color = 'green', marker = '_', label='Lower_CI') 
		plt.plot(std_norm_quantile, std_norm_quantile, color='r')
		plt.xlabel('Theoretical Quantiles')
		plt.ylabel('Sample Quantiles')
		plt.legend(loc='best')
		plt.savefig(output_folder + 'SEED'+ str(SEED) +'OutSampQQ_indivErr' + str(indivError) + 'Idx' + str(index_Xaxis) + '.png')
		plt.show()
		plt.close()
		
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


		# residuals = y_test - mu_star
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
			residualsPlot = ax.scatter(X_test[pivot, 0], X_test[pivot, 1], c= standardised_y_estimate, cmap=cmap, norm = norm0)

		plt.xlabel('Longitude')
		plt.ylabel('Latitude')
		plt.title('Residuals of out-of-sample prediction')
		plt.colorbar(residualsPlot, ax=ax)
		plt.savefig(output_folder + 'Residuals_seed' + str(SEED) + 'numMo' + str(numMo) + 'IndivErr' + str(indivError) +  '_outSampleStd.png')
		plt.show()
		plt.close()
	   
		# numpy2ri.activate() 
		# plot_seq = r.pretty(np.arange(6,42), 20)
		# jet_colors = r.colorRampPalette(r.c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))
		# pal = jet_colors(len(plot_seq) - 1)
		# pal = np.array(pal)
		# numpy2ri.deactivate()
		# cmap = mpl.colors.ListedColormap(list(pal))
		cmap = plt.cm.jet
		bounds = np.linspace(min_all, max_all, 20)
		norm0 = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
		
		# fig, ax = plt.subplots()
		# # testObs = ax.scatter(X_test[:, 0], X_test[:, 1], c= y_test, cmap=plt.cm.jet, vmin=min_all, vmax=max_all)
		# testObs = ax.scatter(X_test[:, 0], X_test[:, 1], c= y_test, cmap=cmap, vmin=min_all, vmax=max_all)
		# # plt.colorbar()
		# for i, txt in enumerate(np.round(y_test, 1)):
		#     ax.annotate(txt, (X_test[:, 0][i], X_test[:, 1][i]))
		# plt.colorbar(testObs, ax = ax)
		# plt.title('Out-of-sample test observations')
		# plt.xlabel('Longitude')
		# plt.ylabel('Latitude')
		# plt.savefig(output_folder + 'SEED'+ str(SEED) + 'TestObs.png')
		# plt.show()
		# plt.close()

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
		cell_coords = cell_coords - np.array([cell_width/2., cell_len/2.]) # this line center the cooridnated at the center of the cell
			
		tmp = X_tildZs - cell_coords 
		
		# get the center coordinates of X_tildZs
		centre_MoCoords = np.array([tmp[i][0] for i in range(len(X_tildZs))])
		X_mo  = centre_MoCoords
		
		# X_mo = np.array(list(chain.from_iterable(X_tildZs))) # This line of code is for the case where only one point for X_tildZs
		legend_elements = [Line2D([0], [0], marker='o', color='w', label='Observations', markerfacecolor=None, markeredgecolor='k', markersize=8),\
			Line2D([0], [0], marker='s',color='w', markerfacecolor=None, markeredgecolor='k', markersize=8, label='Test points')] 

		fig, ax = plt.subplots()
		if indivError:
			maxYtest = y_test[np.argsort(np.abs(residuals))[-3:]]
		else:
			y_testTmp = y_test[pivot]
			maxYtest = y_testTmp[np.argsort(np.abs(residuals))[-3:]]
		maxPlot = ax.scatter(max_coords[:, 0], max_coords[:, 1], c= maxYtest, cmap=cmap, norm =norm0, marker = 's')
		# for i, txt in enumerate(np.round(maxYtest, 1)):
		#     ax.annotate(txt, (max_coords[:, 0][i], max_coords[:, 1][i]))
		trainObs = ax.scatter(X_train[:, 0], X_train[:, 1], c= y_train_withMean, cmap=cmap, norm =norm0)
		# for i, txt in enumerate(np.round(y_train, 1)):
		#     ax.annotate(txt, (X_train[:, 0][i], X_train[:, 1][i]))
		# plt.colorbar(maxPlot, ax = ax)
		# plt.savefig('SEED'+ str(SEED) + 'TrainObsAndTestMaxObs.png')
		plt.colorbar(trainObs, ax = ax)
		# plt.title('Observations & test points')
		plt.xlabel('Longitude')
		plt.ylabel('Latitude')
		# plt.legend(loc='best')
		ax.legend(handles=legend_elements, loc='best')

		plt.savefig(output_folder + 'SEED'+ str(SEED) + 'TrainObs.png')
		plt.show()
		plt.close()

		legend_elements = [Line2D([0], [0], marker='o', color='w', label='Model outputs', markerfacecolor=None, markeredgecolor='k', markersize=8), \
			Line2D([0], [0], marker='s', color='w', markerfacecolor=None, markeredgecolor='k', markersize=8, label='Test points')] 

		fig, ax = plt.subplots()
		if indivError:
			maxYtest = y_test[np.argsort(np.abs(residuals))[-3:]]
		else:
			y_testTmp = y_test[pivot]
			maxYtest = y_testTmp[np.argsort(np.abs(residuals))[-3:]]
		maxPlot = ax.scatter(max_coords[:, 0], max_coords[:, 1], c= maxYtest, cmap=cmap, norm =norm0, marker = 's')
		# for i, txt in enumerate(np.round(maxYtest, 1)):
		#     ax.annotate(txt, (max_coords[:, 0][i], max_coords[:, 1][i]))

		modelOutputs = ax.scatter(X_mo[:, 0], X_mo[:, 1],  c = y_tildZs_withMean, cmap=cmap, norm = norm0)
		# for i, txt in enumerate(np.round(y_tildZs, 1)):
		#     ax.annotate(txt, (X_mo[:, 0][i], X_mo[:, 1][i]))

		# plt.colorbar(maxPlot, ax = ax)
		# plt.savefig('SEED'+ str(SEED) + 'Mo' + str(numMo) + 'andTestObsMax.png')
		plt.colorbar(modelOutputs, ax = ax)
		# plt.title('Model outputs & test points')
		plt.xlabel('Longitude')  
		plt.ylabel('Latitude')
		ax.legend(handles=legend_elements, loc='best')
		plt.savefig(output_folder + 'SEED'+ str(SEED) + 'Mo' + str(numMo) + '.png')
		plt.show()
		plt.close()

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

	print('In-sample RMSE condition on Zhat/Ztilde for seed' + str(SEED) + ' is :' + str(rmse))

	rmse_out = open(output_folder + 'rmse_inSample.pkl', 'wb')
	pickle.dump(rmse, rmse_out) 
	rmse_out.close()

	if not crossValFlag:
		index = np.arange(len(y_test))
		if marginZhat:
			mu_star = np.zeros(C_hatZs.shape[0])
			rmse = np.sqrt(np.mean((y_test - mu_star)**2))
			print('marginZhat in-sample RMSE for seed' + str(SEED) + ' is :' + str(rmse))
			cov_of_predic = C_hatZs
			l_chol_ChatZs = compute_L_chol(C_hatZs) # This is pivoted cholesky decomposition
			standardised_y_estimate = linalg.solve_triangular(l_chol_ChatZs, y_test - mu_star, lower=True)
			std_yEst_out = open(output_folder + 'std_yEst_inSampleMarginZhat.pkl', 'wb')
			pickle.dump(standardised_y_estimate, std_yEst_out)
		elif conditionZhat:
			l_chol_CtildeZs = compute_L_chol(C_tildZs)
			u1 =  linalg.solve_triangular(l_chol_CtildeZs.T, linalg.solve_triangular(l_chol_CtildeZs, y_tildZs - mu_tildZs, lower=True))
			mu_star  = np.dot(avg_pointAreal_upper, u1)
			rmse = np.sqrt(np.mean((y_test - mu_star)**2))
			print('conditionZhat in-sample RMSE for seed' + str(SEED) + ' is :' + str(rmse))
			u2 =  linalg.solve_triangular(l_chol_CtildeZs.T, linalg.solve_triangular(l_chol_CtildeZs, avg_pointAreal_lower, lower=True))
			conditon_cov = C_hatZs - np.dot(avg_pointAreal_upper, u2)
			cov_of_predic = conditon_cov
			G, pivot = compute_chol_pivoted(conditon_cov) # This is pivoted cholesky decomposition
			inv_G = linalg.inv(G)
			standardised_y_estimate = np.dot(inv_G, y_test - mu_star)
			std_yEst_out = open(output_folder + 'std_yEst_inSampleConditionZhat.pkl', 'wb')
			pickle.dump(standardised_y_estimate, std_yEst_out)
		elif conZhatZtilde:
			u0 = linalg.solve_triangular(l_chol_C.T, linalg.solve_triangular(l_chol_C, K_star.T, lower=True))
			cov_of_predic = cov_matrix(X_test, np.exp(log_sigma_Zs), np.exp(log_phi_Zs)) -  np.dot(K_star, u0) +  np.diag(np.repeat(np.exp(log_obs_noi_scale)**2, X_test.shape[0]))
			print ('Shape of X_test is ' + str(X_test.shape))
			print( 'Shape of cov_of_predic is ' + str(cov_of_predic.shape))
			l_chol_cov_of_predic = compute_L_chol(cov_of_predic) # This is pivoted cholesky decomposition
			standardised_y_estimate = linalg.solve_triangular(l_chol_cov_of_predic, y_test - mu_star, lower=True)
			std_yEst_out = open(output_folder + 'std_yEst_inSampleConZhatZtilde.pkl', 'wb')
			pickle.dump(standardised_y_estimate, std_yEst_out)
	
		margin_var = np.diag(cov_of_predic)
		if indivError:
			standardised_y_estimate = (y_test - mu_star)/np.sqrt(margin_var)
			std_yEst_out = open(output_folder + 'std_yEst_inSampleIndivErr.pkl', 'wb')
			pickle.dump(standardised_y_estimate, std_yEst_out)

		if not useSimData:
			plt.figure()
			# plt.scatter(index, standardised_y_estimate, facecolors='none', edgecolors='k', linewidths=1.2)
			if marginZhat:
				plt.scatter(index, standardised_y_estimate, facecolors='none', edgecolors='k', linewidths=1.2)
				plt.xlabel('Index')
			else:
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
			if marginZhat:
				plt.savefig(output_folder + 'SEED'+ str(SEED) +'stdPredicErr_inSampleMarginZhat.png')
			elif conditionZhat:                
				plt.savefig(output_folder + 'SEED'+ str(SEED) +'PredErr_inSampConZhat_IndivErr' + str(indivError) + 'Idx' + str(index_Xaxis) + '.png')    
			else:
				plt.savefig(output_folder + 'SEED'+ str(SEED) +'PredErr_inSampConZhatZtilde_IndivErr' + str(indivError) + 'Idx' + str(index_Xaxis) + '.png')
			
			plt.show()
			plt.close()

			lower_chol =  np.linalg.cholesky(cov_of_predic)
			num_Outputs =  cov_of_predic.shape[0]
			samples_yHat = []
			for i in range(1000):
				tmp = np.dot(lower_chol, np.random.normal(0., 1., num_Outputs))
				tmp = np.dot(inv_G, tmp)
				tmp = np.sort(tmp)
				samples_yHat.append(tmp)
			samples_yHat = np.array(samples_yHat)
			print('shape of samples_yHat is ' + str(samples_yHat.shape))
			Lqunatile = np.quantile(samples_yHat, 0.025, axis=0)
			Uqunatile = np.quantile(samples_yHat, 0.975, axis=0)

			std_norm_quantile = np.array([stats.norm.ppf((i-0.5)/num_Outputs) for i in range(1, num_Outputs+1)])

			plt.figure
			sm.qqplot(standardised_y_estimate, line='45')
			plt.savefig(output_folder + 'SEED'+ str(SEED) +'QQ_inSampConZhat_IndivErr' + str(indivError) + 'Idx' + str(index_Xaxis) + 'NoCI.png')   
			plt.show()
			plt.close()

			plt.figure()
			# sm.qqplot(standardised_y_estimate, line='45')
			plt.scatter(std_norm_quantile, np.sort(standardised_y_estimate),  marker = '.', color ='b', label='Truth')
			plt.scatter(std_norm_quantile, Uqunatile, color = 'k', marker = '_', label='Upper_CI') 
			plt.scatter(std_norm_quantile, Lqunatile, color = 'green', marker = '_', label='Lower_CI') 
			plt.plot(std_norm_quantile, std_norm_quantile, color='r')
			plt.xlabel('Theoretical Quantiles')
			plt.ylabel('Sample Quantiles')
			plt.legend(loc='best')
			if marginZhat:
				plt.savefig(output_folder + 'SEED'+ str(SEED) +'normalQQ_inSampleMarginZhat.png')
			elif conditionZhat:
				plt.savefig(output_folder + 'SEED'+ str(SEED) +'QQ_inSampConZhat_IndivErr' + str(indivError) + 'Idx' + str(index_Xaxis) + '.png')   
			else:
				plt.savefig(output_folder + 'SEED'+ str(SEED) +'QQ_inSampConZhatZtilde_IndivErr' + str(indivError) + 'Idx' + str(index_Xaxis) + '.png')
			
			plt.show()
			plt.close()

			# residuals = y_test - mu_star
			residuals = standardised_y_estimate

			max_coords = X_test[np.argsort(np.abs(residuals))[-3:], :]
			print('max_coords of residuals is ' + str(max_coords))

			maxAbs = np.array([np.abs(residuals.min()), np.abs(residuals.max())]).max()

			fig, ax = plt.subplots()
			cmap = mpl.colors.ListedColormap(["#00007F", "blue",'cyan', 'white', 'green', "red", "#7F0000"])
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
				residualsPlot = ax.scatter(X_test[:, 0], X_test[:, 1], c= standardised_y_estimate, cmap=cmap, norm=norm0)
			else:
				residualsPlot = ax.scatter(X_test[pivot, 0], X_test[pivot, 1], c= standardised_y_estimate, cmap=cmap, norm=norm0)
			plt.xlabel('Longitude')
			plt.ylabel('Latitude')
			plt.title('Residuals of in-sample prediction')
			plt.colorbar(residualsPlot, ax=ax)
			if marginZhat:
				plt.savefig(output_folder + 'Residuals_seed' + str(SEED) + 'numMo' + str(numMo) + 'IndivErr' + str(indivError) + '_insampleMarginZhat.png')
			elif conditionZhat:   
				plt.savefig(output_folder + 'Residuals_seed' + str(SEED) + 'numMo' + str(numMo) + 'IndivErr' + str(indivError) + '_insampleConditionZhatStd.png')
			else:
				plt.savefig(output_folder + 'Residuals_seed' + str(SEED) + 'numMo' + str(numMo) + 'IndivErr' + str(indivError) + '_insample.png')            
			plt.show()
			plt.close()
	
	if marginZhat:
		pass
	elif conditionZhat:
		pass
	else:# The following code is for Zhat | Zhat, Ztilde
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
	# exit(-1)
	########## The following codes are for in-sample model outputs #################
	X_test = X_tildZs
	y_test = y_tildZs
	index = np.arange(len(y_test))
	# The following is just for Ztilde ~ MVN(mu, COV)
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

	if marginZtilde:
		mu_star =  mu_test
		print('length of y_test, mu_star' + str((len(y_test), len(mu_star))))
		rmse = np.sqrt(np.mean((y_test - mu_star)**2))
		print('In-sample RMSE of y_tildZs for seed' + str(SEED) + ' is :' + str(rmse))
		rmse_out = open(output_folder + 'rmse_inSample_ytildZs.pkl', 'wb')
		pickle.dump(rmse, rmse_out) 
		rmse_out.close()
		cov_of_predic = C_tildZs 
		print ('Shape of X_test is ' + str(X_test.shape))
		print( 'Shape of cov_of_predic is ' + str(cov_of_predic.shape))

		#The following two lines are for normal cholesky decomposition
		l_chol_cov_of_predic = compute_L_chol(cov_of_predic)
		standardised_y_estimate = linalg.solve_triangular(l_chol_cov_of_predic, y_test - mu_star, lower=True)

		std_yEst_out = open(output_folder + 'std_yEst_inSample.pkl', 'wb')
		pickle.dump(standardised_y_estimate, std_yEst_out)
	else:# The following is  for Ztilde|Zhat ~ MVN(mu, COV`)
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
		# The following two lines are for normal cholesky decomposition
		# l_chol_cov_of_predic = compute_L_chol(cov_of_predic)
		# standardised_y_estimate = linalg.solve_triangular(l_chol_cov_of_predic, mu_star - y_test, lower=True)
		# The following three lines are for pivoted cholesky decomposition
		G, pivot = compute_chol_pivoted(cov_of_predic)
		inv_G = linalg.inv(G)
		standardised_y_estimate = np.dot(inv_G,  y_test - mu_star)
		std_yEst_out = open(output_folder + 'std_yEst_inSampleCon.pkl', 'wb')
		pickle.dump(standardised_y_estimate, std_yEst_out)

	margin_var = np.diag(cov_of_predic)
	if indivError:
		standardised_y_estimate = (y_test - mu_star)/np.sqrt(margin_var)
		std_yEst_out = open(output_folder + 'std_MoEst_inSampleIndivErr.pkl', 'wb')
		pickle.dump(standardised_y_estimate, std_yEst_out)

	if not useSimData:
		plt.figure()
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
		if marginZtilde:
			plt.savefig(output_folder + 'SEED'+ str(SEED) + 'PreErr_inSampZtilde_indivErr' + str(indivError) + 'idx' + str(index_Xaxis) + '.png')
		else:
			plt.savefig(output_folder + 'SEED'+ str(SEED) + 'PreErr_inSampZtildeConZhat_indivErr' + str(indivError) + 'idx' + str(index_Xaxis) + '.png')
		plt.show()
		plt.close()

		lower_chol =  np.linalg.cholesky(cov_of_predic)
		num_Outputs =  cov_of_predic.shape[0]
		samples_yTilde = []
		for i in range(1000):
			tmp = np.dot(lower_chol, np.random.normal(0., 1., num_Outputs))
			tmp = np.dot(inv_G, tmp)
			tmp = np.sort(tmp)
			samples_yTilde.append(tmp)
		samples_yTilde = np.array(samples_yTilde)
		print('shape of samples_yTilde is ' + str(samples_yTilde.shape))
		Lqunatile = np.quantile(samples_yTilde, 0.025, axis=0)
		Uqunatile = np.quantile(samples_yTilde, 0.975, axis=0)

		std_norm_quantile = np.array([stats.norm.ppf((i-0.5)/num_Outputs) for i in range(1, num_Outputs+1)])

		plt.figure
		sm.qqplot(standardised_y_estimate, line='45')
		plt.savefig(output_folder + 'SEED'+ str(SEED) + 'QQ_inSampZtildeConZhat_indivErr' + str(indivError) + 'idx' + str(index_Xaxis) + 'NoCI.png')
		plt.show()
		plt.close()

		plt.figure()
		# sm.qqplot(standardised_y_estimate, line='45')
		plt.scatter(std_norm_quantile, np.sort(standardised_y_estimate),  marker = '.', color ='b', label='Truth')
		plt.scatter(std_norm_quantile, Uqunatile, color = 'k', marker = '_', label='Upper_CI') 
		plt.scatter(std_norm_quantile, Lqunatile, color = 'green', marker = '_', label='Lower_CI')  
		plt.plot(std_norm_quantile, std_norm_quantile, color='r')
		plt.xlabel('Theoretical Quantiles')
		plt.ylabel('Sample Quantiles')
		plt.legend(loc='best')
		if marginZtilde:
			plt.savefig(output_folder + 'SEED'+ str(SEED) + 'QQ_inSampZtilde_indivErr' + str(indivError) + 'idx' + str(index_Xaxis) + '.png')
		else:
			plt.savefig(output_folder + 'SEED'+ str(SEED) + 'QQ_inSampZtildeConZhat_indivErr' + str(indivError) + 'idx' + str(index_Xaxis) + '.png')
		plt.show()
		plt.close()

		# residuals = y_test - mu_star
		residuals = standardised_y_estimate
		tmp = X_tildZs - cell_coords 
	
		# get the center coordinates of X_tildZs(X_test in this case)
		centre_MoCoords = np.array([tmp[i][0] for i in range(len(X_tildZs))])
		X_mo  = centre_MoCoords

		max_coords = X_mo[np.argsort(np.abs(residuals))[-3:], :]
		print('max_coords of residuals is ' + str(max_coords))

		maxAbs = np.array([np.abs(residuals.min()), np.abs(residuals.max())]).max()
		print('maxAbs is' + str(maxAbs))

		fig, ax = plt.subplots()
		cmap = mpl.colors.ListedColormap(["#00007F", "blue",'cyan', 'white', 'green', "red", "#7F0000"])
		cmap.set_under("crimson")
		cmap.set_over('black')

		if maxAbs >4:
			bounds = np.array([-np.ceil(maxAbs), -4, -2, -1, 1, 2, 4, np.ceil(maxAbs)])
		else:
			 bounds = np.array([-np.ceil(maxAbs), -2, -1, 1, 2,  np.ceil(maxAbs)])
			 cmap = mpl.colors.ListedColormap(["blue", 'cyan', 'white', 'green',  "red"])
		norm0 = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

		# residualsPlot = ax.scatter(X_mo[:, 0], X_mo[:, 1], c= y_test - mu_star, cmap=cmap, vmin = -maxAbs, vmax = maxAbs)
		if indivError:
			residualsPlot = ax.scatter(X_mo[:, 0], X_mo[:, 1], c= standardised_y_estimate, cmap=cmap, norm = norm0)
		else:
			residualsPlot = ax.scatter(X_mo[pivot, 0], X_mo[pivot, 1], c= standardised_y_estimate, cmap=cmap, norm=norm0)
		plt.xlabel('Longitude')
		plt.ylabel('Latitude')
		plt.title('Residuals of in-sample prediction')
		plt.colorbar(residualsPlot, ax=ax)
		if marginZtilde:
			plt.savefig(output_folder + 'Residuals_seed' + str(SEED) + 'numMo' + str(numMo) + 'IndivErr' + str(indivError) + '_insampleYtildZs.png')
		else:
			plt.savefig(output_folder + 'Residuals_seed' + str(SEED) + 'numMo' + str(numMo) + 'IndivErr' + str(indivError) + '_insampleYtildZsConStd.png')
		plt.show()
		plt.close()

	return succRate

def plot_qq_parUncerty(parUncertyOverSeeds = False, numMo = 500, SEED=None, indivError = False, index_Xaxis =True):
	if parUncertyOverSeeds:
		samples_Zs = [] 
		samples_yHat = [] 
		samples_yTilde = []
		seeds = [120, 121] + list(range(123,143)) + list(range(144, 160)) + list(range(161, 167)) + list(range(168, 177)) + list(range(178, 184)) + \
		list(range(185,189)) + list(range(190,198))+ [199, 200, 201,202,204,205,206,208,209,210,212,213,215,216,217,218,219]
		for seed in seeds:
			input_folder = 'DataImogenFrGridMoNotCentre/FPstart2016020612_FR_numObs_128_numMo_' + str(numMo) + '/seed' + str(seed) + '/'
			tmp_in =  open(input_folder + 'all_samples_parUncerty.pkl', 'rb')
			all_samples = pickle.load(tmp_in)
			samples_Zs_tmp = all_samples[0]
			samples_yHat_tmp = all_samples[1]
			samples_yTilde_tmp = all_samples[2]

			samples_Zs.append(samples_Zs_tmp)
			samples_yHat.append(samples_yHat_tmp)
			samples_yTilde.append(samples_yTilde_tmp)

		samples_Zs = np.array(samples_Zs)
		samples_yHat = np.array(samples_yHat) 
		samples_yTilde = np.array(samples_yTilde)
		print(samples_Zs.shape) 
		print(samples_yHat.shape) 
		print(samples_yTilde.shape) 

		input_folder = 'DataImogenFrGridMoNotCentre/FPstart2016020612_FR_numObs_128_numMo_500/seed120/'

		std_yEst_Zs_in = open(input_folder + 'std_yEst_outSample.pkl', 'rb')
		std_yEst_Zs = pickle.load(std_yEst_Zs_in)

		std_yEst_yHat_in = open(input_folder + 'std_yEst_inSampleConditionZhat.pkl', 'rb')
		std_yEst_yHat = pickle.load(std_yEst_yHat_in)

		std_yEst_yTilde_in = open(input_folder + 'std_yEst_inSampleCon.pkl', 'rb')
		std_yEst_yTilde = pickle.load(std_yEst_yTilde_in)

	else:
		samples_Zs = [] 
		samples_yHat = [] 
		samples_yTilde = []
		for i in range(1000):
			input_folder = 'DataImogenFrGridMoNotCentre/FPstart2016020612_FR_numObs_128_numMo_' + str(args.numMo) + '/seed' + str(SEED) + '/idx_theta' + str(i) + '/'
			tmp_in =  open(input_folder + 'all_samples_parUncerty.pkl', 'rb')
			all_samples = pickle.load(tmp_in)

			samples_Zs_tmp = all_samples[0]
			samples_yHat_tmp = all_samples[1]
			samples_yTilde_tmp = all_samples[2]

			samples_Zs.append(samples_Zs_tmp)
			samples_yHat.append(samples_yHat_tmp)
			samples_yTilde.append(samples_yTilde_tmp)

		samples_Zs = np.array(samples_Zs)
		samples_yHat = np.array(samples_yHat) 
		samples_yTilde = np.array(samples_yTilde)
		print(samples_Zs.shape) 
		print(samples_yHat.shape) 
		print(samples_yTilde.shape) 

		input_folder = 'DataImogenFrGridMoNotCentre/FPstart2016020612_FR_numObs_128_numMo_' + str(numMo) + '/seed' + str(SEED)+'/'

		std_yEst_Zs_in = open(input_folder + 'std_yEst_outSample.pkl', 'rb')
		std_yEst_Zs = pickle.load(std_yEst_Zs_in)

		std_yEst_yHat_in = open(input_folder + 'std_yEst_inSampleConditionZhat.pkl', 'rb')
		std_yEst_yHat = pickle.load(std_yEst_yHat_in)

		std_yEst_yTilde_in = open(input_folder + 'std_yEst_inSampleCon.pkl', 'rb')
		std_yEst_yTilde = pickle.load(std_yEst_yTilde_in)

	if parUncertyOverSeeds:
		output_folder = 'DataImogenFrGridMoNotCentre/FPstart2016020612_FR_numObs_128_numMo_' + str(numMo) + '/parUncertyOverSeeds' 
	else:
		output_folder = 'DataImogenFrGridMoNotCentre/FPstart2016020612_FR_numObs_128_numMo_' + str(numMo) + '/seed' + str(SEED) + '/parUncerty' 
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	output_folder +=  '/'
	print('output_folder in plot_qq_parUncerty ' + str(output_folder))
	#QQ plot for out-of-sample predictons of Zhat
	Lqunatile = np.quantile(samples_Zs, 0.025, axis=0)
	Uqunatile = np.quantile(samples_Zs, 0.975, axis=0)
	num_Outputs = samples_Zs.shape[1]
	std_norm_quantile = np.array([stats.norm.ppf((i-0.5)/num_Outputs) for i in range(1, num_Outputs+1)])
	print(Lqunatile, Uqunatile)
	exit(-1)

	plt.figure
	sm.qqplot(std_yEst_Zs, line='45')
	plt.savefig(output_folder + 'SEED'+ str(SEED) +'OutSampQQ_indivErr' + str(indivError) + 'Idx' + str(index_Xaxis) + 'NoCI.png')
	plt.show()
	plt.close()

	plt.figure()
	# sm.qqplot(standardised_y_estimate, line='45')
	plt.scatter(std_norm_quantile, np.sort(std_yEst_Zs), marker = '.', color ='b', label='Truth')
	plt.scatter(std_norm_quantile, Uqunatile, color = 'k', marker = '_', label='Upper_CI') 
	plt.scatter(std_norm_quantile, Lqunatile, color = 'green', marker = '_', label='Lower_CI') 
	plt.plot(std_norm_quantile, std_norm_quantile, color='r')
	plt.xlabel('Theoretical Quantiles')
	plt.ylabel('Sample Quantiles')
	plt.legend(loc='best')
	plt.savefig(output_folder + 'SEED'+ str(SEED) +'OutSampQQ_indivErr' + str(indivError) + 'Idx' + str(index_Xaxis) + '.png')
	plt.show()
	plt.close()
	# QQ plot for insample predeiction for Zhat
	Lqunatile = np.quantile(samples_yHat, 0.025, axis=0)
	Uqunatile = np.quantile(samples_yHat, 0.975, axis=0)
	num_Outputs = samples_yHat.shape[1]
	
	std_norm_quantile = np.array([stats.norm.ppf((i-0.5)/num_Outputs) for i in range(1, num_Outputs+1)])

	plt.figure
	sm.qqplot(std_yEst_yHat, line='45')
	plt.savefig(output_folder + 'SEED'+ str(SEED) +'QQ_inSampConZhat_IndivErr' + str(indivError) + 'Idx' + str(index_Xaxis) + 'NoCI.png')   
	plt.show()
	plt.close()

	plt.figure()
	plt.scatter(std_norm_quantile, np.sort(std_yEst_yHat),  marker = '.', color ='b', label='Truth')
	plt.scatter(std_norm_quantile, Uqunatile, color = 'k', marker = '_', label='Upper_CI') 
	plt.scatter(std_norm_quantile, Lqunatile, color = 'green', marker = '_', label='Lower_CI') 
	plt.plot(std_norm_quantile, std_norm_quantile, color='r')
	plt.xlabel('Theoretical Quantiles')
	plt.ylabel('Sample Quantiles')
	plt.legend(loc='best')
	plt.savefig(output_folder + 'SEED'+ str(SEED) +'QQ_inSampConZhat_IndivErr' + str(indivError) + 'Idx' + str(index_Xaxis) + '.png')   
	plt.show()
	plt.close()
	# QQ plot for insample predicitons for Ztilde
	Lqunatile = np.quantile(samples_yTilde, 0.025, axis=0)
	Uqunatile = np.quantile(samples_yTilde, 0.975, axis=0)
	num_Outputs = samples_yTilde.shape[1]

	std_norm_quantile = np.array([stats.norm.ppf((i-0.5)/num_Outputs) for i in range(1, num_Outputs+1)])

	plt.figure
	sm.qqplot(std_yEst_yTilde, line='45')
	plt.savefig(output_folder + 'SEED'+ str(SEED) + 'QQ_inSampZtildeConZhat_indivErr' + str(indivError) + 'idx' + str(index_Xaxis) + 'NoCI.png')
	plt.show()
	plt.close()

	plt.figure()
	# sm.qqplot(standardised_y_estimate, line='45')
	plt.scatter(std_norm_quantile, np.sort(std_yEst_yTilde),  marker = '.', color ='b', label='Truth')
	plt.scatter(std_norm_quantile, Uqunatile, color = 'k', marker = '_', label='Upper_CI') 
	plt.scatter(std_norm_quantile, Lqunatile, color = 'green', marker = '_', label='Lower_CI')  
	plt.plot(std_norm_quantile, std_norm_quantile, color='r')
	plt.xlabel('Theoretical Quantiles')
	plt.ylabel('Sample Quantiles')
	plt.legend(loc='best')
	plt.savefig(output_folder + 'SEED'+ str(SEED) + 'QQ_inSampZtildeConZhat_indivErr' + str(indivError) + 'idx' + str(index_Xaxis) + '.png')
	plt.show()
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
	p.add_argument('-predicMo', dest='predicMo', default=False,  type=lambda x: (str(x).lower() == 'true'),  help='flag for whether to predict the value where model outputs are produced')
	p.add_argument('-poly_deg', type=int, dest='poly_deg', default=2, help='degree of the polynomial function of the additive model bias')
	p.add_argument('-indivError', dest='indivError', default=False,  type=lambda x: (str(x).lower() == 'true'),  help='flag for whether to individual std errors for GP diagnosis')
	p.add_argument('-index_Xaxis', dest='index_Xaxis', default=True,  type=lambda x: (str(x).lower() == 'true'),  help='flag for whether X-axis is index for GP diagnosis')
	p.add_argument('-parUncerty', dest='parUncerty', default=False,  type=lambda x: (str(x).lower() == 'true'),  help='flag for whether taking into account parameter uncertainty for a fixed seed for GP diagnosis')
	p.add_argument('-parUncertyOverSeeds', dest='parUncertyOverSeeds', default=True,  type=lambda x: (str(x).lower() == 'true'),  help='flag for whether taking into account parameter uncertainty over seeds for GP diagnosis')
	p.add_argument('-idx_theta', dest='idx_theta', type=int, default=0, help='index for the 1000 thetas for a fixed seed')

	args = p.parse_args() 
	# plot_qq_parUncerty(args.parUncertyOverSeeds, args.numMo, args.SEED)
	# exit(-1)
	
	if args.useSimData: 
		# input_folder = os.getcwd() + '/dataRsimGammaTransformErrorInZtilde/numObs_200_numMo_' + str(args.numMo) + '/seed' + str(args.SEED) + '/'
		# input_folder = os.getcwd() + '/dataRsimNoFrGammaTransformArealRes25Cods100butArealZs100/numObs_200_numMo_' + str(args.numMo) + '/seed' + str(args.SEED) + '/'
		input_folder = os.getcwd() + '/dataSimulated/numObs_200_numMo_' + str(args.numMo) + '/seed' + str(args.SEED) + '/'
	else:
		input_folder = os.getcwd() + '/DataImogenFrGridMoNotCentre/FPstart2016020612_FR_numObs_128_numMo_' + str(args.numMo) + '/seed' + str(args.SEED) + '/'
		# if args.SEED == 120:
		#     if args.poly_deg ==5:
		#         input_folder = os.getcwd() + '/Data/FPstart2016020612_FR_numObs_128_numMo_' + str(args.numMo) + '/seed' + str(args.SEED) + '/'
		#     else:
		#         input_folder = os.getcwd() + '/Data/FPstart2016020612_FR_numObs_128_numMo_' + str(args.numMo) + '/seed' + str(args.SEED) + 'PolyDeg2/'
		# else:
		#     input_folder = os.getcwd() + '/Data/FPstart2016020612_FR_numObs_128_numMo_' + str(args.numMo) + '/seed' + str(args.SEED) + '/'

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

	if args.parUncertyOverSeeds:
		samples_Zs, samples_yHat, samples_yTilde = \
		gpValid_parUncertyOverSeeds.predic_gpRegression(mu, X_train, y_train, X_test, y_test, X_tildZs, y_tildZs, args.crossValFlag, args.SEED, args.numMo, \
	args.useSimData, args.grid, args.predicMo, args.poly_deg, args.indivError, args.index_Xaxis)

		output_folder = os.getcwd() + '/DataImogenFrGridMoNotCentre/FPstart2016020612_FR_numObs_128_numMo_' + str(args.numMo) + '/seed' + str(args.SEED)
		if not os.path.exists(output_folder):
			os.makedirs(output_folder)
		output_folder +=  '/'
		print('output_folder for parUncertyOverSeeds ' + str(output_folder))

		tmp = [samples_Zs, samples_yHat, samples_yTilde]
		all_samples_out = open(output_folder + 'all_samples_parUncerty.pkl', 'wb')
		pickle.dump(tmp, all_samples_out)
	elif args.parUncerty:
		num_theta = len(mu)
		lower_chol = np.linalg.cholesky(cov)
		
		theta_tmp = mu + np.dot(lower_chol, np.random.normal(0., 1., num_theta))
		print('theta ' + str(args.idx_theta) + ' is ' + str(theta_tmp))
		samples_Zs, samples_yHat, samples_yTilde = \
		gpValid_parUncerty.predic_gpRegression(args.idx_theta, theta_tmp, X_train, y_train, X_test, y_test, X_tildZs, y_tildZs, args.crossValFlag, args.SEED, args.numMo, \
	args.useSimData, args.grid, args.predicMo, args.poly_deg, args.indivError, args.index_Xaxis)

		output_folder = os.getcwd() + '/DataImogenFrGridMoNotCentre/FPstart2016020612_FR_numObs_128_numMo_' + str(args.numMo) + '/seed' + str(args.SEED) + '/idx_theta' + str(args.idx_theta)
		if not os.path.exists(output_folder):
			os.makedirs(output_folder)
		output_folder += '/'
		print('output_folder for parUncerty ' + str(output_folder))
		tmp = [samples_Zs, samples_yHat, samples_yTilde]
		all_samples_out = open(output_folder + 'all_samples_parUncerty.pkl', 'wb')
		pickle.dump(tmp, all_samples_out)
	else:
		predic_accuracy = predic_gpRegression(mu, X_train, y_train, X_test, y_test, X_tildZs, y_tildZs, args.crossValFlag, args.SEED, args.numMo, \
			args.useSimData, args.grid, args.predicMo, args.poly_deg, args.indivError, args.index_Xaxis)
		 
