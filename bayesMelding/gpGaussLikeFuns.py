import numpy as np
from scipy import linalg
import computeN3Cost
import numbers

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


