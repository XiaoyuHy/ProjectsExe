#!/usr/bin/python -tt   #This line is to solve any difference between spaces and tabs
import numpy as np
import gpGaussLikeFuns
import computeN3Cost
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import chain
import pickle

def sim_hatTildZs_With_Plots(areal_res = 20, point_res=100, sigma = 1., w = [1.], obs_noi_scale = 0.1, b= 1., num_hatZs=200, num_tildZs = 150):
    lower_bound = np.array([0.] * 2)
    upper_bound = np.array([1.] * 2)
    x1, x2 = np.meshgrid(np.linspace(lower_bound[0], upper_bound[0], point_res),
                         np.linspace(lower_bound[1], upper_bound[1], point_res))

    #obtain cororinates for each area
    res_per_areal = point_res/areal_res
    areal_coordinate = []
    for i in range(areal_res):
        tmp = [np.vstack([x1[i*res_per_areal:(i+1)*res_per_areal, j*res_per_areal:(j+1)*res_per_areal].ravel(), x2[i*res_per_areal:(i+1)*res_per_areal, j*res_per_areal:(j+1)*res_per_areal].ravel()]).T for j in range(areal_res)]
        areal_coordinate.append(tmp)
    areal_coordinate = np.array(list(chain.from_iterable(areal_coordinate)))

    #save all the areal coordinates
    all_X_tildZs_out = open('all_X_tildZs.pickle', 'wb')
    pickle.dump(areal_coordinate, all_X_tildZs_out)
    all_X_tildZs_out.close()

    #generate Zs
    X = np.vstack([x1.ravel(), x2.ravel()]).T
    num_Zs = X.shape[0]
    cov = gpGaussLikeFuns.cov_matrix(X=X, sigma=sigma, w=w)
    l_chol_cov = gpGaussLikeFuns.compute_L_chol(cov)
    all_y_Zs = np.dot(l_chol_cov, np.random.normal(size=[num_Zs, 1]).reshape(num_Zs))
    np.savetxt('all_y_Zs_res' + str(point_res) + '.txt', all_y_Zs)

    all_X_Zs = np.vstack([x1.ravel(), x2.ravel()]).T
    np.savetxt('all_X_Zs_res' + str(point_res) + '.txt', all_X_Zs)

    #sample hatZs
    idx = np.random.randint(0, num_Zs, num_hatZs)
    X_hatZs = all_X_Zs[idx, :] 
    y_hatZs = all_y_Zs[idx]
    #add normal noise with scale = obs_noi_scale
    y_hatZs = y_hatZs + np.random.normal(loc=0., scale = obs_noi_scale, size=num_hatZs)
    np.savetxt('X_hatZs_res' + str(point_res) + '.txt', X_hatZs)
    np.savetxt('y_hatZs_res' + str(point_res) + '.txt', y_hatZs)

    plt.figure()
    im = plt.imshow(np.flipud(all_y_Zs.reshape((point_res,point_res))), extent=(lower_bound[0], upper_bound[0],lower_bound[1], upper_bound[1]), cmap = plt.matplotlib.cm.jet)
    plt.scatter(X_hatZs[:,0], X_hatZs[:,1], s=12, c='k', marker = 'o')
    cb=plt.colorbar(im)
    cb.set_label('$\^{Z(s)}$')
    plt.title('min = %.2f , max = %.2f , avg = %.2f' % (all_y_Zs.min(), all_y_Zs.max(), all_y_Zs.mean()))
    plt.xlabel('$x1$')
    plt.ylabel('$x2$')
    plt.grid()
    plt.savefig('d2_Zs_res' + str(point_res) + '_noNoi.png')
    plt.close()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x1, x2, all_y_Zs.reshape(point_res,point_res), rstride=1, cstride=1, cmap=plt.matplotlib.cm.jet)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$\^{Z(s)}$')
    plt.savefig('d3_Zs_res' + str(point_res) + '_noNoi.png')
    plt.close()

    #generate the tildZs = b * average of the areal Zs
    mat_Zs = all_y_Zs.reshape(point_res, point_res)
    areal_Zs = []
    for i in range(areal_res):
        tmp = [mat_Zs[i*res_per_areal:(i+1)*res_per_areal, j*res_per_areal:(j+1)*res_per_areal] for j in range(areal_res)]
        areal_Zs.append(tmp)
    areal_Zs = np.array(list(chain.from_iterable(areal_Zs)))
    all_y_tildZs = np.array([b * np.mean(areal_Zs[i]) for i in range(len(areal_Zs))])
    # save all the tildZs
    all_y_tildZs_out = open('all_y_tildZs.pickle', 'wb')
    pickle.dump(all_y_tildZs, all_y_tildZs_out)
    all_y_tildZs_out.close()

    
    #sample tildZs and the corresponding coordinates
    idx_sampleTildZs = np.random.randint(0, len(all_y_tildZs), num_tildZs)
    y_tildZs = all_y_tildZs[idx_sampleTildZs]
    X_tildZs = areal_coordinate[idx_sampleTildZs] 
    #add normal noise with scale = model_deviation_scale
    # model_deviation_scale = 0.1
    # y_tildZs = y_tildZs + np.random.normal(loc=0., scale = model_deviation_scale, size=num_tildZs)

    #save the samples of tildZs
    y_tildZs_out = open('y_tildZs.pickle', 'wb')
    pickle.dump(y_tildZs, y_tildZs_out)
    y_tildZs_out.close()

    X_tildZs_out = open('X_tildZs.pickle', 'wb')
    pickle.dump(X_tildZs, X_tildZs_out)
    X_tildZs_out.close()

    lower_bound = np.array([0.] * 2)
    upper_bound = np.array([1.] * 2)
    x1, x2 = np.meshgrid(np.linspace(lower_bound[0], upper_bound[0], areal_res),
                         np.linspace(lower_bound[1], upper_bound[1], areal_res))
    X = np.vstack([x1.ravel(), x2.ravel()]).T
    X_tildZs_arealRes = X[idx_sampleTildZs, :]

    plt.figure()
    im = plt.imshow(np.flipud(all_y_tildZs.reshape((areal_res, areal_res))), extent=(lower_bound[0], upper_bound[0],lower_bound[1], upper_bound[1]), cmap = plt.matplotlib.cm.jet)
    plt.scatter(X_tildZs_arealRes[:,0], X_tildZs_arealRes[:,1], s=12, c='k', marker = 'o')
    cb=plt.colorbar(im)
    cb.set_label('$\~{Z(s)}$')
    plt.title('min = %.2f , max = %.2f , avg = %.2f' % (all_y_tildZs.min(), all_y_tildZs.max(), all_y_tildZs.mean()))
    plt.xlabel('$x1$')
    plt.ylabel('$x2$')
    plt.grid()
    plt.savefig('d2_tildZs_res' + str(areal_res) + '_noNoi.png')
    plt.close()

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot_surface(x1, x2, all_y_tildZs.reshape(areal_res, areal_res), rstride=1, cstride=1, cmap=plt.matplotlib.cm.jet)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$\~{Z(s)}$')
    plt.savefig('d3_tildZs_res' + str(areal_res) + '_noNoi.png')
    plt.close()

#covranice matrix between areas
def cov_areal(areal_coordinate, w, sigma=1., b=1., OMEGA = 1e-6):
    num_areas = len(areal_coordinate)
    cov_areas = []
    for i in range(num_areas):
        tmp = [avg_cov_two_areal(areal_coordinate[i], areal_coordinate[j], w) for j in range(num_areas)]
        cov_areas.append(tmp)
    covAreas = np.hstack(cov_areas).reshape(num_areas,num_areas) + np.diag(np.repeat(OMEGA, num_areas))
    return covAreas

#average covranice between two areas
def avg_cov_two_areal(x, y, w, sigma=1., b=1.):
    # compute the K_star for n_test test points
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

    cov_of_two_vec = sigma * np.exp(- 0.5 * square_dist) # is a matrix of size (n_train, n_test)
    avg = b**2 * np.float(np.sum(cov_of_two_vec))/(n_x * n_y)
    return avg

#sample tildZs from the areal covaraince matrix - not necessarily though
def sample_tildZs_from_arealCov(w=[1.], sigma=1, b=1., num_tildZs=150):
    #read the samples of tildZs
    all_X_tildZs_in = open('all_X_tildZs.pickle', 'rb')
    all_X_tildZs = pickle.load(all_X_tildZs_in)
    num_sample = all_X_tildZs.shape[0]
    cov = cov_areal(all_X_tildZs, w)
    l_chol_cov = gpGaussLikeFuns.compute_L_chol(cov)
    all_y_tildZs = np.dot(l_chol_cov, np.random.normal(size=[num_sample, 1]).reshape(num_sample))
    return all_y_tildZs.shape

if __name__ == '__main__':
    computeN3Cost.init(0)
    #sim_hatTildZs_With_Plots()
    res = sample_tildZs_from_arealCov()
    print res
    








