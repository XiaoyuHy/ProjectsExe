#!/usr/bin/python -tt   #This line is to solve any difference between spaces and tabs
import numpy as np
import gpGaussLikeFuns
import computeN3Cost
#plt.switch_backend('agg') # This line is for running code on cluster to make pyplot working on cluster
# from mpl_toolkits.mplot3d import Axes3D
from itertools import chain
import pickle
from scipy import integrate
from scipy import linalg
import argparse
import os
import scipy.stats as stats
# import matplotlib.pyplot as plt
import random
# from rpy2.robjects import r
# from mpl_toolkits.mplot3d import Axes3D

def fun_a_bias(x, a_bias_coefficients = [2., 3., 15.]):
	a_bias_coefficients = np.array(a_bias_coefficients)
	a_bias = np.dot(a_bias_coefficients, np.concatenate((x, [1.])))
	return a_bias

def sim_hatTildZs_With_Plots(SEED = 200, phi_Zs = [0.8], gp_deltas_modelOut = True, phi_deltas_of_modelOut = [0.2], \
	sigma_Zs = 1.0, sigma_deltas_of_modelOut = 0.5, obs_noi_scale = 0.5, b= 0.6, areal_res = 200, point_res=1000, num_hatZs=200, num_tildZs = 150, \
	 a_bias_poly_deg = 2):
	np.random.seed(SEED)
	random.seed(SEED)
	lower_bound = np.array([-12., -6.5])
	upper_bound = np.array([-3., 3.])
	x1, x2 = np.meshgrid(np.linspace(lower_bound[0], upper_bound[0], point_res),  
						 np.linspace(lower_bound[1], upper_bound[1], point_res))
	#obtain coordinates for each area
	res_per_areal = point_res/areal_res
	res_per_areal = int(res_per_areal)
	
	# areal_coordinate = []
	# for i in range(areal_res):
	# 	tmp0 = [np.vstack([x1[i*res_per_areal:(i+1)*res_per_areal, j*res_per_areal:(j+1)*res_per_areal].ravel(), x2[i*res_per_areal:(i+1)*res_per_areal, j*res_per_areal:(j+1)*res_per_areal].ravel()]).T for j in range(areal_res)]
	# 	areal_coordinate.append(tmp0)
	# areal_coordinate = np.array(list(chain.from_iterable(areal_coordinate)))
	# all_X_tildZs = areal_coordinate

	# output_folder = os.getcwd() + '/dataSimulated/seed' + str(SEED)
	# if not os.path.exists(output_folder):
	# 	os.makedirs(output_folder)
	# output_folder += '/'
	# print('output_folder for all simulated data is ' + str(output_folder))

	# #save all the areal coordinates
	# # all_X_tildZs_out = open('dataSimGpDeltas/all_X_tildZs.pickle', 'wb')
	# all_X_tildZs_out = open(output_folder + 'all_X_tildZs.pickle', 'wb')
	# pickle.dump(areal_coordinate, all_X_tildZs_out)
	# all_X_tildZs_out.close()

	# # #generate Zs 
	# # X = np.vstack([x1.ravel(), x2.ravel()]).T
	# # num_Zs = X.shape[0]
	# # cov = gpGaussLikeFuns.cov_matrix(X=X, sigma=sigma_Zs, w=phi_Zs)
	# # l_chol_cov = np.linalg.cholesky(cov)
	# # all_y_Zs = np.dot(l_chol_cov, np.random.normal(size=[num_Zs, 1]).reshape(num_Zs))
	
	# #when point_res = 1000, load data from random fields outputs (Zs) implemented in R
	# r.load('dataRsimulated/rfSimDataSEED' + str(SEED) + '.RData')
	# all_y_Zs = r['rfSimData']
	# all_y_Zs = np.array(all_y_Zs)


	# nparam_density = stats.kde.gaussian_kde(all_y_Zs)
	# x = np.linspace(-3, 3, 200)
	# # x = np.linspace(0, 50, 200)
	# nparam_density = nparam_density(x)

	# # parametric fit: assume normal distribution
	# loc_param, scale_param = stats.norm.fit(all_y_Zs)
	# param_density = stats.norm.pdf(x, loc=loc_param, scale=scale_param)


	# # fig, ax = plt.subplots(figsize=(10, 6))
	# # ax.hist(all_y_Zs, bins=30, normed=True)
	# # ax.plot(x, nparam_density, 'r-', label='non-parametric density (smoothed by Gaussian kernel)')
	# # ax.plot(x, param_density, 'k--', label='parametric density')
	# # # ax.set_ylim([0, 0.15])
	# # ax.legend(loc='best')
	# # plt.savefig(output_folder + 'originSimDataPointRes' + str(point_res) + '.png')
	# # # plt.show()
	# # plt.close()
	# # generate gamma distributed Zs
	# pnorm_Zs = stats.norm.cdf(all_y_Zs)
	# #alpha, loc, scale from gamma fit of FR data of Imogen
	# alpha = 65.4
	# loc = -18.5
	# scale = 0.7
	# rv = stats.gamma(alpha, loc, scale)
	# all_y_Zs = rv.ppf(pnorm_Zs)

	# nparam_density = stats.kde.gaussian_kde(all_y_Zs)
	# # x = np.linspace(-3, 3, 200)
	# x = np.linspace(0, 50, 200)
	# nparam_density = nparam_density(x)

	# # parametric fit: assume normal distribution
	# loc_param, scale_param = stats.norm.fit(all_y_Zs)
	# param_density = stats.norm.pdf(x, loc=loc_param, scale=scale_param)

	# alpha,loc,scale = stats.gamma.fit(all_y_Zs)
	# gamma_density = stats.gamma.pdf(x, alpha, loc, scale)


	# # fig, ax = plt.subplots(figsize=(10, 6))
	# # ax.hist(all_y_Zs, bins=30, normed=True)
	# # ax.plot(x, nparam_density, 'r-', label='non-parametric density (smoothed by Gaussian kernel)')
	# # ax.plot(x, param_density, 'k--', label='parametric density')
	# # ax.plot(x, gamma_density, 'g-', label='gamma density')
	# # # ax.set_ylim([0, 0.15])
	# # ax.legend(loc='best')
	# # plt.savefig(output_folder + 'gammaTransformedWithMean.png')
	# # # plt.show()
	# # plt.close()

	# all_y_Zs = all_y_Zs - np.mean(all_y_Zs)

	# # fig = plt.figure()
	# # ax = Axes3D(fig)
	# # ax.plot_surface(x1, x2, all_y_Zs.reshape(point_res,point_res), rstride=1, cstride=1, cmap=plt.matplotlib.cm.jet)
	# # ax.set_xlabel('$x_1$')
	# # ax.set_ylabel('$x_2$')
	# # ax.set_zlabel('$\^{Z(s)}$')
	# # plt.savefig(output_folder + 'd3_Zs_res' + str(point_res) +  '_noNoi_NoMean.png')
	# # # plt.savefig('dataSimGpDeltas/d3_Zs_res' + str(point_res) +  '_noNoi_NoMean.png')
	# # plt.show()
	# # plt.close()

	# all_y_Zs_out = open(output_folder + 'all_y_Zs.pickle', 'wb')
	# # all_y_Zs_out = open('dataSimGpDeltas/all_y_Zs.pickle', 'wb')
	# pickle.dump(all_y_Zs, all_y_Zs_out)
	# all_y_Zs_out.close()

	# nparam_density = stats.kde.gaussian_kde(all_y_Zs)
	# x = np.linspace(-20, 20, 200)
	# # x = np.linspace(0, 50, 200)
	# nparam_density = nparam_density(x)
	# # parametric fit: assume normal distribution
	# loc_param, scale_param = stats.norm.fit(all_y_Zs)
	# print('norm scale is ' + str(scale_param))
	
	# param_density = stats.norm.pdf(x, loc=loc_param, scale=scale_param)

	# alpha,loc,scale = stats.gamma.fit(all_y_Zs)
	# gamma_density = stats.gamma.pdf(x, alpha, loc, scale)


	# # fig, ax = plt.subplots(figsize=(10, 6))
	# # ax.hist(all_y_Zs, bins=30, normed=True)
	# # ax.plot(x, nparam_density, 'r-', label='non-parametric density (smoothed by Gaussian kernel)')
	# # ax.plot(x, param_density, 'k--', label='parametric density')
	# # ax.plot(x, gamma_density, 'g-', label='gamma density')
	# # # ax.set_ylim([0, 0.15])
	# # ax.legend(loc='best')
	# # plt.savefig(output_folder + 'gammaTransformedNoMean.png')
	# # # plt.show()
	# # plt.close()

	# all_X_Zs = np.vstack([x1.ravel(), x2.ravel()]).T
	# all_X_Zs_out = open(output_folder + 'all_X_Zs.pickle', 'wb')
	# # all_X_Zs_out = open('dataSimGpDeltas/all_X_Zs.pickle', 'wb')
	# pickle.dump(all_X_Zs, all_X_Zs_out)
	# all_X_Zs_out.close()

	# #generate the tildZs = b * average of the areal Zs
	# mat_Zs = all_y_Zs.reshape(point_res, point_res)
	# areal_Zs = []
	# for i in range(areal_res):
	# 	tmp = [mat_Zs[i*res_per_areal:(i+1)*res_per_areal, j*res_per_areal:(j+1)*res_per_areal] for j in range(areal_res)]
	# 	areal_Zs.append(tmp)
	# areal_Zs = np.array(list(chain.from_iterable(areal_Zs)))

	# areal_Zs_out =  open(output_folder + 'areal_Zs.pickle', 'wb')
	# # areal_Zs_out =  open('dataSimGpDeltas/areal_Zs.pickle', 'wb')
	# pickle.dump(areal_Zs, areal_Zs_out)
	# areal_Zs_out.close()

	# all_y_tildZs = np.array([np.mean(areal_Zs[i])  for i in range(len(areal_Zs))])
	# all_y_tildZs_out = open(output_folder + 'all_y_tildZs.pickle', 'wb')
	# pickle.dump(all_y_tildZs, all_y_tildZs_out)
	# all_y_tildZs_out.close()

	# if gp_deltas_modelOut:
	#     #generate delta(s) of model output tildZofs
	#     cov = gpGaussLikeFuns.cov_matrix(X=X, sigma=sigma_deltas_of_modelOut, w=phi_deltas_of_modelOut)
	#     l_chol_cov = np.linalg.cholesky(cov)
	#     all_y_deltas = np.dot(l_chol_cov, np.random.normal(size=[num_Zs, 1]).reshape(num_Zs))

	#     mat_deltas = all_y_deltas.reshape(point_res, point_res)
	#     areal_deltas = []
	#     for i in range(areal_res):
	#         tmp = [mat_deltas[i*res_per_areal:(i+1)*res_per_areal, j*res_per_areal:(j+1)*res_per_areal] for j in range(areal_res)]
	#         areal_deltas.append(tmp)
	#     areal_deltas = np.array(list(chain.from_iterable(areal_deltas)))


	#     all_y_tildZs = np.array([fun_a_bias(np.mean(areal_coordinate[i], axis=0)) + b * np.mean(areal_Zs[i]) \
	#         + np.mean(areal_deltas[i]) for i in range(len(areal_Zs))])

	#     avg_aBias = np.array([fun_a_bias(np.mean(areal_coordinate[i], axis=0))  for i in range(len(areal_Zs))])
	#     avg_deltas = np.array([np.mean(areal_deltas[i]) for i in range(len(areal_Zs))])

	#     all_y_tildZs_out = open('dataSimGpDeltas/all_y_tildZs.pickle', 'wb')
	#     pickle.dump(all_y_tildZs, all_y_tildZs_out)
	#     all_y_tildZs_out.close()

	#     # fig = plt.figure()
	#     # ax = Axes3D(fig)
	#     # ax.plot_surface(x1, x2, all_y_deltas.reshape(point_res,point_res), rstride=1, cstride=1, cmap=plt.matplotlib.cm.jet)
	#     # ax.set_xlabel('$x_1$')
	#     # ax.set_ylabel('$x_2$')
	#     # ax.set_zlabel('$\^{Z(s)}$')
	#     # plt.savefig('dataSimulated/d3_deltas_res' + str(point_res) + '_a_bias_poly_deg' + str(a_bias_poly_deg) + 'SEED' + str(SEED) + \
	#     #  '_lsZs' + str(phi_Zs) + '_sigZs' + str(sigma_Zs) + \
	#     # '_gpdtsMo' + str(gp_deltas_modelOut) + '_lsdtsMo' +  str(phi_deltas_of_modelOut) + '_sigdtsMo' + str(sigma_deltas_of_modelOut) \
	#     #   + '_noNoi.png')
	#     # plt.close()
	# else:
	#     all_y_tildZs = np.array([fun_a_bias(np.mean(areal_coordinate[i], axis=0)) + \
	#         b * np.mean(areal_Zs[i]) \
	#         + np.mean(np.random.normal(0, res_per_areal * np.sqrt(sigma_deltas_of_modelOut), res_per_areal**2)) \
	#         for i in range(len(areal_Zs))])
	  
	# save all the tildZs

	input_folder = os.getcwd() + '/dataSimulated/seed' + str(SEED)
	all_X_Zs_in = open(input_folder + '/all_X_Zs.pickle', 'rb')
	all_X_Zs = pickle.load(all_X_Zs_in)

	all_y_Zs_in = open(input_folder + '/all_y_Zs.pickle', 'rb')
	all_y_Zs = pickle.load(all_y_Zs_in)

	all_X_tildZs_in = open(input_folder + '/all_X_tildZs.pickle', 'rb')
	all_X_tildZs = pickle.load(all_X_tildZs_in)

	all_y_tildZs_in = open(input_folder + '/all_y_tildZs.pickle', 'rb')
	all_y_tildZs = pickle.load(all_y_tildZs_in)
	
	lower_bound = np.array([-12., -6.5])
	upper_bound = np.array([-3., 3.])
	x1, x2 = np.meshgrid(np.linspace(lower_bound[0], upper_bound[0], areal_res),
						 np.linspace(lower_bound[1], upper_bound[1], areal_res))
	# X = np.vstack([x1.ravel(), x2.ravel()]).T
	# X_tildZs_arealRes = X[idx_sampleTildZs, :]


	# plt.figure()
	# im = plt.imshow(np.flipud(all_y_tildZs.reshape((areal_res, areal_res))), extent=(lower_bound[0], upper_bound[0],lower_bound[1], upper_bound[1]), cmap = plt.matplotlib.cm.jet)
	# plt.scatter(X_tildZs_arealRes[:,0], X_tildZs_arealRes[:,1], s=12, c='k', marker = 'o')
	# cb=plt.colorbar(im)
	# cb.set_label('$\~{Z(s)}$')
	# plt.title('min = %.2f , max = %.2f , avg = %.2f' % (all_y_tildZs.min(), all_y_tildZs.max(), all_y_tildZs.mean()))
	# plt.xlabel('$x1$')
	# plt.ylabel('$x2$')
	# plt.grid()
	# plt.savefig('dataSimulated/d2_tildZs_res' + str(areal_res) + '_a_bias_poly_deg' + str(a_bias_poly_deg) + 'SEED' + str(SEED) + '_noNoi.png')
	# plt.close()

	# fig = plt.figure()
	# ax = Axes3D(fig)
	# ax.plot_surface(x1, x2, all_y_tildZs.reshape(areal_res, areal_res), rstride=1, cstride=1, cmap=plt.matplotlib.cm.jet)
	# ax.set_xlabel('$x_1$')
	# ax.set_ylabel('$x_2$')
	# ax.set_zlabel('$\~{Z(s)}$')
	# plt.savefig(output_folder + 'd3_tildZs_res' + str(areal_res) + '_noNoi.png')
	# plt.savefig('dataSimGpDeltas/d3_tildZs_res' + str(areal_res) + '_noNoi.png')
	# plt.show()
	# plt.close()

	# idx = np.random.randint(0, len(all_y_Zs), num_hatZs)
	idx = random.sample(list(np.arange(len(all_y_Zs))), num_hatZs) # 07/08/2018 This randomly geneated integers are unique
	X_hatZs = all_X_Zs[idx, :] 
	y_hatZs = all_y_Zs[idx]
	#add normal noise with scale = obs_noi_scale
	y_hatZs = y_hatZs + np.random.normal(loc=0., scale = obs_noi_scale, size = num_hatZs)

	# plt.figure()
	# im = plt.imshow(np.flipud(all_y_Zs.reshape((point_res,point_res))), extent=(lower_bound[0], upper_bound[0],lower_bound[1], upper_bound[1]), cmap = plt.matplotlib.cm.jet)
	# plt.scatter(X_hatZs[:,0], X_hatZs[:,1], s=12, c='k', marker = 'o')
	# cb=plt.colorbar(im)
	# cb.set_label('${Z(s)}$')
	# plt.title('min = %.2f , max = %.2f , avg = %.2f' % (all_y_Zs.min(), all_y_Zs.max(), all_y_Zs.mean()))
	# plt.xlabel('$x1$')
	# plt.ylabel('$x2$')
	# plt.grid()
	# plt.savefig(output_folder + 'd2_Zs_res' + str(point_res) + '_noNoi.png')
	# # plt.show()
	# plt.close()

	numMO = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
	
	# numMO = np.array([50])
	moIncreNum=50

	modelOutputX = []
	modelOutputy = []
	X_tildZs = []
	y_tildZs = []

	for idxNumMo in range(len(numMO)):
		num_Mo = len(all_y_tildZs)
		mask_idx = np.arange(num_Mo)
		mask = np.zeros(num_Mo, dtype=bool)
			
		include_idx = np.array(random.sample(list(mask_idx), moIncreNum))
		
		mask[include_idx] = True

		X_tildZs_tmp = all_X_tildZs[mask]
		X_tildZs = np.array(list(X_tildZs_tmp) + modelOutputX)
		print('shape of X_tildZs is ' + str(X_tildZs.shape))
		y_tildZs = np.array(list(all_y_tildZs[mask]) + modelOutputy) 
		y_tildZs = y_tildZs + np.random.normal(loc=0., scale = 0.5, size = len(y_tildZs))

		print('shape of y_tildZs is ' + str(y_tildZs.shape))

		all_X_tildZs = all_X_tildZs[~mask]
		all_y_tildZs = all_y_tildZs[~mask]
		print('size of dataMo is ' + str(all_X_tildZs.shape))
		modelOutputX = list(X_tildZs)
		modelOutputy = list(y_tildZs)

		xtil1 = np.array([X_tildZs[i][:,0] for i in range(len(X_tildZs))])

		xtil2 = np.array([X_tildZs[i][:,1] for i in range(len(X_tildZs))])
		print('SEED, numMO is ' + str((SEED, numMO[idxNumMo])))

		output_folder = os.getcwd() +'/dataSimulated/numObs_' + str(num_hatZs) + \
		'_numMo_' + str(numMO[idxNumMo]) +  '/seed' + str(SEED)
		# output_folder = os.getcwd() +'/dataSimGpDeltas/numObs_' + str(num_hatZs) + \
		# '_numMo_' + str(numMO[idxNumMo]) +  '/seed' + str(SEED)
		if not os.path.exists(output_folder):
			os.makedirs(output_folder)
		output_folder += '/'
		print('output_folder is ' + str(output_folder))

		# plt.figure()
		# plt.scatter(xtil1, xtil2)
		# plt.savefig(output_folder + 'rndmodelOut_' + str(numMO[idxNumMo]) + '.png')
		# # plt.show()
		# plt.close()

		areal_hatZs = []
		for i in range(len(y_tildZs)):
			idx_min_dist = np.argmin(np.array([np.linalg.norm(X_tildZs[i] - X_hatZs[j]) for j in range(len(y_hatZs))]))
			areal_hatZs.append(y_hatZs[idx_min_dist])
		areal_hatZs = np.array(areal_hatZs)

		#save the samples of tildZs
		X_hatZs_out = open(output_folder + 'X_hatZs.pkl', 'wb')
		pickle.dump(X_hatZs, X_hatZs_out)
		X_hatZs_out.close()

		y_hatZs_out = open(output_folder + 'y_hatZs.pkl', 'wb')
		pickle.dump(y_hatZs, y_hatZs_out)
		y_hatZs_out.close()

		X_tildZs_out = open(output_folder + 'X_tildZs.pkl', 'wb')
		pickle.dump(X_tildZs, X_tildZs_out)
		X_tildZs_out.close()

		y_tildZs_out = open(output_folder + 'y_tildZs.pkl', 'wb')
		pickle.dump(y_tildZs, y_tildZs_out)
		y_tildZs_out.close()

		areal_hatZs_out = open(output_folder + 'areal_hatZs.pkl', 'wb')
		pickle.dump(areal_hatZs, areal_hatZs_out)
		areal_hatZs_out.close()
	


def gen_simData(SEED=200, num_hatZs=200,  obs_noi_scale = 0.1, moIncreNum=50):
	np.random.seed(SEED)
	random.seed(SEED)
	all_X_Zs_in = open('dataSimulated/all_X_Zs.pickle', 'rb')
	all_X_Zs = pickle.load(all_X_Zs_in)

	all_y_Zs_in = open('dataSimulated/all_y_Zs.pickle', 'rb')
	all_y_Zs = pickle.load(all_y_Zs_in)

	all_X_tildZs_in = open('dataSimulated/all_X_tildZs.pickle', 'rb')
	all_X_tildZs = pickle.load(all_X_tildZs_in)

	all_y_tildZs_in = open('dataSimulated/all_y_tildZs.pickle', 'rb')
	all_y_tildZs = pickle.load(all_y_tildZs_in)

	areal_Zs_in =  open('dataSimulated/areal_Zs.pickle', 'rb')
	areal_Zs = pickle.load(areal_Zs_in)

	# all_X_Zs_in = open('dataSimGpDeltas/all_X_Zs.pickle', 'rb')
	# all_X_Zs = pickle.load(all_X_Zs_in)

	# all_y_Zs_in = open('dataSimGpDeltas/all_y_Zs.pickle', 'rb')
	# all_y_Zs = pickle.load(all_y_Zs_in)

	# all_X_tildZs_in = open('dataSimGpDeltas/all_X_tildZs.pickle', 'rb')
	# all_X_tildZs = pickle.load(all_X_tildZs_in)

	# all_y_tildZs_in = open('dataSimGpDeltas/all_y_tildZs.pickle', 'rb')
	# all_y_tildZs = pickle.load(all_y_tildZs_in)

	# areal_Zs_in =  open('dataSimGpDeltas/areal_Zs.pickle', 'rb')
	# areal_Zs = pickle.load(areal_Zs_in)
	#sample hatZs
	idx = np.random.randint(0, len(all_y_Zs), num_hatZs)
	X_hatZs = all_X_Zs[idx, :] 
	y_hatZs = all_y_Zs[idx]
	#add normal noise with scale = obs_noi_scale
	y_hatZs = y_hatZs + np.random.normal(loc=0., scale = obs_noi_scale, size = num_hatZs)
	
	numMO = np.array([50])
	# numMO = np.array([50])

	modelOutputX = []
	modelOutputy = []
	X_tildZs = []
	y_tildZs = []

	for idxNumMo in range(len(numMO)):
		num_Mo = len(all_y_tildZs)
		mask_idx = np.arange(num_Mo)
		mask = np.zeros(num_Mo, dtype=bool)
			
		include_idx = np.array(random.sample(list(mask_idx), moIncreNum))
		
		mask[include_idx] = True

		X_tildZs_tmp = all_X_tildZs[mask]
		X_tildZs = np.array(list(X_tildZs_tmp) + modelOutputX)
		print('shape of X_tildZs is ' + str(X_tildZs.shape))
		y_tildZs = np.array(list(all_y_tildZs[mask]) + modelOutputy)

		print('shape of y_tildZs is ' + str(y_tildZs.shape))

		all_X_tildZs = all_X_tildZs[~mask]
		all_y_tildZs = all_y_tildZs[~mask]
		print('size of dataMo is ' + str(all_X_tildZs.shape))
		modelOutputX = list(X_tildZs)
		modelOutputy = list(y_tildZs)

		xtil1 = np.array([X_tildZs[i][:,0] for i in range(len(X_tildZs))])
		xtil2 = np.array([X_tildZs[i][:,1] for i in range(len(X_tildZs))])
		print('SEED, numMO is ' + str((SEED, numMO[idxNumMo])))

		output_folder = os.getcwd() +'/dataSimulated/numObs_' + str(num_hatZs) + \
		'_numMo_' + str(numMO[idxNumMo]) +  '/seed' + str(SEED)
		# output_folder = os.getcwd() +'/dataSimGpDeltas/numObs_' + str(num_hatZs) + \
		# '_numMo_' + str(numMO[idxNumMo]) +  '/seed' + str(SEED)
		if not os.path.exists(output_folder):
			os.makedirs(output_folder)
		output_folder += '/'
		print('output_folder is ' + str(output_folder))

		plt.figure()
		plt.scatter(xtil1, xtil2)
		plt.savefig(output_folder + 'rndmodelOut_' + str(numMO[idxNumMo]) + '.png')
		# plt.show()
		plt.close()

		areal_hatZs = []
		for i in range(len(y_tildZs)):
			idx_min_dist = np.argmin(np.array([np.linalg.norm(X_tildZs[i] - X_hatZs[j]) for j in range(len(y_hatZs))]))
			areal_hatZs.append(y_hatZs[idx_min_dist])
		areal_hatZs = np.array(areal_hatZs)

		#save the samples of tildZs
		X_hatZs_out = open(output_folder + 'X_hatZs.pkl', 'wb')
		pickle.dump(X_hatZs, X_hatZs_out)
		X_hatZs_out.close()

		y_hatZs_out = open(output_folder + 'y_hatZs.pkl', 'wb')
		pickle.dump(y_hatZs, y_hatZs_out)
		y_hatZs_out.close()

		X_tildZs_out = open(output_folder + 'X_tildZs.pkl', 'wb')
		pickle.dump(X_tildZs, X_tildZs_out)
		X_tildZs_out.close()

		y_tildZs_out = open(output_folder + 'y_tildZs.pkl', 'wb')
		pickle.dump(y_tildZs, y_tildZs_out)
		y_tildZs_out.close()

		areal_hatZs_out = open(output_folder + 'areal_hatZs.pkl', 'wb')
		pickle.dump(areal_hatZs, areal_hatZs_out)
		areal_hatZs_out.close()

		# return [X_hatZs, y_hatZs, X_tildZs, y_tildZs, areal_hatZs]


if __name__ == '__main__':
	p = argparse.ArgumentParser()
	p.add_argument('-SEED', type=int, dest='SEED', default=200, help='The simulation index')
	p.add_argument('-o', type=str, dest='output', default=None, help='Output folder')
	p.add_argument('-lsZs', type=float, dest='lsZs', default=0.8, help='lengthscale of the GP covariance for Zs')
	p.add_argument('-lsdtsMo', type=float, dest='lsdtsMo', default=0.2, help='lengthscale of the GP covariance for deltas of model output')
	p.add_argument('-sigZs', type=float, dest='sigZs', default=1.0, help='sigma (marginal variance) of the GP covariance for Zs')
	p.add_argument('-sigdtsMo', type=float, dest='sigdtsMo', default=0.5, help='sigma (marginal variance) of the GP covariance for deltas of model output')
	p.add_argument('-gpdtsMo', dest='gpdtsMo', default=False,  type=lambda x: (str(x).lower() == 'true'), \
		help='flag for whether deltas of model output is a GP')
	p.add_argument('-numMo', type=int, dest='numMo', default=50, help='Number of model outputs used in modelling')
	p.add_argument('-numObs', type=int, dest='numObs', default=200, help='Number of observations used in modelling')
	args = p.parse_args()
	# if args.output is None: args.output = os.getcwd()
	# output_folder = args.output
	# if not os.path.exists(output_folder):
	#     os.makedirs(output_folder)
	# output_folder += '/'
	sim_hatTildZs_With_Plots(SEED =args.SEED, phi_Zs = [args.lsZs], gp_deltas_modelOut = args.gpdtsMo, \
		phi_deltas_of_modelOut = [args.lsdtsMo], sigma_Zs = args.sigZs, sigma_deltas_of_modelOut = args.sigdtsMo)
	# gen_simData(args.SEED, args.numObs)







