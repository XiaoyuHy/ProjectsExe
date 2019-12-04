import netCDF4 as nc4
import numpy as np
from matplotlib import pyplot as plt 
# plt.switch_backend('agg') # This line is for running code on cluster to make pyplot working on cluster
from itertools import chain
import pickle
import argparse
import fnmatch
import os
import random
from rpy2.robjects.packages import importr
from rpy2.robjects import r


def loadNetCdf(SEED=260, num_hatZs=128, num_tildZs = 150, crossValFlag=False, zeroMeanHatZs=True, moIncreNum=50, fold=10, cntry = 'FR', flagUseAvgMo = False):
	np.random.seed(SEED)
	random.seed(SEED)
	analysis_files = fnmatch.filter(os.listdir('../../realData/theo/'), '*analysis*nc')
	obs_files = np.array(fnmatch.filter(os.listdir('../../realData/theo_obs/'), '*obs*nc'))
	substr_analysis_files = np.array([analysis_files[i][:17] for i in range(len(analysis_files))])
	substr_obs_files = np.array([obs_files[i][:17] for i in range(len(obs_files))])
	id_Imogen = np.where(substr_analysis_files == 'FPstart2016020612')[0][0]

	# for i in range(len(analysis_files)):
	i = id_Imogen
	analysis_file = analysis_files[i]
	
	r.load('my_dataMo_FR_elev.RData')
	data = r['dataMoFr']
	dataMo_within_squareAround_FR = np.array(data)
    
    # ####### scale the elev_fp to range [0, 1], though not necessary #####################################################
	# tmp_elev_fp = dataMo_within_squareAround_FR[:,2]
	# scale_elev_fp = (tmp_elev_fp - np.min(tmp_elev_fp))/(np.max(tmp_elev_fp) - np.min(tmp_elev_fp))
	# dataMo_within_squareAround_FR[:,2] = scale_elev_fp
	# print(np.min(dataMo_within_squareAround_FR, axis=0), np.max(dataMo_within_squareAround_FR, axis=0))
	
	# plt.figure()
	# plt.hist(dataMo_within_squareAround_FR[:,2])
	# plt.show()
	# plt.close()
	# exit(-1)

	r.load('my_dataObs_FR.RData')
	data = r['dataObsFr']
	dataObs_within_squareAround_FR = np.array(data)
	print('size of model output is ' + str(dataMo_within_squareAround_FR.shape))
	print('size of obs is ' + str(dataObs_within_squareAround_FR.shape))
  
	numObs = dataObs_within_squareAround_FR.shape[0]

	print('number of obs is ' + str(numObs))

	print('maximum of model output is ' + str(np.sort(dataMo_within_squareAround_FR[:, 3])[-3:]))
 
	# idx = np.random.randint(0, dataObs_within_squareAround_FR.shape[0], num_hatZs)# This randomly geneated integers are NOT unique
	idx = np.array(random.sample(range(numObs), num_hatZs)) # 07/08/2018 This randomly geneated integers are unique
	# print 'idx for seed ' + str(SEED) + ' is ' + str(idx)
	X_hatZs = dataObs_within_squareAround_FR[idx, :2]

	y_hatZs = dataObs_within_squareAround_FR[idx, 2]
	print('maximum of obs is ' + str(y_hatZs.max()))

	output_folder = os.getcwd() + '/Data/'+ analysis_file[:17] + '_' + str(cntry) + '_numObs_' + str(num_hatZs) + '/seed' + str(SEED)
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	output_folder += '/'

	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	output_folder += '/'

	all_X_Zs = dataMo_within_squareAround_FR[:, :3]
	all_X_Zs_out = open(output_folder + 'all_X_Zs.pickle', 'wb')
	pickle.dump(all_X_Zs, all_X_Zs_out)
	all_X_Zs_out.close()

	
	# plt.figure()
	# plt.scatter(X_hatZs[:, 0], X_hatZs[:, 1])
	# plt.savefig('obs_'+ str(numObs) + '.png')
	# # plt.show()
	# plt.close()
	# input_folder = 'Data/FPstart2016020612_FR_numObs_128_numMo_500/seed' + str(SEED) + '/predicMo/'
	# meanPredic_in = open(input_folder + 'meanPredic_outSample.pkl', 'rb')
	# predicMo = pickle.load(meanPredic_in) 
	# print('maximum of predicMo is ' + str(predicMo.max()))

	# input_folder =  'Data/FPstart2016020612_FR_numObs_128_numMo_500/seed' + str(SEED) + '/'
	# X_tildZs_in = open(input_folder + 'X_tildZs.pkl', 'rb')
	# X_tildZs = pickle.load(X_tildZs_in) 
	# X_tildZs = np.array(list(chain.from_iterable(X_tildZs)))
	# print(X_tildZs.shape)
  
	# y_tildZs_in = open(input_folder + 'y_tildZs.pkl', 'rb')
	# y_tildZs = pickle.load(y_tildZs_in)
	# print('maximum of y_tildZs is ' + str(y_tildZs.max()))


	# plt.figure()
	# plt.scatter(X_tildZs[:, 0], X_tildZs[:, 1],  c = y_tildZs, \
	#     cmap=plt.cm.jet, vmin=0, vmax=y_hatZs.max())
	# plt.colorbar()
	# plt.savefig('SEED'+ str(SEED) + 'Mo.png')
	# plt.show()
	# plt.close()


	# plt.figure()
	# plt.scatter(X_tildZs[:, 0], X_tildZs[:, 1],  c = y_tildZs, \
	#     cmap=plt.cm.jet, vmin=0, vmax=y_hatZs.max())
	# plt.scatter(X_hatZs[:, 0], X_hatZs[:, 1], c= y_hatZs, cmap=plt.cm.jet, vmin=0, vmax=y_hatZs.max(), edgecolors='black')
	# plt.colorbar()
	# plt.savefig('SEED'+ str(SEED) + 'MoAndObs.png')
	# plt.show()
	# plt.close()

	# plt.figure()
	# plt.scatter(X_tildZs[:, 0], X_tildZs[:, 1],  c = predicMo, \
	#     cmap=plt.cm.jet, vmin=0, vmax=y_hatZs.max())
	# plt.scatter(X_hatZs[:, 0], X_hatZs[:, 1], c= y_hatZs, cmap=plt.cm.jet, vmin=0, vmax=y_hatZs.max(), edgecolors='black')
	# plt.colorbar()
	# plt.savefig('SEED'+ str(SEED) + 'predicMoAndObs.png')
	# plt.show()
	# plt.close()
	# exit(-1)

	if zeroMeanHatZs:
		mean_y_hatZs = np.mean(y_hatZs)
		print('mean_y_hatZs is ' + str(mean_y_hatZs))
		
		mean_out = open(output_folder + 'mean.pickle', 'wb')
		pickle.dump(mean_y_hatZs, mean_out)
		mean_out.close()


		y_hatZs = y_hatZs - mean_y_hatZs #remove the mean of the observations
		dataMo_within_squareAround_FR[:, 3] = dataMo_within_squareAround_FR[:, 3] - mean_y_hatZs
		print('maximum of model output without mean of Obs is ' + str(np.sort(dataMo_within_squareAround_FR[:, 3])[-3:]))

		all_y_Zs = dataMo_within_squareAround_FR[:, 3]
		all_y_Zs_out = open(output_folder + 'all_y_Zs.pickle', 'wb')
		pickle.dump(all_y_Zs, all_y_Zs_out)
		all_y_Zs_out.close()


	numMO = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
	# numMO = np.array([50])

	modelOutputX = []
	modelOutputy = []
	elev_fp_Output = []
	X_tildZs = []
	y_tildZs = []
	elev_fp = []
	
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
	
	for idxNumMo in range(len(numMO)):

		num_Mo = dataMo_within_squareAround_FR.shape[0]
		mask_idx = np.arange(num_Mo)
		mask = np.zeros(num_Mo, dtype=bool)
			
		include_idx = np.array(random.sample(list(mask_idx), moIncreNum))
		
		mask[include_idx] = True

		X_tildZs_tmp = dataMo_within_squareAround_FR[mask, :2]
		# The following line is for only one point of model output
		# X_tildZs = np.array([X_tildZs_tmp[i].reshape(1, X_tildZs_tmp.shape[1]) for i in range(X_tildZs_tmp.shape[0])] + modelOutputX)
		# The following line creats 10*10 points of coordinates for one model output
		X_tildZs = np.array([X_tildZs_tmp[i] + cell_coords for i in range(X_tildZs_tmp.shape[0])] + modelOutputX)
		print('shape of X_tildZs is ' + str(X_tildZs.shape))
		
		y_tildZs = np.array(list(dataMo_within_squareAround_FR[mask, 3]) + modelOutputy)
		elev_fp = np.array(list(dataMo_within_squareAround_FR[mask, 2]) + elev_fp_Output)
		
		print('shape of y_tildZs is ' + str(y_tildZs.shape))
		print('shape of elev_fp is ' + str(elev_fp.shape))

		dataMo_within_squareAround_FR = dataMo_within_squareAround_FR[~mask, :]
		print('size of dataMo is ' + str(dataMo_within_squareAround_FR.shape))
		modelOutputX = list(X_tildZs)
		modelOutputy = list(y_tildZs)
		elev_fp_Output = list(elev_fp)

		xtil1 = np.array([X_tildZs[i][:,0] for i in range(len(X_tildZs))])
		xtil2 = np.array([X_tildZs[i][:,1] for i in range(len(X_tildZs))])
		print('SEED, numMO is ' + str((SEED, numMO[idxNumMo])))

		# idx_sampleTildZs = random.sample(np.arange(dataMo_within_squareAround_FR.shape[0]), numMO[idxNumMo])
		# X_tildZs_tmp = dataMo_within_squareAround_FR[idx_sampleTildZs, :2]
		# X_tildZs = np.array([X_tildZs_tmp[i].reshape(1, X_tildZs_tmp.shape[1]) for i in range(X_tildZs_tmp.shape[0])])
		# y_tildZs = dataMo_within_squareAround_FR[idx_sampleTildZs, 2]
		# xtil1 = np.array([X_tildZs[i][:,0] for i in range(len(X_tildZs_tmp))])
		# xtil2 = np.array([X_tildZs[i][:,1] for i in range(len(X_tildZs_tmp))])

		output_folder ='Data/'+ analysis_file[:17] + '_' + str(cntry) + '_numObs_' + str(num_hatZs) + \
		'_numMo_' + str(numMO[idxNumMo]) +  '/seed' + str(SEED)
		if not os.path.exists(output_folder):
			os.makedirs(output_folder)
		output_folder += '/'

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
	   
		num_hatZs = X_hatZs.shape[0]
		if crossValFlag:
			mask0 = np.ones(num_hatZs, dtype=bool)
			mask_idx = np.arange(X_hatZs.shape[0])
			num_per_fold = num_hatZs/fold
			for i in range(fold):
				include_idx = mask_idx[i * num_per_fold:(i + 1) * num_per_fold]
				mask = np.copy(mask0)
				mask[include_idx] = False
				X_train = X_hatZs[mask, :] 
				y_train = y_hatZs[mask]
				X_test = X_hatZs[~mask, :]
				y_test = y_hatZs[~mask]
				if i == fold -1:
					include_idx = mask_idx[i * num_per_fold:(i + 1) * num_per_fold + num_hatZs % fold]
					mask = np.copy(mask0)
					mask[include_idx] = False
					X_train = X_hatZs[mask, :] 
					y_train = y_hatZs[mask]
					X_test = X_hatZs[~mask, :]
					y_test = y_hatZs[~mask]
				output_folder_cv = output_folder + 'fold_' + str(i)
				if not os.path.exists(output_folder_cv):
					os.makedirs(output_folder_cv)
				output_folder_cv += '/'
				X_train_out = open(output_folder_cv + 'X_train.pkl', 'wb')
				pickle.dump(X_train, X_train_out)
				if zeroMeanHatZs:
					y_train_out = open(output_folder_cv + 'y_train.pkl', 'wb')
				else:
					y_train_out = open(output_folder_cv + 'y_train_withMean.pkl', 'wb')
				pickle.dump(y_train, y_train_out)
				X_test_out = open(output_folder_cv  + 'X_test.pkl', 'wb')
				pickle.dump(X_test, X_test_out)
				if zeroMeanHatZs:
					y_test_out = open(output_folder_cv + 'y_test.pkl', 'wb')
				else:
					y_test_out = open(output_folder_cv + 'y_test_withMean.pkl', 'wb')
				pickle.dump(y_test, y_test_out)
				X_tildZs_out = open(output_folder_cv + 'X_tildZs.pkl', 'wb')
				pickle.dump(X_tildZs, X_tildZs_out) 
				X_tildZs_out.close()
				y_tildZs_out = open(output_folder_cv + 'y_tildZs.pkl', 'wb')
				pickle.dump(y_tildZs, y_tildZs_out) 
				y_tildZs_out.close()
			return [X_train, y_train, X_test, y_test, X_tildZs, y_tildZs]
		else:
			X_hatZs_out = open(output_folder + 'X_hatZs.pkl', 'wb')
			pickle.dump(X_hatZs, X_hatZs_out) 
			X_hatZs_out.close()
			elev_fp_out = open(output_folder + 'elev_fp.pkl', 'wb')
			pickle.dump(elev_fp, elev_fp_out) 
			elev_fp_out.close()
			if zeroMeanHatZs:
				y_hatZs_out = open(output_folder + 'y_hatZs.pkl', 'wb')
			else:
				y_hatZs_out = open(output_folder + 'y_hatZs_withMean.pkl', 'wb')
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
	p.add_argument('-SEED', type=int, dest='SEED', default=120, help='The simulation index')
	p.add_argument('-numObs', type=int, dest='numObs', default=128, help='Number of observations used in modelling')
	p.add_argument('-numMo', type=int, dest='numMo', default=50, help='Number of model outputs used in modelling')
	p.add_argument('-crossValFlag', dest='crossValFlag', default=False,  type=lambda x: (str(x).lower() == 'true'), \
		help='whether to validate the model using cross validation')
	p.add_argument('-zeroMeanHatZs', dest='zeroMeanHatZs', default=True,  type=lambda x: (str(x).lower() == 'true'), \
		help='whether to zero mean for y_hatZs')
	args = p.parse_args()
	loadNetCdf(args.SEED, args.numObs, args.numMo, args.crossValFlag, args.zeroMeanHatZs)



	



