import numpy as np
from matplotlib import pyplot as plt 
import pickle
import argparse
import os
import matplotlib.colors

def plot(useSimData):
	
	numMo = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
	if useSimData:
		#The following seeds are for R simulated Gamma transformed data see dataRsimGammaTransformErrorInZtilde
		# seeds = np.array(list(np.arange(200, 206)) + list(np.arange(207, 263)) + list(np.arange(264, 282)) + list(np.arange(283, 300)))
        #The following seeds are for R simulated, more skewed Gamma transformed data (arealRes 25, res of areal cooridated 10 by 10, but res of areal Zs 40*40) see dataSimulated
		seeds = (200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218)
	else:
		seeds = range(120, 220)
	
	rmse_inSamp = []
	avgVar_inSamp = []
	predicAccuracy_inSamp = []

	rmse_outSamp = []
	avgVar_outSamp = []
	predicAccuracy_outSamp = []

	for seed in seeds:
		if useSimData:
			# input_folder = 'dataRsimGammaTransformErrorInZtilde/kriging/numObs_200/seed' + str(seed) + '/'
			input_folder = 'dataSimulated/kriging/numObs_200/seed' + str(seed) + '/'
		else:
			input_folder = 'kriging/seed' + str(seed) + '/' # This is for results of the FR only data
		

		# rmse_in = open(input_folder + 'rmse_krig_inSample.pkl', 'rb')
		# rmse_tmp = pickle.load(rmse_in)
		# rmse_inSamp.append(rmse_tmp) 

		# avgVar_in = open(input_folder + 'avgVar_krig_inSample.pkl', 'rb')
		# avgVar_tmp = pickle.load(avgVar_in) 
		# avgVar_inSamp.append(avgVar_tmp)

		# accuracy_in = open(input_folder + 'predicAccuracy_krig_inSample.pkl', 'rb')
		# predicAccuracy_tmp  = pickle.load(accuracy_in) 
		# predicAccuracy_inSamp.append(predicAccuracy_tmp)

		rmse_in = open(input_folder + 'rmse_krig_outSample.pkl', 'rb')
		rmse_tmp = pickle.load(rmse_in)
		rmse_outSamp.append(rmse_tmp) 

		avgVar_in = open(input_folder + 'avgVar_krig_outSample.pkl', 'rb')
		avgVar_tmp = pickle.load(avgVar_in) 
		avgVar_outSamp.append(avgVar_tmp)

		accuracy_in = open(input_folder + 'predicAccuracy_krig_outSample.pkl', 'rb')
		predicAccuracy_tmp  = pickle.load(accuracy_in) 
		predicAccuracy_outSamp.append(predicAccuracy_tmp)


	# rmse_krig_inSample = np.array(rmse_inSamp)
	# avg_rmse_krig_inSample = np.mean(rmse_krig_inSample)
	# print('avg_rmse_krig_inSample is ' + str(avg_rmse_krig_inSample))
	# median_rmse_krigInsample = np.quantile(rmse_krig_inSample, 0.5, axis=0)
	# print('median_rmse_krigInsample is ' + str(median_rmse_krigInsample))

	# avgVar_krig_inSample = np.array(avgVar_inSamp)
	# avg_avgVar_krig_inSample = np.mean(avgVar_krig_inSample)
	# print('avg_avgVar_krig_inSample is ' + str(avg_avgVar_krig_inSample))
	# median_var_krigInsample = np.quantile(avgVar_krig_inSample, 0.5, axis=0)
	# print('median_var_krigInsample is ' + str(median_var_krigInsample))
	

	# predicAccuracy_krig_inSample = np.array(predicAccuracy_inSamp)
	# avg_predicAccuracy_krig_inSample = np.mean(predicAccuracy_krig_inSample)
	# print('avg_predicAccuracy_krig_inSample is ' + str(avg_predicAccuracy_krig_inSample))
	# median_cov_krigInsample = np.quantile(predicAccuracy_krig_inSample, 0.5, axis=0)
	# print('median_cov_krigInsample is' + str(median_cov_krigInsample))

	rmse_krig_outSample = np.array(rmse_outSamp)
	avg_rmse_krig_outSample = np.mean(rmse_krig_outSample)
	# print('rmse_krig_outSample is ' + str(rmse_krig_outSample))
	print('avg_rmse_krig_outSample is ' + str(avg_rmse_krig_outSample))
	# upperquantile_rmse_krig = np.quantile(rmse_krig_outSample, 0.75, axis=0)
	# lowerquantile_rmse_krig = np.quantile(rmse_krig_outSample, 0.25, axis=0)
	# median_rmse_krig = np.quantile(rmse_krig_outSample, 0.5, axis=0)

	avgVar_krig_outSample = np.array(avgVar_outSamp)
	avg_avgVar_krig_outSample = np.mean(avgVar_krig_outSample)
	# print('avgVar_krig_outSample is ' + str(avgVar_krig_outSample))
	print('avg_avgVar_krig_outSample is ' + str(avg_avgVar_krig_outSample))
	# upperquantile_var_krig = np.quantile(avgVar_krig_outSample, 0.75, axis=0)
	# lowerquantile_var_krig = np.quantile(avgVar_krig_outSample, 0.25, axis=0)
	# median_var_krig = np.quantile(avgVar_krig_outSample, 0.5, axis=0)

	predicAccuracy_krig_outSample = np.array(predicAccuracy_outSamp)
	avg_predicAccuracy_krig_outSample = np.mean(predicAccuracy_krig_outSample)
	print('avg_predicAccuracy_krig_outSample is ' + str(avg_predicAccuracy_krig_outSample))
	# upperquantile_cov_krig = np.quantile(predicAccuracy_krig_outSample, 0.75, axis=0)
	# lowerquantile_cov_krig = np.quantile(predicAccuracy_krig_outSample, 0.25, axis=0)
	# median_cov_krig = np.round(np.quantile(predicAccuracy_krig_outSample, 0.5, axis=0),3)
	# print('median_cov_krig out-of-sample is ' + str(median_cov_krig))

	rmse_bm_inSample = []
	avgVar_bm_inSample = []
	predicAccuracy_bm_inSample = []

	rmse_bm_outSample = []
	avgVar_bm_outSample = []
	predicAccuracy_bm_outSample = []

	for seed in seeds:
		rmse_inSamp = []
		avgVar_inSamp = []
		predicAccuracy_inSamp = []

		rmse_outSamp = []
		avgVar_outSamp = []
		predicAccuracy_outSamp = []
		for j in range(len(numMo)):
			if useSimData:
			# 	input_folder = 'dataRsimGammaTransformErrorInZtilde/numObs_200_numMo_' + str(numMo[j]) \
			# + '/seed' + str(seed) + '/'
				input_folder = 'dataSimulated/numObs_200_numMo_' + str(numMo[j]) \
			+ '/seed' + str(seed) + '/'
			else:
				input_folder = 'Data/FPstart2016020612_FR_numObs_128_numMo_' + str(numMo[j]) \
				+ '/seed' + str(seed) + '/'

			# rmse_in = open(input_folder + 'rmse_inSample.pkl', 'rb')
			# rmse_tmp = pickle.load(rmse_in)
			# rmse_inSamp.append(rmse_tmp) 

			# avgVar_in = open(input_folder + 'avgVar_inSample.pkl', 'rb')
			# avgVar_tmp = pickle.load(avgVar_in) 
			# avgVar_inSamp.append(avgVar_tmp)

			# accuracy_in = open(input_folder + 'predicAccuracy_inSample.pkl', 'rb')
			# predicAccuracy_tmp  = pickle.load(accuracy_in) 
			# predicAccuracy_inSamp.append(predicAccuracy_tmp)
			
			rmse_in = open(input_folder + 'rmse_outSample.pkl', 'rb')
			rmse_tmp = pickle.load(rmse_in)
			rmse_outSamp.append(rmse_tmp) 

			avgVar_in = open(input_folder + 'avgVar_outSample.pkl', 'rb')
			avgVar_tmp = pickle.load(avgVar_in) 
			avgVar_outSamp.append(avgVar_tmp)

			accuracy_in = open(input_folder + 'predicAccuracy_outSample.pkl', 'rb')
			predicAccuracy_tmp  = pickle.load(accuracy_in) 
			predicAccuracy_outSamp.append(predicAccuracy_tmp)
		

		# rmse_inSample = np.array(rmse_inSamp)
		# avgVar_inSample = np.array(avgVar_inSamp)
		# predicAccuracy_inSample = np.array(predicAccuracy_inSamp)

		# rmse_bm_inSample.append(rmse_inSample)
		# avgVar_bm_inSample.append(avgVar_inSample)
		# predicAccuracy_bm_inSample.append(predicAccuracy_inSample)

		rmse_outSample = np.array(rmse_outSamp)
		avgVar_outSample = np.array(avgVar_outSamp)
		predicAccuracy_outSample = np.array(predicAccuracy_outSamp)

		rmse_bm_outSample.append(rmse_outSample)
		avgVar_bm_outSample.append(avgVar_outSample)
		predicAccuracy_bm_outSample.append(predicAccuracy_outSample)

	# rmse_bm_inSample = np.array(rmse_bm_inSample)
	# avg_rmse_bm_inSample = np.mean(rmse_bm_inSample, axis=0)
	# print('avg_rmse_bm_inSample is ' + str(avg_rmse_bm_inSample))
	# median_rmse_bmInsample = np.quantile(rmse_bm_inSample, 0.5, axis=0)
	# print('median_rmse_bmInsample is ' + str(median_rmse_bmInsample))


	# avgVar_bm_inSample = np.array(avgVar_bm_inSample)
	# avg_avgVar_bm_inSample = np.mean(avgVar_bm_inSample, axis =0)
	# print('avg_avgVar_bm_inSample is ' + str(avg_avgVar_bm_inSample))
	# median_var_bmInsample = np.quantile(avgVar_bm_inSample, 0.5, axis=0)
	# print('median_var_bmInsample is ' + str(median_var_bmInsample))

	# predicAccuracy_bm_inSample = np.array(predicAccuracy_bm_inSample)
	# avg_predicAccuracy_bm_inSample = np.mean(predicAccuracy_bm_inSample, axis=0)
	# print('avg_predicAccuracy_bm_inSample is ' + str(avg_predicAccuracy_bm_inSample))
	# median_cov_bmInsample = np.round(np.quantile(predicAccuracy_bm_inSample, 0.5, axis=0), 3)
	# print('median_cov_bmInsample is' + str(median_cov_bmInsample))


	rmse_bm_outSample = np.array(rmse_bm_outSample)
	# print('rmse_bm_outSample is ' + str(rmse_bm_outSample.T)) 
	avg_rmse_bm_outSample = np.mean(rmse_bm_outSample, axis=0)
	# upperquantile_rmse = np.quantile(rmse_bm_outSample, 0.75, axis=0)
	# lowerquantile_rmse = np.quantile(rmse_bm_outSample, 0.25, axis=0)
	# median_rmse = np.quantile(rmse_bm_outSample, 0.5, axis=0)

	# print rmse_bm_outSample[:,0] > rmse_krig_outSample
	# exit(-1)

	avgVar_bm_outSample = np.array(avgVar_bm_outSample)
	# print('avgVar_bm_outSample is ' + str(avgVar_bm_outSample.T))
	avg_avgVar_bm_outSample  = np.mean(avgVar_bm_outSample, axis=0)
	# upperquantile_var = np.quantile(avgVar_bm_outSample, 0.75, axis=0)
	# lowerquantile_var = np.quantile(avgVar_bm_outSample, 0.25, axis=0)
	# median_var = np.quantile(avgVar_bm_outSample, 0.5, axis=0)

	predicAccuracy_bm_outSample = np.array(predicAccuracy_bm_outSample)
	print('predicAccuracy_bm_outSample is ' + str(predicAccuracy_bm_outSample))
	avg_predicAccuracy_bm_outSample = np.mean(predicAccuracy_bm_outSample, axis=0)
	print('avg_predicAccuracy_bm_outSample is ' + str(avg_predicAccuracy_bm_outSample))
	# upperquantile_cov = np.quantile(predicAccuracy_bm_outSample, 0.75, axis=0)
	# lowerquantile_cov = np.quantile(predicAccuracy_bm_outSample, 0.25, axis=0)
	# median_cov = np.round(np.quantile(predicAccuracy_bm_outSample, 0.5, axis=0), 3)
	# print('median_cov_bmOutsample is' + str(median_cov))

	# exit(-1)

	# plt.figure()
	# plt.plot(numMo, median_rmse_bmInsample, 'g-', label= 'BM median')
	# plt.axhline(median_rmse_krigInsample, color = 'g', linestyle = '--', label= 'Kriging median')
	# plt.xlabel('Number of model output')
	# plt.ylabel('RMSE')
	# plt.legend(loc='upper left')
	# plt.title('In sample prediction')
	# plt.savefig('RMSE_inSample.png')
	# plt.show()
	# plt.close()

	# plt.figure()
	# plt.plot(numMo, median_var_bmInsample, 'g-', label= 'BM median')
	# plt.axhline(median_var_krigInsample, color = 'g', linestyle = '--', label='Kriging median')
	# plt.xlabel('Number of model output')
	# plt.ylabel('Average width of prediction variance')
	# plt.legend(loc='upper left')
	# plt.title('In sample prediction')
	# plt.savefig('avgVar_inSample.png')
	# plt.show()
	# plt.close()

	# plt.figure()
	# plt.plot(numMo, median_cov_bmInsample, 'g-', label= 'BM median')
	# plt.axhline(median_cov_krigInsample, color = 'g', linestyle = '--', label='Kriging median')
	# plt.xlabel('Number of model output')
	# plt.ylabel('Prediction accuracy (%)')
	# plt.legend(loc='lower left')
	# plt.title('In sample prediction')
	# plt.savefig('predicAccuracy_inSample.png')
	# plt.show()
	# plt.close()
	output_folder = 'dataSimulated/'
	plt.figure()
	# plt.plot(numMo, median_rmse, 'c-', label= 'BM median')
	plt.plot(numMo, avg_rmse_bm_outSample, 'g-', label= 'DA mean')
	# plt.plot(numMo, upperquantile_rmse, 'b-', label= 'BM 0.75 quantile')
	# plt.plot(numMo, lowerquantile_rmse, 'r-', label= 'BM 0.25 quantile')
	# plt.axhline(median_rmse_krig, color = 'k', linestyle = '--', label='Kriging median')
	plt.axhline(avg_rmse_krig_outSample, color = 'g', linestyle = '--', label= 'Kriging mean')
	# plt.axhline(upperquantile_rmse_krig, color = 'b', linestyle = '--', label='Kriging 0.75')
	# plt.axhline(lowerquantile_rmse_krig, color = 'r', linestyle = '--', label='Kriging 0.25')
	plt.xlabel('Number of model output')
	plt.ylabel('RMSE')
	plt.legend(loc='upper left')
	# plt.title('Out of sample prediction')
	plt.savefig(output_folder + 'RMSE_outSample.png')
	plt.show()
	plt.close()

	plt.figure()
	# plt.plot(numMo, median_var, 'c-', label= 'BM median')
	plt.plot(numMo, avg_avgVar_bm_outSample, 'g-', label= 'DA mean')
	# plt.plot(numMo, upperquantile_var, 'b-', label= 'BM 0.75 quantile')
	# plt.plot(numMo, lowerquantile_var, 'r-', label= 'BM 0.25 quantile')
	# plt.axhline(median_var_krig, color = 'k', linestyle = '--', label='Kriging median')
	plt.axhline(avg_avgVar_krig_outSample, color = 'g', linestyle = '--', label='Kriging mean')
	# plt.axhline(upperquantile_var_krig, color = 'b', linestyle = '--', label='Kriging 0.75')
	# plt.axhline(lowerquantile_var_krig, color = 'r', linestyle = '--', label='Kriging 0.25')
	plt.xlabel('Number of model output')
	plt.ylabel('Average width of 0.95 confidence interval')
	plt.legend(loc='upper left')
	# plt.title('Out of sample prediction')
	plt.savefig(output_folder + 'avgVar_outSample.png')
	plt.show()
	plt.close()

	plt.figure()
	# plt.plot(numMo, median_cov, 'c-', label= 'BM median')
	plt.plot(numMo, avg_predicAccuracy_bm_outSample * 100, 'g-', label= 'DA mean')
	# plt.plot(numMo, upperquantile_cov, 'b-', label= 'BM 0.75 quantile')
	# plt.plot(numMo, lowerquantile_cov, 'r-', label= 'BM 0.25 quantile')
	# plt.axhline(median_cov_krig, color = 'k', linestyle = '--', label='Kriging median')
	plt.axhline(avg_predicAccuracy_krig_outSample * 100, color = 'g', linestyle = '--', label='Kriging mean')
	# plt.axhline(upperquantile_cov_krig, color = 'b', linestyle = '--', label='Kriging 0.75')
	# plt.axhline(lowerquantile_cov_krig, color = 'r', linestyle = '--', label='Kriging 0.25')
	if useSimData:
		plt.ylim(90, 96)
	plt.xlabel('Number of model output')
	plt.ylabel('Coverage probability (%)')
	plt.legend(loc='lower left')
	# plt.title('Out of sample prediction')
	plt.savefig(output_folder + 'predicAccuracy_outSample.png')
	plt.show()
	plt.close()   
def resGridDataFusionVsKrig(numMo):
	# seeds = np.array(list(np.arange(200, 206)) + list(np.arange(207, 263)) + list(np.arange(264, 282)) + list(np.arange(283, 300)))
	# seeds = range(200, 220)
	seeds = (200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218)
	rmse_outSamp = []
	avgVar_outSamp = []
	predicAccuracy_outSamp = []

	for seed in seeds:
		input_folder = 'bmVsKrigGridScale/kriging/numObs_200/seed' + str(seed) + '/'

		rmse_in = open(input_folder + 'rmse_krig_outSample.pkl', 'rb')
		rmse_tmp = pickle.load(rmse_in)
		rmse_outSamp.append(rmse_tmp) 

		avgVar_in = open(input_folder + 'avgVar_krig_outSample.pkl', 'rb')
		avgVar_tmp = pickle.load(avgVar_in) 
		avgVar_outSamp.append(avgVar_tmp)

		accuracy_in = open(input_folder + 'predicAccuracy_krig_outSample.pkl', 'rb')
		predicAccuracy_tmp  = pickle.load(accuracy_in) 
		predicAccuracy_outSamp.append(predicAccuracy_tmp)

	rmse_krig_outSample = np.array(rmse_outSamp)
	rmse_krig_outSample = np.sqrt(np.mean(rmse_krig_outSample**2, axis=0))
	print(rmse_krig_outSample)

	# # lower_bound = np.array([0.] * 2)
	# # upper_bound = np.array([1.] * 2)
	lower_bound = np.array([-12., -6.5])
	upper_bound = np.array([-3., 3.])
	point_res = 1000

	output_folder = os.getcwd() + '/bmVsKrigGridScale/kriging/numObs_200/numMo' + str(numMo) 

	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	output_folder += '/'

	print(rmse_krig_outSample.max())

	
	cmap = plt.cm.jet
	# extract all colors from the .jet map
	cmaplist = [cmap(i) for i in range(cmap.N)]
	# force the first color entry to be grey
	# cmaplist[0] = (.5,.5,.5,1.0)
	# create the new map
	cmap0 = cmap.from_list('Custom cmap', cmaplist, cmap.N)

	# define the bins and normalize
	bounds = np.linspace(0, np.ceil(rmse_krig_outSample.max()),20)
	norm0 = matplotlib.colors.BoundaryNorm(bounds, cmap.N)


	plt.figure()
	im = plt.imshow(np.flipud(rmse_krig_outSample.reshape((point_res,point_res))), extent=(lower_bound[0], upper_bound[0],lower_bound[1], upper_bound[1]), \
		cmap  =cmap0, norm = norm0)
	cb=plt.colorbar(im)
	cb.set_label('${RMSE}$')
	plt.xlabel('$lon$')
	plt.ylabel('$lat$')
	# plt.title('RMSE - Kriging')
	# plt.grid()
	plt.savefig(output_folder + 'Krig_predic_RMSE.png')
	plt.show()
	plt.close()
	

	avgVar_krig_outSample = np.array(avgVar_outSamp)
	avgVar_krig_outSample = np.mean(avgVar_krig_outSample, axis = 0)
	print(avgVar_krig_outSample)


	bounds = np.linspace(6.0, np.ceil(avgVar_krig_outSample.max()),20)
	norm1 = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

	plt.figure()
	im = plt.imshow(np.flipud(avgVar_krig_outSample.reshape((point_res,point_res))), extent=(lower_bound[0], upper_bound[0],lower_bound[1], upper_bound[1]), \
		cmap =cmap0, norm = norm1)
	cb=plt.colorbar(im)
	cb.set_label('${STD}$')
	plt.xlabel('$lon$')
	plt.ylabel('$lat$')
	# plt.title('Width of confidence interval - Kriging')
	# plt.grid()
	plt.savefig(output_folder + 'Krig_predic_std.png')
	plt.show()
	plt.close()

	predicAccuracy_krig_outSample = np.array(predicAccuracy_outSamp)
	predicAccuracy_krig_outSample = np.sum(predicAccuracy_krig_outSample, axis=0)/19.
	print (predicAccuracy_krig_outSample.min(), predicAccuracy_krig_outSample.max()) 
	print (np.sum((predicAccuracy_krig_outSample<0.90).astype(int)))
	print ('avergae predicAccuracy_krig_outSample is ' + str(np.mean(predicAccuracy_krig_outSample)))
	print ('median predicAccuracy_krig_outSample is ' + str(np.median(predicAccuracy_krig_outSample)))


	bounds = np.linspace(0.8, 1.0, 20)
	norm2 = matplotlib.colors.BoundaryNorm(bounds, cmap.N)

	plt.figure()
	im = plt.imshow(np.flipud(predicAccuracy_krig_outSample.reshape((point_res,point_res))), extent=(lower_bound[0], upper_bound[0],lower_bound[1], upper_bound[1]), \
		cmap =cmap0, norm = norm2)
	cb=plt.colorbar(im)
	cb.set_label('${Coverage}$')
	plt.xlabel('$lon$')
	plt.ylabel('$lat$')
	# plt.title('Coverage probability - Kriging')
	# plt.grid()
	plt.savefig(output_folder + 'Krig_predic_coverage.png')
	plt.show()
	plt.close()


	# seeds = np.array(list(np.arange(200, 206)) + list(np.arange(207, 263)) + list(np.arange(264, 282)) + list(np.arange(283, 300)))
	# seeds = range(200, 220)
	

	rmse_outSamp = []
	avgVar_outSamp = []
	predicAccuracy_outSamp = []

	for seed in seeds:     
		input_folder = 'bmVsKrigGridScale/numObs_200_numMo_' + str(numMo) + '/seed' + str(seed) + '/'
		rmse_in = open(input_folder + 'rmse_outSample.pkl', 'rb')
		rmse_tmp = pickle.load(rmse_in)
		rmse_outSamp.append(rmse_tmp) 

		avgVar_in = open(input_folder + 'avgVar_outSample.pkl', 'rb')
		avgVar_tmp = pickle.load(avgVar_in) 
		avgVar_outSamp.append(avgVar_tmp)

		accuracy_in = open(input_folder + 'predicAccuracy_outSample.pkl', 'rb')
		predicAccuracy_tmp  = pickle.load(accuracy_in) 
		predicAccuracy_outSamp.append(predicAccuracy_tmp)

	output_folder = 'bmVsKrigGridScale/numObs_200_numMo_' + str(numMo) + '/' 

	rmse_outSample = np.array(rmse_outSamp)
	rmse_bm_outSample = np.sqrt(np.mean(rmse_outSample**2, axis=0))
	print (rmse_bm_outSample)

	plt.figure()
	im = plt.imshow(np.flipud(rmse_bm_outSample.reshape((point_res,point_res))), extent=(lower_bound[0], upper_bound[0],lower_bound[1], upper_bound[1]), \
		cmap =cmap0, norm=norm0)
	# im = plt.imshow(np.flipud(rmse_bm_outSample.reshape((point_res,point_res))), extent=(lower_bound[0], upper_bound[0],lower_bound[1], upper_bound[1]), \
	# 	cmap =plt.matplotlib.cm.jet, vmin =0, vmax = rmse_bm_outSample.max())
	cb=plt.colorbar(im)
	cb.set_label('${RMSE}$')
	plt.xlabel('$lon$')
	plt.ylabel('$lat$')
	# plt.title('RMSE - Data assimilation')
	# plt.grid()
	plt.savefig(output_folder + 'DF_predic_RMSE.png')
	plt.show()
	plt.close()

	avgVar_outSample = np.array(avgVar_outSamp)
	avgVar_bm_outSample = np.mean(avgVar_outSample, axis = 0)
	print (avgVar_bm_outSample)

	plt.figure()
	im = plt.imshow(np.flipud(avgVar_bm_outSample.reshape((point_res,point_res))), extent=(lower_bound[0], upper_bound[0],lower_bound[1], upper_bound[1]), \
		cmap =cmap0, norm = norm1)
	# im = plt.imshow(np.flipud(avgVar_bm_outSample.reshape((point_res,point_res))), extent=(lower_bound[0], upper_bound[0],lower_bound[1], upper_bound[1]), \
	# 	cmap =plt.matplotlib.cm.jet, vmin =0, vmax = avgVar_bm_outSample.max())
	cb=plt.colorbar(im)
	cb.set_label('${STD}$')
	plt.xlabel('$lon$')
	plt.ylabel('$lat$')
	# plt.title('Width of confidence interval - Data assimilation')
	# plt.grid()
	plt.savefig(output_folder + 'DF_predic_std.png')
	plt.show()
	plt.close()

	predicAccuracy_outSample = np.array(predicAccuracy_outSamp)
	predicAccuracy_bm_outSample = np.sum(predicAccuracy_outSample, axis=0)/19.
	print (predicAccuracy_bm_outSample.min(), predicAccuracy_bm_outSample.max())
	print (np.sum((predicAccuracy_bm_outSample<0.90).astype(int)))
	print ('avergae predicAccuracy_bm_outSample is ' + str(np.mean(predicAccuracy_bm_outSample)))
	print ('median predicAccuracy_bm_outSample is ' + str(np.median(predicAccuracy_bm_outSample)))

	plt.figure()
	im = plt.imshow(np.flipud(predicAccuracy_bm_outSample.reshape((point_res,point_res))), extent=(lower_bound[0], upper_bound[0],lower_bound[1], upper_bound[1]), \
		cmap =cmap0, norm = norm2)
	# im = plt.imshow(np.flipud(predicAccuracy_bm_outSample.reshape((point_res,point_res))), extent=(lower_bound[0], upper_bound[0],lower_bound[1], upper_bound[1]), \
	# 	cmap =plt.matplotlib.cm.jet, vmin =0.8, vmax = predicAccuracy_bm_outSample.max())
	cb=plt.colorbar(im)
	cb.set_label('${Coverage}$')
	plt.xlabel('$lon$')
	plt.ylabel('$lat$')
	# plt.title('Coverage probability - Data assimilation')
	# plt.grid()
	plt.savefig(output_folder + 'DF_predic_coverage.png')
	plt.show()
	plt.close()


if __name__ == '__main__':
	p = argparse.ArgumentParser()
	p.add_argument('-useSimData', dest='useSimData', default=True,  type=lambda x: (str(x).lower() == 'true'),  help='flag for whether to use simulated data')
	p.add_argument('-numMo', type=int, dest='numMo', default=300, help='Number of model outputs used in modelling')
	args = p.parse_args()
	# resGridDataFusionVsKrig(args.numMo)
	plot(args.useSimData)





