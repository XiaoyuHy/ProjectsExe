
import argparse
import scipy.stats as stats
import statsmodels.api as sm
from matplotlib import pyplot as plt
import pickle
import numpy as np
import os

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

def CI_parUntercy():
	succRate = [] 
	for i in range(1000):
		input_folder = 'DataImogenFrGridMoNotCentre/FPstart2016020612_FR_numObs_128_numMo_500/seed120/idx_theta' + str(i) + '/'
		tmp_in =  open(input_folder + 'predicAccuracy_outSample.pkl', 'rb')
		successRate = pickle.load(tmp_in)
		succRate.append(successRate)
	succRate = np.array(succRate)
	mean_succRate = np.mean(succRate)
	std_succRate = np.std(succRate)
	median_succRate = np.quantile(succRate, 0.5)
	Lqunatile = np.quantile(succRate, 0.025)
	Uqunatile = np.quantile(succRate, 0.975)	
	print(' Mean,std_succRate, Lqunatile, median_succRate, Uqunatile of credible interval for SEED 120 numMo 500 is ' + str((mean_succRate, std_succRate, Lqunatile, median_succRate, Uqunatile))) 

if __name__ == '__main__':
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
	p.add_argument('-parUncerty', dest='parUncerty', default=True,  type=lambda x: (str(x).lower() == 'true'),  help='flag for whether taking into account parameter uncertainty for a fixed seed for GP diagnosis')
	p.add_argument('-parUncertyOverSeeds', dest='parUncertyOverSeeds', default=False,  type=lambda x: (str(x).lower() == 'true'),  help='flag for whether taking into account parameter uncertainty over seeds for GP diagnosis')
	p.add_argument('-idx_theta', dest='idx_theta', type=int, default=0, help='index for the 1000 thetas for a fixed seed')
	p.add_argument('-parUncertyCI', dest='parUncertyCI', default=True,  type=lambda x: (str(x).lower() == 'true'),  help='flag for whether taking into account parameter uncertainty for a fixed seed for GP credible interval diagnosis')

	args = p.parse_args() 
	# plot_qq_parUncerty(args.parUncertyOverSeeds, args.numMo, args.SEED)
	CI_parUntercy()
	