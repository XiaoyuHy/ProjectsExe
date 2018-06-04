import numpy as np
import pickle
count0 = []
count1 = []
seeds =200
for seed in range(100, 300):
	fileName = 'Output200seeds/SEED_'+ str(seed) + '_withPrior_False_fixMb_True_onlyOptimCovPar_False_poly_deg_2_lsZs_0.1_lsdtsMo_0.2_sigZs_1.5_sigdtsMo_1.0_gpdtsMo_True_useGradsFlag_True_repeat5/resOptim.pkl'
	res_in = open(fileName, 'rb')
	res = pickle.load(res_in)
	res_in.close()
	count0.append(res['count_in_confiInterv'])
	count1.append(res['count_in_confiInterv_rounded'])
count0 = np.array(count0)
count1 = np.array(count1)

print 'number of parameters within 95 percent confidence interval without rounding is ' + str(np.sum(count0))
print 'number of parameters within 95 percent confidence interval with rounding is ' + str(np.sum(count1))

confiLev0 = np.float(np.sum(count0))/(seeds *9)
print 'confidencelevel of ' + str(seeds) + ' seeds without rounding is '+ str(confiLev0)
confiLev1 = np.float(np.sum(count1))/(seeds *9)
print 'confidencelevel of ' + str(seeds) + ' seeds with rounding is '+ str(confiLev1)
