
#!/usr/bin/python -tt   #This line is to solve any difference between spaces and tabs
import numpy as np
import pandas as pd
import pickle
from sklearn import linear_model
import simData
import argparse
import os

def read_Sim_Data(SEED):
	#read the samples of hatZs
	X_hatZs = np.array(pd.read_csv('simDataFiles/X_hatZs_res100_a_bias_poly_deg2SEED' + str(SEED) + '.txt', sep=" ", header=None))
	y_hatZs = np.array(pd.read_csv('simDataFiles/y_hatZs_res100_a_bias_poly_deg2SEED' + str(SEED) + '.txt', sep=" ", header=None)).reshape(X_hatZs.shape[0])

	#read the samples of tildZs
	X_tildZs_in = open('simDataFiles/X_tildZs_a_bias_poly_deg2SEED' + str(SEED) + '.pickle', 'rb')
	X_tildZs = pickle.load(X_tildZs_in)

	y_tildZs_in = open('simDataFiles/y_tildZs_a_bias_poly_deg2SEED' + str(SEED) + '.pickle', 'rb')
	y_tildZs = pickle.load(y_tildZs_in)

	return[X_hatZs, y_hatZs, X_tildZs, y_tildZs]

if __name__ == '__main__':
	p = argparse.ArgumentParser()
	p.add_argument('-SEED', type=int, dest='SEED', default=0, help='The simulation index')
	p.add_argument('-o', type=str, dest='output', default=None, help='Output folder')
	args = p.parse_args()
	if args.output is None: args.output = os.getcwd()
	output_folder = args.output + '/SEED_' + str(args.SEED) 
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)
	output_folder += '/'
	print 'Output: ' + output_folder
	#sim_hatTildZs_With_Plots()
	_, _, X_tildZs, y_tildZs = simData.sim_hatTildZs_With_Plots(SEED = args.SEED)
	tmp = [np.mean(X_tildZs[i], axis =0) for i in range(len(X_tildZs))]
	X = np.array(tmp)
	regr = linear_model.LinearRegression()
	regr.fit(X, y_tildZs)
	coefficients = regr.coef_
	intercept = regr.intercept_
	print coefficients, intercept
	# print regr.get_params()
  
