#!/usr/bin/python -tt   #This line is to solve any difference between spaces and tabs
import pickle
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

def trace_plot(sample_par, size = 1000):

	plt.figure()
	plt.plot(np.arange(size), sample_par[0,:])
	plt.savefig('sample_log_sigma_size' + str(size) +'.png')
	plt.close()

	plt.figure()
	plt.plot(np.arange(size), sample_par[1,:])
	plt.savefig('sample_log_length_scale_size' + str(size) + '.png')
	plt.close()

	plt.figure()
	plt.plot(np.arange(size), sample_par[2,:])
	plt.savefig('sample_log_obs_noi_scale_size' + str(size) + '.png')
	plt.close()

def read_sample(size = 1000):
	sample_par_in = open('sample_par_size' + str(size) + '.pickle', 'rb')
	sample_par = pickle.load(sample_par_in)
	sample_par = np.exp(sample_par)
	mod = stats.mode(sample_par, axis = 1)
	mean = np.mean(sample_par, axis =1)
	print 'shape is ' + str(sample_par.shape)
	print 'mod is ' + str(mod)
	print 'mean is ' + str(mean)
	return sample_par


if __name__ == '__main__':
	size = 1100
	res = read_sample(size = size)
	# exit(-1)
	trace_plot(res, size = size)
