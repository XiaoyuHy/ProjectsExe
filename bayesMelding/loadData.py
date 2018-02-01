import numpy as np 
import pandas as pd 


def gp_reg(num = 50, dim=2):
	data = np.array(pd.read_csv('Data/concrete.csv'))
	X = data[:, :-1]
	y = data[:, -1]

	n = X.shape[0]
	d = X.shape[1]
	for i in range(d):
		mu = np.mean(X[:,i])
		sd = np.float(np.std(X[:,i])) 
		if sd==0:
			raise
		else:
			X[:,i] = X[:,i] - mu
			X[:,i] = X[:,i]/sd
	return [X[:num, :dim], y[:num]] 

def gp_cla(num = 10, dim=3):
	X = np.array(pd.read_csv('Data/glass_X.txt',sep=' ', header=None))
	y = np.array(pd.read_csv('Data/glass_y.txt', sep=' ', header=None))

	n = X.shape[0]
	d = X.shape[1]
	for i in range(d):
		mu = np.mean(X[:,i])
		sd = np.float(np.std(X[:,i])) 
		if sd==0:
			raise
		else:
			X[:,i] = X[:,i] - mu
			X[:,i] = X[:,i]/sd
	return [X[:num, :dim], y[:num]] 


# def test():
#     X, y = gp_cla()    
#     print X.shape

# if __name__ == '__main__':
#     test()