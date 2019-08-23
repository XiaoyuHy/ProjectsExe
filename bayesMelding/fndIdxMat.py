
import numpy as np
import itertools
import pickle
from matplotlib import pyplot as plt
import matplotlib.colors
import matplotlib as mpl

# The following is the function that find the index of one submatrix (bb) in the matrix a
def findIdxMat(a, b):
	d = np.array(list(a[:, :2]))
	c = []
	for i in range(len(b)):
		dx = [idx for idx, val in enumerate(d) if np.bool(np.sum(val == b[i]))]
		c.append(dx)
	c = np.array(list(itertools.chain.from_iterable(c)))
	return c   

def plot():
	Zmos_in = open('z_mos.pickle', 'rb')   
	z_mos = pickle.load(Zmos_in)
	cmap = plt.cm.jet
		# define the bins and normalize
	# norm = mpl.colors.Normalize(vmin=6, vmax=42)
	# bounds = np.round(np.linspace(6, 42, 22),0)
	bounds = np.arange(6,43)
	norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
	print(cmap.N)

	fig = plt.figure()
	fig.set_rasterized(True)
	ax = fig.add_subplot(111)
	ax.set_rasterized(True)
	im=ax.imshow(np.flipud(np.array(z_mos).reshape((500, 500))), extent=(-11.7, -3.21, -6.2, 3.0), cmap  =cmap, norm = norm)
   
	plt.xlabel('$Longitude$')
	plt.ylabel('$Latitude$')
	plt.colorbar(im)
	plt.show()
	plt.close()
	  


if __name__ == '__main__':
	# plot()
	# exit(-1)
	a = np.arange(9).reshape(3,3)
	b =  np.array([0,1,6,7]).reshape(2,2)
	c = findIdxMat(a, b)
	print(c)