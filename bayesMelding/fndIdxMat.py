
import numpy as np
import itertools

# The following is the function that find the index of one submatrix (bb) in the matrix a
def findIdxMat(a, b):
	d = np.array(list(a[:, :2]))
	c = []
	for i in range(len(b)):
		dx = [idx for idx, val in enumerate(d) if np.bool(np.sum(val == b[i]))]
		c.append(dx)
	c = np.array(list(itertools.chain.from_iterable(c)))
	return c      


if __name__ == '__main__':
	a = np.arange(9).reshape(3,3)
	b =  np.array([0,1,6,7]).reshape(2,2)
	c = findIdxMat(a, b)
	print(c)