import numpy as np
from scipy import linalg
import computeN3Cost
import numbers
import pickle
import os
import argparse
from itertools import chain
# import statsmodels.api as sm
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import pyplot as plt
# # import matplotlib.colors
# import matplotlib as mpl
# plt.switch_backend('agg') # This line is for running code on cluster to make pyplot working on cluster
from rpy2.robjects.packages import importr
from rpy2.robjects import r
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
# from matplotlib.lines import Line2D
import scipy.stats as stats


def subplotsR():
	numpy2ri.activate() 
	r.png('frMap.png')
	flds = importr('fields')
	mp0 = importr("maps")
	mp  =mp0.map("world", region="France")
	r.plot(np.array(mp.rx('x')), np.array(mp.rx('y')), type="l", xlab ='lon', ylab='lat')
	exit(-1)
	x1 = np.arange(1,11)
	y1 = np.arange(1,15)
	z1 = r.outer(x1,y1,"+") 
	print(z1)
	plot_seq = r.pretty(np.arange(1,25), 20)
	jet_colors = r.colorRampPalette(r.c("#00007F", "blue", "#007FFF", "cyan", "#7FFF7F", "yellow", "#FF7F00", "red", "#7F0000"))
	pal = jet_colors(len(plot_seq) - 1)
	x_plot =  np.linspace(-11.7, -3.21, 10)
	y_plot = np.linspace(-6.2, 3.0, 10)	

	r.png('testParfun.png', height=288) 
	r.par(mfrow=r.c(1,2))
	# r.par(mar=r.c(0.5, 2.5, 0.5, 0.5))
	r.image(x1,y1, z1, breaks = plot_seq, col=pal, main='(a)')
	# r.par(mar=r.c(0.5, 4.5, 0.5, 0.5))
	x2 = np.arange(1,11)
	y2 = np.arange(1,15)
	z2 = r.outer(x2,y2,"+") 
	flds.image_plot(x2,y2, z2, breaks = plot_seq, col=pal)
	numpy2ri.deactivate()





if __name__ == '__main__':
	subplotsR()