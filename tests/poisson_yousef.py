import os
from os import path
import subprocess as sb
import numpy as np

import scipy.stats as st
import scipy.stats.distributions as di
import scipy

from sklearn.lda import LDA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN

import samcnet.mh as mh
from samcnet.mixturepoisson import *
from samcnet.lori import *

np.seterr(all='ignore') # Careful with this

# Run Yousef/Jianping RNA Synthetic
currdir = path.abspath('.')
synloc = path.expanduser('~/GSP/research/samc/synthetic/rnaseq')

try:
    os.chdir(synloc)
    #sb.check_call(path.join(synloc, 
	#'gen -i params/mcmcparams -sr 0.05 -lr 9 -hr 10').split())
finally:
    os.chdir(currdir)

data = np.loadtxt(path.join(synloc, 'out','trn.txt'),
	delimiter=',', skiprows=1)
Ntrn = data.shape[0]
data0 = data[:Ntrn/2,:]
data1 = data[Ntrn/2:,:]
norm_data = (data - data.mean(axis=0)) / np.sqrt(data.var(axis=0,ddof=1))
norm_data0 = norm_data[:Ntrn/2,:]
norm_data1 = norm_data[Ntrn/2:,:]
test = np.loadtxt(path.join(synloc, 'out','tst.txt'),
	delimiter=',', skiprows=1)
norm_test = (test - test.mean(axis=0)) / np.sqrt(test.var(axis=0,ddof=1))
N = test.shape[0]
D = data.shape[1]

trn_labels = np.hstack(( np.zeros(Ntrn/2), np.ones(Ntrn/2) ))
tst_labels = np.hstack(( np.zeros(N/2), np.ones(N/2) ))
sklda = LDA()
skknn = KNN(3, warn_on_equidistant=False)
sksvm = SVC()
sklda.fit(norm_data, trn_labels)
skknn.fit(norm_data, trn_labels)
sksvm.fit(norm_data, trn_labels)
print("skLDA error: %f" % (1-sklda.score(norm_test, tst_labels)))
print("skKNN error: %f" % (1-skknn.score(norm_test, tst_labels)))
print("skSVM error: %f" % (1-sksvm.score(norm_test, tst_labels)))

labels = np.hstack((np.zeros(N/2), np.ones(N/2)))
bayes0 = GaussianBayes(np.zeros(D), 1, 8, np.eye(D)*3, norm_data0)
bayes1 = GaussianBayes(np.zeros(D), 1, 8, np.eye(D)*3, norm_data1)

# Gaussian Analytic
gc = GaussianCls(bayes0, bayes1)
print("Gaussian Analytic error: %f" % gc.approx_error_data(norm_test, labels))

# MPM Model
dist0 = MPMDist(data0,kmax=3,mumove=0.03,lammove=0.03,priorkappa=80)
dist1 = MPMDist(data1,kmax=3,mumove=0.03,lammove=0.03,priorkappa=80)
mpm = MPMCls(dist0, dist1) 
mh = mh.MHRun(mpm, burn=100, thin=20)
mh.sample(1e3,verbose=False)
numlam = 10
print("MPM Sampler error: %f" % mpm.approx_error_data(mh.db, test, labels,numlam=numlam))

n,gext,grid = get_grid_data(np.vstack(( data0, data1 )), positive=True)

gavg = mpm.calc_gavg(mh.db, grid, numlam=numlam).reshape(-1,n)
def myplot(ax,g,data0,data1):
    ax.plot(data0[:,0], data0[:,1], 'g.',label='0', alpha=0.3)
    ax.plot(data1[:,0], data1[:,1], 'r.',label='1', alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    im = ax.imshow(g, extent=gext, aspect='equal', origin='lower')
    p.colorbar(im,ax=ax)
    ax.contour(g, [0.0], extent=gext, aspect=1, origin='lower', cmap = p.cm.gray)
p.figure()
myplot(p.gca(),gavg,data0,data1)
p.show()
