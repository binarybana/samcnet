
from samcnet.mixturepoisson import *
import numpy as np
import samcnet.mh as mh
from samcnet.lori import *
import scipy.stats as st
import scipy.stats.distributions as di
import scipy

from sklearn.lda import LDA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN

np.seterr(all='ignore') # Careful with this

######## Generate Data ########
def gen_data(mu, cov, n):
    lams = MVNormal(mu, cov).rvs(n)
    ps = np.empty_like(lams)
    for i in xrange(lams.shape[0]):
	for j in xrange(lams.shape[1]):
	    ps[i,j] = di.poisson.rvs(10* np.exp(lams[i,j]))
    return ps

D = 5
mu0 = np.zeros(D) - 0.5
mu1 = np.zeros(D) + 0.5
#rho0 = -0.4
#rho1 = 0.4
#cov0 = np.array([[1, rho0],[rho0, 1]])
#cov1 = np.array([[1, rho1],[rho1, 1]])
cov0 = np.eye(D) 
cov1 = np.eye(D) 

rseed = np.random.randint(10**6)
dseed = 1
#dseed = np.random.randint(1000)

print("rseed: %d" % rseed)
print("dseed: %d" % dseed)

#np.random.seed(dseed)

trn_data0 = gen_data(mu0,cov0,30)
trn_data1 = gen_data(mu1,cov1,30)

tst_data0 = gen_data(mu0,cov0,3000)
tst_data1 = gen_data(mu1,cov1,3000)

#np.random.seed(rseed)

trn_data = np.vstack(( trn_data0, trn_data1 ))
tst_data = np.vstack(( tst_data0, tst_data1 ))

######## /Generate Data ########

########## Comparison #############
Ntrn = trn_data.shape[0]
norm_trn_data = (trn_data - trn_data.mean(axis=0)) / np.sqrt(trn_data.var(axis=0,ddof=1))
norm_trn_data0 = norm_trn_data[:Ntrn/2,:]
norm_trn_data1 = norm_trn_data[Ntrn/2:,:]
norm_tst_data = (tst_data - tst_data.mean(axis=0)) / np.sqrt(tst_data.var(axis=0,ddof=1))
N = tst_data.shape[0]
D = trn_data.shape[1]

trn_labels = np.hstack(( np.zeros(Ntrn/2), np.ones(Ntrn/2) ))
tst_labels = np.hstack(( np.zeros(N/2), np.ones(N/2) ))
sklda = LDA()
skknn = KNN(3, warn_on_equidistant=False)
sksvm = SVC()
sklda.fit(norm_trn_data, trn_labels)
skknn.fit(norm_trn_data, trn_labels)
sksvm.fit(norm_trn_data, trn_labels)
print("skLDA error: %f" % (1-sklda.score(norm_tst_data, tst_labels)))
print("skKNN error: %f" % (1-skknn.score(norm_tst_data, tst_labels)))
print("skSVM error: %f" % (1-sksvm.score(norm_tst_data, tst_labels)))


# Gaussian Analytic
bayes0 = GaussianBayes(np.zeros(D), 1, 8, np.eye(D)*3, norm_trn_data0)
bayes1 = GaussianBayes(np.zeros(D), 1, 8, np.eye(D)*3, norm_trn_data1)
gc = GaussianCls(bayes0, bayes1)
print("Gaussian Analytic error: %f" % gc.approx_error_data(norm_tst_data, tst_labels))

# MPM Model
dist0 = MPMDist(trn_data0,kmax=1)
dist1 = MPMDist(trn_data1,kmax=1)
mpm = MPMCls(dist0, dist1) 
mh = mh.MHRun(mpm, burn=100, thin=20)
mh.sample(3e3,verbose=False)
print("MPM Sampler error: %f" % mpm.approx_error_data(mh.db, tst_data, tst_labels))
