import os
import sys
import tempfile
import yaml
import zlib
import numpy as np
import simplejson as js
import subprocess as sb
from time import time,sleep
from os import path
from scipy.stats.mstats import mquantiles

try:
    from sklearn.lda import LDA
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier as KNN
    from sklearn.feature_selection import SelectKBest, f_classif

    import samcnet.mh as mh
    from samcnet.mixturepoisson import *
    from samcnet.lori import *
    from samcnet.data import *
except ImportError as e:
    sys.exit("Make sure LD_LIBRARY_PATH is set correctly and that the build"+\
            " directory is populated by waf.\n\n %s" % str(e))

if 'WORKHASH' in os.environ:
    try:
        server = os.environ['SERVER']
    except:
        sys.exit("ERROR in worker: Need SERVER environment variable defined.")

if 'PARAM' in os.environ:
    params = yaml.load(os.environ['PARAM'])
else:
    params = {}

iters = setv(params, 'iters', int(1e4), int)

num_feat = setv(params, 'num_feat', 4, int)
seed = setv(params, 'seed', np.random.randint(10**8), int)
rseed = setv(params, 'rseed', np.random.randint(10**8), int)

Ntrn = setv(params, 'Ntrn', 10, int)
Ntst = setv(params, 'Ntst', 3000, int)
f_glob = setv(params, 'f_glob', 5, int)
subclasses = setv(params, 'subclasses', 0, int)
f_het = setv(params, 'f_het', 0, int)
f_rand = setv(params, 'f_rand', 10, int)
rho = setv(params, 'rho', 0.6, float)
f_tot = setv(params, 'f_tot', f_glob+f_het*subclasses+f_rand, float)
blocksize = setv(params, 'blocksize', 5, int)
mu0 = setv(params, 'mu0', -2.0, float)
mu1 = setv(params, 'mu1', -1.0, float)
sigma0 = setv(params, 'sigma0', 0.2, float)
sigma1 = setv(params, 'sigma1', 0.6, float)

lowd = setv(params, 'lowd', 9.0, float)
highd = setv(params, 'highd', 11.0, float)

output = {}
output['errors'] = {}
errors = output['errors']
np.seterr(all='ignore') # Careful with this
rseed = np.random.randint(10**8)

t1 = time()

trn_data, trn_labels, tst_data, tst_labels = data_yj(params)
norm_trn_data, norm_tst_data = norm(trn_data, tst_data)

norm_trn_data0, norm_trn_data1 = split(norm_trn_data, trn_labels)
norm_tst_data0, norm_tst_data1 = split(norm_tst_data, tst_labels)
trn_data0, trn_data1 = split(trn_data, trn_labels)
tst_data0, tst_data1 = split(tst_data, tst_labels)

#################### CLASSIFICATION ################
sklda = LDA()
skknn = KNN(3, warn_on_equidistant=False)
sksvm = SVC()
sklda.fit(norm_trn_data, trn_labels)
skknn.fit(norm_trn_data, trn_labels)
sksvm.fit(norm_trn_data, trn_labels)
errors['lda'] = (1-sklda.score(norm_tst_data, tst_labels))
errors['knn'] = (1-skknn.score(norm_tst_data, tst_labels))
errors['svm'] = (1-sksvm.score(norm_tst_data, tst_labels))
print("skLDA error: %f" % errors['lda'])
print("skKNN error: %f" % errors['knn'])
print("skSVM error: %f" % errors['svm'])

kappa = 10
bayes0 = GaussianBayes(np.zeros(num_feat), 1, kappa, np.eye(num_feat)*(kappa-1-num_feat), norm_trn_data0)
bayes1 = GaussianBayes(np.zeros(num_feat), 1, kappa, np.eye(num_feat)*(kappa-1-num_feat), norm_trn_data1)

# Gaussian Analytic
gc = GaussianCls(bayes0, bayes1)
errors['gauss'] = gc.approx_error_data(norm_tst_data, tst_labels)
print("Gaussian Analytic error: %f" % errors['gauss'])

# MPM Model
dist0 = MPMDist(trn_data0,kmax=1,priorkappa=180,lammove=0.002,mumove=0.08)
dist1 = MPMDist(trn_data1,kmax=1,priorkappa=180,lammove=0.002,mumove=0.08)
mpm = MPMCls(dist0, dist1) 
mhmc = mh.MHRun(mpm, burn=1000, thin=50)
mhmc.sample(iters,verbose=False)
errors['mpm'] = mpm.approx_error_data(mhmc.db, tst_data, tst_labels,numlam=50)
print("MPM Sampler error: %f" % errors['mpm'])
output['acceptance'] = float(mhmc.accept_loc)/mhmc.total_loc
mhmc.clean_db()

kappa = 200
S0 = (np.ones(4) + (np.eye(4)-1)*0.4) * (kappa - 4 - 1) *0.2
S1 = (np.ones(4) + (np.eye(4)-1)*0.4) * (kappa - 4 - 1) *0.6
priormu1 = np.ones(4)*0.6
priorsigma = np.ones(4) * 0.1
dist0 = MPMDist(trn_data0,kmax=1,priorkappa=280,lammove=0.002,mumove=0.08,S=S0,kappa=kappa,
        priorsigma=priorsigma)
dist1 = MPMDist(trn_data1,kmax=1,priorkappa=280,lammove=0.002,mumove=0.08,S=S1,kappa=kappa,
        priormu=priormu1, priorsigma=priorsigma)
mpm = MPMCls(dist0, dist1) 
mhmc = mh.MHRun(mpm, burn=1000, thin=50)
mhmc.sample(iters,verbose=False)
errors['mpm_prior'] = mpm.approx_error_data(mhmc.db, tst_data, tst_labels,numlam=50)
print("MPM prior Sampler error: %f" % errors['mpm_prior'])
output['acceptance_prior'] = float(mhmc.accept_loc)/mhmc.total_loc
mhmc.clean_db()

output['seed'] = seed
output['time'] = time()-t1

if 'WORKHASH' in os.environ:
    import zmq
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.connect('tcp://'+server+':7000')

    wiredata = zlib.compress(js.dumps(output))
    #wiredata = s.read_db()
    socket.send(os.environ['WORKHASH'], zmq.SNDMORE)
    socket.send(wiredata)
    socket.recv()
    socket.close()
    ctx.term()


