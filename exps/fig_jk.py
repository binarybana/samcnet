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
import scipy.stats.distributions as di

try:
    from sklearn.lda import LDA
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier as KNN
    from sklearn.feature_selection import SelectKBest, f_classif

    import samcnet.mh as mh
    from samcnet.mixturepoisson import *
    from samcnet.lori import *
    from samcnet.data import *
    from samcnet.calibrate import *
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

seed = setv(params, 'seed', np.random.randint(10**8), int)
rseed = setv(params, 'rseed', np.random.randint(10**8), int)

# Synthetic Params
Ntrn = setv(params, 'Ntrn', 20, int)
Ntst = setv(params, 'Ntst', 3000, int)
mu0 = setv(params, 'mu0', np.random.randn()*0.2, float)
mu1 = setv(params, 'mu1', np.random.randn()*0.2, float)
sigma0 = setv(params, 'sigma0', di.invgamma.rvs(3), float)
sigma1 = setv(params, 'sigma1', di.invgamma.rvs(3), float)

### For YJ ####
f_glob = setv(params, 'f_glob', 10, int)
subclasses = setv(params, 'subclasses', 2, int)
f_het = setv(params, 'f_het', 20, int)
f_rand = setv(params, 'f_rand', 20, int)
rho = setv(params, 'rho', np.random.rand(), float)
f_tot = setv(params, 'f_tot', f_glob+f_het*subclasses+f_rand, float)
blocksize = setv(params, 'blocksize', 5, int)
############

### For JK ###
num_gen_feat = setv(params, 'num_gen_feat', 20, int)
lowd = setv(params, 'lowd', 9.0, float)
highd = setv(params, 'highd', 11.0, float)
#kappa = setv(params, 'kappa', 2000, float)
#kappa = setv(params, 'kappa', 22.0, float)
##############

# Final number of features
num_feat = setv(params, 'num_feat', 4, int)

# MCMC
mumove = setv(params, 'mumove', 0.08, float)
lammove = setv(params, 'lammove', 0.01, float)
priorkappa = setv(params, 'priorkappa', 150, int)
burn = setv(params, 'burn', 3000, int)
thin = setv(params, 'thin', 40, int)
numlam = setv(params, 'numlam', 40, int)

output = {}
output['errors'] = {}
errors = output['errors']
np.seterr(all='ignore') # Careful with this

sel, rawdata, normdata = get_data(data_yj, params)

norm_trn_data = normdata.loc[sel['trn'], sel['feats']]
norm_tst_data = normdata.loc[sel['tst'], sel['feats']]
tst_data = rawdata.loc[sel['tst'], sel['feats']]

t1 = time()
#################### CLASSIFICATION ################
sklda = LDA()
skknn = KNN(3, warn_on_equidistant=False)
sksvm = SVC()
sklda.fit(norm_trn_data, sel['trnl'])
skknn.fit(norm_trn_data, sel['trnl'])
sksvm.fit(norm_trn_data, sel['trnl'])
errors['lda'] = (1-sklda.score(norm_tst_data, sel['tstl']))
errors['knn'] = (1-skknn.score(norm_tst_data, sel['tstl']))
errors['svm'] = (1-sksvm.score(norm_tst_data, sel['tstl']))
print("skLDA error: %f" % errors['lda'])
print("skKNN error: %f" % errors['knn'])
print("skSVM error: %f" % errors['svm'])

lorikappa = 10
bayes0 = GaussianBayes(np.zeros(num_feat), 1, lorikappa, 
        np.eye(num_feat)*(lorikappa-1-num_feat), 
        normdata.loc[sel['trn0'], sel['feats']])
bayes1 = GaussianBayes(np.zeros(num_feat), 1, lorikappa,
        np.eye(num_feat)*(lorikappa-1-num_feat), 
        normdata.loc[sel['trn1'], sel['feats']])

# Gaussian Analytic
gc = GaussianCls(bayes0, bayes1)
errors['gauss'] = gc.approx_error_data(norm_tst_data, sel['tstl'])
print("Gaussian Analytic error: %f" % errors['gauss'])

# MPM Model
dist0 = MPMDist(rawdata.loc[sel['trn0'],sel['feats']],priorkappa=priorkappa,
        lammove=lammove,mumove=mumove)
dist1 = MPMDist(rawdata.loc[sel['trn1'],sel['feats']],priorkappa=priorkappa,
        lammove=lammove,mumove=mumove)
mpm = MPMCls(dist0, dist1) 
mhmc = mh.MHRun(mpm, burn=burn, thin=thin)
mhmc.sample(iters,verbose=False)
errors['mpm'] = mpm.approx_error_data(mhmc.db, tst_data, sel['tstl'],numlam=numlam)
print("MPM Sampler error: %f" % errors['mpm'])

output['acceptance'] = float(mhmc.accept_loc)/mhmc.total_loc
mhmc.clean_db()
########################################
########################################
########################################
########################################
########################################
# Calibrated MPM Model
p0, p1 = calibrate(rawdata, sel, params)

dist0 = MPMDist(rawdata.loc[sel['trn0'],sel['feats']],priorkappa=priorkappa,
        lammove=lammove,mumove=mumove,**p0)
dist1 = MPMDist(rawdata.loc[sel['trn1'],sel['feats']],priorkappa=priorkappa,
        lammove=lammove,mumove=mumove,**p1)
mpmc = MPMCls(dist0, dist1) 
mhmcc = mh.MHRun(mpmc, burn=burn, thin=thin)
mhmcc.sample(iters,verbose=False)
errors['mpmc_calib'] = mpmc.approx_error_data(mhmcc.db, tst_data, sel['tstl'],numlam=numlam)
print("mpmc Calibrated error: %f" % errors['mpmc_calib'])

output['acceptance_calib'] = float(mhmcc.accept_loc)/mhmcc.total_loc
mhmcc.clean_db()
########################################
########################################
########################################
########################################
########################################
priorsigma = np.ones(4)*0.1
pm0 = np.ones(4) * mu0
pm1 = np.ones(4) * mu1
dist0 = MPMDist(rawdata.loc[sel['trn0'],sel['feats']],priorkappa=priorkappa,
        lammove=lammove,mumove=mumove,
        priormu=pm0,priorsigma=priorsigma)
dist1 = MPMDist(rawdata.loc[sel['trn1'],sel['feats']],priorkappa=priorkappa,
        lammove=lammove,mumove=mumove,
        priormu=pm1,priorsigma=priorsigma)
mpmp = MPMCls(dist0, dist1) 
mhmcp = mh.MHRun(mpmp, burn=burn, thin=thin)
mhmcp.sample(iters,verbose=False)
errors['mpm_prior'] = mpmp.approx_error_data(mhmcp.db, tst_data, sel['tstl'],numlam=numlam)
print("MPM prior Sampler error: %f" % errors['mpm_prior'])
output['acceptance_prior'] = float(mhmcp.accept_loc)/mhmcp.total_loc
mhmcp.clean_db()
########################################
########################################
########################################
########################################

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


