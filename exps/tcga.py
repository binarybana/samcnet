import os
import sys
import tempfile
import yaml
import zlib
import numpy as np
import pandas as pa
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

def setv(p,s,d,conv=None): 
    if s not in p:
        p[s] = str(d)
        return d
    elif conv is not None:
        return conv(p[s])
    else:
        p[s]

iters = setv(params, 'iters', int(1e4), int)

num_feat = setv(params, 'num_feat', 4, int)
seed = setv(params, 'seed', np.random.randint(10**8), int)

Ntrn = setv(params, 'Ntrn', 5, int)
Ntst = setv(params, 'Ntst', 3000, int)
f_glob = setv(params, 'f_glob', 2, int)
subclasses = setv(params, 'subclasses', 2, int)
f_het = setv(params, 'f_het', 2, int)
f_rand = setv(params, 'f_rand', 10, int)
rho = setv(params, 'rho', 0.6, float)
f_tot = setv(params, 'f_tot', f_glob+f_het*subclasses+f_rand, float)
blocksize = setv(params, 'blocksize', 2, int)
mu0 = setv(params, 'mu0', 0.0, float)
mu1 = setv(params, 'mu1', 0.6, float)
sigma0 = setv(params, 'sigma0', 0.2, float)
sigma1 = setv(params, 'sigma1', 0.6, float)

lowd = setv(params, 'lowd', 9.0, float)
highd = setv(params, 'highd', 11.0, float)

output = {}
output['errors'] = {}
errors = output['errors']
np.seterr(all='ignore') # Careful with this
rseed = np.random.randint(10**8)

def data_tcga(params):
    from sklearn.feature_selection import SelectKBest, f_classif
    store = pa.HDFStore('/home/bana/largeresearch/seq-data/store.h5', complib='blosc', complevel=6)
    brca_all = store['brca_norm_all'].as_matrix()
    paad_res = store['paad_norm'].as_matrix()

    alldata = np.hstack(( paad_res, brca_all  )).T
    alllabels = np.hstack(( np.ones(paad_res.shape[1]), np.zeros(brca_all.shape[1]) ))

    assert Ntrn <= 40
    #trn_data = np.hstack(( paad_res[:, np.random.choice(40,Ntrn)], brca_all[:, np.random.choice(brca_all.shape[0], Ntrn)] )).T
    trn_data = np.hstack(( paad_res[:,:Ntrn], brca_all[:,:Ntrn] )).T
    trn_labels = np.hstack(( np.ones(Ntrn), np.zeros(Ntrn) ))

    tst_data = np.hstack(( paad_res[:,Ntrn:], brca_all[:,Ntrn:] )).T
    tst_labels = np.hstack(( np.ones(paad_res.shape[1] - Ntrn), np.zeros(brca_all.shape[1] - Ntrn) ))

    #grab only some columns
    selector = SelectKBest(f_classif, k=4)
    selector.fit(trn_data, trn_labels)
    pvind = selector.pvalues_.argsort()
    D = 4
    istart = 8000
    return trn_data[:,pvind[istart:istart+D]], trn_labels, tst_data[:, pvind[istart:istart+D]], tst_labels

t1 = time()

trn_data, trn_labels, tst_data, tst_labels = data_tcga(params)

def norm(data):
    return (data - data.mean(axis=0)) / np.sqrt(data.var(axis=0,ddof=1))

norm_trn_data = norm(trn_data)
norm_tst_data = norm(tst_data)

def split(data):
    N = data.shape[0]/2
    return data[:N,:], data[N:,:]

norm_trn_data0, norm_trn_data1 = split(norm_trn_data)
trn_data0, trn_data1 = split(trn_data)
#norm_tst_data0, norm_tst_data1 = split(norm_tst_data)
#tst_data0, tst_data1 = split(tst_data)

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

bayes0 = GaussianBayes(np.zeros(num_feat), 1, 8, np.eye(num_feat)*100, norm_trn_data0)
bayes1 = GaussianBayes(np.zeros(num_feat), 1, 8, np.eye(num_feat)*100, norm_trn_data1)

# Gaussian Analytic
gc = GaussianCls(bayes0, bayes1)
errors['gauss'] = gc.approx_error_data(norm_tst_data, tst_labels)
print("Gaussian Analytic error: %f" % errors['gauss'])

# MPM Model
iters = 4e3
pm0 = np.ones(4) * 10
pm1 = np.ones(4) * 10
dist0 = MPMDist(trn_data0,kmax=1,priorkappa=50,lammove=0.01,mumove=0.08,priormu=pm0)
dist1 = MPMDist(trn_data1,kmax=1,priorkappa=50,lammove=0.01,mumove=0.08,priormu=pm1)
mpm = MPMCls(dist0, dist1) 
mhmc = mh.MHRun(mpm, burn=1000, thin=50)
mhmc.sample(iters,verbose=False)
errors['mpm'] = mpm.approx_error_data(mhmc.db, tst_data, tst_labels,numlam=50)
print("MPM Sampler error: %f" % errors['mpm'])
#mhmc.db.close()

output['acceptance'] = float(mhmc.accept_loc)/mhmc.total_loc

#priorsigma = np.ones(4)*0.1
#pm0 = np.ones(4) * mu0
#pm1 = np.ones(4) * mu1
#ud = True
#dist0 = MPMDist(trn_data0,kmax=1,priorkappa=200,lammove=0.01,mumove=0.08,#S=S0,kappa=kappa,
        #priormu=pm0,priorsigma=priorsigma, usedata=ud)
#dist1 = MPMDist(trn_data1,kmax=1,priorkappa=200,lammove=0.01,mumove=0.08,#S=S1,kappa=kappa,
        #priormu=pm1, priorsigma=priorsigma, usedata=ud)
#mpm = MPMCls(dist0, dist1) 
#mhmc = mh.MHRun(mpm, burn=1000, thin=50)
#mhmc.sample(iters,verbose=False)
#errors['mpm_prior'] = mpm.approx_error_data(mhmc.db, tst_data, tst_labels,numlam=50)
#print("MPM prior Sampler error: %f" % errors['mpm_prior'])
#output['acceptance_prior'] = float(mhmc.accept_loc)/mhmc.total_loc

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

#mhmc.db.close()

