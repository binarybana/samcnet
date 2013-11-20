import os
import sys
import tempfile
import yaml
import zlib
import sha
import numpy as np
import pandas as pa
import simplejson as js
import subprocess as sb
from time import time,sleep
from os import path
from scipy.stats.mstats import mquantiles
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.lda import LDA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.feature_selection import SelectKBest, f_classif

import samcnet.mh as mh
from samcnet.mixturepoisson import *
from samcnet.lori import *
from samcnet.data import *
from samcnet.calibrate import *

if 'WORKHASH' in os.environ:
    try:
        server = os.environ['SERVER']
    except:
        sys.exit("ERROR in worker: Need SERVER environment variable defined.")

if 'PARAM' in os.environ:
    params = yaml.load(os.environ['PARAM'])
else:
    params = {}

np.seterr(all='ignore') # Careful with this
num_feat = setv(params, 'num_feat', 4, int)
rseed = setv(params, 'rseed', np.random.randint(10**8), int)
seed = setv(params, 'seed', np.random.randint(10**8), int)

low = setv(params, 'low_filter', 1, int) 
high = setv(params, 'high_filter', 10, int)

# MCMC
mumove = setv(params, 'mumove', 0.08, float)
lammove = setv(params, 'lammove', 0.01, float)
priorkappa = setv(params, 'priorkappa', 150, int)
iters = setv(params, 'iters', int(1e4), int)
burn = setv(params, 'burn', 3000, int)
thin = setv(params, 'thin', 40, int)
numlam = setv(params, 'numlam', 40, int)
d = setv(params, 'd', 10, int)

np.random.seed(seed)

Ntrn = setv(params, 'Ntrn', 40, int)
assert Ntrn >= 40

sel, rawdata, normdata = get_data(data_tcga, params)

for Ntrn in [40, 35, 30, 25, 20, 15, 10, 5]:
    output = {}
    output['errors'] = {}
    errors = output['errors']

    ### Select Ntrn number of training samples
    numsub = sel['trn0'].sum() - Ntrn
    sel = subsample(sel, numsub)

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
            lammove=lammove,mumove=mumove,d=d)
    dist1 = MPMDist(rawdata.loc[sel['trn1'],sel['feats']],priorkappa=priorkappa,
            lammove=lammove,mumove=mumove,d=d)
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
    record_hypers(output, p0, p1)

    dist0 = MPMDist(rawdata.loc[sel['trn0'],sel['feats']],priorkappa=priorkappa,
            lammove=lammove,mumove=mumove,d=d,**p0)
    dist1 = MPMDist(rawdata.loc[sel['trn1'],sel['feats']],priorkappa=priorkappa,
            lammove=lammove,mumove=mumove,d=d,**p1)
    mpmc = MPMCls(dist0, dist1) 
    mhmcc = mh.MHRun(mpmc, burn=burn, thin=thin)
    mhmcc.sample(iters,verbose=False)
    errors['mpmc_calib'] = mpmc.approx_error_data(mhmcc.db, tst_data, sel['tstl'],numlam=numlam)
    print("mpmc Calibrated error: %f" % errors['mpmc_calib'])

    output['acceptance_calib'] = float(mhmcc.accept_loc)/mhmcc.total_loc
    mhmcc.clean_db()

    output['seed'] = seed
    output['time'] = time()-t1

    if 'WORKHASH' in os.environ:
        jobhash, paramhash = os.environ['WORKHASH'].split('|')
        param = yaml.dump({'Ntrn':Ntrn}).strip()
        paramhash = sha.sha(param).hexdigest()
        # submit paramhash
        import redis
        r = redis.StrictRedis(server)
        r.hset('params:sources', paramhash, param)

        os.environ['WORKHASH'] = jobhash.strip() + '|' + paramhash

        import zmq
        ctx = zmq.Context()
        socket = ctx.socket(zmq.REQ)
        socket.connect('tcp://'+server+':7000')

        wiredata = zlib.compress(js.dumps(output))
        socket.send(os.environ['WORKHASH'], zmq.SNDMORE)
        socket.send(wiredata)
        socket.recv()
        socket.close()
        ctx.term()
