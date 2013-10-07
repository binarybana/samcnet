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
iters = setv(params, 'iters', int(1e4), int)
num_feat = setv(params, 'num_feat', 4, int)
rseed = setv(params, 'rseed', np.random.randint(10**8), int)
seed = setv(params, 'seed', np.random.randint(10**8), int)
np.random.seed(seed)

Ntrn = setv(params, 'Ntrn', 40, int)
assert Ntrn >= 40

trn_data, trn_labels, tst_data, tst_labels = data_tcga(params)

for Ntrn in [40, 35, 30, 25, 20, 15, 10, 5]:
    output = {}
    output['errors'] = {}
    errors = output['errors']

    trn_data = np.vstack (( trn_data[trn_labels==0,:][:Ntrn], trn_data[trn_labels==1,:][:Ntrn] ))
    trn_labels = np.hstack(( np.zeros(Ntrn), np.ones(Ntrn) ))

    print(trn_data.shape)
    print(trn_labels.shape)

    norm_trn_data, norm_tst_data = norm(trn_data, tst_data)
    norm_trn_data0, norm_trn_data1 = split(norm_trn_data, trn_labels)
    trn_data0, trn_data1 = split(trn_data, trn_labels)
    tst_data0, tst_data1 = split(tst_data, tst_labels)

    #################### CLASSIFICATION ################
    t1 = time()
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
    output['time_others'] = time()-t1

    # Gaussian Analytic
    t1 = time()
    kappa = 10
    bayes0 = GaussianBayes(np.zeros(num_feat), 1, kappa, np.eye(num_feat)*(kappa-1-num_feat), norm_trn_data0)
    bayes1 = GaussianBayes(np.zeros(num_feat), 1, kappa, np.eye(num_feat)*(kappa-1-num_feat), norm_trn_data1)

    gc = GaussianCls(bayes0, bayes1)
    errors['gauss'] = gc.approx_error_data(norm_tst_data, tst_labels)
    print("Gaussian Analytic error: %f" % errors['gauss'])
    output['time_gauss_obc'] = time()-t1

    # MPM Model
    t1 = time()
    up = True
    pm0 = np.ones(num_feat) * 0
    pm1 = np.ones(num_feat) * 0
    dist0 = MPMDist(trn_data0,kmax=1,priorkappa=90,lammove=0.01,mumove=0.18,priormu=pm0,d=10.0,usepriors=up)
    dist1 = MPMDist(trn_data1,kmax=1,priorkappa=90,lammove=0.01,mumove=0.18,priormu=pm1,d=10.0,usepriors=up)
    mpm = MPMCls(dist0, dist1) 
    mhmc = mh.MHRun(mpm, burn=3000, thin=20)
    mhmc.sample(iters,verbose=False)
    errors['mpm'] = mpm.approx_error_data(mhmc.db, tst_data, tst_labels,numlam=40)
    print("MPM Sampler error: %f" % errors['mpm'])
    output['time_mp_obc'] = time()-t1
    mhmc.clean_db()

    output['seed'] = seed

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

