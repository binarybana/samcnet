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
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.lda import LDA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.feature_selection import SelectKBest, f_classif

import samcnet.mh as mh
from samcnet.mixturepoisson import *
from samcnet.lori import *

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
rseed = np.random.randint(10**8)
#seed = 123
seed = setv(params, 'seed', np.random.randint(10**8), int)
Ntrn = setv(params, 'Ntrn', 40, int)

np.random.seed(seed)

output = {}
output['errors'] = {}
errors = output['errors']
np.seterr(all='ignore') # Careful with this

def data_tcga(params):
    store = pa.HDFStore(os.path.expanduser('~/largeresearch/seq-data/store.h5'))
    luad = store['lusc_norm'].as_matrix()
    lusc = store['luad_norm'].as_matrix()

    alldata = np.hstack(( lusc, luad  )).T
    alllabels = np.hstack(( np.ones(lusc.shape[1]), np.zeros(luad.shape[1]) ))

    luad_inds = np.arange(luad.shape[1])
    lusc_inds = np.arange(lusc.shape[1])
    np.random.shuffle(luad_inds)
    np.random.shuffle(lusc_inds)
    trn_data = np.round(np.hstack(( 
        lusc[:,lusc_inds[:Ntrn]], 
        luad[:,luad_inds[:Ntrn]] )).T)
    trn_labels = np.hstack(( np.zeros(Ntrn), np.ones(Ntrn) ))

    tst_data = np.round(np.hstack(( 
        lusc[:,lusc_inds[Ntrn:]], 
        luad[:,luad_inds[Ntrn:]] )).T)
    tst_labels = np.hstack(( np.zeros(lusc.shape[1] - Ntrn), 
        np.ones(luad.shape[1] - Ntrn) ))

    #grab only some columns
    good_cols = (trn_data.mean(axis=0) < 10) & (trn_data.mean(axis=0) > 1)
    low_trn_data = trn_data[:, good_cols]
    low_tst_data = tst_data[:, good_cols]
    selector = SelectKBest(f_classif, k=4)
    selector.fit(low_trn_data, trn_labels)
    pvind = selector.pvalues_.argsort()

    # Univariate T test
    #pvals = np.array([st.ttest_ind(low_trn_data[trn_labels==0,i], low_trn_data[trn_labels==1,i])[1] 
        #for i in range(low_trn_data.shape[1])], dtype=np.float)
    #pvind1 = pvals.argsort()
    #print (pvind1 - pvind).sum()

    feats = np.random.choice(np.arange(low_trn_data.shape[1]), num_feat, replace=False)
    output['feature_start'] = list(feats)
    store.close()
    return low_trn_data[:,feats], trn_labels, low_tst_data[:,feats], tst_labels

def data_test(params):
    trn_data = np.vstack(( np.zeros((10,2)), np.ones((10,2))+2 )) 
    trn_labels = np.hstack(( np.ones(10), np.zeros(10) ))
    tst_data = np.vstack(( np.zeros((1000,2)), np.ones((1000,2)) ))
    tst_labels = np.hstack(( np.ones(1000), np.zeros(1000) ))
    return trn_data, trn_labels, tst_data, tst_labels


trn_data, trn_labels, tst_data, tst_labels = data_tcga(params)
#trn_data, trn_labels, tst_data, tst_labels = data_test(params)
def norm(data):
    return (data - data.mean(axis=0)) / np.sqrt(data.var(axis=0,ddof=1))

norm_trn_data = norm(trn_data)
norm_tst_data = norm(tst_data)

def split(data, labels):
    return data[labels==0,:], data[labels==1,:]

norm_trn_data0, norm_trn_data1 = split(norm_trn_data, trn_labels)
trn_data0, trn_data1 = split(trn_data, trn_labels)
tst_data0, tst_data1 = split(tst_data, tst_labels)

#p.close("all")
#p.figure()
#p.plot(trn_data0[:,0], trn_data0[:,1], 'g.',label='0', alpha=0.5)
#p.plot(trn_data1[:,0], trn_data1[:,1], 'r.',label='1', alpha=0.5)
#p.legend(fontsize=8, loc='best')

#p.figure()
#p.plot(tst_data0[:,0], tst_data0[:,1], 'g.',label='0', alpha=0.5)
#p.plot(tst_data1[:,0], tst_data1[:,1], 'r.',label='1', alpha=0.5)
#p.legend(fontsize=8, loc='best')

#p.show()
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
iters = 8e3
pm0 = np.ones(num_feat) * 1
pm1 = np.ones(num_feat) * 1
dist0 = MPMDist(trn_data0,kmax=1,priorkappa=90,lammove=0.01,mumove=0.18,priormu=pm0,d=10.0,usepriors=up)
dist1 = MPMDist(trn_data1,kmax=1,priorkappa=90,lammove=0.01,mumove=0.18,priormu=pm1,d=10.0,usepriors=up)
mpm = MPMCls(dist0, dist1) 
mhmc = mh.MHRun(mpm, burn=3000, thin=20)
mhmc.sample(iters,verbose=False)
errors['mpm'] = mpm.approx_error_data(mhmc.db, tst_data, tst_labels,numlam=40)
print("MPM Sampler error: %f" % errors['mpm'])
output['time_mp_obc'] = time()-t1
mhmc.db.close()


def myplot(ax,g,data0,data1,gext):
    ax.plot(data0[:,0], data0[:,1], 'g.',label='0', alpha=0.5)
    ax.plot(data1[:,0], data1[:,1], 'r.',label='1', alpha=0.5)
    ax.legend(fontsize=8, loc='best')

    im = ax.imshow(g, extent=gext, aspect=1.0, origin='lower')
    #p.colorbar(im,ax=ax)
    ax.contour(g, extent=gext, aspect=1.0, origin='lower')
    #ax.contour(g, [0.0], extent=gext, aspect=1.0, origin='lower', cmap = p.cm.gray)

def jitter(x):
    rand = np.random.randn
    return x + rand(*x.shape)*0.1

def plot_all(n, gext, grid, data0, data1, g0, g1, gavg):
    Z = np.exp(g0)+np.exp(g1)
    eg0 = np.exp(g0)/Z
    eg1 = np.exp(g1)/Z
    err = np.minimum(eg0,eg1)
    err = err.reshape(-1,n)

    lx,hx,ly,hy = gext
    asp = float(hx-lx) / (hy-ly)
    alp = 1.0
    ms = 8

    p.figure()
    p.subplot(2,2,1)
    p.plot(data0[:,0], data0[:,1], 'g^',label='0', markersize=ms, alpha=alp)
    p.plot(data1[:,0], data1[:,1], 'ro',label='1', markersize=ms, alpha=alp)
    p.legend(fontsize=8, loc='best')
    #p.contour(gavg, extent=gext, aspect=1, origin='lower', cmap = p.cm.gray)
    #p.contour(gavg, [0.0], extent=gext, aspect=1, origin='lower', cmap = p.cm.gray)
    #p.imshow(gavg, extent=gext, aspect=1, origin='lower')
    #p.imshow(g0.reshape(-1,n), extent=gext, aspect=asp, origin='lower')
    #p.colorbar()
    p.contour(g0.reshape(-1,n), extent=gext, aspect=asp, origin='lower', cmap = p.cm.Greens)

    p.subplot(2,2,2)
    p.plot(data0[:,0], data0[:,1], 'g^',label='0', markersize=ms, alpha=alp)
    p.plot(data1[:,0], data1[:,1], 'ro',label='1', markersize=ms, alpha=alp)
    p.legend(fontsize=8, loc='best')
    #p.contour(g0.reshape(-1,n), extent=gext, aspect=1, origin='lower', cmap = p.cm.Greens)
    #p.contour(g1.reshape(-1,n), extent=gext, aspect=1, origin='lower', cmap = p.cm.Reds)
    #p.contour((g1-g0).reshape(-1,n), [0.0], extent=gext, aspect=1, origin='lower', cmap = p.cm.gray)
    #p.imshow((g1-g0).reshape(-1,n), extent=gext, aspect=1, origin='lower')
    #p.imshow(g1.reshape(-1,n), extent=gext, aspect=asp, origin='lower')
    #p.colorbar()
    p.contour(g1.reshape(-1,n), extent=gext, aspect=asp, origin='lower', cmap = p.cm.Reds)

    p.subplot(2,2,3)
    p.plot(data0[:,0], data0[:,1], 'g^',label='0', markersize=ms, alpha=alp)
    p.plot(data1[:,0], data1[:,1], 'ro',label='1', markersize=ms, alpha=alp)
    p.legend(fontsize=8, loc='best')
    #p.imshow(err, extent=gext, origin='lower', aspect=asp)
    #p.colorbar()
    p.contour((g1-g0).reshape(-1,n), [0.0], extent=gext, aspect=asp, origin='lower', cmap = p.cm.gray)
    #p.contour(eg0.reshape(-1,n), extent=gext, aspect=1, origin='lower', cmap = p.cm.Greens)
    #p.contour(eg1.reshape(-1,n), extent=gext, aspect=1, origin='lower', cmap = p.cm.Reds)

    p.subplot(2,2,4)
    p.plot(data0[:,0], data0[:,1], 'g^',label='0', markersize=ms)
    p.plot(data1[:,0], data1[:,1], 'ro',label='1', markersize=ms)
    p.legend(fontsize=8, loc='best')
    p.contour((g1-g0).reshape(-1,n), [0.0], extent=gext, aspect=asp, origin='lower', cmap = p.cm.gray)
    CS = p.contour(err, [0.4, 0.3, 0.2, 0.1, 0.05], extent=gext, aspect=asp, origin='lower')
    p.clabel(CS, inline=1, fontsize=10, aspect=asp)
    p.show()

def plot_concise(n, gext, grid, data0, data1, g0, g1, gavg):
    p.figure()
    Z = np.exp(g0)+np.exp(g1)
    eg0 = np.exp(g0)/Z
    eg1 = np.exp(g1)/Z
    err = np.minimum(eg0,eg1)
    err = err.reshape(-1,n)
    ms=8

    lx,hx,ly,hy = gext
    asp = float(hx-lx) / (hy-ly)
    p.plot(data0[:,0], data0[:,1], 'g^',label='0', markersize=ms)
    p.plot(data1[:,0], data1[:,1], 'ro',label='1', markersize=ms)
    p.legend(fontsize=8, loc='best')
    
    cont = (g0.max() + g1.max()) / 2.0 - 0.6
    #print("g0.max() = %f" % g0.max())
    #print("g1.max() = %f" % g1.max())
    #print("cont = %f" % cont)
    p.contour(g0.reshape(-1,n), [cont], extent=gext, aspect=asp, origin='lower', cmap = p.cm.gray)
    p.contour(g1.reshape(-1,n), [cont], extent=gext, aspect=asp, origin='lower', cmap = p.cm.gray)
    p.imshow(err, extent=gext, origin='lower', aspect=asp, alpha=0.4, cmap = p.cm.Reds)
    p.contour((g1-g0).reshape(-1,n), [0.0], extent=gext, aspect=asp, origin='lower', cmap = p.cm.gray, linewidth=15.0)
    CS = p.contour(err, [0.4, 0.3, 0.2, 0.1, 0.05], extent=gext, aspect=asp, origin='lower')
    p.clabel(CS, inline=1, fontsize=10, aspect=asp)
    p.show()

#n,gext,grid = get_grid_data(np.vstack(( trn_data0, trn_data1 )), positive=True)
#gavg = mpm.calc_gavg(mhmc.db, grid, numlam=20).reshape(-1,n)
#myplot(p.subplot(3,1,1),gavg,trn_data0,trn_data1,gext)

#g0 = mpm.dist0.calc_db_g(mhmc.db, mhmc.db.root.object.dist0, grid)
#g1 = mpm.dist1.calc_db_g(mhmc.db, mhmc.db.root.object.dist1, grid)

#myplot(p.subplot(3,1,3),err.reshape(-1,n),jitter(tst_data0),jitter(tst_data1),gext)

#plot_all(n, gext, grid, trn_data0, trn_data1, g0,g1,gavg)
#plot_concise(n, gext, grid, trn_data0, trn_data1, g0,g1,gavg)

#p.figure()
#n,gext,grid = get_grid_data(np.vstack(( norm_trn_data0, norm_trn_data1 )), positive=False)
#myplot(p.gca(),sksvm.decision_function(grid).reshape(-1,n),norm_trn_data0,norm_trn_data1,gext)
#p.figure()
#myplot(p.gca(),gc.calc_gavg(grid).reshape(-1,n),norm_trn_data0,norm_trn_data1,gext)
#p.show()

#p.figure()
#myplot(p.subplot(1,1,1),gavg,jitter(tst_data0),jitter(tst_data1),gext)
#p.axis(gext)
#p.figure()
#mpm.dist0.plot_traces(mhmc.db, '/object/dist0', ['mu','lam','sigma'])
#mpm.dist1.plot_traces(mhmc.db, '/object/dist1', ['mu','lam'])
#p.show()

#output['acceptance'] = float(mhmc.accept_loc)/mhmc.total_loc

output['seed'] = seed

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

