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

num_feat = setv(params, 'num_feat', 2, int)
seed = setv(params, 'seed', np.random.randint(10**8), int)
rseed = setv(params, 'rseed', np.random.randint(10**8), int)

Ntrn = setv(params, 'Ntrn', 20, int)
Ntst = setv(params, 'Ntst', 3000, int)
mu0 = setv(params, 'mu0', 0.0, float)
mu1 = setv(params, 'mu1', 0.6, float)
sigma0 = setv(params, 'sigma0', 0.2, float)
sigma1 = setv(params, 'sigma1', 0.6, float)
kappa = setv(params, 'kappa', 30.0, float)

lowd = setv(params, 'lowd', 9.0, float)
highd = setv(params, 'highd', 11.0, float)

num_gen_feat = setv(params, 'num_gen_feat', 20, int)
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
rseed = np.random.randint(10**8)

sel, rawdata, normdata = get_data(data_jk, params)
norm_trn_data = normdata.loc[sel['trn'], sel['feats']]
norm_tst_data = normdata.loc[sel['tst'], sel['feats']]
tst_data = rawdata.loc[sel['tst'], sel['feats']]

t1 = time()
#################### CLASSIFICATION ################
########################################
########################################
########################################
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

bayes0 = GaussianBayes(np.zeros(num_feat), 1, kappa, 
        np.eye(num_feat)*(kappa-1-num_feat), 
        normdata.loc[sel['trn0'], sel['feats']])
bayes1 = GaussianBayes(np.zeros(num_feat), 1, kappa,
        np.eye(num_feat)*(kappa-1-num_feat), 
        normdata.loc[sel['trn1'], sel['feats']])

# Gaussian Analytic
gc = GaussianCls(bayes0, bayes1)
errors['gauss'] = gc.approx_error_data(norm_tst_data, sel['tstl'])
print("Gaussian Analytic error: %f" % errors['gauss'])

########################################
########################################
########################################
########################################
########################################
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
#dist0 = MPMDist(rawdata.loc[sel['trn0'],sel['feats']],kmax=1,priorkappa=200,
        #lammove=0.01,mumove=0.08,#S=S0,kappa=kappa,
        #priormu=pm0,priorsigma=priorsigma, usedata=ud)
#dist1 = MPMDist(rawdata.loc[sel['trn0'],sel['feats']],kmax=1,priorkappa=200,
        #lammove=0.01,mumove=0.08,#S=S1,kappa=kappa,
        #priormu=pm1, priorsigma=priorsigma, usedata=ud)
mpmp = MPMCls(dist0, dist1) 
mhmcp = mh.MHRun(mpmp, burn=burn, thin=thin)
mhmcp.sample(iters,verbose=False)
errors['mpm_prior'] = mpmp.approx_error_data(mhmcp.db, tst_data, sel['tstl'],numlam=numlam)
print("MPM prior Sampler error: %f" % errors['mpm_prior'])
output['acceptance_prior'] = float(mhmcp.accept_loc)/mhmcp.total_loc
########################################
########################################
########################################
########################################
import pylab as p
n,gext,grid = get_grid_data(np.vstack(( rawdata.loc[sel['trn0'],sel['feats']], 
    rawdata.loc[sel['trn1'],sel['feats']])), positive=True)

def myplot(ax,g,data,sel,gext):
    data0 = data.loc[sel['trn0'], sel['feats']]
    data1 = data.loc[sel['trn1'], sel['feats']]
    ax.plot(data0.iloc[:,0], data0.iloc[:,1], 'g.',label='0', alpha=0.5)
    ax.plot(data1.iloc[:,0], data1.iloc[:,1], 'r.',label='1', alpha=0.5)
    ax.legend(fontsize=8, loc='best')

    im = ax.imshow(g, extent=gext, aspect=1.0, origin='lower')
    p.colorbar(im,ax=ax)
    ax.contour(g, [0.0], extent=gext, aspect=1.0, origin='lower', cmap = p.cm.gray)

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
    p.contour(g0.reshape(-1,n), [cont], extent=gext, aspect=asp, origin='lower', cmap = p.cm.gray)
    p.contour(g1.reshape(-1,n), [cont], extent=gext, aspect=asp, origin='lower', cmap = p.cm.gray)
    p.imshow(err, extent=gext, origin='lower', aspect=asp, alpha=0.4, cmap = p.cm.Reds)
    p.contour((g1-g0).reshape(-1,n), [0.0], extent=gext, aspect=asp, origin='lower', cmap = p.cm.gray, linewidth=15.0)
    CS = p.contour(err, [0.4, 0.3, 0.2, 0.1, 0.05], extent=gext, aspect=asp, origin='lower')
    p.clabel(CS, inline=1, fontsize=10, aspect=asp)
    p.show()

##def jitter(x):
    ##rand = np.random.rand
    ##n = x.shape[0]
    ##return (x.T + rand(n)).T
#def jitter(x):
    #rand = np.random.rand
    #return x + rand(*x.shape)-0.5

p.close("all")
gavg = mpm.calc_gavg(mhmc.db, grid, numlam=numlam).reshape(-1,n)
myplot(p.subplot(3,1,1),gavg,rawdata,sel,gext)
gavgc = mpmc.calc_gavg(mhmcc.db, grid, numlam=numlam).reshape(-1,n)
myplot(p.subplot(3,1,2),gavgc,rawdata,sel,gext)
gavgp = mpmp.calc_gavg(mhmcp.db, grid, numlam=numlam).reshape(-1,n)
myplot(p.subplot(3,1,3),gavgp,rawdata,sel,gext)

p.show()

#g0 = mpm1.dist0.calc_db_g(mhmc1.db, mhmc1.db.root.object.dist0, grid)
#g1 = mpm1.dist1.calc_db_g(mhmc1.db, mhmc1.db.root.object.dist1, grid)

##myplot(p.subplot(3,1,3),err.reshape(-1,n),jitter(tst_data0),jitter(tst_data1),gext)

#plot_all(n, gext, grid, trn_data0, trn_data1, g0,g1,gavg)
#plot_concise(n, gext, grid, trn_data0, trn_data1, g0,g1,gavg)

##n,gext,grid = get_grid_data(np.vstack(( norm_trn_data0, norm_trn_data1 )), positive=False)
##myplot(p.subplot(3,1,3),sksvm.decision_function(grid).reshape(-1,n),norm_trn_data0,norm_trn_data1,gext)

#p.figure()
#myplot(p.subplot(1,1,1),gavg,jitter(tst_data0),jitter(tst_data1),gext)
#p.axis(gext)
#mpm1.dist0.plot_traces(mhmc1.db, '/object/dist0', ['sigma'])

output['seed'] = seed
output['time'] = time()-t1

if 'WORKHASH' in os.environ:
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

#mhmc.clean_db()

