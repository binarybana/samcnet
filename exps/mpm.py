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
    import samcnet.samc as samc
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

num_feat = setv(params, 'num_feat', 2, int)
#seed = setv(params, 'seed', 1234, int)
seed = setv(params, 'seed', np.random.randint(10**8), int)

Ntrn = setv(params, 'Ntrn', 6, int)
Ntst = setv(params, 'Ntst', 3000, int)
f_glob = setv(params, 'f_glob', 1, int)
subclasses = setv(params, 'subclasses', 2, int)
f_het = setv(params, 'f_het', 1, int)
f_rand = setv(params, 'f_rand', 0, int)
rho = setv(params, 'rho', 0.6, float)
f_tot = setv(params, 'f_tot', f_glob+f_het*subclasses+f_rand, int) 
blocksize = setv(params, 'blocksize', 1, int)
mu0 = setv(params, 'mu0', -0.2, float)
mu1 = setv(params, 'mu1', 0.2, float)
sigma0 = setv(params, 'sigma0', 0.5, float)
sigma1 = setv(params, 'sigma1', 0.2, float)

lowd = setv(params, 'lowd', 9.0, float)
highd = setv(params, 'highd', 11.0, float)

numlam = setv(params, 'numlam', 20, int)

output = {}
output['errors'] = {}
errors = output['errors']
np.seterr(all='ignore') # Careful with this
rseed = np.random.randint(10**8)

def data_yj(params):
    # Run Yousef/Jianping RNA Synthetic
    currdir = path.abspath('.')
    synloc = path.expanduser('~/GSP/research/samc/synthetic/rnaseq')

    YJparams=""".d NoOfTrainSamples0 {Ntrn}
.d NoOfTrainSamples1 {Ntrn}
.d NoOfTestSamples0 {Ntst}
.d NoOfTestSamples1 {Ntst}
.d TotalFeatures {f_tot}
.d GlobalFeatures {f_glob}
.d HeteroSubTypes {subclasses}
.d HeteroFeaturesPerSubType {f_het}
.d RandomFeatures {f_rand}
.d CorrBlockSize {blocksize}
.d CorrType 1
.f Rho {rho}
.d ScrambleFlag 0
.f Mu_0 {mu0}
.f Mu_1 {mu1}
.f Sigma_0 {sigma0}
.f Sigma_1 {sigma1}
""".format(**params)

    try:
        os.chdir(synloc)
        fid,fname = tempfile.mkstemp(dir='params')
        fname = path.basename(fname)
        fid = os.fdopen(fid,'w')
        fid.write(YJparams)
        fid.close()
        inspec = 'gen -i params/%s -sr 0.05 -lr %f -hr %f -seed %d' % \
                (fname, lowd, highd, seed)
        spec = path.join(synloc, inspec).split()
        sb.check_call(spec)
    finally:
        os.chdir(currdir)

    rawdata = np.loadtxt(path.join(synloc, 'out','%s_trn.txt'%fname),
        delimiter=',', skiprows=1)
    trn_labels = np.hstack(( np.zeros(Ntrn), np.ones(Ntrn) ))
    selector = SelectKBest(f_classif, k=num_feat)
    selector.fit(rawdata, trn_labels)
    trn_data = selector.transform(rawdata)
    D = trn_data.shape[1]
    raw_tst_data = np.loadtxt(path.join(synloc, 'out','%s_tst.txt'%fname),
            delimiter=',', skiprows=1)
    tst_data = selector.transform(raw_tst_data)
    N = tst_data.shape[0]
    tst_labels = np.hstack(( np.zeros(N/2), np.ones(N/2) ))
    return trn_data, trn_labels, tst_data, tst_labels

def gen_data(mu, cov, n):
    lams = MVNormal(mu, cov).rvs(n)
    ps = np.empty_like(lams)
    for i in xrange(lams.shape[0]):
        for j in xrange(lams.shape[1]):
            ps[i,j] = di.poisson.rvs(di.uniform.rvs(lowd,highd-lowd)* np.exp(lams[i,j]))
    return ps

def data_jason(params):
    np.random.seed(seed)

    D = num_feat
    lmu0 = np.ones(D) * mu0
    lmu1 = np.ones(D) * mu1
    #rho0 = -0.4
    #rho1 = 0.4
    #cov0 = np.array([[1, rho0],[rho0, 1]])
    #cov1 = np.array([[1, rho1],[rho1, 1]])
    #cov0 = np.eye(D) 
    #cov1 = np.eye(D) 
    cov0 = sample_invwishart(np.eye(D)*(10-D-1)*sigma0, 10) * 0.1
    cov1 = sample_invwishart(np.eye(D)*(10-D-1)*sigma1, 10) * 0.1
    #v1,v2 = 0.2,1.3
    #cov1 = cov0
    #trn_data0 = np.vstack (( gen_data(np.array([v1, v2]),cov0,Ntrn/2), gen_data(np.array([v2,v1]),cov0,Ntrn/2) ))
    #tst_data0 = np.vstack (( gen_data(np.array([v1, v2]),cov0,Ntst/2), gen_data(np.array([v2,v1]),cov0,Ntst/2) ))
    
    trn_data0 = gen_data(lmu0,cov0,Ntrn)
    trn_data1 = gen_data(lmu1,cov1,Ntrn)

    tst_data0 = gen_data(lmu0,cov0,Ntst)
    tst_data1 = gen_data(lmu1,cov1,Ntst)

    trn_data = np.vstack(( trn_data0, trn_data1 ))
    tst_data = np.vstack(( tst_data0, tst_data1 ))
    trn_labels = np.hstack(( np.zeros(Ntrn), np.ones(Ntrn) ))
    tst_labels = np.hstack(( np.zeros(Ntst), np.ones(Ntst) ))
    np.random.seed(rseed)
    return trn_data, trn_labels, tst_data, tst_labels

t1 = time()

#trn_data, trn_labels, tst_data, tst_labels = data_yj(params)
trn_data, trn_labels, tst_data, tst_labels = data_jason(params)

def norm(data):
    return (data - data.mean(axis=0)) / np.sqrt(data.var(axis=0,ddof=1))

norm_trn_data = norm(trn_data)
norm_tst_data = norm(tst_data)

def split(data):
    N = data.shape[0]/2
    return data[:N,:], data[N:,:]

norm_trn_data0, norm_trn_data1 = split(norm_trn_data)
norm_tst_data0, norm_tst_data1 = split(norm_tst_data)
trn_data0, trn_data1 = split(trn_data)
tst_data0, tst_data1 = split(tst_data)

#################### CLASSIFICATION ################
sklda = LDA()
skknn = KNN(3)
sksvm = SVC()
sklda.fit(norm_trn_data, trn_labels)
skknn.fit(norm_trn_data, trn_labels)
sksvm.fit(norm_trn_data, trn_labels)
errors['lda'] = (1-sklda.score(norm_tst_data, tst_labels))
errors['knn'] = (1-skknn.score(norm_tst_data, tst_labels))
errors['svm'] = (1-sksvm.score(norm_tst_data, tst_labels))

bayes0 = GaussianBayes(np.zeros(num_feat), 1, 8, np.eye(num_feat)*3, norm_trn_data0)
bayes1 = GaussianBayes(np.zeros(num_feat), 1, 8, np.eye(num_feat)*3, norm_trn_data1)

# Gaussian Analytic
gc = GaussianCls(bayes0, bayes1)
errors['gauss'] = gc.approx_error_data(norm_tst_data, tst_labels)

# MPM Model
#d0 = np.asarray(mquantiles(trn_data0, 0.75, axis=1)).reshape(-1)
#d1 = np.asarray(mquantiles(trn_data1, 0.75, axis=1)).reshape(-1)
#dist0 = MPMDist(trn_data0,kmax=1,priorkappa=150,lammove=0.01,mumove=0.08,d=d0)
#dist1 = MPMDist(trn_data1,kmax=1,priorkappa=150,lammove=0.01,mumove=0.08,d=d1)

#p.figure()
def myplot(ax,g,data0,data1,gext):
    ax.plot(data0[:,0], data0[:,1], 'g.',label='0', alpha=0.5)
    ax.plot(data1[:,0], data1[:,1], 'r.',label='1', alpha=0.5)
    ax.legend(fontsize=8, loc='best')

    #im = ax.imshow(g, extent=gext, aspect=1.0, origin='lower')
    #p.colorbar(im,ax=ax)
    ax.contour(g, [0.0], extent=gext, aspect=1.0, origin='lower', cmap = p.cm.gray)

n,gext,grid = get_grid_data(np.vstack(( trn_data0, trn_data1 )), positive=True)

iters=8e3
up = True
dist0 = MPMDist(trn_data0,kmax=1,priorkappa=100,lammove=0.05,mumove=0.08,usepriors=up)
dist1 = MPMDist(trn_data1,kmax=1,priorkappa=100,lammove=0.05,mumove=0.08,usepriors=up)
mpm1 = MPMCls(dist0, dist1) 
mhmc1 = mh.MHRun(mpm1, burn=1000, thin=50)
mhmc1.sample(iters,verbose=False)
errors['mpm'] = mpm1.approx_error_data(mhmc1.db, tst_data, tst_labels,numlam=numlam)
print("")
print("skLDA error: %f" % errors['lda'])
print("skKNN error: %f" % errors['knn'])
print("skSVM error: %f" % errors['svm'])
print("gauss error: %f" % errors['gauss'])
print("my MP error: %f" % errors['mpm'])

p.close("all")
gavg = mpm1.calc_gavg(mhmc1.db, grid, numlam=numlam).reshape(-1,n)
#myplot(p.subplot(3,1,1),gavg,trn_data0,trn_data1,gext)

g0 = mpm1.dist0.calc_db_g(mhmc1.db, mhmc1.db.root.object.dist0, grid)
g1 = mpm1.dist1.calc_db_g(mhmc1.db, mhmc1.db.root.object.dist1, grid)

#def jitter(x):
    #rand = np.random.rand
    #n = x.shape[0]
    #return (x.T + rand(n)).T
def jitter(x):
    rand = np.random.rand
    return x + rand(*x.shape)-0.5

#myplot(p.subplot(3,1,3),err.reshape(-1,n),jitter(tst_data0),jitter(tst_data1),gext)

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

plot_all(n, gext, grid, trn_data0, trn_data1, g0,g1,gavg)
plot_concise(n, gext, grid, trn_data0, trn_data1, g0,g1,gavg)

#n,gext,grid = get_grid_data(np.vstack(( norm_trn_data0, norm_trn_data1 )), positive=False)
#myplot(p.subplot(3,1,3),sksvm.decision_function(grid).reshape(-1,n),norm_trn_data0,norm_trn_data1,gext)

p.figure()
myplot(p.subplot(1,1,1),gavg,jitter(tst_data0),jitter(tst_data1),gext)
p.axis(gext)
p.show()

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

#mhmc2.db.close()
