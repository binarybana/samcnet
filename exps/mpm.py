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

Ntrn = setv(params, 'Ntrn', 20, int)
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
    cov0 = sample_invwishart(np.eye(D)*10*sigma0, 10)
    cov1 = sample_invwishart(np.eye(D)*10*sigma1, 10)
    #cov1 = cov0
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

bayes0 = GaussianBayes(np.zeros(num_feat), 1, 8, np.eye(num_feat)*3, norm_trn_data0)
bayes1 = GaussianBayes(np.zeros(num_feat), 1, 8, np.eye(num_feat)*3, norm_trn_data1)

# Gaussian Analytic
gc = GaussianCls(bayes0, bayes1)
errors['gauss'] = gc.approx_error_data(norm_tst_data, tst_labels)
print("Gaussian Analytic error: %f" % errors['gauss'])

# MPM Model
#d0 = np.asarray(mquantiles(trn_data0, 0.75, axis=1)).reshape(-1)
#d1 = np.asarray(mquantiles(trn_data1, 0.75, axis=1)).reshape(-1)
#dist0 = MPMDist(trn_data0,kmax=1,priorkappa=150,lammove=0.01,mumove=0.08,d=d0)
#dist1 = MPMDist(trn_data1,kmax=1,priorkappa=150,lammove=0.01,mumove=0.08,d=d1)
dist0 = MPMDist(trn_data0,kmax=1,priorkappa=150,lammove=0.01,mumove=0.08)
dist1 = MPMDist(trn_data1,kmax=1,priorkappa=150,lammove=0.01,mumove=0.08)
mpm = MPMCls(dist0, dist1) 
mh = mh.MHRun(mpm, burn=1000, thin=50)
mh.sample(iters,verbose=False)
errors['mpm'] = mpm.approx_error_data(mh.db, tst_data, tst_labels,numlam=50)
print("MPM Sampler error: %f" % errors['mpm'])

output['acceptance'] = float(mh.accept_loc)/mh.total_loc
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

mh.db.close()
