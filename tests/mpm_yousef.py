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
    #samples = int(params['samples'])
    #assert samples < 50, "Need to modify mcmcparams"

output = {}
output['errors'] = {}
errors = output['errors']
np.seterr(all='ignore') # Careful with this

# Run Yousef/Jianping RNA Synthetic
currdir = path.abspath('.')
synloc = path.expanduser('~/GSP/research/samc/synthetic/rnaseq')

params=""".d NoOfTrainSamples0 20
.d NoOfTrainSamples1 20
.d NoOfTestSamples0 3000
.d NoOfTestSamples1 3000
.d TotalFeatures 104
.d GlobalFeatures 0
.d HeteroSubTypes 2
.d HeteroFeaturesPerSubType 2
.d RandomFeatures 100
.d CorrBlockSize 2
.d CorrType 1
.f Rho 0.5
.d ScrambleFlag 0
.f Mu_0 0.000000
.f Mu_1 0.500000
.f Sigma_0 0.200000
.f Sigma_1 0.600000
"""

seed = np.random.randint(10**10)
try:
    os.chdir(synloc)
    fid,fname = tempfile.mkstemp(dir='params')
    fid = os.fdopen(fid,'w')
    fid.write(params)
    fid.close()
    inspec = 'gen -i params/%s -sr 0.05 -lr 9 -hr 11 -seed %d' % \
            (path.basename(fname), seed)
    spec = path.join(synloc, inspec).split()
    sb.check_call(spec)
finally:
    os.chdir(currdir)

rawdata = np.loadtxt(path.join(synloc, 'out','%s_trn.txt'%fname),
	delimiter=',', skiprows=1)

Ntrn = rawdata.shape[0]
trn_labels = np.hstack(( np.zeros(Ntrn/2), np.ones(Ntrn/2) ))

selector = SelectKBest(f_classif, k=3)
selector.fit(rawdata, trn_labels)
trn_data = selector.transform(rawdata)
D = trn_data.shape[1]

trn_data0 = trn_data[:Ntrn/2,:]
trn_data1 = trn_data[Ntrn/2:,:]
norm_trn_data = (trn_data - trn_data.mean(axis=0)) / np.sqrt(trn_data.var(axis=0,ddof=1))
norm_trn_data0 = norm_trn_data[:Ntrn/2,:]
norm_trn_data1 = norm_trn_data[Ntrn/2:,:]

raw_tst_data = np.loadtxt(path.join(synloc, 'out','%s_tst.txt'%fname),
	delimiter=',', skiprows=1)
tst_data = selector.transform(raw_tst_data)
norm_tst_data = (tst_data - tst_data.mean(axis=0)) / np.sqrt(tst_data.var(axis=0,ddof=1))
N = tst_data.shape[0]

trn_labels = np.hstack(( np.zeros(Ntrn/2), np.ones(Ntrn/2) ))
tst_labels = np.hstack(( np.zeros(N/2), np.ones(N/2) ))
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

labels = np.hstack((np.zeros(N/2), np.ones(N/2)))
bayes0 = GaussianBayes(np.zeros(D), 1, 8, np.eye(D)*3, norm_trn_data0)
bayes1 = GaussianBayes(np.zeros(D), 1, 8, np.eye(D)*3, norm_trn_data1)

# Gaussian Analytic
gc = GaussianCls(bayes0, bayes1)
errors['gauss'] = gc.approx_error_data(norm_tst_data, labels)
print("Gaussian Analytic error: %f" % errors['gauss'])

# MPM Model
dist0 = MPMDist(trn_data0,kmax=1,priorkappa=180,lammove=0.02,mumove=0.02)
dist1 = MPMDist(trn_data1,kmax=1,priorkappa=180,lammove=0.02,mumove=0.02)
mpm = MPMCls(dist0, dist1) 
mh = mh.MHRun(mpm, burn=1000, thin=50)
mh.sample(1e4,verbose=False)
errors['mpm'] = mpm.approx_error_data(mh.db, tst_data, labels,numlam=50)
print("MPM Sampler error: %f" % errors['mpm'])

output['acceptance'] = float(mh.accept_loc)/mh.total_loc
output['seed'] = seed

def myplot(ax,g,data0,data1,gext):
    ax.plot(data0[:,0], data0[:,1], 'g.',label='0', alpha=0.3)
    ax.plot(data1[:,0], data1[:,1], 'r.',label='1', alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    im = ax.imshow(g, extent=gext, aspect='equal', origin='lower')
    p.colorbar(im,ax=ax)
    ax.contour(g, [0.0], extent=gext, aspect=1, origin='lower', cmap = p.cm.gray)

n,gext,grid = get_grid_data(np.vstack(( data0, data1 )), positive=True)
gavg = mpm.calc_gavg(mh.db, grid, numlam=numlam).reshape(-1,n)
myplot(p.subplot(2,1,1),gavg,data0,data1,gext)

n,gext,grid = get_grid_data(np.vstack(( norm_data0, norm_data1 )), positive=False)
myplot(p.subplot(2,1,2),sksvm.decision_function(grid).reshape(-1,n),norm_data0,norm_data1,gext)

p.show()

if 'WORKHASH' in os.environ:
    import zmq,time
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
