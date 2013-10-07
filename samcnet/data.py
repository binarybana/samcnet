import os
import sys
import tempfile
import subprocess as sb
import numpy as np
import pandas as pa
from os import path

from sklearn.feature_selection import SelectKBest, f_classif

import scipy.stats as st
import scipy.stats.distributions as di

from statsmodels.sandbox.distributions.mv_normal import MVT,MVNormal

from lori import sample_invwishart

def norm(data1, data2):
    mu = data1.mean(axis=0)
    std = np.sqrt(data1.var(axis=0, ddof=1))
    return (data1 - mu) / std, (data2 - mu) / std

def split(data, labels):
    return data[labels==0,:], data[labels==1,:]

param_template = """.d NoOfTrainSamples0 {Ntrn}
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
"""

def setv(p,s,d,conv=None): 
    if s not in p:
        p[s] = d
        return d
    elif conv is not None:
        return conv(p[s])
    else:
        p[s]

def data_yj(params):
    Ntrn = params['Ntrn']
    num_feat = params['num_feat']
    lowd = params['lowd']
    highd = params['highd']
    seed = params['seed']
    # Run Yousef/Jianping RNA Synthetic
    currdir = path.abspath('.')
    synloc = path.expanduser('~/GSP/research/samc/synthetic/rnaseq')

    YJparams = param_template.format(**params)

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
    except Exception as e:
        print "ERROR in data_yj: " + str(e)
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

def gen_data(mu, cov, n, lowd, highd):
    lams = MVNormal(mu, cov).rvs(n)
    ps = np.empty_like(lams)
    for i in xrange(lams.shape[0]):
        for j in xrange(lams.shape[1]):
            ps[i,j] = di.poisson.rvs(di.uniform.rvs(lowd,highd-lowd)* np.exp(lams[i,j]))
    return ps

def data_jk(params):
    D = params['num_feat']
    Ntrn = params['Ntrn']
    Ntst = params['Ntst']
    sigma0 = params['sigma0']
    sigma1 = params['sigma1']
    mu0 = params['mu0']
    mu1 = params['mu1']
    kappa = params['kappa']
    seed = params['seed']
    rseed = params['rseed']
    lowd = params['lowd']
    highd = params['highd']

    np.random.seed(seed)

    lmu0 = np.ones(D) * mu0
    lmu1 = np.ones(D) * mu1
    #rho0 = -0.4
    #rho1 = 0.4
    #cov0 = np.array([[1, rho0],[rho0, 1]])
    #cov1 = np.array([[1, rho1],[rho1, 1]])
    #cov0 = np.eye(D) 
    #cov1 = np.eye(D) 
    cov0 = sample_invwishart(np.eye(D)*(kappa-D-1)*sigma0, kappa) 
    cov1 = sample_invwishart(np.eye(D)*(kappa-D-1)*sigma1, kappa) 
    #v1,v2 = 0.2,1.3
    #cov1 = cov0
    #trn_data0 = np.vstack (( gen_data(np.array([v1, v2]),cov0,Ntrn/2), 
        #gen_data(np.array([v2,v1]),cov0,Ntrn/2) ))
    #tst_data0 = np.vstack (( gen_data(np.array([v1, v2]),cov0,Ntst/2), 
        #gen_data(np.array([v2,v1]),cov0,Ntst/2) ))
    
    trn_data0 = gen_data(lmu0,cov0,Ntrn,lowd,highd)
    trn_data1 = gen_data(lmu1,cov1,Ntrn,lowd,highd)

    tst_data0 = gen_data(lmu0,cov0,Ntst,lowd,highd)
    tst_data1 = gen_data(lmu1,cov1,Ntst,lowd,highd)

    trn_data = np.vstack(( trn_data0, trn_data1 ))
    tst_data = np.vstack(( tst_data0, tst_data1 ))
    trn_labels = np.hstack(( np.zeros(Ntrn), np.ones(Ntrn) ))
    tst_labels = np.hstack(( np.zeros(Ntst), np.ones(Ntst) ))
    np.random.seed(rseed)
    return trn_data, trn_labels, tst_data, tst_labels

def data_tcga(params):
    Ntrn = params['Ntrn']
    num_feat = params['num_feat']
    store = pa.HDFStore(os.path.expanduser('~/largeresearch/seq-data/store.h5'))
    luad = store['lusc_norm'].as_matrix()
    lusc = store['luad_norm'].as_matrix()
    print luad.shape

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
    store.close()
    return low_trn_data[:,feats], trn_labels, low_tst_data[:,feats], tst_labels

def data_karen(params):
    Ntrn = params['Ntrn']
    num_feat = params['num_feat']
    datapath = os.path.expanduser('~/GSP/research/samc/samcnet/data/')
    store = pa.HDFStore(datapath+'karen-clean1.h5')
    data = store['data']
    store.close()

    #num_cols = pa.Index(map(str.strip,open(datapath+'colon_rat.txt','r').readlines()))
    num_cols = data.columns - pa.Index(['Diet', 'treatment'])
    numdata = data[num_cols]

    aom = data.treatment == 'AOM'
    aom_inds = data.index[aom]
    saline_inds = data.index - aom_inds

    trn_inds = pa.Index(np.random.choice(aom_inds, Ntrn, replace=False)) \
            + pa.Index(np.random.choice(saline_inds, Ntrn, replace=False))
    trn_labels = np.array((data.loc[trn_inds, 'treatment']=='AOM').astype(np.int64) * 1)

    tst_inds = data.index - trn_inds
    tst_labels = np.array((data.loc[tst_inds, 'treatment']=='AOM').astype(np.int64) * 1)

    #grab only some columns
    low = params['low_filter']
    high = params['high_filter']
    good_cols = numdata.columns[(numdata.mean() <= high) & (numdata.mean() >= low)]

    print("# Good columns: {}, # Total columns: {}".format(
        len(good_cols), numdata.shape[1]))

    # T Tests (same ranking as F tests)
    #pvals = np.array([st.ttest_ind(numdata.loc[aom,col], 
        #numdata.loc[~aom,col])[1] for col in good_cols], dtype=np.float)
    #pvind = pvals.argsort()

    # F Tests
    selector = SelectKBest(f_classif, k=4)
    selector.fit(numdata.loc[:, good_cols].as_matrix().astype(np.float), aom)
    pvind = selector.pvalues_.argsort()
    #print(selector.pvalues_[pvind2[:50]])

    num_candidates = params['num_candidates']
    candidates = pvind[:num_candidates]
    feats = np.random.choice(candidates, num_feat, replace=False)

    return numdata.ix[trn_inds, good_cols[feats]].as_matrix(), trn_labels, \
            numdata.ix[tst_inds, good_cols[feats]].as_matrix(), tst_labels

def data_test(params):
    trn_data = np.vstack(( np.zeros((10,2)), np.ones((10,2))+2 )) 
    trn_labels = np.hstack(( np.ones(10), np.zeros(10) ))
    tst_data = np.vstack(( np.zeros((1000,2)), np.ones((1000,2)) ))
    tst_labels = np.hstack(( np.ones(1000), np.zeros(1000) ))
    return trn_data, trn_labels, tst_data, tst_labels

if __name__ == '__main__':
    params = {}

    def setv(p,s,d,conv=None): 
        if s not in p:
            p[s] = d
            return d
        elif conv is not None:
            return conv(p[s])
        else:
            p[s]

    iters = setv(params, 'iters', int(1e4), int)
    num_feat = setv(params, 'num_feat', 4, int)
    seed = setv(params, 'seed', np.random.randint(10**8), int)
    rseed = setv(params, 'rseed', np.random.randint(10**8), int)
    Ntrn = setv(params, 'Ntrn', 20, int)
    Ntst = setv(params, 'Ntst', 3000, int)
    f_glob = setv(params, 'f_glob', 2, int)
    subclasses = setv(params, 'subclasses', 2, int)
    f_het = setv(params, 'f_het', 1, int)
    f_rand = setv(params, 'f_rand', 0, int)
    rho = setv(params, 'rho', 0.6, float)
    f_tot = setv(params, 'f_tot', f_glob+f_het*subclasses+f_rand, int) 
    blocksize = setv(params, 'blocksize', 1, int)
    mu0 = setv(params, 'mu0', -1.2, float)
    mu1 = setv(params, 'mu1', -0.2, float)
    sigma0 = setv(params, 'sigma0', 0.5, float)
    sigma1 = setv(params, 'sigma1', 0.2, float)
    kappa = setv(params, 'kappa', 10.0, float)
    lowd = setv(params, 'lowd', 9.0, float)
    highd = setv(params, 'highd', 11.0, float)
    numlam = setv(params, 'numlam', 20, int)

    data_yj(params)
    data_jk(params)
    data_tcga(params)
    data_karen(params)
    data_test(params)
