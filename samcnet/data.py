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
    Ntst = params['Ntst']
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
        inspec = 'gen -i params/%s -c 0.05 -l %f -h %f -s %d' % \
                (fname, lowd, highd, seed)
        spec = path.join(synloc, inspec).split()
        sb.check_call(spec)
    except Exception as e:
        print "ERROR in data_yj: " + str(e)
    finally:
        os.chdir(currdir)

    try:
        trn_path = path.join(synloc, 'out','%s_trn.txt'%fname)
        tst_path = path.join(synloc, 'out','%s_tst.txt'%fname)

        raw_trn_data = np.loadtxt(trn_path,
            delimiter=',', skiprows=1)
        selector = SelectKBest(f_classif, k=num_feat)
        trn_labels = np.hstack(( np.zeros(Ntrn), np.ones(Ntrn) ))
        selector.fit(raw_trn_data, trn_labels)

        raw_tst_data = np.loadtxt(tst_path,
                delimiter=',', skiprows=1)
    except Exception as e:
        print "ERROR in data_yj: " + str(e)
    finally:
        os.remove(trn_path)
        os.remove(tst_path)

    trn0, trn1, tst0, tst1 = gen_labels(Ntrn, Ntrn, Ntst, Ntst)
    rawdata = np.vstack(( raw_trn_data, raw_tst_data ))

    pvind = selector.pvalues_.argsort()

    np.random.shuffle(pvind)

    feats = np.zeros(rawdata.shape[1], dtype=bool)
    feats[pvind[:num_feat]] = True
    calib = ~feats

    return rawdata, trn0, trn1, tst0, tst1, feats, calib 

def gen_data(mu, cov, n, lowd, highd):
    lams = MVNormal(mu, cov).rvs(n)
    ps = np.empty_like(lams)
    ps = di.poisson.rvs(di.uniform.rvs(lowd,highd-lowd,size=lams.shape) * np.exp(lams))
    return ps

def data_jk(params):
    D = params['num_gen_feat']
    num_feat = params['num_feat']
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
    if kappa >= 1000:
        cov0 = sample_invwishart(np.eye(D)*sigma0, D+2) 
        cov1 = sample_invwishart(np.eye(D)*sigma1, D+2) 
        cov0 = np.diag(np.diag(cov0))
        cov1 = np.diag(np.diag(cov1))
    else:
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

    rawdata = np.vstack(( trn_data0, trn_data1, tst_data0, tst_data1 ))

    trn0, trn1, tst0, tst1 = gen_labels(Ntrn, Ntrn, Ntst, Ntst)

    #selector = SelectKBest(f_classif, k=num_feat)
    #trn_data = np.vstack(( trn_data0, trn_data1 ))
    #trn_labels = np.hstack(( np.zeros(Ntrn), np.ones(Ntrn) ))
    #selector.fit(trn_data, trn_labels)
    #pvind = selector.pvalues_.argsort()

    #np.random.shuffle(pvind)

    feats = np.zeros(rawdata.shape[1], dtype=bool)
    #feats[pvind[:num_feat]] = True
    feats[:num_feat] = True
    calib = ~feats
    return rawdata, trn0, trn1, tst0, tst1, feats, calib

def data_tcga(params):
    Ntrn = params['Ntrn']
    num_feat = params['num_feat']
    low = params['low_filter']
    high = params['high_filter']

    if 'c' in params:
        #Interpret Ntrn as total training samples
        Ntrn0 = int(round(params['c'] * Ntrn))
        Ntrn1 = int(round((1-params['c']) * Ntrn))
    else:
        Ntrn0, Ntrn1 = Ntrn, Ntrn

    print("Training data sizes: {}, {}".format(Ntrn0, Ntrn1))

    store = pa.HDFStore(os.path.expanduser('~/largeresearch/seq-data/store.h5'))
    luad = store['lusc_norm'].as_matrix()
    lusc = store['luad_norm'].as_matrix()

    # Grab random training set and use the rest as testing
    luad_inds = np.arange(luad.shape[1])
    lusc_inds = np.arange(lusc.shape[1])
    np.random.shuffle(luad_inds)
    np.random.shuffle(lusc_inds)
    trn_data = np.round(np.hstack(( 
        lusc[:,lusc_inds[:Ntrn0]], 
        luad[:,luad_inds[:Ntrn1]] )).T)
    tst_data = np.round(np.hstack(( 
        lusc[:,lusc_inds[Ntrn0:]], 
        luad[:,luad_inds[Ntrn1:]] )).T)

    # Generate labels
    trn_labels = np.hstack(( np.zeros(Ntrn0), np.ones(Ntrn1) ))
    trn0, trn1, tst0, tst1 = gen_labels(Ntrn0, Ntrn1, lusc.shape[1]-Ntrn0, luad.shape[1]-Ntrn1)

    # Select a subset of the features, then select a further subset based on
    # Univariate F tests
    good_cols = (trn_data.mean(axis=0) < high) & (trn_data.mean(axis=0) > low)
    low_trn_data = trn_data[:, good_cols]
    low_tst_data = tst_data[:, good_cols]
    #selector = SelectKBest(f_classif, k=4)
    #selector.fit(low_trn_data, trn_labels)
    #pvind = selector.pvalues_.argsort()

    ##########
    #tst_labels = np.hstack(( np.zeros(lusc_inds.size-Ntrn0), np.ones(luad_inds.size-Ntrn1) ))
    #pvind2 = selector.pvalues_.argsort()
    #selector2 = SelectKBest(f_classif, k=4)
    #selector2.fit(low_tst_data, tst_labels)

    #print low, high
    #print trn_data.shape[1]
    #print good_cols.sum()
    #print selector.pvalues_[pvind[:10]]
    #print selector.pvalues_[pvind[-10:]]

    #print selector2.pvalues_[pvind2[:10]]
    #print selector2.pvalues_[pvind2[-10:]]
    ###################

    rawdata = np.vstack(( low_trn_data, low_tst_data ))

    feats_ind = np.random.choice(np.arange(low_trn_data.shape[1]), num_feat, replace=False)
    feats = np.zeros(rawdata.shape[1], dtype=bool)
    feats[feats_ind] = True
    #feats[:num_feat] = True
    calib = ~feats

    store.close()
    return rawdata, trn0, trn1, tst0, tst1, feats, calib

def data_karen(params):
    Ntrn = params['Ntrn']
    num_feat = params['num_feat']
    low = params['low_filter']
    high = params['high_filter']
    num_candidates = params['num_candidates']
    cat = params['split_category']
    use_candidates = params['use_candidates']

    datapath = os.path.expanduser('~/GSP/research/samc/samcnet/data/')
    store = pa.HDFStore(datapath+'karen-clean2.h5')
    data = store['data']
    store.close()

    #num_cols = pa.Index(map(str.strip,open(datapath+'colon_rat.txt','r').readlines()))
    num_cols = data.columns - pa.Index(['oil', 'fiber', 'treatment'])
    numdata = data[num_cols]

    cls0 = data.loc[:,cat].iloc[0]

    aom = data[cat] == cls0
    aom_inds = data.index[aom]
    saline_inds = data.index - aom_inds
    trn_inds = pa.Index(np.random.choice(aom_inds, Ntrn, replace=False)) \
            + pa.Index(np.random.choice(saline_inds, Ntrn, replace=False))
    tst_inds = data.index - trn_inds
    trn_labels = np.array((data.loc[trn_inds, cat]==cls0).astype(np.int64) * 1)

    # Feature selection, first stage
    good_cols = numdata.columns[(numdata.mean() <= high) & (numdata.mean() >= low)]

    print("# Good columns: {}, # Total columns: {}".format(
        len(good_cols), numdata.shape[1]))

    # F Tests
    selector = SelectKBest(f_classif, k=4)
    selector.fit(numdata.loc[:, good_cols].as_matrix().astype(np.float), aom)
    pvind = selector.pvalues_.argsort()
    #print(selector.pvalues_[pvind2[:50]])

    if not use_candidates:
        np.random.shuffle(pvind)

    rawdata = numdata[good_cols].as_matrix()

    candidates = pvind[:num_candidates]
    feats_ind = np.random.choice(candidates, num_feat, replace=False)
    feats = np.zeros(rawdata.shape[1], dtype=bool)
    feats[feats_ind] = True
    calib = ~feats

    # It's hideous!!!
    # Wow... converting from indices/numeric locations to a boolean array
    # has never looked so ugly... but it works for now.
    trn0 = pa.Series(np.zeros(data.index.size), index=data.index, dtype=bool)
    trn1 = pa.Series(np.zeros(data.index.size), index=data.index, dtype=bool)
    tst0 = pa.Series(np.zeros(data.index.size), index=data.index, dtype=bool)
    tst1 = pa.Series(np.zeros(data.index.size), index=data.index, dtype=bool)

    trn0.loc[trn_inds & data.index[(data.loc[:, cat] == cls0)]] = True
    trn1.loc[trn_inds & data.index[(data.loc[:, cat] != cls0)]] = True
    tst0.loc[tst_inds & data.index[(data.loc[:, cat] == cls0)]] = True
    tst1.loc[tst_inds & data.index[(data.loc[:, cat] != cls0)]] = True

    trn0 = np.array(trn0)
    trn1 = np.array(trn1)
    tst0 = np.array(tst0)
    tst1 = np.array(tst1)
    return rawdata, trn0, trn1, tst0, tst1, feats, calib

def data_test(params):
    trn_data = np.vstack(( np.zeros((10,2)), np.ones((10,2))+2 )) 
    trn_labels = np.hstack(( np.ones(10), np.zeros(10) ))
    tst_data = np.vstack(( np.zeros((1000,2)), np.ones((1000,2)) ))
    tst_labels = np.hstack(( np.ones(1000), np.zeros(1000) ))
    return trn_data, trn_labels, tst_data, tst_labels

def gen_labels(a,b,c,d):
    trn0 = np.hstack(( np.ones(a), np.zeros(b), np.zeros(c+d) )).astype(bool)
    trn1 = np.hstack(( np.zeros(a), np.ones(b), np.zeros(c+d) )).astype(bool)
    tst0 = np.hstack(( np.zeros(a+b), np.ones(c), np.zeros(d) )).astype(bool)
    tst1 = np.hstack(( np.zeros(a+b), np.zeros(c), np.ones(d) )).astype(bool)
    return trn0, trn1, tst0, tst1

def plot_p_hist(data, sel, label):
    selector = SelectKBest(f_classif, k=4)
    print sel['tst'].shape
    print data.loc[sel['tst'],:].shape
    selector.fit(data.loc[sel['tst'], :], sel['tstl'])
    pvals = selector.pvalues_
    pvind = pvals.argsort()
    pvals[np.isnan(pvals)] = 1

    import pylab as p
    p.hist(pvals, bins=np.logspace(-5,0,30), label=label, log=True, histtype='step')

#def norm(data1, data2):
    #mu = data1.mean(axis=0)
    #std = np.sqrt(data1.var(axis=0, ddof=1))
    #return (data1 - mu) / std, (data2 - mu) / std

#def split(data, labels):
    #return data[labels==0,:], data[labels==1,:]

#def shuffle_features(trn, tst):
    #D = trn.shape[1]
    #assert D == tst.shape[1]
    #ind = np.arange(D)
    #np.random.shuffle(ind)
    #return trn[:,ind], tst[:,ind]

def subsample(sel, N=1):
    """ 
    Takes N samples from each class's training sample and moves them to holdout 
    """
    s0 = np.random.choice(sel['trn0'].nonzero()[0], size=N, replace=False)
    s1 = np.random.choice(sel['trn1'].nonzero()[0], size=N, replace=False)

    sel['trn0'][s0] = False
    sel['trn'][s0] = False
    sel['tst0'][s0] = True
    sel['tst'][s0] = True

    sel['trn1'][s1] = False
    sel['trn'][s1] = False
    sel['tst1'][s1] = True
    sel['tst'][s1] = True

    # FIXME: Hack
    sel['trnl'] = (sel['trn0']*1 + sel['trn1']*2 - 1)[sel['trn']]
    sel['tstl'] = (sel['tst0']*1 + sel['tst1']*2 - 1)[sel['tst']]

    return sel

def get_data(method, params):
    """
    Returns a selector dictionary, rawdata matrix and normalized data matrix
    where the selector dictionary has the following keys defined:
    trn, trn0, trn1, tst, tst0, tst1, feats, calib
    where the last two are for the features and calibration features
    """
    rawdata, trn0, trn1, tst0, tst1, feats, calib = method(params)
    trn = trn0 | trn1
    tst = tst0 | tst1

    # FIXME: Hack
    trnl = (trn0*1 + trn1*2 - 1)[trn]
    tstl = (tst0*1 + tst1*2 - 1)[tst]

    num_calibs = setv(params, 'num_calibs', 5, int)

    if num_calibs == 0 or params['num_gen_feat'] == params['num_feat']:
        # This should really be done cleaner, but I can accrue a little
        # technical debt every now and then right?... right?!
        sel = dict(trn0=trn0, trn1=trn1, trn=trn, tst0=tst0, tst1=tst1, tst=tst, 
                tstl=tstl, trnl=trnl)
    else:
        size_calibs = setv(params, 'size_calibs', 2, int)

        subcalibs = calib.nonzero()[0]
        clip_calibs = subcalibs.size - (num_calibs * size_calibs)
        assert clip_calibs >= 0, clip_calibs
        np.random.shuffle(subcalibs)
        subcalibs = np.split(subcalibs[:-clip_calibs], num_calibs)
        sel = dict(trn0=trn0, trn1=trn1, trn=trn, tst0=tst0, tst1=tst1, tst=tst, 
                feats=feats, calib=calib,
                tstl=tstl, trnl=trnl,
                subcalibs=subcalibs)

    # Normalize
    mu = rawdata[trn,:].mean(axis=0)
    std = np.sqrt(rawdata[trn,:].var(axis=0, ddof=1))
    normdata = (rawdata - mu) / std
    
    return sel, pa.DataFrame(rawdata), pa.DataFrame(normdata)

def setv(p,s,d,conv=None): 
    if s not in p:
        p[s] = d
        return d
    elif conv is not None:
        return conv(p[s])
    else:
        p[s]

def get_test_params():
    params = {}

    iters = setv(params, 'iters', int(1e4), int)
    num_feat = setv(params, 'num_feat', 5, int)
    num_gen_feat = setv(params, 'num_gen_feat', 30, int)
    seed = setv(params, 'seed', np.random.randint(10**8), int)
    rseed = setv(params, 'rseed', np.random.randint(10**8), int)
    Ntrn = setv(params, 'Ntrn', 20, int)
    Ntst = setv(params, 'Ntst', 300, int)
    f_glob = setv(params, 'f_glob', 10, int)
    subclasses = setv(params, 'subclasses', 2, int)
    f_het = setv(params, 'f_het', 20, int)
    f_rand = setv(params, 'f_rand', 20, int)
    rho = setv(params, 'rho', 0.6, float)
    f_tot = setv(params, 'f_tot', f_glob+f_het*subclasses+f_rand, int) 
    blocksize = setv(params, 'blocksize', 5, int)
    mu0 = setv(params, 'mu0', 0.8, float)
    mu1 = setv(params, 'mu1', -0.8, float)
    sigma0 = setv(params, 'sigma0', 0.6, float)
    sigma1 = setv(params, 'sigma1', 0.2, float)
    kappa = setv(params, 'kappa', 32.0, float)
    lowd = setv(params, 'lowd', 9.0, float)
    highd = setv(params, 'highd', 11.0, float)
    numlam = setv(params, 'numlam', 20, int)
    low = setv(params, 'low_filter', 1, int)
    high = setv(params, 'high_filter', 10, int)
    num_candidates = setv(params, 'num_candidates', 50, int)
    cat = setv(params, 'split_category', 'treatment', str)
    use_candidates = setv(params, 'use_candidates', False, bool)
    return params

if __name__ == '__main__':
    import pylab as p
    p.figure()

    params = get_test_params()
    sel, raw, norm = get_data(data_tcga, params)
    print(sel['tst'].sum(), sel['tst0'].sum(), sel['tst1'].sum())

    sys.exit()
    #plot_p_hist(raw, sel, 'tcga low')

    sel, raw, norm = get_data(data_karen, params)
    plot_p_hist(raw, sel, 'karen low')

    params['low_filter'] = 0
    params['high_filter'] = 10000
    sel, raw, norm = get_data(data_karen, params)
    plot_p_hist(raw, sel, 'karen all')

    sel, raw, norm = get_data(data_jk, params)
    plot_p_hist(raw, sel, 'ic')

    p.legend(loc='best')
    #p.xscale('log')
    p.show()
    sys.exit()

    def test(out):
        sel, raw, norm = out
        assert raw.shape == norm.shape
        for k,v in sel.iteritems():
            if type(v) == np.ndarray:
                assert v.sum() > 0, str(k) + str(v.shape)
                assert v.sum() < max(raw.shape)
                #assert v.shape[0] == raw.shape[0] or v.shape[0] == raw.shape[1]

    test(get_data(data_yj, params))
    test(get_data(data_jk, params))
    test(get_data(data_tcga, params))
    test(get_data(data_karen, params))

