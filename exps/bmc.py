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
import scipy.stats.distributions as di

from sklearn.lda import LDA
from sklearn.feature_selection import SelectKBest, f_classif
from samcnet.data import *
from samcnet.calibrate import *

params = {}

seed = setv(params, 'seed', np.random.randint(10**8), int)
rseed = setv(params, 'rseed', np.random.randint(10**8), int)

# Synthetic Params
Ntrn = setv(params, 'Ntrn', 1000, int)
Ntst = setv(params, 'Ntst', 1000, int)
mu0 = setv(params, 'mu0', np.random.randn()*0.2, float)
mu1 = setv(params, 'mu1', np.random.randn()*0.2, float)
sigma0 = setv(params, 'sigma0', di.invgamma.rvs(3), float)
sigma1 = setv(params, 'sigma1', di.invgamma.rvs(3), float)

### For YJ ####
f_glob = setv(params, 'f_glob', 10, int)
subclasses = setv(params, 'subclasses', 2, int)
f_het = setv(params, 'f_het', 20, int)
f_rand = setv(params, 'f_rand', 20, int)
rho = setv(params, 'rho', np.random.rand(), float)
f_tot = setv(params, 'f_tot', f_glob+f_het*subclasses+f_rand, float)
blocksize = setv(params, 'blocksize', 5, int)
############

### For JK ###
num_gen_feat = setv(params, 'num_gen_feat', 20, int)
lowd = setv(params, 'lowd', 9.0, float)
highd = setv(params, 'highd', 11.0, float)
#kappa = setv(params, 'kappa', 2000, float)
kappa = setv(params, 'kappa', 22.0, float)
##############

# Final number of features
num_feat = setv(params, 'num_feat', 4, int)

output = {}
output['errors'] = {}
errors = output['errors']
#np.seterr(all='ignore') # Careful with this


#################### CLASSIFICATION ################
def yj():
    params['mu0'] = np.random.randn()*0.2
    params['mu1'] = np.random.randn()*0.2
    params['sigma0'] = di.invgamma.rvs(3)
    params['sigma1'] = di.invgamma.rvs(3)
    sel, rawdata, normdata = get_data(data_yj, params)
    norm_trn_data = normdata.loc[sel['trn'], sel['feats']]
    norm_tst_data = normdata.loc[sel['tst'], sel['feats']]

    sklda = LDA()
    sklda.fit(norm_trn_data, sel['trnl'])
    error = (1-sklda.score(norm_tst_data, sel['tstl']))
    print("skLDA error: %f" % error)
    return error

def jk():
    params['mu0'] = np.random.randn()*0.2
    params['mu1'] = np.random.randn()*0.2
    params['sigma0'] = di.invgamma.rvs(3)
    params['sigma1'] = di.invgamma.rvs(3)
    sel, rawdata, normdata = get_data(data_jk, params)
    norm_trn_data = normdata.loc[sel['trn'], sel['feats']]
    norm_tst_data = normdata.loc[sel['tst'], sel['feats']]
    tst_data = rawdata.loc[sel['tst'], sel['feats']]

    sklda = LDA()
    sklda.fit(norm_trn_data, sel['trnl'])
    error = (1-sklda.score(norm_tst_data, sel['tstl']))
    print("skLDA error: %f" % error)
    return error

yjs = [yj() for i in range(1000)]
print("")
jks = [jk() for i in range(1000)]
