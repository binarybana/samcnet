import numpy as np
import samcnet.mh as mh
from samcnet.mixturepoisson import *

def rho_matrix(p, diag, offdiag):
    assert np.abs(diag) >= np.abs(offdiag)
    return np.diag(np.ones(p) * diag) + (np.ones((p,p)) - np.eye(p)) * offdiag

def calc_avgs(db):
    #p['priormu'] = db.mu.read().mean(axis=0)
    #p['priorsigma'] = db.mu.read().std(axis=0)

    ## Will eventually want to sample these two many times....
    #eS = db.sigma.read().mean(axis=0)
    #varS = db.sigma.read().std(axis=0)

    #p['kappa'] = 2*eS[0,0]**2 / varS[0,0] + D + 3
    #sig = (p['kappa'] - D - 1) * eS[0,0]
    #rho = eS[0,1] / eS[0,0]
    #p['S'] = rho_matrix(D, eS[0,0], rho)
    D = db.mu.read()[0].size
    mumean = db.mu.read().mean()
    muvar = db.mu.read().std(axis=0).mean()
    sigmean = db.sigma.read().mean(axis=0)
    sigvar = db.sigma.read().std(axis=0)
    sigdiagmean = (sigmean[0,0] + sigmean[1,1]) / 2.0
    sigdiagvar = (sigvar[0,0] + sigvar[1,1]) / 2.0
    rho = sigmean[0,1] / sigdiagmean
    kappa = 2*sigdiagmean**2 / sigdiagvar + D + 3
    return np.r_[mumean, muvar, sigdiagvar, rho, kappa]

def get_calibration_params(params, D):
    meanp = params.mean(axis=0)
    return dict(
            priormu=np.ones(D)*meanp[0], 
            priorsigma=np.ones(D)*meanp[1], 
            kappa=int(meanp[4]),
            S=rho_matrix(D, meanp[2], meanp[3]*meanp[2]))

def get_calibration(db):
    p0 = get_calibration_params(db.root.object.dist0)
    p1 = get_calibration_params(db.root.object.dist1)
    return p0, p1

def calibrate(rawdata, sel, params):
    iters = params['iters']
    num_feat = params['num_feat']
    burn = params['burn']
    thin = params['thin']

    paramlog0 = np.empty((0,5), dtype=float)
    paramlog1 = np.empty((0,5), dtype=float)
    for feats in sel['subcalibs']:
        dist0 = MPMDist(rawdata.loc[sel['trn0'],feats],
            priorkappa=params['priorkappa'],
            lammove=params['lammove'],
            mumove=params['mumove'],
            usepriors=False)
        dist1 = MPMDist(rawdata.loc[sel['trn1'],feats],
            priorkappa=params['priorkappa'],
            lammove=params['lammove'],
            mumove=params['mumove'],
            usepriors=False)
        mpm = MPMCls(dist0, dist1) 
        mhmc = mh.MHRun(mpm, burn=burn, thin=thin, verbose=False)
        mhmc.sample(iters,verbose=False)
        paramlog0 = np.vstack(( paramlog0, calc_avgs(mhmc.db.root.object.dist0) ))
        paramlog1 = np.vstack(( paramlog1, calc_avgs(mhmc.db.root.object.dist1) ))
        mhmc.clean_db()

    p0 = get_calibration_params(paramlog0, num_feat)
    p1 = get_calibration_params(paramlog1, num_feat)
    return p0, p1
