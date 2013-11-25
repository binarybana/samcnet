import numpy as np
import samcnet.mh as mh
from samcnet.mixturepoisson import *

def rho_matrix(p, diag, offdiag):
    assert np.abs(diag) >= np.abs(offdiag)
    return np.diag(np.ones(p) * diag) + (np.ones((p,p)) - np.eye(p)) * offdiag

def calc_avgs(db):
    D = db.mu.read()[0].size
    mumean = db.mu.read().mean()
    muvar = db.mu.read().std(axis=0).mean()
    sigmean = db.sigma.read().mean(axis=0)
    return np.r_[mumean, sigmean[0,0], sigmean[1,1], sigmean[0,1]]

def get_calibration_params(params, D):
    meanp = params.mean(axis=0)
    mumean = meanp[0]
    muvar = params[:,0].var(ddof=1)

    diags = params[:,[1,2]]
    sigdiagmean = diags.mean()
    sigoffmean = params[:,3].mean()

    sigdiagvar = 1./(diags.size - 1) * ((sigdiagmean - diags.flatten())**2).sum()
    sigma2 = 2 * sigdiagmean * (sigdiagmean**2/sigdiagvar + 1)
    rho = sigoffmean/sigdiagmean
    kappa = 2*sigdiagmean**2 / sigdiagvar + D + 3

    S=rho_matrix(D, sigma2, sigma2*rho)
    try:
        np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        S = rho_matrix(D, sigma2, 0.0)
    return dict(
            priormu=np.ones(D) * mumean,
            priorsigma=np.ones(D) * muvar,
            kappa=int(kappa),
            S=S)

def record_hypers(output, p0, p1):
    for k in p0.keys():
        if type(p0[k]) == np.ndarray:
            output['p0_'+k] = list(p0[k].flat)
            output['p1_'+k] = list(p1[k].flat)
        else:
            output['p0_'+k] = p0[k]
            output['p1_'+k] = p1[k]

def calibrate(rawdata, sel, params):
    iters = params['iters']
    num_feat = params['num_feat']
    burn = params['burn']
    thin = params['thin']
    d = params.get('d', 10)

    paramlog0 = np.empty((0,4), dtype=float)
    paramlog1 = np.empty((0,4), dtype=float)
    for feats in sel['subcalibs']:
        dist0 = MPMDist(rawdata.loc[sel['trn0'],feats],
            priorkappa=params['priorkappa'],
            lammove=params['lammove'],
            mumove=params['mumove'],
            d=d,
            usepriors=False)
        dist1 = MPMDist(rawdata.loc[sel['trn1'],feats],
            priorkappa=params['priorkappa'],
            lammove=params['lammove'],
            mumove=params['mumove'],
            d=d,
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
