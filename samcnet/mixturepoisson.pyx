from __future__ import division

# cython: profile=True
cimport cython
import sys

import numpy as np
cimport numpy as np

import pylab as p
import tables as t
import matplotlib as mpl
import random
from math import pi

import scipy.stats as st
import scipy.stats.distributions as di
import scipy
from scipy.special import betaln

from statsmodels.sandbox.distributions.mv_normal import MVT,MVNormal

cdef extern from "math.h":
    double log2(double)
    double log(double)
    double exp(double)
    double lgamma(double)
    double gamma(double)

def sample_invwishart(lmbda,dof):
    # TODO make a version that returns the cholesky
    # TODO allow passing in chol/cholinv of matrix parameter lmbda
    n = lmbda.shape[0]
    lmbda = np.asarray(lmbda)
    chol = np.linalg.cholesky(lmbda)

    if (dof <= 81+n) and (dof == np.round(dof)):
        x = np.random.randn(dof,n)
    else:
        x = np.diag(np.sqrt(st.chi2.rvs(dof-np.arange(n))))
        x[np.triu_indices_from(x,1)] = np.random.randn(n*(n-1)/2)
    R = np.linalg.qr(x,'r')
    T = scipy.linalg.solve_triangular(R.T,chol.T,lower=True).T
    return np.dot(T,T.T)

def logp_invwishart(mat, kappa, s):
    ''' Return log probability from an inverse wishart with
    DOF kappa, covariance matrix 'mat' and prior matrix 's' '''
    mat = np.asarray(mat)
    s = np.asarray(s)
    D = s.shape[0]
    mlgamma = 0.0
    for i in range(D):
        mlgamma += lgamma(kappa/2 + (1-i+1)/2)
    return -(kappa+D+1)/2 * log(np.linalg.det(mat)) \
            - 0.5*np.trace(np.dot(s,np.linalg.inv(mat))) \
            + kappa/2 * log(np.linalg.det(s)) \
            - kappa*D/2 * log(2) \
            - D*(D-1)/4 * log(pi) * mlgamma

def logp_normal(x, mu, sigma, nu=1.0):
    ''' Return log probabilities from a multivariate normal with
    scaling parameter nu, mean mu, and covariance matrix sigma.'''
    x = np.asarray(x)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    k = mu.size
    if x.ndim > 1:
        axis = 1
    else:
        axis = 0
    t1 = -0.5*k*log(2*pi) 
    t2 = -0.5*log(np.linalg.det(sigma))
    t3 = - nu/2 * (np.dot((x-mu), np.linalg.inv(sigma)) \
            * (x-mu)).sum(axis=axis)
    return t1+t2+t3

cdef class MixturePoissonSampler:
    cdef public:
        object data0, data1, mu0, mu1, sigma0, sigma1, w0, w1, S, priormu
        int D, kmax, k0, k1, d0, d1
        double c, Ec, kappa, comp_geom, nu

        object oldmu0, oldmu1, oldsigma0, oldsigma1, oldw0, oldw1
        int oldk0, oldk1, oldd0, oldd1
        double oldc

    def __init__(self, data0, data1):
        self.data0 = data0
        self.data1 = data1

        ##### Proposal variances ######
        #self.propdof = 200
        #self.propmu = 0.2

        ######## Starting point of MCMC Run #######
        assert data0.shape[1] == data1.shape[1], "Datasets must be of same featuresize"
        self.D = data0.shape[1]
        self.c = np.random.rand()
        self.kmax = 4
        self.k0 = 1
        self.k1 = 1
        self.d0 = 10
        self.d1 = 10

        self.Ec = data0.shape[0] / (data1.shape[0] + data0.shape[0])

        self.mu0 = np.ones((self.D,self.kmax))
        self.mu1 = np.ones((self.D,self.kmax))
        self.sigma0 = sample_invwishart(np.eye(self.D), 5)
        self.sigma1 = sample_invwishart(np.eye(self.D), 5)

        self.w0 = np.random.dirichlet((1,)*self.k0)
        self.w1 = np.random.dirichlet((1,)*self.k1)

        ##### Prior Quantities ######
        self.kappa = 5
        self.S = np.eye(self.D)
        self.comp_geom = 0.6
        self.priormu = np.zeros(self.D)
        self.nu = 3.0

        ###### Bookeeping ######
        self.oldmu0 = None
        self.oldmu1 = None
        self.oldsigma0 = None
        self.oldsigma1 = None
        self.oldc = 0.0
        self.oldk0 = 0
        self.oldk1 = 0
        self.oldd0 = 0
        self.oldd1 = 0
        self.oldw0 = None
        self.oldw1 = None

    def propose(self):
        """ 
        We do one of a couple of things:
        0) Modify number of mixture components (k)
        1) Modify weights of mixture components (w)
        2) Modify means of mixture components (mu)
        3) Modify covariance matrices (sigma)
        On every step let's modify d
        """

        self.oldmu0 = self.mu0.copy()
        self.oldmu1 = self.mu1.copy()
        self.oldsigma0 = self.sigma0.copy()
        self.oldsigma1 = self.sigma1.copy()
        self.oldc = self.c
        self.oldk0 = self.k0
        self.oldk1 = self.k1
        self.oldd0 = self.d0
        self.oldd1 = self.d1
        self.oldw0 = self.w0.copy()
        self.oldw1 = self.w1.copy()

        scheme = np.random.randint(4)
        if scheme == 0:
            if self.k0 > 1 and self.k0 < self.kmax:
                mod = np.random.choice((-1,1))
            elif self.k0 == 1:
                mod = 1
            else:
                mod = -1
            if mod == 1: # Add one
                self.mu0[:,self.k0] = self.mu0[:,self.k0-1]
                self.w0 *= 0.8
                self.w0 = np.hstack((self.w0, 0.2))
                self.k0 += 1
            else: # remove one
                self.w0 = self.w0[:-1]
                self.w0 /= self.w0.sum()
                self.k0 -= 1

            if self.k1 > 1 and self.k0 < self.kmax: # TODO YUCK! DRY
                mod = np.random.choice((-1,1))
            elif self.k1 == 1:
                mod = 1
            else:
                mod = -1
            if mod == 1: # Add one
                self.mu1[:,self.k1] = self.mu1[:,self.k1-1]
                self.w1 *= 0.8
                self.w1 = np.hstack((self.w1, 0.2))
                self.k1 += 1
            else: # remove one
                self.w1 = self.w1[:-1]
                self.w1 /= self.w1.sum()
                self.k1 -= 1

        elif scheme == 1: # Modify weights
            self.w0 = np.random.dirichlet((1,)*self.k0)
            self.w1 = np.random.dirichlet((1,)*self.k1)

        elif scheme == 2: # Modify means
            self.mu0 += np.random.randn(self.D, self.kmax)
            self.mu1 += np.random.randn(self.D, self.kmax)
        elif scheme == 3: # Modify covariances
            self.sigma0 = sample_invwishart(np.eye(self.D), 5)
            self.sigma1 = sample_invwishart(np.eye(self.D), 5)
        
        #modify di's
        self.d0 += np.random.randn()*0.5
        self.d1 += np.random.randn()*0.5
        self.d0 = np.clip(self.d0, 9,10)
        self.d1 = np.clip(self.d1, 9,10)

        return scheme

    def reject(self):
        self.mu0 = self.oldmu0
        self.mu1 = self.oldmu1
        self.sigma0 = self.oldsigma0
        self.sigma1 = self.oldsigma1
        self.c = self.oldc
        self.k0 = self.oldk0
        self.k1 = self.oldk1
        self.d0 = self.oldd0
        self.d1 = self.oldd1
        self.w0 = self.oldw0
        self.w1 = self.oldw1

    def energy(self):
        cdef double accumlan, accumdat, accumK, accumD, sum = 0.0
        cdef int i,j,m,d,numlam = 100

        cdef np.ndarray[np.double_t, ndim=3, mode="c"] lams0 = np.empty((numlam, self.D, self.k0), np.double)
        cdef np.ndarray[np.double_t, ndim=3, mode="c"] lams1 = np.empty((numlam, self.D, self.k1), np.double)
        
        #class 0 negative log likelihood
        numcom = self.k0
        numdat = self.data0.shape[0]
        # pre-generate all lambda values
        for k in range(self.k0):
            lams0[:,:,k] = MVNormal(self.mu0[:,k], self.sigma0).rvs(numlam)
        accumlan = 0.0
        for i in range(numlam):
            accumdat = 0.0
            for j in xrange(numdat):
                accumK = 0.0
                for m in xrange(self.k0):
                    accumD = 1.0
                    for d in xrange(self.D):
                        accumD *= di.poisson.pmf(self.data0[j,d], self.d0*exp(lams0[i,d,m]))
                    accumK += accumD * self.w0[m]
                accumdat *= accumK
            accumlan += accumdat
        if accumlan != 0.0:
            sum += log(accumlan)


        #class 1 negative log likelihood
        numcom = self.k1
        numdat = self.data1.shape[0]
        # pre-generate all lambda values
        for k in range(self.k0):
            lams1[:,:,k] = MVNormal(self.mu1[:,k], self.sigma1).rvs(numlam)
        accumlan = 0.0
        for i in range(numlam):
            accumdat = 0.0
            for j in xrange(numdat):
                accumK = 0.0
                for m in xrange(self.k1):
                    accumD = 1.0
                    for d in xrange(self.D):
                        accumD *= di.poisson.pmf(self.data1[j,d], self.d1*exp(lams1[i,d,m]))
                    accumK += accumD * self.w1[m]
                accumdat *= accumK
            accumlan += accumdat
        if accumlan != 0.0:
            sum += log(accumlan)

        #Class proportion c (from page 3, eq 1)
        # Should we do this? I don't think it's necessary for this model...
        #sum -= log(self.c)*(self.dist0.alpha+self.dist0.n-1) \
                #+ log(1-self.c)*(self.dist1.alpha+self.dist1.n-1) \
                #- betaln(self.dist0.alpha + self.dist0.n, self.dist1.alpha + self.dist1.n)

        #Now add in the priors...
        sum -= logp_invwishart(self.sigma0, self.kappa, self.S)
        sum -= logp_invwishart(self.sigma1, self.kappa, self.S)
        sum -= di.geom.logpmf(self.k0, self.comp_geom)
        sum -= di.geom.logpmf(self.k1, self.comp_geom)
        for k in xrange(self.k0):
            sum -= logp_normal(self.mu0[:,k], self.priormu, self.S, self.nu)
        for k in xrange(self.k1):
            sum -= logp_normal(self.mu1[:,k], self.priormu, self.S, self.nu)

        return sum

    def init_db(self, db, size):
        """ Takes a Pytables Group object (group) and the total number of samples expected and
        expands or creates the necessary groups.
        """
        D = self.dist0.D
        objroot = db.root.object
        db.createEArray(objroot.objfxn, 'mu0', t.Float64Atom(shape=(D,self.kmax)), (0,), expectedrows=size)
        db.createEArray(objroot.objfxn, 'mu1', t.Float64Atom(shape=(D,self.kmax)), (0,), expectedrows=size)
        db.createEArray(objroot.objfxn, 'sigma0', t.Float64Atom(shape=(D,D)), (0,), expectedrows=size)
        db.createEArray(objroot.objfxn, 'sigma1', t.Float64Atom(shape=(D,D)), (0,), expectedrows=size)
        db.createEArray(objroot.objfxn, 'k0', t.Float64Atom(), (0,), expectedrows=size)
        db.createEArray(objroot.objfxn, 'k1', t.Float64Atom(), (0,), expectedrows=size)
        db.createEArray(objroot.objfxn, 'w0', t.Float64Atom(shape=(self.kmax,)), (0,), expectedrows=size)
        db.createEArray(objroot.objfxn, 'w1', t.Float64Atom(shape=(self.kmax,)), (0,), expectedrows=size)
        db.createEArray(objroot.objfxn, 'd0', t.Float64Atom(), (0,), expectedrows=size)
        db.createEArray(objroot.objfxn, 'd1', t.Float64Atom(), (0,), expectedrows=size)

    def save_iter_db(self, db):
        """ Saves objective function (and possible samples depending on verbosity) to
        Pytables db
        """ 
        root = db.root.object
        root.objfxn.mu0.append((self.mu0,))
        root.objfxn.mu1.append((self.mu1,))
        root.objfxn.sigma0.append((self.sigma0,))
        root.objfxn.sigma1.append((self.sigma1,))
        root.objfxn.k0.append((self.k0,))
        root.objfxn.k1.append((self.k1,))
        root.objfxn.w0.append((self.w0,))
        root.objfxn.w1.append((self.w1,))
        root.objfxn.d0.append((self.d0,))
        root.objfxn.d1.append((self.d1,))

    def approx_error_data(self, db, data, labels, partial=False):
        preds = self.calc_gavg(db, data, partial) < 0
        return np.abs(preds-labels).sum()/float(labels.shape[0])

    def calc_gavg(self, db, pts, partial=False):
        of = db.root.object.objfxn
        temp = db.root.samc.theta_trace.read()
        parts = np.exp(temp - temp.max())
        if partial:
            inds = np.argsort(parts)[::-1][:partial]
            g0 = self.calc_g(pts, parts[inds], of.mu0.read()[inds], of.sigma0.read()[inds],
                    of.k0.read()[inds], of.w0.read()[inds], of.d0.read()[inds])
            g1 = self.calc_g(pts, parts[inds], of.mu1.read()[inds], of.sigma1.read()[inds],
                    of.k1.read()[inds], of.w1.read()[inds], of.d1.read()[inds])
            Ec = of.c.read()[inds].mean()
        else:
            g0 = self.calc_g(pts, parts, of.mu0.read(), of.sigma0.read(),
                    of.k0.read(), of.w0.read(), of.d0.read())
            g1 = self.calc_g(pts, parts, of.mu1.read(), of.sigma1.read(),
                    of.k1.read(), of.w1.read(), of.d1.read())
            Ec = self.Ec
        efactor = log(Ec) - log(1-Ec)
        return g0 - g1 + efactor

    def calc_g(self, pts, parts, mus, sigmas, ks, ws, ds):
        """ Returns weighted (parts) average logp for all pts """
        res = np.zeros(pts.shape[0])
        accumlan = np.zeros(np.ones(pts.shape[0]))
        accumD = np.ones(pts.shape[0])
        for i in range(parts.size):
            numlam = 100
            #class 0 negative log likelihood
            numcom = self.ks[i]
            # generate lambda values
            lams = np.dstack(( MVNormal(mus[i][:,k], self.sigmas[i]).rvs(numlam) for k in range(self.ks[i]) ))
            accumlan[:] = 0
            for j in range(numlam):
                for m in xrange(self.k0):
                    accumD[:] = 1
                    for d in xrange(self.D):
                        accumD *= di.poisson.pmf(pts[:,d], self.ds[i]*exp(lams[j,d,m]))
                    accumlan += accumD * self.w0[m]

            res += parts[i] * accumlan
        return np.log(res / parts.sum())

