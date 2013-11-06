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
from numpy import inf

import scipy.stats as st
import scipy.stats.distributions as di
import scipy
import scipy.special as spec
from scipy.special import betaln

from statsmodels.sandbox.distributions.mv_normal import MVT,MVNormal

cdef extern from "math.h":
    double log2(double)
    double log(double)
    double exp(double)
    double lgamma(double)
    double gamma(double)
    double pow(double,double)

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

def logp_invwishart(mat, kappa, s, logdets):
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
            + kappa/2 * logdets \
            - kappa*D/2 * log(2) \
            - D*(D-1)/4 * log(pi) * mlgamma

cpdef double logp_normal(np.ndarray x, np.ndarray mu, np.ndarray sigma, 
        double logdetsigma, np.ndarray invsigma, double nu=1.0):
    ''' Return log probabilities from a multivariate normal with
    scaling parameter nu, mean mu, and covariance matrix sigma.'''
    cdef int k = mu.size
    cdef np.ndarray diff = x-mu
    cdef double t1 = -0.5*k*log(2*pi) 
    cdef double t2 = -0.5*logdetsigma
    cdef double t3 = -nu/2 * (np.dot(diff, invsigma) * diff).sum()
    return t1+t2+t3

cpdef logp_uni_normal(x, mu, sigma): 
    return -(x-mu)*(x-mu)/(2*sigma) - 0.5*log(sigma) - 0.5*log(2*pi)

cdef class MPMParams:
    cdef public:
        np.ndarray mu, sigma, w, lam, invsigma, d
        int k
        double energy, logdetsigma
    def __init__(self,D,kmax,numsamps):
        self.mu = np.zeros((D,kmax), np.double)
        self.sigma = np.zeros((D,D), np.double)
        self.invsigma = np.zeros((D,D), np.double)
        self.w = np.zeros(kmax, np.double)
        self.lam = np.zeros((D,numsamps), np.double)
        self.d = np.zeros(numsamps, np.double)
        self.logdetsigma = 0.0
        self.energy = -inf
    cdef void set(MPMParams self, MPMParams other):
        self.mu[:] = other.mu
        self.sigma[:] = other.sigma
        self.invsigma[:] = other.invsigma
        self.w[:] = other.w
        self.lam[:] = other.lam
        self.k = other.k
        self.d[:] = other.d
        self.energy = other.energy
        self.logdetsigma = other.logdetsigma
    def copy(self):
        return (self.mu.copy(), self.sigma.copy(), self.k, self.d.copy(), 
                self.w.copy(), self.lam.copy(), self.energy)

cdef class MPMDist:
    cdef public:
        MPMParams curr, old
        object usepriors, usedata
        np.ndarray data, S, priormu, priorsigma
        int D, kmax, n
        double kappa, comp_geom, green_factor, logdetS, priorkappa, mumove, lammove, wmove, birthmove

    def __init__(self, data, kappa=None, S=None, comp_geom=None, 
            priormu=None, priorsigma=None, kmax=None, priorkappa=None,
            mumove=None, lammove=None, d=None, startmu=None, startk=None,
            wmove=None,birthmove=None,usepriors=True,usedata=True):
        self.green_factor = 0.0

        data = np.array(data)
        self.data = data
        self.n = data.shape[0]
        self.D = data.shape[1]

        ##### Prior Quantities ######
        self.kappa = 10.0 if kappa is None else kappa
        self.priorkappa = 10.0 if priorkappa is None else priorkappa
        self.S = np.eye(self.D) * (self.kappa-1-self.D) if S is None else S.copy()
        self.logdetS = log(np.linalg.det(self.S))
        self.comp_geom = 0.6 if comp_geom is None else comp_geom
        self.priormu = np.zeros(self.D) if priormu is None else priormu.copy()
        self.priorsigma = np.ones(self.D)*5.0 if priorsigma is None else priorsigma.copy()
        self.mumove = 0.04 if mumove is None else mumove
        self.lammove = 0.05 if lammove is None else lammove
        self.wmove = 0.02 if wmove is None else wmove
        self.birthmove = 1.0 if birthmove is None else birthmove

        self.usepriors=usepriors
        self.usedata=usedata

        self.kmax = 1 if kmax is None else kmax
        ######## Starting point of MCMC Run #######
        self.curr = MPMParams(self.D, self.kmax, self.n)
        self.old = MPMParams(self.D, self.kmax, self.n)

        self.curr.k = 1 if startk is None else startk
        if d is None:
            self.curr.d = 10 * np.ones(self.n) 
        elif type(d) is float:
            self.curr.d = d * np.ones(self.n)
        else:
            self.curr.d = d.copy()

        assert self.curr.d.size == self.n, "d must be vector of length n"

        nuc = np.clip(np.log(self.data.mean(axis=0)/self.curr.d.mean()), 
                -5, 20).reshape(self.D,1)
        self.curr.mu = np.repeat(nuc,
                self.kmax, axis=1) if startmu is None else startmu.copy()
        assert self.curr.mu.shape[0] == self.D, "Wrong startmu dimensions"
        assert self.curr.mu.shape[1] == self.kmax, "Wrong startmu dimensions"
        self.curr.mu[:, self.curr.k:] = 0.0
        self.curr.sigma = self.S.copy() / self.kappa
        self.curr.logdetsigma = log(np.linalg.det(self.curr.sigma))
        self.curr.invsigma = np.linalg.inv(self.curr.sigma)
        self.curr.w[:self.curr.k] = np.random.dirichlet((1,)*self.curr.k)

        self.curr.lam = np.clip(np.log(data.T/self.curr.d), -5, 20)

    def copy(self):
        return self.curr.copy()

    def propose(self):
        """ 
        We do one of a couple of things:
        0) Add mixture component (birth)
        1) Remove mixture component (death)
        2) Modify parameters (w, mu, sigma, d, lam)
        """
        cdef int i,j
        cdef double corrfac
        self.old.set(self.curr)
        self.curr.energy = -inf
        cdef MPMParams curr = self.curr
        self.green_factor = 0.0
        cdef np.ndarray[np.double_t, ndim=1, mode="c"] addfac 

        if curr.k == 1 and curr.k == self.kmax:
            scheme = 2
        elif curr.k == 1:
            scheme = np.random.choice(range(3), p=[1./8, 0, 7./8])
        elif curr.k == self.kmax:
            scheme = np.random.choice(range(3), p=[0, 1./8, 7./8])
        else:
            scheme = np.random.choice(range(3), p=[1./8, 1./8, 6./8])

        if scheme == 0: # birth
            curr.mu[:,curr.k] = curr.mu[:,curr.k-1] 
            addfac = np.random.randn(self.D) * self.birthmove
            for i in xrange(self.D):
                self.green_factor -= logp_uni_normal(addfac[i], 0.0, self.birthmove**2)
            curr.mu[:,curr.k] += addfac.T 
            # TODO: Still not sure if ^^^ should also be on 
            #the death side of things too

            curr.w *= 0.9
            curr.w[curr.k] = 0.1
            self.green_factor += (curr.k)*log(0.9)
            curr.k += 1
            curr.mu[:, curr.k:] = 0.0
            #print("Birth, propto k: %d, green_factor: %f" % (curr.k, self.green_factor))

        elif scheme == 1: # death
            curr.w[curr.k-1] = 0.0
            corrfac = curr.w[:curr.k-1].sum()
            curr.w /= corrfac
            curr.k -= 1
            curr.mu[:, curr.k-1] = 0.5*curr.mu[:, curr.k-1] + 0.5*curr.mu[:, curr.k]
            curr.mu[:, curr.k:] = 0.0
            self.green_factor = -curr.k*log(corrfac) # Negative is because we are wanting 1/corrfac
            #print("Death, propto k: %d, green_factor: %f" % (curr.k, self.green_factor))

        elif scheme == 2:  # modify params

            if curr.k != 1:
                i = np.random.randint(curr.k)
                curr.w[:curr.k] = curr.w[:curr.k] + np.random.randn(curr.k) * self.wmove
                np.clip(curr.w, 0.05, np.inf, out=curr.w)
                curr.w[:curr.k] = curr.w[:curr.k] / curr.w[:curr.k].sum()
            else:
                i = 0

            # Modify means
            curr.mu[:,i] += np.random.randn(self.D) * self.mumove

            #modify di's
            #curr.d += np.random.randn(self.n)*0.2
            #np.clip(curr.d, 8,12, out=curr.d)

            # Modify covariances
            curr.sigma = sample_invwishart(curr.sigma * (self.priorkappa-1-self.D),
                    self.priorkappa) 
            curr.invsigma = np.linalg.inv(curr.sigma)
            curr.logdetsigma = log(np.linalg.det(curr.sigma))
        
            curr.lam += np.random.randn(self.D, self.n) * self.lammove
        return scheme

    def reject(self):
        self.curr.set(self.old)

    ## FIXME These need to be updated for self.curr.lam
    #def optim(self, x, grad):
        #""" 
        #For use with NLopt. Assuming k = 1 
        #"""
        #cdef:
            #int d = self.D
            #int k = self.kmax
            #int ind = 0
        #self.curr.mu[:,0] = x[ind:ind+d]
        #ind += d
        #self.curr.sigma.flat = x[ind:ind+d*d]
        #ind += d*d
        #self.curr.d = x[ind]
        #self.curr.w[0] = 1.0
        #self.curr.k = 1
        #self.energy = -inf

        #try:
            #return self.energy(1000)
        #except:
            #return np.inf

    #def get_dof(self):
        #""" Assuming k = 1 """
        #d = self.D
        #return d + d*d + 1 

    #def get_params(self):
        #""" Assuming k = 1 """
        #return np.hstack(( self.mu[:,0].flat, self.sigma.flat, self.d ))

    def energy(self, force = False):
        if self.curr.energy != -inf and not force: # Cached 
            return self.curr.energy #- self.green_factor #hmm... is this right?

        cdef double lam,dat,accumdat,accumK,sum = 0.0
        cdef int i,j,m,d
        curr = self.curr
        
        accumdat = 0.0
        if self.usedata:
            for j in xrange(self.n):
                for d in xrange(self.D):
                    dat = self.data[j,d]
                    lam = curr.d[j]*exp(curr.lam[d,j])
                    accumdat += dat*log(lam) - lgamma(dat+1) - lam
            sum -= accumdat

            #Now add in the priors...
            # Lambdas
            for j in xrange(self.n):
                accumK = 0.0
                if curr.k == 1:
                    sum -= logp_normal(curr.lam[:,j], curr.mu[:,0], curr.sigma, 
                            curr.logdetsigma, curr.invsigma)
                else:
                    for i in xrange(curr.k):
                        accumK += exp(logp_normal(curr.lam[:,j], curr.mu[:,i], curr.sigma, 
                                curr.logdetsigma, curr.invsigma)) * curr.w[i]
                    sum -= log(accumK)

        if self.usepriors:
            # Sigma
            sum -= logp_invwishart(curr.sigma, self.kappa, self.S, self.logdetS)

            # k
            sum -= di.geom.logpmf(curr.k, self.comp_geom)

            # mu
            for k in xrange(curr.k):
                for i in xrange(self.D):
                    sum -= logp_uni_normal(curr.mu[i,k], self.priormu[i], self.priorsigma[i])

        self.curr.energy = sum
        return sum - self.green_factor

    def init_db(self, db, node, size):
        """ Takes a Pytables db and Group object (node) and the total number of samples expected and
        expands or creates the necessary groups.
        """
        D = self.D
        db.createEArray(node, 'mu', t.Float64Atom(shape=(D,self.kmax)), (0,), expectedrows=size)
        db.createEArray(node, 'lam', t.Float64Atom(shape=(D,self.n)), (0,), expectedrows=size)
        db.createEArray(node, 'sigma', t.Float64Atom(shape=(D,D)), (0,), expectedrows=size)
        db.createEArray(node, 'k', t.Int64Atom(), (0,), expectedrows=size)
        db.createEArray(node, 'w', t.Float64Atom(shape=(self.kmax,)), (0,), expectedrows=size)
        db.createEArray(node, 'd', t.Float64Atom(shape=(self.n,)), (0,), expectedrows=size)

    def save_iter_db(self, db, node):
        """ Saves objective function (and possible samples depending on verbosity) to
        Pytables db group object
        """ 
        node.mu.append((self.curr.mu,))
        node.lam.append((self.curr.lam,))
        node.sigma.append((self.curr.sigma,))
        node.k.append((self.curr.k,))
        node.w.append((self.curr.w,))
        node.d.append((self.curr.d,))

    def calc_db_g(self, db, node, pts, partial=None, numlam=10):
        if db.root._v_attrs['mcmc_type'] == 'samc':
            temp = db.root.samc.theta_trace.read()
            parts = np.exp(temp - temp.max())
            if partial:
                inds = np.argsort(parts)[::-1][:partial]
            else:
                inds = np.arange(parts.size)
            return self.calc_g(pts, parts[inds], node.mu.read()[inds],
                    node.sigma.read()[inds], node.k.read()[inds], 
                    node.w.read()[inds], node.d.read()[inds], numlam)
        elif db.root._v_attrs['mcmc_type'] == 'mh':
            if partial:
                print("Instead of a partial posterior sum calculation, why not take less samples?")
            parts = np.ones(node.k.read().shape[0])
            return self.calc_g(pts, parts, node.mu.read(),
                    node.sigma.read(), node.k.read(), node.w.read(), node.d.read(), numlam)

    def calc_curr_g(self, object pts, numlam=10):
        parts = np.array([1])
        return self.calc_g(pts, parts, [self.curr.mu], [self.curr.sigma],
                [self.curr.k], [self.curr.w], [self.curr.d], numlam)

    #@cython.boundscheck(False)
    #@cython.wraparound(False)
    def calc_g(self, pts_raw, parts, mus, sigmas, ks, ws, ds, numlam):
        """ Returns weighted (parts) average logp for all pts """
        cdef int i,m,d,s,j,numpts,numcom
        cdef double dmean, lam, loglam
        numpts = pts_raw.shape[0]
        cdef double accumcom, accumlam, accumD
        cdef np.ndarray[np.double_t, ndim=2, mode="c"] pts = np.array(pts_raw, copy=True, order='c').astype(np.double)
        cdef np.ndarray[np.double_t, ndim=1, mode="c"] res = np.zeros(numpts)
        #cdef np.ndarray[np.double_t, ndim=1, mode="c"] accumD = np.zeros(numpts)
        #cdef np.ndarray[np.double_t, ndim=1, mode="c"] accumcom = np.zeros(numpts)
        #cdef np.ndarray[np.double_t, ndim=1, mode="c"] accumlam = np.zeros(numpts)
        cdef np.ndarray[np.double_t, ndim=2, mode="c"] Z = spec.gammaln(pts+1)
        cdef np.ndarray[np.double_t, ndim=2, mode="c"] wsfast = np.vstack(ws)
        cdef np.ndarray[np.double_t, ndim=3, mode="c"] lams 
        for i in range(parts.size):
            numcom = int(ks[i])
            dmean = ds[i].mean() # TODO, this should probably be input or something
            lams = np.dstack(( MVNormal(mus[i][:,m], sigmas[i]).rvs(numlam) for m in xrange(numcom) ))
            numtotlam = lams.shape[0]
            for j in xrange(numpts):
                accumcom = 0.0
                for m in xrange(numcom):
                    accumlam = 0.0
                    for s in xrange(numtotlam): 
                        accumD = 0.0
                        for d in xrange(self.D): 
                            lam = dmean*exp(lams[s,d,m])
                            loglam = log(lam)
                            accumD += pts[j,d]*loglam - lam - Z[j,d]
                        accumlam += exp(accumD)
                    accumcom += accumlam/numtotlam * wsfast[i,m]
                res[j] += parts[i] * accumcom
        return np.log(res / parts.sum())

    def plot_traces(self, db, node, names=None):
        for curr in db.getNode(node):
            if issubclass(type(curr), t.Array):
                if names is None or curr.name in names:
                    p.figure()
                    data = curr.read()
                    data = data.reshape(data.shape[0], -1)
                    for i in xrange(data.shape[1]):
                        p.plot(data[:,i], label=str(i))
                    p.legend(loc='best', fontsize=8)
                    p.title(curr.name)

    #cdef inline double logp_point(self, double pt, double lam, double d):
        #return pt*(log(d)+lam) - lgamma(pt+1) - lam

cdef class MPMCls:
    cdef public:
        MPMDist dist0, dist1
        double Ec
        int lastmod  # Lastmod is a dirty flag that also tells us where we're dirty

    def __init__(self, dist0, dist1):
        self.dist0 = dist0
        self.dist1 = dist1
        assert self.dist0.D == self.dist1.D, "Datasets must be of same featuresize"
        self.Ec = self.dist0.n / (self.dist0.n + self.dist1.n)

    def propose(self):
        """ 
        At each step we either modify dist0 or dist1
        """
        self.lastmod = np.random.randint(2)
        if self.lastmod == 0:
            return self.dist0.propose()
        elif self.lastmod == 1: 
            return self.dist1.propose()

    def reject(self):
        if self.lastmod == 0:
            self.dist0.reject()
        elif self.lastmod == 1:
            self.dist1.reject()
        else:
            raise Exception("Rejection not possible")

    def optim(self, x, grad):
        """ 
        For use with NLopt. Assuming k = 1 
        """
        half = x.size/2
        try:
            return self.dist0.optim(x[:half],grad) + self.dist1.optim(x[half:],grad)
        except:
            return np.inf

    def get_dof(self):
        return self.dist0.get_dof() + self.dist0.get_dof()

    def get_params(self):
        """ Assuming k = 1 """
        return np.hstack(( self.dist0.get_params().flat, self.dist1.get_params().flat ))

    def copy(self):
        return (self.dist0.copy(), self.dist1.copy())

    def energy(self):
        return self.dist0.energy() + self.dist1.energy()

    def init_db(self, db, node, size):
        """ Takes a Pytables db and the total number of samples expected and
        expands or creates the necessary groups.
        """
        filt = t.Filters(complib='bzip2', complevel=7, fletcher32=True)
        db.createGroup(node, 'dist0', 'Dist0 trace', filters=filt)
        db.createGroup(node, 'dist1', 'Dist1 trace', filters=filt)
        node._v_attrs['c'] = self.Ec
        self.dist0.init_db(db, node.dist0, size)
        self.dist1.init_db(db, node.dist1, size)

    def save_iter_db(self, db, node):
        """ Saves objective function (and possible samples depending on verbosity) to
        Pytables db
        """ 
        self.dist0.save_iter_db(db, node.dist0)
        self.dist1.save_iter_db(db, node.dist1)

    def approx_error_data(self, db, data, labels, partial=False, numlam=10):
        preds = self.calc_gavg(db, data, partial, numlam=numlam) < 0
        return np.abs(preds-labels).sum()/float(labels.shape[0])

    def calc_gavg(self, db, pts, partial=False, numlam=10):
        if type(db) == str:
            db = t.openFile(db,'r')
        g0 = self.dist0.calc_db_g(db, db.root.object.dist0, pts, partial, numlam=numlam)
        g1 = self.dist1.calc_db_g(db, db.root.object.dist1, pts, partial, numlam=numlam)
        Ec = db.root.object._v_attrs['c']
        efactor = log(Ec) - log(1-Ec)
        return g0 - g1 + efactor

    def calc_curr_g(self, pts, numlam=10):
        g0 = self.dist0.calc_curr_g(pts, numlam)
        g1 = self.dist1.calc_curr_g(pts, numlam)
        efactor = log(self.Ec) - log(1-self.Ec)
        return g0 - g1 + efactor

#cpdef int bench1(int N):
    #cdef int i
    #x = di.poisson(10).rvs(20).astype(np.double)
    #for i in xrange(N):
        #test_method(x)

#cpdef int bench2(int N):
    #cdef int i
    #cdef double [:] x = di.poisson(10).rvs(20).astype(np.double)
    #for i in xrange(N):
        #test_method(x)

#def benchp(N):
    #cdef int i
    #x = di.poisson(10).rvs(20).astype(np.double)
    #for i in xrange(N):
        #test_method(x)

#cpdef double [:] test_method(double[:] arr):
    #return di.poisson.logpmf(arr, 10.0)
