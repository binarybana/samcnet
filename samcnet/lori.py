from __future__ import division

import numpy as np
import pylab as p
import tables as t
import matplotlib as mpl
import random
from math import log, exp, pi, lgamma

import scipy.stats as st
import scipy.stats.distributions as di
import scipy
from scipy.special import betaln

from statsmodels.sandbox.distributions.mv_normal import MVT,MVNormal
#from sklearn.qda import QDA 

#import sys
#sys.path.append('/home/bana/GSP/research/samc/code')
#sys.path.append('/home/bana/GSP/research/samc/code/build')

# Borrowed from https://github.com/mattjj/pymattutil/blob/master/stats.py
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

class Johnson():
    def __init__(self, **kw):
        seed = np.random.randint(10**6)
        #seed = 10000
        print "Seed is %d" % seed
        np.random.seed(seed)

        #### Ground Truth Parameters 
        true = {0:{'mu':0.0, 'sigma':0.7413, 'gamma':-1.2, 'delta':0.9}, 
                1:{'mu':0.0, 'sigma':0.7413, 'gamma':1.2, 'delta':0.9},
                'type0':'SU',
                'type1':'SU',
                'n':100,
                'c':0.5}

        for k in kw:
            if type(params[k]) == dict:
                true[k].update(params[k])
            else:
                true[k] = params[k]
        self.true = true
        self.n = true['n'] # Data points
        self.n0 = int(true['c'] * self.n)
        self.n1 = self.n - self.n0 # not really random, but we're not so interested in this
        # source of variation
        self.true['n0'] = self.n0
        self.true['n1'] = self.n1

        ######## Generate Data ########
        
        self.dist0 = di.johnsonsu if true['type0'] == 'SU' else di.johnsonsb
        self.dist1 = di.johnsonsu if true['type1'] == 'SU' else di.johnsonsb
        self.true0 = self.dist0(self.true[0]['gamma'], self.true[0]['delta'],
                    loc=self.true[0]['mu'], scale=self.true[0]['sigma'])
        self.true1 = self.dist1(self.true[1]['gamma'], self.true[1]['delta'],
                    loc=self.true[1]['mu'], scale=self.true[1]['sigma'])

        data0 = self.true0.rvs(size=self.n0)
        data1 = self.true1.rvs(size=self.n1)

        self.data = np.hstack(( data0, data1 ))

        self.mask0 = np.hstack((
            np.ones(self.n0, dtype=np.bool),
            np.zeros(self.n1, dtype=np.bool)))
        self.mask1 = np.logical_not(self.mask0)

        ######## Starting point of MCMC Run #######
        self.init_params()

        ###### Bookeeping ######
        self.oldmu0 = None
        self.oldsigma0 = None
        self.oldgamma0 = None
        self.olddelta0 = None
        self.oldmu1 = None
        self.oldsigma1 = None
        self.oldgamma1 = None
        self.olddelta1 = None
        self.oldc = None

        # Proposal amounts for      c    mu0  mu1  0 sig1 gamma0 gam1 del0 del1
        self.propscales = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) 
        #self.propscales = np.array([0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) 

        ### Prior for c ###
        self.alpha0 = 1.0
        self.alpha1 = 1.0

    def init_params(self):
        self.mu0 = self.data[self.mask0].mean()
        self.mu1 = self.data[self.mask1].mean()
        self.sigma0 = 1.0
        self.sigma1 = 1.0
        self.gamma0 = -0.5
        self.gamma1 = 0.5
        self.delta0 = 1.0
        self.delta1 = 1.0
        self.c = 0.5

    def propose(self):
        self.oldmu0 = self.mu0
        self.oldmu1 = self.mu1
        self.oldsigma0 = self.sigma0
        self.oldsigma1 = self.sigma1
        self.oldgamma0 = self.gamma0
        self.oldgamma1 = self.gamma1
        self.olddelta0 = self.delta0
        self.olddelta1 = self.delta1
        self.oldc = self.c

        if np.random.rand() < 0.01: # Random restart
            self.init_params()
        else:

            rands = (np.random.rand(9) - 0.5) * self.propscales
            self.c += rands[0]
            if self.c >= 1.0:
                self.c -= self.c - 1 + 0.01
            elif self.c <= 0.0:
                self.c = abs(self.c) + 0.01

            self.mu0 += rands[1]
            self.mu1 += rands[2]
            self.sigma0 += rands[3]
            if self.sigma0 <= 0.0:
                self.sigma0 = 0.1
            self.sigma1 = rands[4]
            if self.sigma1 <= 0.0:
                self.sigma1 = 0.1
            self.gamma0 += rands[5]
            self.gamma1 += rands[6]
            self.delta0 += rands[7]
            self.delta1 += rands[8]

            # TODO Probably need to check if this is actually
            # a valid set of parameters

        return 0

    def copy(self):
        return (self.mu0, self.mu1, self.sigma0, self.sigma1,
                self.gamma0, self.gamma1, self.c)

    def reject(self):
        self.mu0 = self.oldmu0
        self.mu1 = self.oldmu1
        self.sigma0 = self.oldsigma0
        self.sigma1 = self.oldsigma1
        self.gamma0 = self.oldgamma0
        self.gamma1 = self.oldgamma1
        self.delta0 = self.olddelta0
        self.delta1 = self.olddelta1
        self.c = self.oldc

    def energy(self):
        sum = 0.0
        #class 0 negative log likelihood
        points = self.data[self.mask0]
        if points.size > 0:
            sum -= self.dist0.logpdf(points, self.gamma0, self.delta0,
                    loc=self.mu0, scale=self.sigma0).sum()

        #class 1 negative log likelihood
        points = self.data[self.mask1]
        if points.size > 0:
            sum -= self.dist1.logpdf(points, self.gamma1, self.delta1,
                    loc=self.mu1, scale=self.sigma1).sum()
                
        #Class proportion c (from page 3, eq 1)
        sum -= log(self.c)*(self.alpha0+self.n0-1) + log(1-self.c)*(self.alpha1+self.n1-1) \
                - betaln(self.alpha0 + self.n0, self.alpha1 + self.n1)

        #Now add in the priors...
        # TODO TODO FIXME
        return sum

    def get_grid(self):
        samples = np.array([self.true0.mean(), self.true1.mean()])
        n = 100
        l,h = samples.min(), samples.max()
        spread = (self.true0.std() + self.true1.std())
        l -= spread * 0.5
        h += spread * 0.5
        grid = np.linspace(l,h,n)
        return n, (l,h), grid


    def draw_from_true(self, cls, n=20):
        if cls == 0:
            return self.true0.rvs(size=n)
        else:
            return self.true1.rvs(size=n)

    def true_densities(self, line):
        return (self.true0.logpdf(line), self.true1.logpdf(line))

    def calc_densities(self, line):
        fx0 = self.dist0.logpdf(line, self.gamma0, self.delta0, loc=self.mu0, scale=self.sigma0)
        fx1 = self.dist1.logpdf(line, self.gamma1, self.delta1, loc=self.mu1, scale=self.sigma1)
        return (fx0, fx1)

    def init_db(self, db, size):
        """ Takes a Pytables Group object (group) and the total number of samples expected and
        expands or creates the necessary groups.
        """
        objroot = db.root.object
        db.createEArray(objroot.objfxn, 'mu0', t.Float64Atom(), (0,), expectedrows=size)
        db.createEArray(objroot.objfxn, 'mu1', t.Float64Atom(), (0,), expectedrows=size)
        db.createEArray(objroot.objfxn, 'sigma0', t.Float64Atom(), (0,), expectedrows=size)
        db.createEArray(objroot.objfxn, 'sigma1', t.Float64Atom(), (0,), expectedrows=size)
        db.createEArray(objroot.objfxn, 'gamma0', t.Float64Atom(), (0,), expectedrows=size)
        db.createEArray(objroot.objfxn, 'gamma1', t.Float64Atom(), (0,), expectedrows=size)
        db.createEArray(objroot.objfxn, 'delta0', t.Float64Atom(), (0,), expectedrows=size)
        db.createEArray(objroot.objfxn, 'delta1', t.Float64Atom(), (0,), expectedrows=size)
        db.createEArray(objroot.objfxn, 'c', t.Float64Atom(), (0,), expectedrows=size)
        objroot._v_attrs.true_dict = self.true

    def save_iter_db(self, db):
        """ Saves objective function (and possible samples depending on verbosity) to
        Pytables db
        """ 
        root = db.root.object
        root.objfxn.mu0.append((self.mu0,))
        root.objfxn.mu1.append((self.mu1,))
        root.objfxn.sigma0.append((self.sigma0,))
        root.objfxn.sigma1.append((self.sigma1,))
        root.objfxn.gamma0.append((self.gamma0,))
        root.objfxn.gamma1.append((self.gamma1,))
        root.objfxn.delta0.append((self.delta0,))
        root.objfxn.delta1.append((self.delta1,))
        root.objfxn.c.append((self.c,))

class Classifier(object):
    def __init__(self, dist0, dist1):
        self.dist0 = dist0
        self.dist1 = dist1

        assert(dist0.D == dist1.D)
        self.D = dist0.D
        self.n = dist0.n + dist1.n
        pass

    def predict(self, data):
        """ Given data, return vector with predicted labels """
        return map(self.predict_single, data)

    def error(self, data, labels):
        """ Given data and true labels return (numcorrect, numwrong) """
        difs = (self.predict(data) - labels).nonzero()[0].sum()
        return (labels.shape[0] - difs, difs)

class GaussianDist(object):
    def __init__(self, truemu, truesigma, n, priormu, nu, kappa, S, alpha=1):
        """ Initialize Gaussian distribution with data and priors
        then calculate the analytic posterior """
        self.truemu = truemu.copy()
        self.truesigma = truesigma.copy()

        self.n = n
        self.data = self.gen_data(n)

        self.priormu = priormu.copy()
        self.nu = nu
        self.kappa = kappa
        self.S = S.copy()
        self.alpha = alpha
        self.D = self.data.shape[1]

        self.train_analytic()

    def gen_data(self, n):
        return np.random.multivariate_normal(self.truemu, self.truesigma, n)

    def train_analytic(self):
        """ Calculating the analytic distribution posteriors given on page 15 of Lori's 
        Optimal Classification eq 34. """
        self.nustar = self.nu + self.n

        samplemean = self.data.mean(axis=0)
        samplecov = np.cov(self.data.T)

        self.mustar = (self.nu * self.priormu + self.n * samplemean) \
                / (self.nu + self.n)
        self.kappastar = self.kappa + self.n
        self.Sstar = self.S + (self.n-1)*samplecov + self.nu*self.n/(self.nu+self.n)\
                * np.outer((samplemean - self.priormu), (samplemean - self.priormu))
                
        # Now calculate effective class conditional densities from eq 55 page 21
        self.fx = MVT(
                self.mustar, 
                (self.nustar+1)/(self.kappastar-self.D+1)/self.nustar * self.Sstar, 
                self.kappastar - self.D + 1)

    def eval_posterior(self, grid_n, grid):
        return self.fx.logpdf(grid).reshape(grid_n, -1)

    def eval_true(self, grid_n, grid):
        return MVNormal(self.truemu, self.truesigma).logpdf(grid).reshape(grid_n, -1)

class GaussianCls(Classifier):
    def __init__(self, dist0, dist1, c=None):
        """ Trains Classifier """
        super(GaussianCls, self).__init__(dist0, dist1)
        self.Ec = c or (self.dist0.n + self.dist0.alpha) / (self.n + self.dist0.alpha + self.dist1.alpha)
        # Expectation of C from page 3 eq. 1 using beta conjugate prior
        self.efactor = log(self.Ec) - log(1-self.Ec)

    def predict_single(self, data):
        if self.dist0.fx.logpdf(data) - self.dist1.fx.logpdf(data) + self.efactor > 0: 
            return 0
        else:
            return 1
        # TODO Might have these switched

    def calc_densities(self, grid, n, record):
        return (self.dist0.fx.logpdf(grid) - self.dist1.fx.logpdf(grid) + self.efactor).reshape(grid_n, -1)

def gen_dists():
    D = 2
    priorsigma0 = np.eye(D)*0.3
    priorsigma1 = np.eye(D)*0.3
    nu0 = 12.0
    nu1 = 2.0
    kappa0 = 6.0
    kappa1 = 6.0
    
    #### Ground Truth Parameters 
    c = 0.83 # Ground truth class marginal
    sigma0 = sample_invwishart(priorsigma0, nu0)
    sigma1 = sample_invwishart(priorsigma1, nu1)
    mu0 = np.zeros(D)
    mu1 = np.ones(D)
    #mu0 = np.random.multivariate_normal(self.priormu0, sigma0/self.nu0, size=1)[0]
    #mu1 = np.random.multivariate_normal(self.priormu1, sigma1/self.nu1, size=1)[0]

    ###########
    n = 30
    n0 = st.binom.rvs(n, c)
    n1 = n - n0
    dist0 = GaussianDist(mu0, sigma0, n0, mu0, nu0, kappa0, priorsigma0)
    dist1 = GaussianDist(mu1, sigma1, n1, mu1, nu1, kappa1, priorsigma1)
    return c, dist0, dist1

class GaussianSampler(Classifier):
    def __init__(self, dist0, dist1):
        super(GaussianSampler, self).__init__(dist0, dist1)

        ##### Proposal variances ######
        self.propdof = 200
        self.propmu = 0.2

        ######## Starting point of MCMC Run #######
        self.c = np.random.rand()
        self.mu0 = dist0.data.mean(axis=0)
        self.mu1 = dist1.data.mean(axis=0)
        self.sigma0 = sample_invwishart(self.dist0.S, self.dist0.kappa)
        self.sigma1 = sample_invwishart(self.dist1.S, self.dist1.kappa)

        ###### Bookeeping ######
        self.oldmu0 = None
        self.oldmu1 = None
        self.oldsigma0 = None
        self.oldsigma1 = None
        self.oldc = None

        self.mtype = ''

    def propose(self):
        self.oldmu0 = self.mu0.copy()
        self.oldmu1 = self.mu1.copy()
        self.oldsigma0 = self.sigma0.copy()
        self.oldsigma1 = self.sigma1.copy()
        self.oldc = self.c

        if np.random.rand() < 0.01: # Random restart
            self.mu0 = self.dist0.data.mean(axis=0)
            self.mu1 = self.dist1.data.mean(axis=0)
            self.sigma0 = sample_invwishart(self.dist0.S, self.dist0.kappa)
            self.sigma1 = sample_invwishart(self.dist1.S, self.dist1.kappa)
        else:
            self.mu0 += (np.random.rand(self.D)-0.5)*self.propmu
            self.mu1 += (np.random.rand(self.D)-0.5)*self.propmu
            self.sigma0 = sample_invwishart(self.sigma0*self.propdof, self.propdof)
            self.sigma1 = sample_invwishart(self.sigma1*self.propdof, self.propdof)

        add = np.random.randn()*0.1
        self.c += add
        if self.c >= 1.0:
            self.c -= self.c - 1 + 0.01
        elif self.c <= 0.0:
            self.c = abs(self.c) + 0.01
        return 0

    def copy(self):
        return (self.mu0.copy(), self.mu1.copy(), self.sigma0.copy(), self.sigma1.copy(), self.c)

    def reject(self):
        self.mu0 = self.oldmu0.copy()
        self.mu1 = self.oldmu1.copy()
        self.sigma0 = self.oldsigma0.copy()
        self.sigma1 = self.oldsigma1.copy()
        self.c = self.oldc

    def energy(self):
        sum = 0.0
        #class 0 negative log likelihood
        points = self.dist0.data
        if points.size > 0:
            sum -= logp_normal(points, self.mu0, self.sigma0).sum()

        #class 1 negative log likelihood
        points = self.dist1.data
        if points.size > 0:
            sum -= logp_normal(points, self.mu1, self.sigma1).sum()
                
        #Class proportion c (from page 3, eq 1)
        sum -= log(self.c)*(self.dist0.alpha+self.dist0.n-1) \
                + log(1-self.c)*(self.dist1.alpha+self.dist1.n-1) \
                - betaln(self.dist0.alpha + self.dist0.n, self.dist1.alpha + self.dist1.n)

        #Now add in the priors...
        sum -= logp_invwishart(self.sigma0, self.dist0.kappa, self.dist0.S)
        sum -= logp_invwishart(self.sigma1, self.dist1.kappa, self.dist1.S)
        sum -= logp_normal(self.mu0, self.dist0.priormu, self.sigma0, self.dist0.nu)
        sum -= logp_normal(self.mu1, self.dist1.priormu, self.sigma1, self.dist1.nu)

        return sum

    def calc_densities(self, grid, n, record):
        fx0 = np.exp(logp_normal(grid, record['mu0'], record['sigma0'])).reshape(n,n)
        fx1 = np.exp(logp_normal(grid, record['mu1'], record['sigma1'])).reshape(n,n)
        return (fx0, fx1)

    def init_db(self, db, size):
        """ Takes a Pytables Group object (group) and the total number of samples expected and
        expands or creates the necessary groups.
        """
        objroot = db.root.object
        db.createEArray(objroot.objfxn, 'mu0', t.Float64Atom(shape=(2,)), (0,), expectedrows=size)
        db.createEArray(objroot.objfxn, 'mu1', t.Float64Atom(shape=(2,)), (0,), expectedrows=size)
        db.createEArray(objroot.objfxn, 'sigma0', t.Float64Atom(shape=(2,2)), (0,), expectedrows=size)
        db.createEArray(objroot.objfxn, 'sigma1', t.Float64Atom(shape=(2,2)), (0,), expectedrows=size)
        db.createEArray(objroot.objfxn, 'c', t.Float64Atom(), (0,), expectedrows=size)
        #objroot._v_attrs.true_dict = self.true
        #temp = {}
        #temp['entropy'] = 'Entropy in bits'
        #temp['kld']  = 'KL-Divergence from true network in bits'
        #temp['edge_distance']  = 'Proportion of incorrect edges |M-X|/n^2'
        #objroot._v_attrs.descs = temp

    def save_iter_db(self, db):
        """ Saves objective function (and possible samples depending on verbosity) to
        Pytables db
        """ 
        root = db.root.object
        root.objfxn.mu0.append((self.mu0,))
        root.objfxn.mu1.append((self.mu1,))
        root.objfxn.sigma0.append((self.sigma0,))
        root.objfxn.sigma1.append((self.sigma1,))
        root.objfxn.c.append((self.c,))

def get_grid(cls):
    samples = np.vstack((cls.dist0.gen_data(30), cls.dist1.gen_data(30)))
    lx,hx,ly,hy = np.vstack((samples.min(axis=0), samples.max(axis=0))).T.flatten()
    xspread, yspread = hx-lx, hy-ly
    n = 30
    lx -= xspread * 0.2
    hx += xspread * 0.2
    ly -= yspread * 0.2
    hy += yspread * 0.2
    gextent = (lx,hx,ly,hy)
    grid = np.dstack(np.meshgrid(
                    np.linspace(lx,hx,n),
                    np.linspace(ly,hy,n))).reshape(-1,2)
    return n, gextent, grid

#def calc_gavg(cmean,fx0avg,fx1avg):
    #return fx0avg*cmean / (fx1avg*(1-cmean))

#def plot_run(c, db):
    ## Plot the data
    #p.figure()
    #splot = p.subplot(2,2,1)
    #p.title('Data')
    #p.grid(True)

    ## Data Plot
    #p.plot(c.data[c.mask0,0], c.data[c.mask0,1], 'r.', label='class 0')
    #p.plot(c.data[c.mask1,0], c.data[c.mask1,1], 'g.', label='class 1')
    #p.legend(loc='best')
    #plot_ellipse(splot, c.true['mu0'], c.true['sigma0'], 'red')
    #plot_ellipse(splot, c.true['mu1'], c.true['sigma1'], 'green')

    ## MCMC Mean Samples Plot
    #splot = p.subplot(2,2,2, sharex=splot, sharey=splot)
    #p.grid(True)
    #p.title('MCMC Mean Samples')

    #plot_ellipse(splot, c.true['mu0'], c.true['sigma0'], 'red')
    #plot_ellipse(splot, c.true['mu1'], c.true['sigma1'], 'green')
    #p.plot(np.vstack(db['mu0'])[:,0], np.vstack(db['mu0'])[:,1], 'r.', alpha=0.3)
    #p.plot(np.vstack(db['mu1'])[:,0], np.vstack(db['mu1'])[:,1], 'g.', alpha=0.3)

    #mu0mean = np.vstack(db['mu0']).mean(axis=0)
    #mu1mean = np.vstack(db['mu1']).mean(axis=0)

    #p.plot(mu0mean[0], mu0mean[1], 'ro', markersize=5)
    #p.plot(mu1mean[0], mu0mean[1], 'go', markersize=5)

    ############# Plot Gavg from mcmc samples
    #grid_n, extent, grid = c.get_grid()
    #cmean, fx0avg, fx1avg = c.estimate_sample_means(grid_n, grid, db) 
    #afx0, afx1, agavg = c.calc_analytic(grid_n, grid)
    
    #splot = p.subplot(2,2,3, sharex=splot, sharey=splot)
    #gmin, gmax = agavg.min(), agavg.max()
    #gavg = np.clip(calc_gavg(cmean, fx0avg, fx1avg), np.exp(gmin), np.exp(gmax))
    #p.imshow(np.log(gavg), extent=extent, origin='lower')
    #p.colorbar()
    #p.contour(np.log(gavg), [0.0], extent=extent, origin='lower', cmap = p.cm.gray)

    #p.plot(c.data[c.mask0,0], c.data[c.mask0,1], 'r.')
    #p.plot(c.data[c.mask1,0], c.data[c.mask1,1], 'g.')
    #plot_ellipse(splot, c.true['mu0'], c.true['sigma0'], 'red')
    #plot_ellipse(splot, c.true['mu1'], c.true['sigma1'], 'green')

    ############# Plot analytic G function
    #splot = p.subplot(2,2,4, sharex=splot, sharey=splot)
    #p.imshow(agavg, extent=extent, origin='lower')
    #p.colorbar()
    #p.contour(agavg, [0.0], extent=extent, origin='lower', cmap = p.cm.gray)
    #p.contour(np.log(gavg), [0.0], extent=extent, origin='lower', cmap = p.cm.gray,
            #linestyles='dotted')

    #p.plot(c.data[c.mask0,0], c.data[c.mask0,1], 'r.')
    #p.plot(c.data[c.mask1,0], c.data[c.mask1,1], 'g.')

    #plot_ellipse(splot, c.true['mu0'], c.true['sigma0'], 'red')
    #plot_ellipse(splot, c.true['mu1'], c.true['sigma1'], 'green')

#def plot_cross(c,db,ind=None):
    #grid_n, extent, grid = c.get_grid()
    #cmean, fx0avg, fx1avg = c.estimate_sample_means(grid_n, grid, db) 
    #afx0, afx1, agavg = c.calc_analytic(grid_n, grid)
    #p.figure()
    #x = grid[:grid_n,0]
    #p.subplot(3,1,1)
    #i0 = np.argmax(afx0.sum(axis=1))
    #i1 = np.argmax(afx1.sum(axis=1))

    #p.plot(x,afx0[i0,:], 'r',label='analyticfx0')
    #p.plot(x,np.log(fx0avg[i0,:]), 'g', label='avgfx0')
    #p.xlabel('slice at index %d from bottom' % i0)
    #p.legend(loc='best')
    #p.grid(True)
    #p.subplot(3,1,2)
    #p.plot(x,afx1[i1,:], 'r',label='analyticfx1')
    #p.plot(x,np.log(fx1avg[i1,:]), 'g', label='avgfx1')
    #p.xlabel('slice at index %d from bottom' % i1)
    #p.legend(loc='best')
    #p.grid(True)

    #p.subplot(3,1,3)
    #ind = ind if ind else (i0+i1)/2
    #p.plot(x,log(cmean)+np.log(fx0avg)[ind,:], 'r', label='avg0')
    #p.plot(x,log(c.Ec)+afx0[ind,:], 'r--',label='true0')
    #p.plot(x,log(1-cmean)+np.log(fx1avg)[ind,:], 'g', label='avg0')
    #p.plot(x,log(1-c.Ec)+afx1[ind,:], 'g--',label='true1')
    #p.xlabel('slice at index %d from bottom' % ind)
    #p.legend(loc='best')
    #p.grid(True)

#def plot_eff(c):
    #p.figure()
    #splot = p.subplot(2,1,1)
    #p.imshow(np.log(c.fx0avg), extent=c.gextent, origin='lower')
    #p.title('fx0')
    #p.colorbar()
    #p.plot(c.data[np.arange(c.n0),0], c.data[np.arange(c.n0),1], 'r.')
    #plot_ellipse(splot, c.true['mu0'], c.true['sigma0'], 'red')

    #############
    #splot = p.subplot(2,1,2, sharex=splot, sharey=splot)
    #p.imshow(np.log(c.fx1avg), extent=c.gextent, origin='lower')
    #p.colorbar()
    #p.title('fx1')
    #p.plot(c.data[np.arange(c.n0,c.n),0], c.data[np.arange(c.n0,c.n),1], 'g.')
    #plot_ellipse(splot, c.true['mu1'], c.true['sigma1'], 'green')

def plot_ellipse(splot, mean, cov, color):
    v, w = np.linalg.eigh(cov)
    u = w[0] / np.linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi  # convert to degrees
    # filled gaussian at 2 standard deviation
    ell = mpl.patches.Ellipse(mean, 2 * v[0] ** 0.5, 2 * v[1] ** 0.5,
                                            180 + angle, color=color)
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.3)
    splot.add_artist(ell)

if __name__ == '__main__':
    import samc
    import pylab as p

    #seed = np.random.randint(10**6)
    seed = 32
    print "Seed is %d" % seed
    np.random.seed(seed)

    ## First generate true distributions and data
    c, dist0, dist1 = gen_dists()

    ## Now test Gaussian Analytic calculation
    gc = GaussianCls(dist0, dist1)

    c = GaussianSampler(dist0,dist1)
    s = samc.SAMCRun(c, burn=1e2, stepscale=1000, refden=1, thin=10)
    s.sample(2e3, temperature=1)

    s.compute_means(False)

    n, gextent, grid = get_grid(gc)
    #lp0,lp1 = j.calc_densities(line)
    #tlp0,tlp1 = j.true_densities(line)
    sp = p.subplot(2,2,1)
    p.axis(gextent)
    plot_ellipse(sp, dist0.truemu, dist0.truesigma, 'green')
    plot_ellipse(sp, dist1.truemu, dist1.truesigma, 'red')

    p.plot(dist0.data[:,0], dist0.data[:,1], 'go')
    p.plot(dist1.data[:,0], dist1.data[:,1], 'ro')

    p.subplot(2,2,2)
    p.axis(gextent)
    p.plot(dist0.data[:,0], dist0.data[:,1], 'go')
    p.plot(dist1.data[:,0], dist1.data[:,1], 'ro')

    p0 = dist0.eval_posterior(n, grid)
    p1 = dist1.eval_posterior(n, grid)
    p.imshow(p0-p1+gc.efactor, extent=gextent, origin='lower')
    p.colorbar()
    p.contour(p0-p1+gc.efactor, [0.0], extent=gextent, origin='lower', cmap = p.cm.gray)

    p.subplot(2,2,3)

    p.show()

