from __future__ import division

import numpy as np
import pylab as p
import matplotlib as mpl
import random
from math import log, exp, pi, lgamma

import scipy.stats as st
import scipy.stats.distributions as di
import scipy
from scipy.special import betaln

from statsmodels.sandbox.distributions.mv_normal import MVT,MVNormal
#from sklearn.qda import QDA

import sys
sys.path.append('/home/bana/GSP/research/samc/code')
sys.path.append('/home/bana/GSP/research/samc/code/build')

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

class Classification():
    def __init__(self):
        seed = np.random.randint(10**6)
        print "Seed is %d" % seed
        np.random.seed(seed)

        self.D = 2 # Dimension
        self.n = 30 # Data points

        ##### Prior Values and Confidences ######
        self.priorsigma0 = np.eye(self.D)*0.3
        self.priorsigma1 = np.eye(self.D)*0.3
        self.kappa0 = 6
        self.kappa1 = 6

        self.priormu0 = np.zeros(self.D)
        self.priormu1 = np.ones(self.D)*0.5

        self.nu0 = 12.0
        self.nu1 = 2.0

        self.alpha0 = 1.0
        self.alpha1 = 1.0

        #### Ground Truth Parameters 
        c = 0.83 # Ground truth class marginal

        sigma0 = sample_invwishart(self.priorsigma0, self.nu0)
        sigma1 = sample_invwishart(self.priorsigma1, self.nu1)

        #mu0 = np.zeros(self.D)
        #mu1 = np.ones(self.D)
        mu0 = np.random.multivariate_normal(self.priormu0, sigma0/self.nu0, size=1)[0]
        mu1 = np.random.multivariate_normal(self.priormu1, sigma1/self.nu1, size=1)[0]

        ##### Record true values for plotting, comparison #######
        self.true = {'c':c, 
                'mu0': mu0.copy(), 
                'sigma0': sigma0.copy(), 
                'mu1': mu1.copy(), 
                'sigma1': sigma1.copy(),
                'seed': seed}

        self.n0 = st.binom.rvs(self.n, c)
        self.n1 = self.n - self.n0
        self.data = np.vstack(( \
            np.random.multivariate_normal(mu0, sigma0, self.n0),
            np.random.multivariate_normal(mu1, sigma1, self.n1) ))

        self.mask0 = np.hstack((
            np.ones(self.n0, dtype=np.bool),
            np.zeros(self.n1, dtype=np.bool)))
        self.mask1 = np.logical_not(self.mask0)

        ##### Proposal variances ######
        self.propdof = 200
        self.propmu = 0.2

        ######## Starting point of MCMC Run #######
        self.c = np.random.rand()
        self.mu0 = self.data[self.mask0].mean(axis=0)
        self.mu1 = self.data[self.mask1].mean(axis=0)
        self.sigma0 = sample_invwishart(self.priorsigma0, self.kappa0)
        self.sigma1 = sample_invwishart(self.priorsigma1, self.kappa1)

        ###### Bookeeping ######
        self.oldmu0 = None
        self.oldmu1 = None
        self.oldsigma0 = None
        self.oldsigma1 = None
        self.oldc = None

        self.mtype = ''

        #### Calculating the Analytic solution given on page 15 of Lori's 
        #### Optimal Classification eq 34.
        self.nu0star = self.nu0 + self.n0
        self.nu1star = self.nu1 + self.n1

        sample0mean = self.data[self.mask0].mean(axis=0)
        sample1mean = self.data[self.mask1].mean(axis=0)
        sample0cov = np.cov(self.data[self.mask0].T)
        sample1cov = np.cov(self.data[self.mask1].T)

        self.mu0star = (self.nu0*self.priormu0 + self.n0 * sample0mean) \
                / (self.nu0 + self.n0)
        self.mu1star = (self.nu1*self.priormu1 + self.n1 * sample1mean ) \
                / (self.nu1 + self.n1)

        self.kappa0star = self.kappa0 + self.n0
        self.kappa1star = self.kappa1 + self.n1

        self.S0star = self.priorsigma0 + (self.n0-1)*sample0cov + self.nu0*self.n0/(self.nu0+self.nu0)\
                * np.outer((sample0mean - self.priormu0), (sample0mean - self.priormu0))
        self.S1star = self.priorsigma1 + (self.n1-1)*sample1cov + self.nu1*self.n1/(self.nu1+self.nu1)\
                * np.outer((sample1mean - self.priormu1), (sample1mean - self.priormu1))
                
        #### Now calculate effective class conditional densities from eq 55
        #### page 21

        self.fx0 = MVT(
                self.mu0star, 
                (self.nu0star+1)/(self.kappa0star-self.D+1)/self.nu0star * self.S0star, 
                self.kappa0star - self.D + 1)
        self.fx1 = MVT(
                self.mu1star, 
                (self.nu1star+1)/(self.kappa1star-self.D+1)/self.nu1star * self.S1star, 
                self.kappa1star - self.D + 1)
        self.Ec = (self.n0 + self.alpha0) / (self.n + self.alpha0 + self.alpha1)

    def propose(self):
        self.oldmu0 = self.mu0.copy()
        self.oldmu1 = self.mu1.copy()
        self.oldsigma0 = self.sigma0.copy()
        self.oldsigma1 = self.sigma1.copy()
        self.oldc = self.c

        if np.random.rand() < 0.01: # Random restart
            self.mu0 = self.data[self.mask0].mean(axis=0)
            self.mu1 = self.data[self.mask1].mean(axis=0)
            self.sigma0 = sample_invwishart(self.priorsigma0, self.kappa0)
            self.sigma1 = sample_invwishart(self.priorsigma1, self.kappa1)
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
        points = self.data[self.mask0]
        if points.size > 0:
            sum -= logp_normal(points, self.mu0, self.sigma0).sum()

        #class 1 negative log likelihood
        points = self.data[self.mask1]
        if points.size > 0:
            sum -= logp_normal(points, self.mu1, self.sigma1).sum()
                
        #Class proportion c (from page 3, eq 1)
        sum -= log(self.c)*(self.alpha0+self.n0-1) + log(1-self.c)*(self.alpha1+self.n1-1) \
                - betaln(self.alpha0 + self.n0, self.alpha1 + self.n1)

        #Now add in the priors...
        sum -= logp_invwishart(self.sigma0,self.kappa0,self.priorsigma0)
        sum -= logp_invwishart(self.sigma1,self.kappa1,self.priorsigma1)
        sum -= logp_normal(self.mu0, self.priormu0, self.sigma0, self.nu0)
        sum -= logp_normal(self.mu1, self.priormu1, self.sigma1, self.nu1)

        return sum

    def calc_densities(self, grid, n, record):
        fx0 = np.exp(logp_normal(grid, record['mu0'], record['sigma0'])).reshape(n,n)
        fx1 = np.exp(logp_normal(grid, record['mu1'], record['sigma1'])).reshape(n,n)
        return (fx0, fx1)

    def save_to_db(self):
        return (self.mu0,
                self.mu1,
                self.sigma0.copy(),
                self.sigma1.copy(),
                self.c)

    def get_grid(self):
        samples = np.vstack((self.draw_from_true(0), self.draw_from_true(1)))
        n = 30
        lx,hx,ly,hy = np.vstack((samples.min(axis=0), samples.max(axis=0))).T.flatten()
        xspread, yspread = hx-lx, hy-ly
        lx -= xspread * 0.2
        hx += xspread * 0.2
        ly -= yspread * 0.2
        hy += yspread * 0.2
        gextent = (lx,hx,ly,hy)
        grid = np.dstack(np.meshgrid(
                        np.linspace(lx,hx,n),
                        np.linspace(ly,hy,n))).reshape(-1,2)
        return n, gextent, grid

    def calc_analytic(self, grid_n, grid):
        # Expectation of C from page 3 eq. 1 using beta conjugate prior
        ag = (self.fx0.logpdf(grid) - self.fx1.logpdf(grid) \
                + log(self.Ec) - log(1-self.Ec)).reshape(grid_n,-1)
        afx0 = self.fx0.logpdf(grid).reshape(grid_n, -1)
        afx1 = self.fx1.logpdf(grid).reshape(grid_n, -1)
        return afx0, afx1, ag

    #def estimate_sample_means(self, n, grid, db):
        #fxlist = [self.calc_densities(grid,n,x) for x in db]
        #fx0list = np.asarray([x[0] for x in fxlist])
        #fx1list = np.asarray([x[1] for x in fxlist])
        #if self.mhtype=='mh': 
            #fx0avg = np.dstack(fx0list).mean(axis=2)
            #fx1avg = np.dstack(fx1list).mean(axis=2)
            #cmean = db['c'].mean()
        #elif self.mhtype=='samc': 
            #part = np.exp(db['theta'] - db['theta'].max())
            #numerator0 = (part * fx0list).sum(axis=2)
            #numerator1 = (part * fx1list).sum(axis=2)
            #numeratorc = (part * db['c']).sum()
            #denom = part.sum()
            #fx0avg, fx1avg, cmean = numerator0/denom, numerator1/denom, numeratorc/denom
        #else:
            #raise Exception("NOT IMPLEMENTED: Mean calculation for %s method" % self.mhtype)
        #return cmean, fx0avg, fx1avg

    def draw_from_true(self, cls, n=100):
        if cls == 0:
            mu = self.true['mu0']
            sigma = self.true['sigma0']
        else:
            mu = self.true['mu1']
            sigma = self.true['sigma1']
        return np.random.multivariate_normal(mu, sigma, n)

def calc_gavg(cmean,fx0avg,fx1avg):
    return fx0avg*cmean / (fx1avg*(1-cmean))

def plot_run(c, db):
    # Plot the data
    p.figure()
    splot = p.subplot(2,2,1)
    p.title('Data')
    p.grid(True)

    # Data Plot
    p.plot(c.data[c.mask0,0], c.data[c.mask0,1], 'r.', label='class 0')
    p.plot(c.data[c.mask1,0], c.data[c.mask1,1], 'g.', label='class 1')
    p.legend(loc='best')
    plot_ellipse(splot, c.true['mu0'], c.true['sigma0'], 'red')
    plot_ellipse(splot, c.true['mu1'], c.true['sigma1'], 'green')

    # MCMC Mean Samples Plot
    splot = p.subplot(2,2,2, sharex=splot, sharey=splot)
    p.grid(True)
    p.title('MCMC Mean Samples')

    plot_ellipse(splot, c.true['mu0'], c.true['sigma0'], 'red')
    plot_ellipse(splot, c.true['mu1'], c.true['sigma1'], 'green')
    p.plot(np.vstack(db['mu0'])[:,0], np.vstack(db['mu0'])[:,1], 'r.', alpha=0.3)
    p.plot(np.vstack(db['mu1'])[:,0], np.vstack(db['mu1'])[:,1], 'g.', alpha=0.3)

    mu0mean = np.vstack(db['mu0']).mean(axis=0)
    mu1mean = np.vstack(db['mu1']).mean(axis=0)

    p.plot(mu0mean[0], mu0mean[1], 'ro', markersize=5)
    p.plot(mu1mean[0], mu0mean[1], 'go', markersize=5)

    ############ Plot Gavg from mcmc samples
    grid_n, extent, grid = c.get_grid()
    cmean, fx0avg, fx1avg = c.estimate_sample_means(grid_n, grid, db) 
    afx0, afx1, agavg = c.calc_analytic(grid_n, grid)
    
    splot = p.subplot(2,2,3, sharex=splot, sharey=splot)
    gmin, gmax = agavg.min(), agavg.max()
    gavg = np.clip(calc_gavg(cmean, fx0avg, fx1avg), np.exp(gmin), np.exp(gmax))
    p.imshow(np.log(gavg), extent=extent, origin='lower')
    p.colorbar()
    p.contour(np.log(gavg), [0.0], extent=extent, origin='lower', cmap = p.cm.gray)

    p.plot(c.data[c.mask0,0], c.data[c.mask0,1], 'r.')
    p.plot(c.data[c.mask1,0], c.data[c.mask1,1], 'g.')
    plot_ellipse(splot, c.true['mu0'], c.true['sigma0'], 'red')
    plot_ellipse(splot, c.true['mu1'], c.true['sigma1'], 'green')

    ############ Plot analytic G function
    splot = p.subplot(2,2,4, sharex=splot, sharey=splot)
    p.imshow(agavg, extent=extent, origin='lower')
    p.colorbar()
    p.contour(agavg, [0.0], extent=extent, origin='lower', cmap = p.cm.gray)
    p.contour(np.log(gavg), [0.0], extent=extent, origin='lower', cmap = p.cm.gray,
            linestyles='dotted')

    p.plot(c.data[c.mask0,0], c.data[c.mask0,1], 'r.')
    p.plot(c.data[c.mask1,0], c.data[c.mask1,1], 'g.')

    plot_ellipse(splot, c.true['mu0'], c.true['sigma0'], 'red')
    plot_ellipse(splot, c.true['mu1'], c.true['sigma1'], 'green')

def plot_cross(c,db,ind=None):
    grid_n, extent, grid = c.get_grid()
    cmean, fx0avg, fx1avg = c.estimate_sample_means(grid_n, grid, db) 
    afx0, afx1, agavg = c.calc_analytic(grid_n, grid)
    p.figure()
    x = grid[:grid_n,0]
    p.subplot(3,1,1)
    i0 = np.argmax(afx0.sum(axis=1))
    i1 = np.argmax(afx1.sum(axis=1))

    p.plot(x,afx0[i0,:], 'r',label='analyticfx0')
    p.plot(x,np.log(fx0avg[i0,:]), 'g', label='avgfx0')
    p.xlabel('slice at index %d from bottom' % i0)
    p.legend(loc='best')
    p.grid(True)
    p.subplot(3,1,2)
    p.plot(x,afx1[i1,:], 'r',label='analyticfx1')
    p.plot(x,np.log(fx1avg[i1,:]), 'g', label='avgfx1')
    p.xlabel('slice at index %d from bottom' % i1)
    p.legend(loc='best')
    p.grid(True)

    p.subplot(3,1,3)
    ind = ind if ind else (i0+i1)/2
    p.plot(x,log(cmean)+np.log(fx0avg)[ind,:], 'r', label='avg0')
    p.plot(x,log(c.Ec)+afx0[ind,:], 'r--',label='true0')
    p.plot(x,log(1-cmean)+np.log(fx1avg)[ind,:], 'g', label='avg0')
    p.plot(x,log(1-c.Ec)+afx1[ind,:], 'g--',label='true1')
    p.xlabel('slice at index %d from bottom' % ind)
    p.legend(loc='best')
    p.grid(True)

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
