from __future__ import division

import numpy as np
import pylab as p
import matplotlib as mpl
import random
from math import log, exp, pi, lgamma

import scipy.stats as st
import scipy
from scipy.special import betaln

from statsmodels.sandbox.distributions.mv_normal import MVT
#from sklearn.qda import QDA

import sys
sys.path.append('/home/bana/GSP/research/samc/code')
sys.path.append('/home/bana/GSP/research/samc/code/build')

mydb = []

class MHRun():
    def __init__(self, obj, burn, thin=1):
        self.obj = obj
        self.burn = burn
        self.db = None

        self.mapvalue = None
        self.mapenergy = None

        self.thin = thin

        self.iteration = 0

        self.propose = 0
        self.accept = 0

    def sample(self, num):
        self.db = self.obj.init_db(self.db, num)
        minenergy = np.infty

        oldenergy = self.obj.energy()
        for i in range(int(num)):
            self.iteration += 1

            self.obj.propose()
            self.propose+=1

            newenergy = self.obj.energy()

            r = oldenergy - newenergy # ignoring non-symmetric proposals for now
            if r > 0.0 or np.random.rand() < exp(r):
                # Accept
                oldenergy = newenergy
                self.accept += 1
            else:
                self.obj.reject()

            if i>self.burn and i%self.thin == 0:
                self.obj.save_to_db(self.db, 0, oldenergy, i)

            if oldenergy < minenergy:
                minenergy = oldenergy
                self.mapvalue = self.obj.copy()
                self.mapenergy = oldenergy

            #if self.iteration%1e3 == 0:
                #p.plot(self.obj.gavg[10,:], 'b')
                #p.show()

            if self.iteration%1e3 == 0:
                print "Iteration: %9d, best energy: %7f, current energy: %7f" \
                        % (self.iteration, minenergy, oldenergy)

        print "Sampling done, acceptance: %d/%d = %f" \
                % (self.accept, self.propose, float(self.accept)/float(self.propose))
            
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
    #return (-0.5*k*log(2*pi) - 0.5*log(np.linalg.det(sigma)) \
            #- nu/2* (np.dot((x-mu), np.linalg.inv(sigma)) \
            #* (x-mu)).sum(axis=axis)).sum()

class Classification():
    def __init__(self):
        np.random.seed(347)

        self.D = 2 # Dimension
        self.n = 30 # Data points

        ##### Prior Values and Confidences ######
        self.priorsigma0 = np.eye(self.D)*10.3
        self.priorsigma1 = np.eye(self.D)*10.3
        self.kappa0 = 6
        self.kappa1 = 6

        self.priormu0 = np.zeros(self.D)
        self.priormu1 = np.ones(self.D)

        self.nu0 = 12.0
        self.nu1 = 12.0

        self.alpha0 = 1.0
        self.alpha1 = 1.0

        #### Ground Truth Parameters 
        c = 0.82 # Ground truth class marginal

        sigma0 = sample_invwishart(self.priorsigma0, self.nu0)
        sigma1 = sample_invwishart(self.priorsigma1, self.nu1)

        mu0 = np.zeros(self.D)
        mu1 = np.ones(self.D)

        ##### Record true values for plotting, comparison #######
        self.true = {'c':c, 
                'mu0': mu0, 
                'sigma0': sigma0, 
                'mu1': mu1, 
                'sigma1': sigma1}

        # For G function calculation and averaging
        self.grid_n = 20
        lx,hx,ly,hy = (-4,4,-4,4)
        self.gextent = (lx,hx,ly,hy)
        self.grid = np.dstack(np.meshgrid(
                        np.linspace(lx,hx,self.grid_n),
                        np.linspace(ly,hy,self.grid_n))).reshape(-1,2)
        self.fx0avg = np.zeros((self.grid_n, self.grid_n))
        self.fx1avg = np.zeros((self.grid_n, self.grid_n))
        self.numgavg = 0

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
        self.propdof = 100
        self.propmu = 0.3

        ######## Starting point of MCMC Run #######
        # 'Cheat' by starting at the 'right' spot... for now
        self.c = c
        self.mu0 = mu0.copy()
        self.mu1 = mu1.copy()
        self.sigma0 = sigma0.copy()
        self.sigma1 = sigma1.copy()
        #self.c = np.random.rand()
        #self.mu0 = np.random.rand(self.D)
        #self.mu1 = np.random.rand(self.D)
        #self.sigma0 = sample_invwishart(self.priorsigma0, self.kappa0)
        #self.sigma1 = sample_invwishart(self.priorsigma1, self.kappa1)

        
        ###### Bookeeping ######
        self.oldmu0 = None
        self.oldmu1 = None
        self.oldsigma0 = None
        self.oldsigma1 = None
        self.oldc = None
        self.oldlastenergy = None

        self.lastenergy = -np.infty

        #### Calculating the Analytic solution given on page 15 of Lori's 
        #### Optimal Classification eq 34.
        self.nu0star = self.nu0 + self.n0
        self.nu1star = self.nu1 + self.n1

        sample0mean = self.data[self.mask0].mean()
        sample1mean = self.data[self.mask1].mean()
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

        # Expectation of C from page 3 eq. 1 using beta conjugate prior
        self.Ec = (self.n0 + self.alpha0) / (self.n + self.alpha0 + self.alpha1)
        self.analyticg = (self.fx0.logpdf(self.grid) \
                - self.fx1.logpdf(self.grid)).reshape(self.grid_n, -1) \
                + log(self.Ec) - log(1-self.Ec)

        self.analyticfx0 = self.fx0.logpdf(self.grid).reshape(self.grid_n, -1)
        self.analyticfx1 = self.fx1.logpdf(self.grid).reshape(self.grid_n, -1)

    def propose(self):
        self.oldmu0 = self.mu0.copy()
        self.oldmu1 = self.mu1.copy()
        self.oldsigma0 = self.sigma0.copy()
        self.oldsigma1 = self.sigma1.copy()
        self.oldc = self.c
        self.oldlastenergy = self.lastenergy

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
        self.lastenergy = self.oldlastenergy

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

        self.lastenergy = sum
        return sum

    def calc_eff_densities(self):
        fx0 = np.exp(logp_normal(self.grid, self.mu0, self.sigma0)).reshape(self.grid_n,self.grid_n)
        fx1 = np.exp(logp_normal(self.grid, self.mu1, self.sigma1)).reshape(self.grid_n,self.grid_n)
        return (fx0, fx1)

    def init_db(self, db, dbsize):
        dtype = [('thetas',np.double),
                ('energies',np.double),
                ('funcs',np.double)]
        if db == None:
            return np.zeros(dbsize, dtype=dtype)
        elif db.shape[0] != dbsize:
            return np.resize(db, dbsize)
        else:
            raise Exception("DB Not inited")

    def save_to_db(self, db, theta, energy, iteration):
        func = 0.0
        db[iteration] = np.array([theta, energy, func])
        global mydb
        mydb.append(self.copy())

        # Update G function average
        # TODO: Need to test if MCMC is weighted (SAMC) and if it is, 
        # then perform a weighted running average (is this possible with SAMC?)
        # Perhaps we'll have to do this offline... because the weights are not
        # fully known yet.
        self.numgavg += 1
        #self.gavg += (self.calc_eff_densities() - self.gavg) / self.numgavg
        fx0, fx1 = self.calc_eff_densities()
        self.fx0avg += (fx0 - self.fx0avg) / self.numgavg
        self.fx1avg += (fx1 - self.fx1avg) / self.numgavg

def calc_gavg(c,db):
    fx0, fx1 = c.calc_eff_densities()
    cmean = np.array([x[4] for x in db]).mean()
    return fx0*cmean / (fx1*(1-cmean))

def plotrun(c, db):
    # Plot the data
    p.figure()
    splot = p.subplot(2,2,1)
    p.title('Data')
    p.grid(True)
    n0 = c.n0
    n = c.n

    p.plot(c.data[c.mask0,0], c.data[c.mask0,1], 'r.', label='class 0')
    p.plot(c.data[c.mask1,0], c.data[c.mask1,1], 'g.', label='class 1')
    p.legend()
    plot_ellipse(splot, c.true['mu0'], c.true['sigma0'], 'red')
    plot_ellipse(splot, c.true['mu1'], c.true['sigma1'], 'green')

    # Now plot the MCMC run
    splot = p.subplot(2,2,2, sharex=splot, sharey=splot)
    p.grid(True)
    p.title('MCMC Mean Samples')

    plot_ellipse(splot, c.true['mu0'], c.true['sigma0'], 'red')
    plot_ellipse(splot, c.true['mu1'], c.true['sigma1'], 'green')
    p.plot([x[0][0] for x in db], [x[0][1] for x in db], 'r.', alpha=0.3)
    p.plot([x[1][0] for x in db], [x[1][1] for x in db], 'g.', alpha=0.3)

    means = calc_means(db)

    p.plot(means[0], means[1], 'ro', markersize=10)
    p.plot(means[2], means[3], 'go', markersize=10)

    ############
    cmeans = np.array([x[-1] for x in db]).mean()
    
    splot = p.subplot(2,2,3, sharex=splot, sharey=splot)
    gmin, gmax = c.analyticg.min(), c.analyticg.max()
    gavg = np.clip(calc_gavg(c,db), np.exp(gmin), np.exp(gmax))
    p.imshow(np.log(gavg), extent=c.gextent, origin='lower')
    p.colorbar()
    p.contour(np.log(gavg), [0.0], extent=c.gextent, origin='lower', cmap = p.cm.gray)

    p.plot(c.data[np.arange(n0),0], c.data[np.arange(n0),1], 'r.')
    p.plot(c.data[np.arange(n0,c.n),0], c.data[np.arange(n0,c.n),1], 'g.')
    plot_ellipse(splot, c.true['mu0'], c.true['sigma0'], 'red')
    plot_ellipse(splot, c.true['mu1'], c.true['sigma1'], 'green')

    ############
    splot = p.subplot(2,2,4, sharex=splot, sharey=splot)
    p.imshow(c.analyticg, extent=c.gextent, origin='lower')
    p.colorbar()
    p.contour(c.analyticg, [0.0], extent=c.gextent, origin='lower', cmap = p.cm.gray)

    p.plot(c.data[np.arange(n0),0], c.data[np.arange(n0),1], 'r.')
    p.plot(c.data[np.arange(n0,c.n),0], c.data[np.arange(n0,c.n),1], 'g.')

    plot_ellipse(splot, c.true['mu0'], c.true['sigma0'], 'red')
    plot_ellipse(splot, c.true['mu1'], c.true['sigma1'], 'green')

    #splot = p.subplot(2,2,3, sharex=splot, sharey=splot)
    #p.imshow(np.log(c.gavg), extent=c.gextent, origin='lower')
    #p.colorbar()

    #p.plot(c.data[np.arange(n0),0], c.data[np.arange(n0),1], 'r.')
    #p.plot(c.data[np.arange(n0,c.n),0], c.data[np.arange(n0,c.n),1], 'g.')
    #plot_ellipse(splot, c.true['mu0'], c.true['sigma0'], 'red')
    #plot_ellipse(splot, c.true['mu1'], c.true['sigma1'], 'green')

    #############
    #splot = p.subplot(2,2,4, sharex=splot, sharey=splot)
    #p.imshow(c.analyticfx0, extent=c.gextent, origin='lower')
    #p.colorbar()
    #p.plot(c.data[np.arange(n0),0], c.data[np.arange(n0),1], 'r.')
    #p.plot(c.data[np.arange(n0,c.n),0], c.data[np.arange(n0,c.n),1], 'g.')
    #plot_ellipse(splot, c.true['mu0'], c.true['sigma0'], 'red')
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
    #splot.set_xticks(())
    #splot.set_yticks(())

def calc_means(db):
    n = len(db)
    arr = np.empty((n,4))
    for i,rec in enumerate(db):
        arr[i,0] = rec[0][0]
        arr[i,1] = rec[0][1]
        arr[i,2] = rec[1][0]
        arr[i,3] = rec[1][1]

    return arr.mean(axis=0)

if __name__ == '__main__':
    #import samc
    #from samcnet.utils import *
    c = Classification()

    #print c.energy()
    #c.propose()
    #print c.energy()
    #c.reject()
    #print c.energy()
    #c.propose()
    #for i in range(50):
        #c.propose()
    #c.reject()
    #c.energy()

    #s = samc.SAMCRun(c, burn=0, stepscale=1000, refden=2)

    p.close('all')
    
    #p.imshow(c.calc_eff_densities(), origin='lower')
    #p.colorbar()
    
    s = MHRun(c, burn=0)
    s.sample(5e3)

    plotrun(c,mydb)

    p.figure()
    p.subplot(3,1,1)
    i0 = np.argmax(c.analyticfx0.sum(axis=1))
    i1 = np.argmax(c.analyticfx1.sum(axis=1))

    p.plot(c.analyticfx0[i0,:], 'r',label='analyticfx0')
    p.plot(np.log(c.fx0avg[i0,:]), 'g', label='avgfx0')
    p.xlabel('slice at index %d from bottom' % i0)
    p.legend()
    p.grid(True)
    p.subplot(3,1,2)
    p.plot(c.analyticfx1[i1,:], 'r',label='analyticfx1')
    p.plot(np.log(c.fx1avg[i1,:]), 'g', label='avgfx1')
    p.xlabel('slice at index %d from bottom' % i1)
    p.legend()
    p.grid(True)

    p.subplot(3,1,3)
    p.plot(log(c.Ec)+c.analyticfx0[(i1+i0)/2,:], 'r',label='analyticfx0')
    p.plot(log(1-c.Ec)+c.analyticfx1[(i0+i1)/2,:], 'g',label='analyticfx1')
    p.xlabel('slice at index %d from bottom' % i1)
    p.legend(loc='best')
    p.grid(True)

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

    p.show()

