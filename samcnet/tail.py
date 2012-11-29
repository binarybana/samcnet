# A simple example of calculating predictive posteriors in a normal unknown
# mean unknown variance case.

from __future__ import division
from lori import MHRun

import numpy as np
import pylab as p
import matplotlib as mpl
import random
from math import log, exp, pi, lgamma

import scipy.stats as st
import scipy
from scipy.special import betaln
import scipy.stats.distributions as di

from statsmodels.sandbox.distributions.mv_normal import MVT
#from sklearn.qda import QDA

import sys
sys.path.append('/home/bana/GSP/research/samc/code')
sys.path.append('/home/bana/GSP/research/samc/code/build')

mydb = []

class Classification():
    def __init__(self):
        np.random.seed(1234)

        self.n = 1 # Data points

        self.true_mu = 0.0
        self.true_sigma = 1 #di.invgamma.rvs(3)

        # For G function calculation and averaging
        self.grid_n = 100
        low,high = -4, 4
        self.gextent = (low,high)
        self.grid = np.linspace(low,high,self.grid_n)
        self.gavg = np.zeros(self.grid_n)
        self.numgavg = 0

        self.data = di.norm.rvs(size=self.n)
        
        ######## Starting point of MCMC Run #######
        self.mu = 0.0
        self.sigma = 2.0

        ###### Bookeeping ######
        self.oldmu = None
        self.oldsigma = None

    def propose(self):
        self.oldmu = self.mu
        self.oldsigma = self.sigma

        self.mu += np.random.randn()*0.1
        #self.mu = np.random.randn()
        self.sigma = di.invgamma.rvs(1)
        return 0

    def copy(self):
        return (self.mu, self.sigma, di.norm.rvs(loc=self.mu, scale=self.sigma))

    def reject(self):
        self.mu = self.oldmu
        self.sigma = self.oldsigma

    def energy(self):
        sum = 0.0
        sum -= di.norm.logpdf(self.data, loc=self.mu, scale=self.sigma).sum()

        #Now add in the priors...
        #sum -= di.invgamma.logpdf(self.sigma, self.priorgamma)
        sum -= log(self.sigma)*-2 #di.norm.logpdf(self.mu, loc=self.priormu, scale=self.priortau)
        return sum

    def calc_gfunc(self):
        return di.norm.pdf(self.grid, loc=self.mu, scale=self.sigma) 

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
        self.numgavg += 1
        self.gavg += (self.calc_gfunc() - self.gavg) / self.numgavg
        #mydb.append(self.calc_gfunc())

def pnorm(loc,scale):
    p.figure()
    x = np.linspace(-20, 20, 400)
    p.plot(x, di.norm.pdf(x,loc=loc, scale=scale))
    p.show()

if __name__ == '__main__':
    import samc
    from samcnet.utils import *
    c = Classification()

    p.close('all')
    s = MHRun(c, burn=0)
    #s = samc.SAMCRun(c, burn=0, stepscale=1000, refden=0)
    s.sample(5e4)
    plotHist(s)
    p.subplot(4,1,1)
    p.plot(c.grid, c.gavg, 'r')
    p.plot(c.data, np.ones_like(c.data), 'ko')
    p.grid(True)
    x = np.linspace(0.01,2,50)
    p.subplot(4,1,2)
    p.hist(np.vstack(mydb)[:,1],bins=x)

    p.subplot(4,1,3)
    mus = np.vstack(mydb)[:,0]
    counts,bins,_ = p.hist(mus,bins=80)

    xx = np.linspace(bins[0], bins[-1], 300)
    ty = di.t.pdf(xx, *di.t.fit(mus))
    ny = di.norm.pdf(xx, *di.norm.fit(mus))

    p.plot(xx,ty*counts.max()/ty.max(),'g', label='t fit')
    p.plot(xx,ny*counts.max()/ny.max(),'b--', label='normal fit')
    p.legend()

    p.subplot(4,1,4)
    ys = np.vstack(mydb)[:,2]
    counts,bins,_ = p.hist(ys,bins=80)

    xx = np.linspace(bins[0], bins[-1], 300)
    ty = di.t.pdf(xx, *di.t.fit(ys))
    ny = di.norm.pdf(xx, *di.norm.fit(ys))

    p.title("sampled y's")
    p.plot(xx,ty*counts.max()/ty.max(),'g', label='t fit')
    p.plot(xx,ny*counts.max()/ny.max(),'b--', label='normal fit')
    p.legend()

    p.figure()
    p.title('logpdfs')
    xx = np.linspace(bins[0], bins[-1], 300)
    ty = di.t.logpdf(xx, *di.t.fit(ys))
    ny = di.norm.logpdf(xx, *di.norm.fit(ys))

    p.title("sampled y's")
    p.plot(xx,ty,'g', label='t empirical')
    p.plot(xx,ny,'b--', label='normal empirical')

    p.plot(c.grid, np.log(c.gavg), 'r', label='gavg')

    p.plot(c.data, np.ones_like(c.data), 'ko')
    p.grid(True)
    p.legend()

    p.show()

