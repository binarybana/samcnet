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

        self.n = 4 # Data points

        self.true_mu = 0.0
        self.true_sigma = 1 #di.invgamma.rvs(3)

        # For G function calculation and averaging
        self.grid_n = 100
        low,high = -4, 4
        self.gextent = (low,high)
        self.grid = np.linspace(low,high,self.grid_n)
        self.gavg = np.zeros(self.grid_n)
        self.numgavg = 0

        #self.data = di.norm.rvs(size=self.n)
        self.data = np.array([0.0, -0.0, 0.5, -0.5])
        assert self.data.size == self.n
        
        ######## Starting point of MCMC Run #######
        self.mu = 0.0
        self.sigma = 2.0

        ###### Bookeeping ######
        self.oldmu = None
        self.oldsigma = None

        ##### Prior Values and Confidences ######
        self.priorsigma = 2
        self.kappa = 1
        self.priormu = 0
        self.nu = 8.0
        #### Calculating the Analytic solution given on page 15 of Lori's 
        #### Optimal Classification eq 34.
        self.nustar = self.nu + self.n

        samplemean = self.data.mean()
        samplevar = np.cov(self.data)

        self.mustar = (self.nu*self.priormu + self.n * samplemean) \
                / (self.nu + self.n)
        self.kappastar = self.kappa + self.n
        self.Sstar = self.priorsigma + (self.n-1)*samplevar + self.nu*self.n/(self.nu+self.nu)\
                * (samplemean - self.priormu)**2
                
        #### Now calculate effective class conditional densities from eq 55
        #### page 21

        #self.fx = MVT(
                #self.mu0star, 
                #(self.nu0star+1)/(self.kappa0star-self.D+1)/self.nu0star * self.S0star, 
                #self.kappa0star - self.D + 1)
        # So I'm pretty sure this is incorrect below, off by some scaling
        # parameters
        self.fx = MVT(
                [self.mustar], 
                [(self.nustar+1)/(self.kappastar)/self.nustar * self.Sstar / 2],
                self.kappastar /2 )

        self.analyticfx = self.fx.logpdf(self.grid.reshape(-1,1))


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
        sum -= log(self.sigma)*(-0.5) - self.nu/2 * (self.mu-self.priormu)**2/self.sigma
        sum -= log(self.sigma)*(self.kappa+2)/(-2) - 0.5*self.priorsigma/self.sigma
        return sum

    def calc_gfunc(self):
        return di.norm.pdf(self.grid, loc=self.mu, scale=self.sigma) 

    def init_db(self, db, dbsize):
        pass
        #dtype = [('thetas',np.double),
                #('energies',np.double),
                #('funcs',np.double)]
        #if db == None:
            #return np.zeros(dbsize, dtype=dtype)
        #elif db.shape[0] != dbsize:
            #return np.resize(db, dbsize)
        #else:
            #raise Exception("DB Not inited")

    def save_to_db(self, db, theta, energy, iteration):
        #func = 0.0
        #db[iteration] = np.array([theta, energy, func])
        global mydb
        mydb.append(self.copy())

        # Update G function average
        self.numgavg += 1
        self.gavg += (self.calc_gfunc() - self.gavg) / self.numgavg

def pnorm(loc,scale):
    p.figure()
    x = np.linspace(-20, 20, 400)
    p.plot(x, di.norm.pdf(x,loc=loc, scale=scale))
    p.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    import samc
    from samcnet.utils import *
    c = Classification()

    #p.close('all')
    s = MHRun(c, burn=0)
    #s = samc.SAMCRun(c, burn=0, stepscale=1000, refden=0)
    s.sample(1e3)
    #plotHist(s)

    ##################################
    ##################################
    #p.subplot(4,1,1)
    #p.plot(c.grid, c.gavg, 'r')
    #p.plot(c.data, np.ones_like(c.data), 'ko')
    #p.grid(True)
    #x = np.linspace(0.01,2,50)
    #p.subplot(4,1,2)
    #p.hist(np.vstack(mydb)[:,1],bins=x)

    #p.subplot(4,1,3)
    #mus = np.vstack(mydb)[:,0]
    #counts,bins,_ = p.hist(mus,bins=80)

    #xx = np.linspace(bins[0], bins[-1], 300)
    #ty = di.t.pdf(xx, *di.t.fit(mus))
    #ny = di.norm.pdf(xx, *di.norm.fit(mus))

    #p.plot(xx,ty*counts.max()/ty.max(),'g', label='t fit')
    #p.plot(xx,ny*counts.max()/ny.max(),'b--', label='normal fit')
    #p.legend()

    #p.subplot(4,1,4)
    #ys = np.vstack(mydb)[:,2]
    #counts,bins,_ = p.hist(ys,bins=80)

    #xx = np.linspace(bins[0], bins[-1], 300)
    #ty = di.t.pdf(xx, *di.t.fit(ys))
    #ny = di.norm.pdf(xx, *di.norm.fit(ys))
    #ay = c.fx.pdf(xx.reshape(-1,1))

    #p.title("sampled y's")
    #p.plot(xx,ty*counts.max()/ty.max(),'g', label='t fit')
    #p.plot(xx,ny*counts.max()/ny.max(),'b--', label='normal fit')
    
    #p.plot(xx,ay*counts.max()/ay.max(),'k--', label='t analytic')

    #p.legend()

    ##############################
    fig1 = plt.figure()
    #xx = np.linspace(bins[0], bins[-1], 300)
    #ty = di.t.logpdf(xx, *di.t.fit(ys))
    #p.plot(xx,ty,'g', label='t empirical')

    plt.title("predictive posteriors")
    plt.ylabel('logpdfs')
    plt.grid(True)
    plt.hold(True)

    plt.plot(c.data, np.ones_like(c.data), 'ko', label='data')
    plt.plot(c.grid, np.exp(c.analyticfx), 'k--', label='student t')
    if True:
        s.sample(3e3)
        plt.plot(c.grid, c.gavg, 'r', label='gavg')
        ys = np.vstack(mydb)[:,2]
        counts,bins,_ = p.hist(ys,bins=80,normed=True)

    elif False: # Animation
        l, = plt.plot(c.grid, c.gavg, 'r', label='gavg')
        def update_line(num, data, line):
            global c
            line.set_data(c.grid,data[num])
            return line,
        data = [c.gavg.copy()]
        N=int(sys.argv[1])
        for x in range(N):
            s.sample(1e3)
            data.append(c.gavg.copy())

        line_ani = animation.FuncAnimation(fig1, update_line, N, fargs=(data, l),
            interval=50, blit=True)

    plt.show()


