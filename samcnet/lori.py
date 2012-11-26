from __future__ import division

import numpy as np
import pylab as p
import matplotlib as mpl
import random
from math import log, exp, pi, lgamma

import scipy.stats as st
import scipy

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
        np.random.seed(1)
        self.D = 2 # Dimension
        self.N = 30 # Data points
        self.DOF = 3 # For random ground truth COV mats
        c = 0.8 # Ground truth class marginal

        temp0 = sample_invwishart(np.eye(self.D), self.DOF)
        temp1 = sample_invwishart(np.eye(self.D), self.DOF)

        sigma0 = np.dot(temp0.T,temp0)
        sigma1 = np.dot(temp1.T,temp1)

        mu0 = np.zeros(self.D)
        mu1 = np.ones(self.D)

        # For G function calculation and averaging
        self.grid_n = 20
        lx,hx,ly,hy = (-4,4,-4,4)
        self.gextent = (lx,hx,ly,hy)
        self.grid = np.dstack(np.meshgrid(
                        np.linspace(lx,hx,self.grid_n),
                        np.linspace(ly,hy,self.grid_n))).reshape(-1,2)
        self.gavg = np.zeros((self.grid_n, self.grid_n))
        self.numgavg = 0



        count = st.binom.rvs(self.N, c)
        self.data = np.vstack(( \
            np.random.multivariate_normal(mu0, sigma0, count),
            np.random.multivariate_normal(mu1, sigma1, self.N-count) ))

        self.mask = np.hstack((
            np.zeros(count, dtype=np.bool),
            np.ones(self.N-count, dtype=np.bool)))

        ##### Record true values for plotting, comparison #######
        self.true = {'c':c, 
                'count' : count,
                'mu0': mu0, 
                'sigma0': sigma0, 
                'mu1': mu1, 
                'sigma1': sigma1}

        ##### Proposal variances ######
        self.propdof = 50
        self.propmu = 0.3

        ##### Prior Values and Confidences ######
        self.priorsigma0 = np.eye(self.D)
        self.priorsigma1 = np.eye(self.D)
        self.priorDOF0 = 10
        self.priorDOF1 = 10

        self.priormu0 = np.zeros(self.D)
        self.priormu1 = np.ones(self.D)

        self.nu0 = 1.0
        self.nu1 = 1.0

        ######## Starting point of MCMC Run #######
        # 'Cheat' by starting at the right spot... for now
        self.c = c
        self.mu0 = mu0.copy()
        self.mu1 = mu1.copy()
        self.sigma0 = sigma0.copy()
        self.sigma1 = sigma1.copy()
        #self.c = np.random.rand()
        #self.mu0 = np.random.rand(self.D)
        #self.mu1 = np.random.rand(self.D)
        #self.sigma0 = sample_invwishart(self.priorsigma0, self.priorDOF0)
        #self.sigma1 = sample_invwishart(self.priorsigma1, self.priorDOF1)

        
        ###### Bookeeping ######
        self.oldmu0 = None
        self.oldmu1 = None
        self.oldsigma0 = None
        self.oldsigma1 = None
        self.oldc = None

    def calc_analytic(self):
        self.nu0star = None


    def propose(self):
        self.oldmu0 = self.mu0.copy()
        self.oldmu1 = self.mu1.copy()
        self.oldsigma0 = self.sigma0.copy()
        self.oldsigma1 = self.sigma1.copy()
        self.oldc = self.c

        self.mu0 += (np.random.rand(self.D)-0.5)*self.propmu
        self.mu1 += (np.random.rand(self.D)-0.5)*self.propmu

        self.sigma0 = sample_invwishart(self.sigma0*self.propdof, self.propdof)
        self.sigma1 = sample_invwishart(self.sigma1*self.propdof, self.propdof)
        #self.sigma0 = np.dot(self.sigma0, sample_invwishart(np.eye(self.D)*100, 100))
        #self.sigma1 = np.dot(self.sigma1, sample_invwishart(np.eye(self.D)*100, 100))
        # TODO FIXME  temporarily commented these out
        
        self.c = np.random.rand()

        return 0

    def copy(self):
        return (self.mu0.copy(), self.mu1.copy(), self.sigma0.copy(), self.sigma1.copy())

    def reject(self):
        self.mu0 = self.oldmu0.copy()
        self.mu1 = self.oldmu1.copy()
        self.sigma0 = self.oldsigma0.copy()
        self.sigma1 = self.oldsigma1.copy()
        self.c = self.oldc

    def energy(self):
        # First calculate the labels from the Bayes classifier
        # this comes from page 21-22 of Lori's Optimal Bayes Classifier 
        # Part 1 Paper (eq 56).
        #s0i = np.linalg.inv(self.sigma0)
        #s1i = np.linalg.inv(self.sigma1)
        #A = -0.5 * (s1i - s0i)
        #a = np.dot(s1i,self.mu1) - np.dot(s0i,self.mu0)
        #b = -0.5 * np.dot(np.dot(self.mu1.T,s1i),self.mu1) \
            #- np.dot(np.dot(self.mu0.T,s0i),self.mu0) \
            #+ np.log((1-self.c)/self.c \
            #* (np.linalg.det(self.sigma0)/np.linalg.det(self.sigma1))**0.5)

        #g = (np.dot(self.data, A) * self.data).sum(axis=1) \
            #+ np.dot(self.data, a) + b

        #clf = QDA()
        #clf.covariances_ = []
        #clf.set_params(means_=[self.mu0, self.mu1],
                #priors_=[self.c, 1-self.c],
                #covariances_=[self.sigma0, self.sigma1])
        #skpreds = clf.predict(self.data)
        #skscores = clf.predict_log_proba(self.data)

        sum = 0.0
        #class 0 negative log likelihood
        points = self.data[np.logical_not(self.mask)]
        if points.size > 0:
            sum -= logp_normal(points, self.mu0, self.sigma0).sum()

        #class 1 negative log likelihood
        points = self.data[self.mask]
        if points.size > 0:
            sum -= logp_normal(points, self.mu1, self.sigma1).sum()
                
        #Now add in the priors...
        # we'll assume uniform for class conditional (so we can ignore the
        # constant)
        sum -= logp_invwishart(self.sigma0,self.priorDOF0,self.priorsigma0)
        sum -= logp_invwishart(self.sigma1,self.priorDOF1,self.priorsigma1)
        sum -= logp_normal(self.mu0, self.priormu0, self.sigma0, self.nu0).sum()
        sum -= logp_normal(self.mu1, self.priormu1, self.sigma1, self.nu1).sum()
        return sum

    def calc_gfunc(self):
        gridenergies = logp_normal(self.grid, self.mu0, self.sigma0) * (self.c)
        gridenergies -= logp_normal(self.grid, self.mu1, self.sigma1) * (1-self.c)
        return gridenergies.reshape(self.grid_n,self.grid_n)

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

def plotrun(c, db):
    # Plot the data
    splot = p.subplot(2,2,1)
    p.title('Data')
    p.grid(True)
    count = c.true['count']
    n = c.N

    p.plot(c.data[np.arange(count),0], c.data[np.arange(count),1], 'r.')
    p.plot(c.data[np.arange(count,c.N),0], c.data[np.arange(count,c.N),1], 'g.')
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

    splot = p.subplot(2,2,3, sharex=splot, sharey=splot)
    p.imshow(c.gavg, extent=c.gextent, origin='lower')
    p.colorbar()

    p.contour(c.gavg, [0.0], extent=c.gextent, origin='lower', cmap = p.cm.gray)

    p.plot(c.data[np.arange(count),0], c.data[np.arange(count),1], 'r.')
    p.plot(c.data[np.arange(count,c.N),0], c.data[np.arange(count,c.N),1], 'g.')
    plot_ellipse(splot, c.true['mu0'], c.true['sigma0'], 'red')
    plot_ellipse(splot, c.true['mu1'], c.true['sigma1'], 'green')

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
    import samc
    from samcnet.utils import *
    c = Classification()

    print c.energy()
    c.propose()
    print c.energy()
    c.reject()
    #print c.energy()
    #c.propose()
    #for i in range(50):
        #c.propose()
    #c.reject()
    #c.energy()

    #s = samc.SAMCRun(c, burn=0, stepscale=1000, refden=2)

    p.close('all')
    
    #p.imshow(c.calc_gfunc(), origin='lower')
    #p.colorbar()
    
    s = MHRun(c, burn=0)
    s.sample(2e3)
    plotrun(c,mydb)
    p.show()

    #clf = QDA()
    #print clf.fit(c.data, np.hstack((np.zeros(c.true['count']), np.ones(c.N-c.true['count']))))
    #print clf.predict_log_proba(c.data)

