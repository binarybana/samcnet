from samcnet.mixturepoisson import *
import numpy as np
import pylab as p
import tables as t
import samcnet.samc as samc
import samcnet.mh as mh

from samcnet.lori import *
from math import exp,log

import scipy.stats as st
import scipy.stats.distributions as di
import scipy
import subprocess as sb
import os
import os.path as path
from scipy.special import gammaln
from time import time

p.close('all')

def log_poisson(k,lam):
    return log(lam) * k - gammaln(k+1) - lam

######## PARAMS ########
numlam = 10
kappa = 5
priorkappa = 80
S = np.eye(2) * kappa * 0.1
#S = np.array([[1,-.9],[-.9,1]]) * kappa
prior_mu = np.zeros(2) + 0 
prior_sigma = np.zeros(2) + 10
######## /PARAMS ########

######## Generate Data ########
def gen_data(mu, cov, n):
    lams = MVNormal(mu, cov).rvs(n)
    ps = np.empty_like(lams)
    for i in xrange(lams.shape[0]):
	for j in xrange(lams.shape[1]):
	    ps[i,j] = di.poisson.rvs(10 * np.exp(lams[i,j]))
    return ps

rho = -0.0
cov = np.array([[1, rho],[rho, 1]]) * 0.01
mu1 = np.array([log(2), log(2)])
mu2 = np.array([log(2), log(8)])

rseed = np.random.randint(1000)
#rseed = 875
dseed = 36
#dseed = np.random.randint(1000)

print("rseed: %d" % rseed)
print("dseed: %d" % dseed)
np.random.seed(dseed)
ps = np.vstack(( gen_data(mu1,cov,30), gen_data(mu2,cov,30) ))
superps = np.vstack(( gen_data(mu1,cov,1000), gen_data(mu2,cov,1000) ))
np.random.seed(rseed)

n,gext,grid = get_grid_data(ps, positive=True)
#p.plot(superps[:,0], superps[:,1], 'k.', alpha=0.1)
#p.show()
#sys.exit()
######## /Generate Data ########

######## MH Samples ########
startmu = np.array([[log(8),log(8)],[log(2),log(2)],[log(2),log(2)]]).T
#startmu = np.array([[log(3),log(3)],[log(2),log(2)],[log(2),log(2)]]).T
#startmu = np.array([[log(3),log(3)],[log(3),log(3)]]).T
#startmu = np.array([[log(3),log(3)]]).T
dist = MPMDist(ps,kappa=kappa,S=S,priormu=prior_mu,priorsigma=prior_sigma,
	priorkappa=priorkappa,kmax=3,d=np.ones(60)*10, mumove=0.1, lammove=0.01,
	startk=2,startmu=startmu,wmove=0.03,birthmove=0.5)
mh = mh.MHRun(dist, burn=0, thin=50)
iters = 1e3
t1=time()
mh.sample(iters,verbose=False)
print "%d MH iters took %f seconds" % (iters, time()-t1)

t1=time()
gavg = dist.calc_db_g(mh.db, mh.db.root.object, grid, numlam=numlam).reshape(-1,n)
print "Generating gavg using numlam %d took %f seconds" % (numlam, time()-t1)
#gavg = dist.calc_curr_g(grid, numlam=3).reshape(-1,n)

p.subplot(2,1,1)
p.imshow(gavg, extent=gext, aspect=1, origin='lower')
p.colorbar()
p.plot(ps[:,0], ps[:,1], 'k.')

p.subplot(2,1,2)
p.imshow(gavg, extent=gext, aspect=1, origin='lower')
p.colorbar()
p.plot(superps[:,0], superps[:,1], 'k.', alpha=0.1)

dist.plot_traces(mh.db, mh.db.root.object, names=('w','k','mu'))
#dist.plot_traces(mh.db, mh.db.root.object, names=('sigma','mu','lam','k','w'))

p.show()
