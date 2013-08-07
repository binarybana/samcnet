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
priorkappa = 20
S = np.eye(2) * kappa
#S = np.array([[1,-.9],[-.9,1]]) * kappa
prior_mu = np.zeros(2) + 0 
prior_sigma = np.ones(2) + 3
######## /PARAMS ########

######## Generate Data ########
def gen_data(mu, cov, n):
    lams = MVNormal(mu, cov).rvs(n)
    ps = np.empty_like(lams)
    for i in xrange(lams.shape[0]):
	for j in xrange(lams.shape[1]):
	    ps[i,j] = di.poisson.rvs(10* np.exp(lams[i,j]))
    return ps

rho = 0.0
cov = np.array([[1, rho],[rho, 1]])
mu = np.zeros(2)

rseed = np.random.randint(1000)
dseed = 1000
#dseed = np.random.randint(1000)

print("rseed: %d" % rseed)
print("dseed: %d" % dseed)
np.random.seed(dseed)
ps = gen_data(mu,cov,30)
superps = gen_data(mu,cov,3000)
np.random.seed(rseed)

n,gext,grid = get_grid_data(ps, positive=True)
######## /Generate Data ########

######## MH Samples ########
dist = MPMDist(ps,kappa=kappa,S=S,priormu=prior_mu,priorsigma=prior_sigma,
	priorkappa=priorkappa,kmax=1)
mh = mh.MHRun(dist, burn=100, thin=4)
iters = 1e3
t1=time()
mh.sample(iters,verbose=False)
print "%d MH iters took %f seconds" % (iters, time()-t1)

gavg,plams = dist.calc_db_g(mh.db, mh.db.root.object, grid)
#gavg,plams = dist.calc_curr_g(grid, numlam=3)
gavg = gavg.reshape(-1,n)

p.subplot(2,1,1)
p.imshow(gavg, extent=gext, aspect=1, origin='lower')
p.colorbar()
p.plot(ps[:,0], ps[:,1], 'k.')

p.subplot(2,1,2)
p.imshow(gavg, extent=gext, aspect=1, origin='lower')
p.colorbar()
p.plot(superps[:,0], superps[:,1], 'k.', alpha=0.1)

dist.plot_traces(mh.db, mh.db.root.object, names=('sigma','mu','lam'))
######## /MH Samples ########

######## Lambda subsampled MH Samples ########
#p.subplot(2,2,2)
#mus = mh.db.root.object.mu.read()
#sigmas = mh.db.root.object.sigma.read()
#d = mh.db.root.object.d.read()
#numsamps = mus.shape[0]
#mapd = 10
#avgavg = np.zeros(grid.shape[0])
#mhlams = np.vstack(( MVNormal(mus[i,:,0], sigmas[i]).rvs(numlam) for i in xrange(numsamps) ))
#for j in xrange(numsamps * numlam):
    #avgavg[:] += np.exp(log_poisson(grid[:,0], d[j]*np.exp(mhlams[j,0])) \
				#+ log_poisson(grid[:,1], d[j]*np.exp(mhlams[j,1])))
#avgavg /= numlam*numsamps
#avgavg = np.log(avgavg).reshape(-1,n)

#p.imshow(avgavg, extent=gext, aspect=1, origin='lower')
#p.plot(ps[:,0], ps[:,1], 'k.')
######## /Lambda subsampled MH Samples ########

######## MAP sample ########
#mapd, maplam = mh.mapvalue[3], mh.mapvalue[5]
#mlg = (log_poisson(grid[:,0], mapd*np.exp(maplam[0])) \
	#+ log_poisson(grid[:,1], mapd*np.exp(maplam[1]))).reshape(-1,n)
#p.imshow(mlg, extent=gext, aspect=1, origin='lower')
#p.plot(ps[:,0], ps[:,1], 'k.')
######## /MAP sample ########

######## Lambda subsampled map sample ########
#p.subplot(2,2,3)
#mapmu, mapsigma = mh.mapvalue[:2]

#lamdist = MVNormal(mapmu[:,0], mapsigma)
#maplams = lamdist.rvs(numlam)
#avglg = np.zeros(grid.shape[0])
#print(mapsigma)
#for i in xrange(numlam):
    #avglg[:] += np.exp(log_poisson(grid[:,0], mapd*np.exp(maplams[i,0])) \
		#+ log_poisson(grid[:,1], mapd*np.exp(maplams[i,1])))
#avglg /= numlam
#avglg = np.log(avglg).reshape(-1,n)

#p.imshow(avglg, extent=gext, aspect=1, origin='lower')
##p.colorbar()
#p.plot(ps[:,0], ps[:,1], 'k.')
######## Lambda subsampled map sample ########

############ STAN ###############
#ps = ps[:0]
#from samcnet.utils import stan_vec
#open('../stan/data.R','w').write(stan_vec(
    #samples=ps.astype(np.int), 
    #num_samples=ps.shape[0], 
    #num_features=2,
    #kappa = kappa,
    #S = S,
    #prior_mu = prior_mu,
    #prior_sigma = prior_sigma,
    #prior_z_sigma = 3,
    #prior_A_sigma = 3,
    #lowd = 8,
    #highd = 12))
#os.chdir('../stan')
#sb.check_call( 
    #'./mpm --data=data.R --iter=900 --warmup=200 --refresh=100 --samples=samples2.csv'.split())
#os.chdir('../samcnet')
#stan_samps = np.genfromtxt('../stan/samples2.csv', names=True, skip_header=25, delimiter=',')
#numsamps = stan_samps.shape[0]
#stan_lams = np.dstack(( np.vstack(( 
    #stan_samps['lam%d1'%(i+1)], stan_samps['lam%d2'%(i+1)] )) 
    #for i in xrange(ps.shape[0]) ))
#stan_rates = np.dstack(( np.vstack(( 
    #stan_samps['rate%d1'%(i+1)], stan_samps['rate%d2'%(i+1)] )) 
    #for i in xrange(ps.shape[0]) ))
#stan_zs = stan_samps['z']

#avglg = np.zeros(grid.shape[0])
#for i in xrange(numsamps):
    #for j in xrange(ps.shape[0]):
	#avglg[:] += np.exp(log_poisson(grid[:,0], stan_rates[0,i,j]) \
		#+ log_poisson(grid[:,1], stan_rates[1,i,j]))
#avglg /= numsamps
#avglg = np.log(avglg).reshape(-1,n)

#stanmus = np.vstack(( stan_samps['mu1'], stan_samps['mu2'] ))
#stansigmas = [np.array([[stan_samps['sigma11'][i], stan_samps['sigma12'][i]],
    #[stan_samps['sigma21'][i], stan_samps['sigma22'][i]]]) for i in xrange(numsamps)]

#p.subplot(2,2,4)
#p.imshow(avglg, extent=gext, aspect=1, origin='lower')
#p.plot(ps[:,0], ps[:,1], 'k.')

############ /STAN ###########

#p.figure()
#for j in xrange(ps.shape[0]):
    #p.plot(stan_lams[0,:,j], stan_lams[1,:,j], 'r.', alpha=0.1, label='stansubsamps')
#p.plot(mhlams[:,0], mhlams[:,1], 'g.', alpha=0.5, label='mhsubsamps')
#mhlamsamps = mh.db.root.object.lam.read()
#p.plot(mhlamsamps[:,0], mhlamsamps[:,1], 'm.', alpha=0.5, label='mhsamps')
#stanlamsamps = np.vstack(( stan_samps['lam1'], stan_samps['lam2'] ))
#p.plot(stanlamsamps[:,0], stanlamsamps[:,1], 'k.', alpha=0.9, label='stansamps')

#p.legend(fontsize=8, loc='best')

#p.figure()
#p.plot(np.exp(stanlams[:,0]), np.exp(stanlams[:,1]), 'r.', alpha=0.5, label='stan')
#p.plot(np.exp(mhlams[:,0]), np.exp(mhlams[:,1]), 'g.', alpha=0.5, label='mh')
#p.legend()

######## 3D Plot ########
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter

#fig = p.figure()
#ax = fig.gca(projection='3d')
#X = np.linspace(gext[0], gext[1], n)
#Y = np.linspace(gext[0], gext[1], n)
#X, Y = np.meshgrid(X, Y)
#surf = ax.plot_surface(X, Y, avglg, rstride=1, cstride=1, cmap=cm.coolwarm,
	#linewidth=0, antialiased=False)
#fig = p.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(X, Y, avgavg, rstride=1, cstride=1, cmap=cm.coolwarm,
        #linewidth=0, antialiased=False)
######## /3D Plot ########

p.show()
