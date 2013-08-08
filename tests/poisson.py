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
#import nlopt
import subprocess as sb
import os
import os.path as path

from sklearn.lda import LDA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN

np.seterr(all='ignore') # Careful with this

seedr = np.random.randint(10**6)
seedd = 40767
#seed = 32
print "Seed is %d" % seedr
#np.random.seed(seedd)
np.random.seed(seedr)

def myplot(ax,g,data0,data1):
    ax.plot(data0[:,0], data0[:,1], 'g.',label='0', alpha=0.3)
    ax.plot(data1[:,0], data1[:,1], 'r.',label='1', alpha=0.3)
    ax.legend(fontsize=8, loc='best')

    im = ax.imshow(g, extent=gext, aspect='equal', origin='lower')
    p.colorbar(im,ax=ax)
    ax.contour(g, [0.0], extent=gext, aspect=1, origin='lower', cmap = p.cm.gray)

########## Profiling #############
#N = 20
#data0 = np.vstack((
    #np.hstack(( di.poisson.rvs(10*exp(1), size=(N,1)), di.poisson.rvs(10*exp(2), size=(N,1)) )),
    #np.hstack(( di.poisson.rvs(10*exp(2.2), size=(N,1)), di.poisson.rvs(10*exp(2), size=(N,1)) )) ))
#data1 = np.hstack(( di.poisson.rvs(10*exp(2), size=(N,1)), di.poisson.rvs(10*exp(1), size=(N,1)) ))
#dist0 = MPMDist(data0)
#dist1 = MPMDist(data1)
#mpm = MPMCls(dist0, dist1) # TODO allow params input here (or maybe per class)
#s2 = samc.SAMCRun(mpm, burn=0, stepscale=100, refden=1, thin=10, lim_iters=200)
#s2.sample(1e4, temperature=20)
#import pstats, cProfile
#cProfile.runctx("samc.SAMCRun(mpm, burn=0, stepscale=1000, thin=10)", globals(), locals(), "prof.prof")
#cProfile.runctx("[mpm.energy() for i in xrange(10000)]", globals(), locals(), "prof.prof")
#cProfile.runctx("mpm.propose()", globals(), locals(), "prof.prof")

#s = pstats.Stats("prof.prof")
#s.strip_dirs().sort_stats("time").print_stats()
#s.strip_dirs().sort_stats("cumtime").print_stats()

#sys.exit()
########## /Profiling #############

######## Generate Data ########
def gen_data(mu, cov, n):
    lams = MVNormal(mu, cov).rvs(n)
    ps = np.empty_like(lams)
    for i in xrange(lams.shape[0]):
	for j in xrange(lams.shape[1]):
	    ps[i,j] = di.poisson.rvs(10* np.exp(lams[i,j]))
    return ps

def write_data(data0, data1, loc):
    assert data0.shape[0] == data1.shape[0]
    data = np.vstack(( data0, data1 ))
    np.savetxt(loc, data.astype(np.int), header="%d %d %d" % (data.shape+(2,)),
	    delimiter=',', comments='', fmt='%d')

mu0 = np.zeros(2) #- 0.5
mu1 = np.zeros(2) #+ 0.5
rho0 = -0.4
rho1 = 0.4
cov0 = np.array([[1, rho0],[rho0, 1]])
cov1 = np.array([[1, rho1],[rho1, 1]])

rseed = np.random.randint(10000)
dseed = 10000
#dseed = np.random.randint(1000)

print("rseed: %d" % rseed)
print("dseed: %d" % dseed)
np.random.seed(dseed)
ps0 = gen_data(mu0,cov0,30)
ps1 = gen_data(mu1,cov1,30)
superps0 = gen_data(mu0,cov0,3000)
superps1 = gen_data(mu1,cov1,3000)
np.random.seed(rseed)
ps = np.vstack(( ps0, ps1 ))
superps = np.vstack(( superps0, superps1 ))

n,gext,grid = get_grid_data(ps, positive=True)
######## /Generate Data ########

########## Comparison #############
p.close('all')
# Run Yousef/Jianping RNA Synthetic
currdir = path.abspath('.')
synloc = path.expanduser('~/GSP/research/samc/synthetic/rnaseq')

write_data(ps0, ps1, path.join(synloc, 'out', 'trn.txt'))
write_data(superps0, superps1, path.join(synloc, 'out', 'tst.txt'))

try:
    os.chdir(synloc)
    #sb.check_call(path.join(synloc, 
	#'gen -i params/easyparams -sr 0.05 -lr 9 -hr 10').split())
    sb.check_call(path.join(synloc, 
	'cls -t out/trn.txt -s out/tst.txt').split())
finally:
    os.chdir(currdir)
# Grab some info from the run
data = np.loadtxt(path.join(synloc,'out','out'))
lda,knn,svm,num_feats = data[0:4]
print("LDA error: %f" % lda)
print("KNN error: %f" % knn)
print("SVM error: %f" % svm)
feat_inds = data[4:].astype(int)

rawdata = np.loadtxt(path.join(synloc, 'out','trn.txt'),
	delimiter=',', skiprows=1)
data = rawdata[:,feat_inds]
Ntrn = data.shape[0]
data0 = data[:Ntrn/2,:]
data1 = data[Ntrn/2:,:]
norm_data = (data - data.mean(axis=0)) / np.sqrt(data.var(axis=0,ddof=1))
norm_data0 = norm_data[:Ntrn/2,:]
norm_data1 = norm_data[Ntrn/2:,:]
rawtest = np.loadtxt(path.join(synloc, 'out','tst.txt'),
	delimiter=',', skiprows=1)
test = rawtest[:,feat_inds]
norm_test = (test - test.mean(axis=0)) / np.sqrt(test.var(axis=0,ddof=1))
N = test.shape[0]
D = data.shape[1]
#sys.exit()

trn_labels = np.hstack(( np.zeros(Ntrn/2), np.ones(Ntrn/2) ))
tst_labels = np.hstack(( np.zeros(N/2), np.ones(N/2) ))
sklda = LDA()
skknn = KNN(3, warn_on_equidistant=False)
sksvm = SVC()
sklda.fit(norm_data, trn_labels)
skknn.fit(norm_data, trn_labels)
sksvm.fit(norm_data, trn_labels)
print("skLDA error: %f" % (1-sklda.score(norm_test, tst_labels)))
print("skKNN error: %f" % (1-skknn.score(norm_test, tst_labels)))
print("skSVM error: %f" % (1-sksvm.score(norm_test, tst_labels)))

labels = np.hstack((np.zeros(N/2), np.ones(N/2)))
n,gext,grid = get_grid_data(np.vstack(( norm_data0, norm_data1 )))

bayes0 = GaussianBayes(np.zeros(D), 1, 8, np.eye(D)*3, norm_data0)
bayes1 = GaussianBayes(np.zeros(D), 1, 8, np.eye(D)*3, norm_data1)

# Gaussian Analytic
gc = GaussianCls(bayes0, bayes1)
print("Gaussian Analytic error: %f" % gc.approx_error_data(norm_test, labels))
gavg = gc.calc_gavg(grid).reshape(-1,n)
myplot(p.subplot(2,3,1),gavg,norm_data0, norm_data1)

 #Gaussian Sampler
#c = GaussianSampler(bayes0,bayes1,norm_data0,norm_data1)
#s1 = samc.SAMCRun(c, burn=0, stepscale=1000, refden=1, thin=10, lim_iters=200)
#s1.sample(1e3, temperature=1)
#print("Gaussian Sampler error: %f" % c.approx_error_data(s1.db, norm_test, labels))
#gavg = c.calc_gavg(s1.db, grid, 50).reshape(-1,n)
#myplot(p.subplot(2,3,2),gavg)

# MPM Model
n,gext,grid = get_grid_data(np.vstack(( data0, data1 )), positive=True)

dist0 = MPMDist(data0,kmax=1)
dist1 = MPMDist(data1,kmax=1)
mpm = MPMCls(dist0, dist1) 
#s2 = samc.SAMCRun(mpm, burn=0, stepscale=1000, refden=1, thin=10, 
	#lim_iters=100, low_margin=0.2, high_margin=-0.5)
#s2.sample(2e5, temperature=2)
mh = mh.MHRun(mpm, burn=100, thin=20)
mh.sample(5e3,verbose=False)
print("MPM Sampler error: %f" % mpm.approx_error_data(mh.db, test, labels))

numlam = 200

gavg = mpm.calc_gavg(mh.db, grid, numlam=numlam).reshape(-1,n)
#g = mpm.calc_curr_g(grid).reshape(-1,n)
ga1 = mpm.dist0.calc_db_g(mh.db, mh.db.root.object.dist0, grid, numlam=numlam).reshape(-1,n)
ga2 = mpm.dist1.calc_db_g(mh.db, mh.db.root.object.dist1, grid, numlam=numlam).reshape(-1,n)

myplot(p.subplot(2,3,2),gavg,data0,data1)
myplot(p.subplot(2,3,3),ga1,data0,data1)
myplot(p.subplot(2,3,4),ga2,data0,data1)
myplot(p.subplot(2,3,5),gavg,test[:500,:],test[500:,:])
p.subplot(2,3,6)
p.plot(test[500:,0], test[500:,1],'m.',alpha=0.5)
p.plot(test[:500,0], test[:500,1],'g.',alpha=0.5)

p.show()
sys.exit()
########## /Comparison #############

########## SAMC #############
#p.close('all')
#N = 20
#data0 = np.vstack((
    #np.hstack(( di.poisson.rvs(10*exp(1), size=(N,1)), di.poisson.rvs(10*exp(2), size=(N,1)) )),
    #np.hstack(( di.poisson.rvs(10*exp(2.2), size=(N,1)), di.poisson.rvs(10*exp(2), size=(N,1)) )) ))
#data1 = np.hstack(( di.poisson.rvs(10*exp(2), size=(N,1)), di.poisson.rvs(10*exp(1), size=(N,1)) ))

#dist0 = MPMDist(data0)
#dist1 = MPMDist(data1)
#mpm = MPMCls(dist0, dist1)
#np.random.seed(seedr)
#n,gext,grid = get_grid_data(np.vstack(( data0, data1 )) )


#g = mpm.calc_curr_g(grid).reshape(-1,n)
#s = samc.SAMCRun(mpm, burn=0, stepscale=1000, refden=1, thin=10, lim_iters=200)
#s.sample(1e3, temperature=1)
#gavg = mpm.calc_gavg(s.db, grid, 50).reshape(-1,n)

#ga1 = mpm.dist0.calc_db_g(s.db, s.db.root.object.dist0, grid, 50).reshape(-1,n)
#ga2 = mpm.dist1.calc_db_g(s.db, s.db.root.object.dist1, grid, 50).reshape(-1,n)

#myplot(p.subplot(2,2,1),gavg)
##myplot(p.subplot(2,2,2),g)
#myplot(p.subplot(2,2,3),ga1)
#myplot(p.subplot(2,2,4),ga2)

#p.show()
#sys.exit()
########## /SAMC #############

########### NLOpt #############
#print mpm.energy()
#x = mpm.get_params()
#print x
#print mpm.optim(x,None)

#def pvec(x):
	#s = "[ "
	#for i in x:
		#s += "%5.1f," % i
	#s = s[:-1]
	#s+= " ]"
	#return s
#def f(x,grad):
    #e = mpm.optim(x,grad)
    #print "Trying: %8.2f %s" % (e,pvec(x))
    #return e

##opt = nlopt.opt(nlopt.LN_BOBYQA, mpm.get_dof())
#opt = nlopt.opt(nlopt.GN_DIRECT_L, mpm.get_dof())
##opt = nlopt.opt(nlopt.G_MLSL_LDS, mpm.get_dof())
##lopt = nlopt.opt(nlopt.LN_NELDERMEAD, mpm.get_dof())
##lopt.set_ftol_abs(5)
##opt.set_local_optimizer(lopt)
##opt.set_min_objective(mpm.optim)
##opt.set_initial_step(0.5)
#opt.set_min_objective(f)
#opt.set_maxtime(10)
##opt.set_maxeval(100)
##opt.set_ftol_rel(1e-6)
##opt.set_lower_bounds(-10)
##opt.set_upper_bounds(10)
#opt.set_lower_bounds(x-3.0)
#opt.set_upper_bounds(x+3.0)


#xopt = opt.optimize(x)
#xopt_val = opt.last_optimum_value()
#ret = opt.last_optimize_result()

#print "Starting: %9.3f %s" % (mpm.optim(x,None), pvec(x))
#print "Final   : %9.3f %s" % (xopt_val, pvec(xopt))
#print "Return: %d" %ret

#sys.exit()
########## /NLOpt #############

