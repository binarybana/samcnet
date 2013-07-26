from samcnet.mixturepoisson import *
import numpy as np
import pylab as p
import tables as t
import samcnet.samc as samc
from samcnet.lori import *
from math import exp,log

import scipy.stats as st
import scipy.stats.distributions as di
import scipy
import nlopt
import subprocess as sb
import os
import os.path as path

np.seterr(all='ignore') # Careful with this

seedr = np.random.randint(10**6)
seedd = 40767
#seed = 32
print "Seed is %d" % seedr
#np.random.seed(seedd)
np.random.seed(seedr)

def myplot(ax,g):
    ax.plot(data0[:,0], data0[:,1], 'go',label='0')
    ax.plot(data1[:,0], data1[:,1], 'ro',label='1')
    ax.legend(fontsize=8, loc='best')

    im = ax.imshow(g, extent=gext, origin='lower')
    p.colorbar(im,ax=ax)
    ax.contour(g, [0.0], extent=gext, origin='lower', cmap = p.cm.gray)

########## Comparison #############
# Run Yousef/Jianping RNA Synthetic
currdir = path.abspath('.')
synloc = '/home/bana/GSP/research/samc/synthetic/rnaseq'
os.chdir(synloc)
sb.check_call(path.join(synloc, 
    'RNASeq -o out -i params/easyparams -d 2 -m 1 -lr 9 -hr 10 -sr 0.05').split())
os.chdir(currdir)
# Grab some info from the run
data = np.loadtxt(path.join(synloc,'out','out'))
lda,knn,svm = data[0:3]
print("LDA error: %f" % lda)
print("KNN error: %f" % knn)
print("SVM error: %f" % svm)

data = np.loadtxt(path.join(synloc, 'out','trn_norm.txt'),
	delimiter=',')
data0 = data[:30,:]
data1 = data[30:,:]
test = np.loadtxt(path.join(synloc, 'out','tst_norm.txt'),
	delimiter=',')
D = data.shape[1]

labels = np.hstack((np.zeros(500), np.ones(500)))
n,gext,grid = get_grid_data(np.vstack(( data0, data1 )) )

bayes0 = GaussianBayes(np.zeros(D), 1, 8, np.eye(D)*3, data0)
bayes1 = GaussianBayes(np.zeros(D), 1, 8, np.eye(D)*3, data1)

# Gaussian Analytic
gc = GaussianCls(bayes0, bayes1)
print("Gaussian Analytic error: %f" % gc.approx_error_data(test, labels))
gavg = gc.calc_gavg(grid).reshape(-1,n)
myplot(p.subplot(2,3,1),gavg)

 #Gaussian Sampler
#c = GaussianSampler(bayes0,bayes1,data0,data1)
#s1 = samc.SAMCRun(c, burn=0, stepscale=1000, refden=1, thin=10, lim_iters=200)
#s1.sample(1e3, temperature=1)
#print("Gaussian Sampler error: %f" % c.approx_error_data(s1.db, test, labels))
#gavg = c.calc_gavg(s1.db, grid, 50).reshape(-1,n)
#myplot(p.subplot(2,3,2),gavg)

# MPM Model
data = np.loadtxt(path.join(synloc, 'out','trn.txt'),
	delimiter=',')
test = np.loadtxt(path.join(synloc, 'out','tst.txt'),
	delimiter=',')
data0 = data[:30,:]
data1 = data[30:,:]
n,gext,grid = get_grid_data(np.vstack(( data0, data1 )) )

dist0 = MPMDist(data0)
dist1 = MPMDist(data1)
mpm = MPMCls(dist0, dist1) # TODO allow params input here (or maybe per class)
s2 = samc.SAMCRun(mpm, burn=0, stepscale=1000, refden=1, thin=10, lim_iters=200)
s2.sample(1e3, temperature=1)
print("MPM Sampler error: %f" % mpm.approx_error_data(s2.db, test, labels))

gavg = mpm.calc_gavg(s2.db, grid, 50).reshape(-1,n)
#g = mpm.calc_curr_g(grid).reshape(-1,n)
ga1 = mpm.dist0.calc_db_g(s2.db, s2.db.root.object.dist0, grid, 50).reshape(-1,n)
ga2 = mpm.dist1.calc_db_g(s2.db, s2.db.root.object.dist1, grid, 50).reshape(-1,n)

myplot(p.subplot(2,3,2),gavg)
myplot(p.subplot(2,3,3),ga1)
myplot(p.subplot(2,3,4),ga2)
p.subplot(2,3,5)
p.plot(test[:500,0], test[:500,1],'g.',alpha=0.5)
p.plot(test[500:,0], test[500:,1],'r.',alpha=0.5)

p.show()
sys.exit()
########## /Comparison #############

########## SAMC #############
p.close('all')
N = 20
data0 = np.vstack((
    np.hstack(( di.poisson.rvs(10*exp(1), size=(N,1)), di.poisson.rvs(10*exp(2), size=(N,1)) )),
    np.hstack(( di.poisson.rvs(10*exp(2.2), size=(N,1)), di.poisson.rvs(10*exp(2), size=(N,1)) )) ))
data1 = np.hstack(( di.poisson.rvs(10*exp(2), size=(N,1)), di.poisson.rvs(10*exp(1), size=(N,1)) ))

dist0 = MPMDist(data0)
dist1 = MPMDist(data1)
mpm = MPMCls(dist0, dist1)
np.random.seed(seedr)
n,gext,grid = get_grid_data(np.vstack(( data0, data1 )) )


g = mpm.calc_curr_g(grid).reshape(-1,n)
s = samc.SAMCRun(mpm, burn=0, stepscale=1000, refden=1, thin=10, lim_iters=200)
s.sample(1e3, temperature=1)
gavg = mpm.calc_gavg(s.db, grid, 50).reshape(-1,n)

ga1 = mpm.dist0.calc_db_g(s.db, s.db.root.object.dist0, grid, 50).reshape(-1,n)
ga2 = mpm.dist1.calc_db_g(s.db, s.db.root.object.dist1, grid, 50).reshape(-1,n)

myplot(p.subplot(2,2,1),gavg)
#myplot(p.subplot(2,2,2),g)
myplot(p.subplot(2,2,3),ga1)
myplot(p.subplot(2,2,4),ga2)

p.show()
sys.exit()
########## /SAMC #############

########### NLOpt #############
print mpm.energy()
x = mpm.get_params()
print x
print mpm.optim(x,None)

def pvec(x):
	s = "[ "
	for i in x:
		s += "%5.1f," % i
	s = s[:-1]
	s+= " ]"
	return s
def f(x,grad):
    e = mpm.optim(x,grad)
    print "Trying: %8.2f %s" % (e,pvec(x))
    return e

#opt = nlopt.opt(nlopt.LN_BOBYQA, mpm.get_dof())
opt = nlopt.opt(nlopt.GN_DIRECT_L, mpm.get_dof())
#opt = nlopt.opt(nlopt.G_MLSL_LDS, mpm.get_dof())
#lopt = nlopt.opt(nlopt.LN_NELDERMEAD, mpm.get_dof())
#lopt.set_ftol_abs(5)
#opt.set_local_optimizer(lopt)
#opt.set_min_objective(mpm.optim)
#opt.set_initial_step(0.5)
opt.set_min_objective(f)
opt.set_maxtime(10)
#opt.set_maxeval(100)
#opt.set_ftol_rel(1e-6)
#opt.set_lower_bounds(-10)
#opt.set_upper_bounds(10)
opt.set_lower_bounds(x-3.0)
opt.set_upper_bounds(x+3.0)


xopt = opt.optimize(x)
xopt_val = opt.last_optimum_value()
ret = opt.last_optimize_result()

print "Starting: %9.3f %s" % (mpm.optim(x,None), pvec(x))
print "Final   : %9.3f %s" % (xopt_val, pvec(xopt))
print "Return: %d" %ret

sys.exit()
########## /NLOpt #############

########## Profiling #############
#import pstats, cProfile
#cProfile.runctx("samc.SAMCRun(mpm, burn=0, stepscale=1000, thin=10)", globals(), locals(), "prof.prof")
#cProfile.runctx("[mpm.energy() for i in xrange(1000)]", globals(), locals(), "prof.prof")
#cProfile.runctx("mpm.propose()", globals(), locals(), "prof.prof")

#s = pstats.Stats("prof.prof")
#s.strip_dirs().sort_stats("time").print_stats()
#s.strip_dirs().sort_stats("cumtime").print_stats()

########## /Profiling #############
