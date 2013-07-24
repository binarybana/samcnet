from samcnet.mixturepoisson import *
import numpy as np
import pylab as p
import tables as t
import samcnet.samc as samc
from samcnet.lori import get_grid_data
from math import exp,log

import scipy.stats as st
import scipy.stats.distributions as di
import scipy
import nlopt

seedr = np.random.randint(10**6)
seedd = 40767
#seed = 32
#print "Seed is %d" % seed
np.random.seed(seedd)

p.close('all')
N = 20
data0 = np.hstack(( di.poisson.rvs(10*exp(1), size=(N,1)), di.poisson.rvs(10*exp(2), size=(N,1)) ))
data1 = np.hstack(( di.poisson.rvs(10*exp(2), size=(N,1)), di.poisson.rvs(10*exp(1), size=(N,1)) ))

dist0 = MPMDist(data0)
dist1 = MPMDist(data1)
mpm = MPMCls(dist0, dist1)
np.random.seed(seedr)

########## SAMC #############
n,gext,grid = get_grid_data(np.vstack(( data0, data1 )) )

def myplot(ax,g):
    ax.plot(data0[:,0], data0[:,1], 'go',label='0')
    ax.plot(data1[:,0], data1[:,1], 'ro',label='1')
    ax.legend(fontsize=8, loc='best')

    im = ax.imshow(g, extent=gext, origin='lower')
    p.colorbar(im,ax=ax)
    ax.contour(g, [0.0], extent=gext, origin='lower', cmap = p.cm.gray)

sb1 = p.subplot(2,1,1)
sb2 = p.subplot(2,1,2)

g = mpm.calc_curr_g(grid).reshape(-1,n)
s = samc.SAMCRun(mpm, burn=0, stepscale=1000, refden=1, thin=10, lim_iters=200)
s.sample(1e3, temperature=1)
gavg = mpm.calc_gavg(s.db, grid, 50).reshape(-1,n)
#gavg = mpm.calc_gavg(".tmp/samcPoCdx6", grid, 100).reshape(-1,n)

myplot(sb1,gavg)
myplot(sb2,g)

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
