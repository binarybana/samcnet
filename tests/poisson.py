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


mpm = MixturePoissonSampler(data0, data1)
np.random.seed(seedr)
#s = samc.SAMCRun(mpm, burn=0, stepscale=1000, refden=1, thin=10)
#s.sample(1e4, temperature=1)
#print s.mapvalue

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

p.plot(data0[:,0], data0[:,1], 'go')
p.plot(data1[:,0], data1[:,1], 'ro')

lx,hx,ly,hy = p.gca().axis()
n,gext,grid = get_grid_data(np.vstack(( data0, data1 )) )

#gavg = mpm.calc_gavg(s.db, grid, 50).reshape(-1,n)


gavg = mpm.calc_gavg(".tmp/samcQ5VzMP", grid, 100).reshape(-1,n)
    #def calc_g(self, pts, parts, mus, sigmas, ks, ws, ds):
	#g = mpm.calc_g(np.ones((2,2)), np.ones(1), [np.zeros((2,4))], [np.eye(2)], [1], [np.ones(4)], [9])
p.imshow(gavg, extent=gext, origin='lower')
p.colorbar()
p.contour(gavg, [0.0], extent=gext, origin='lower', cmap = p.cm.gray)
		
p.show()

#import pstats, cProfile
#cProfile.runctx("samc.SAMCRun(mpm, burn=0, stepscale=1000, thin=10)", globals(), locals(), "prof.prof")
#cProfile.runctx("[mpm.energy() for i in xrange(1000)]", globals(), locals(), "prof.prof")
#cProfile.runctx("mpm.propose()", globals(), locals(), "prof.prof")

#s = pstats.Stats("prof.prof")
#s.strip_dirs().sort_stats("time").print_stats()
#s.strip_dirs().sort_stats("cumtime").print_stats()

