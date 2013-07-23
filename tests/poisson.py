from samcnet.mixturepoisson import *
import numpy as np
import pylab as p
import tables as t
import samcnet.samc as samc
from math import exp,log

import scipy.stats as st
import scipy.stats.distributions as di
import scipy

seed = np.random.randint(10**6)
#seed = 40767
#seed = 32
print "Seed is %d" % seed
np.random.seed(seed)

p.close('all')

data0 = np.hstack(( di.poisson.rvs(10*exp(1), size=(10,1)), di.poisson.rvs(10*exp(2), size=(10,1)) ))
data1 = np.hstack(( di.poisson.rvs(10*exp(2), size=(10,1)), di.poisson.rvs(10*exp(1), size=(10,1)) ))

mpm = MixturePoissonSampler(data0, data1)
s = samc.SAMCRun(mpm, burn=0, stepscale=1000, refden=1, thin=10)
print("hey")
s.sample(1e3, temperature=1)
print("hey")

