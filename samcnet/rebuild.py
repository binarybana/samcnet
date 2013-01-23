import sys, os, random
############### SAMC Setup ############### 
sys.path.append('build') # Yuck!
sys.path.append('.')
sys.path.append('lib')

import numpy as np
import scipy as sp
import networkx as nx
import simplejson as js

try:
    from samc import SAMCRun
    from bayesnet import BayesNet
    from bayesnetcpd import BayesNetCPD
    from generator import *
except ImportError as e:
    sys.exit("Make sure LD_LIBRARY_PATH is set correctly and that the build"+\
            " directory is populated by waf.\n\n %s" % str(e))

if 'WORKHASH' in os.environ:
    try:
        redis_server = os.environ['REDIS']
        import redis
        r = redis.StrictRedis(redis_server)
    except:
        sys.exit("ERROR in worker: Need REDIS environment variable defined.")
############### /SAMC Setup ############### 

N = 5
iters = 1e2
numdata = 20
priorweight = 5
numtemplate = 5
burn = 10
stepscale=30000
temperature = 1.0

random.seed(123456)
np.random.seed(123456)

groundgraph = generateHourGlassGraph(nodes=N)
data, states, joint = generateData(groundgraph,numdata,method='noisylogic')
#data, states, joint = generateData(groundgraph,numdata,method='dirichlet')
template = sampleTemplate(groundgraph, numtemplate)

print "Joint:"
print joint

random.seed()
np.random.seed()

ground = BayesNetCPD(states, data, template, ground=joint, priorweight=priorweight, gold=True)

b = BayesNetCPD(states, data, template, ground=ground, priorweight=priorweight)
s = SAMCRun(b,burn,stepscale)
s.sample(iters, temperature)

mean = s.estimate_func_mean()

print("Mean is: ", mean)
if 'WORKHASH' in os.environ:
    r.lpush('jobs:done:'+os.environ['WORKHASH'], mean)
