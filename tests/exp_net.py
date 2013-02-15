import sys, os, random
############### SAMC Setup ############### 
print "Starting job"

sys.path.append('build') # Yuck!
sys.path.append('.')
sys.path.append('lib')

import numpy as np
import scipy as sp
import networkx as nx
import samcnet.utils as utils

try:
    from samcnet.samc import SAMCRun
    from samcnet.bayesnet import BayesNet
    from samcnet.bayesnetcpd import BayesNetCPD
    from samcnet.generator import *
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

N = 4
iters = 1e6
numdata = 20
priorweight = 5
numtemplate = 5
burn = 10000
stepscale=100000
temperature = 5.0
thin = 100
refden = 0.0

random.seed(12345)
np.random.seed(12345)

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
s = SAMCRun(b,burn,stepscale,refden,thin)
s.sample(iters, temperature)

res = []
for acc in [lambda x: x[0], lambda x: x[1], lambda x: x[2]]:
    for get in [s.func_mean, s.func_cummean]:
        res.append(get(acc))

res = utils.prepare_data([utils.encode_element(x) for x in res])

if 'WORKHASH' in os.environ:
    r.lpush('jobs:done:'+os.environ['WORKHASH'], res)
