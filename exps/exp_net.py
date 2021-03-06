import sys, os, random
import zlib, cPickle
############### SAMC Setup ############### 
import numpy as np
import scipy as sp
import networkx as nx

from samcnet.samc import SAMCRun
from samcnet.bayesnetcpd import BayesNetSampler, BayesNetCPD
from samcnet import utils
from samcnet.generator import *

if 'WORKHASH' in os.environ:
    try:
        redis_server = os.environ['REDIS']
        import redis
        r = redis.StrictRedis(redis_server)
    except:
        sys.exit("ERROR in worker: Need REDIS environment variable defined.")
############### /SAMC Setup ############### 

N = 8
iters = 3e5
numdata = 40
priorweight = 0.0
numtemplate = 10
burn = 1000
stepscale = 5000
temperature = 10.0
thin = 50
refden = 0.0

random.seed(12345)
np.random.seed(12345)

groundgraph = generateHourGlassGraph(nodes=N)
#joint, states = generateJoint(groundgraph, method='dirichlet')
joint, states = generateJoint(groundgraph, method='noisylogic')
data = generateData(groundgraph, joint, numdata)
template = sampleTemplate(groundgraph, numtemplate)

if 'WORKHASH' in os.environ:
    jobhash = os.environ['WORKHASH']
    if not r.hexists('jobs:grounds', jobhash):
        r.hset('jobs:grounds', jobhash, zlib.compress(cPickle.dumps(groundgraph)))

random.seed()
np.random.seed()

groundbnet = BayesNetCPD(states, data, limparent=3)
groundbnet.set_cpds(joint)

obj = BayesNetCPD(states, data, limparent=3)

b = BayesNetSampler(obj, template, groundbnet, priorweight)
s = SAMCRun(b,burn,stepscale,refden,thin)
s.sample(iters, temperature)

s.compute_means()
#s.compute_means(cummeans=False)

if 'WORKHASH' in os.environ:
    r.lpush('jobs:done:' + jobhash, s.read_db())
s.db.close()
#############
