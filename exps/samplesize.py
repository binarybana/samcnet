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
priorweight = 0.0
numtemplate = 0
burn = 1000
stepscale = 10000
thin = 10
refden = 0.0

random.seed(12345)
np.random.seed(12345)

groundgraph = generateHourGlassGraph(nodes=N)
#joint, states = generateJoint(groundgraph, method='dirichlet')
joint, states = generateJoint(groundgraph, method='noisylogic')
template = sampleTemplate(groundgraph, numtemplate)

if 'WORKHASH' in os.environ:
    jobhash = os.environ['WORKHASH']
    if not r.hexists('jobs:grounds', jobhash):
        r.hset('jobs:grounds', jobhash, zlib.compress(cPickle.dumps(groundgraph)))

random.seed()
np.random.seed()

datasizes = [4, 16, 32, 64, 128, 256]
temps = [1.0, 1.0, 2.0, 2.0, 5.0, 5.0]

for temperature, numdata in zip(temps, datasizes):
    data = generateData(groundgraph, joint, numdata)
    groundbnet = BayesNetCPD(states, data, limparent=3)
    groundbnet.set_cpds(joint)
    obj = BayesNetCPD(states, data, limparent=3)
    b = BayesNetSampler(obj, template, groundbnet, priorweight)
    s = SAMCRun(b,burn,stepscale,refden,thin)
    s.sample(iters, temperature)
    s.compute_means()

    if 'WORKHASH' in os.environ:
        r.lpush('jobs:done:' + jobhash, s.read_db())
        r.lpush('custom:%s:samplesize=%d' % 
                (jobhash, numdata), 
                s.db.root.computed.means._v_attrs['kld'] )

    s.db.close()

