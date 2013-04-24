import sys, os, random
import numpy as np
import scipy as sp
import networkx as nx
import tables as t
import zlib

from samcnet.samc import SAMCRun
from samcnet.treenet import TreeNet, generateTree, generateData
from samcnet import utils

if 'WORKHASH' in os.environ:
    try:
        redis_server = os.environ['REDIS']
        import redis
        r = redis.StrictRedis(redis_server)
    except:
        sys.exit("ERROR in worker: Need REDIS environment variable defined.")

N = 5
comps = 2
iters = 1e3
numdata = 20
#priorweight = 5
#numtemplate = 5
burn = 1
stepscale=10
temperature = 1.0
thin = 1
refden = 0.0

random.seed(123456)
np.random.seed(123456)

groundgraph = generateTree(N, comps)
data = generateData(groundgraph,numdata)
#template = sampleTemplate(groundgraph, numtemplate)

random.seed()
np.random.seed()

ground = TreeNet(N, graph=groundgraph)
b = TreeNet(N, data=data, ground=ground)
s = SAMCRun(b,burn,stepscale,refden,thin,verbose=True)
s.sample(iters, temperature)

s.compute_means()

# All to exercise cde deps
tmp = s.read_db()
import cPickle
txt = zlib.compress(cPickle.dumps([1,2,3]))

if 'WORKHASH' in os.environ:
    r.lpush('jobs:done:'+os.environ['WORKHASH'], s.read_db())
