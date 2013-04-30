import sys, os, random
import numpy as np
import scipy as sp
import networkx as nx
import json as js
import tables as t
import zlib
import cPickle

from samcnet.samc import SAMCRun
from samcnet.treenet import TreeNet, generateTree, generateData
from samcnet import utils
from samcnet.generator import sampleTemplate

if 'WORKHASH' in os.environ:
    try:
        redis_server = os.environ['REDIS']
        import redis
        r = redis.StrictRedis(redis_server)
    except:
        sys.exit("ERROR in worker: Need REDIS environment variable defined.")

N = 10
comps = 2
iters = 4e5
numdata = 30
burn = 1000
stepscale = 200
temperature = 3.0
thin = 10
refden = 0.0
numtemplate = 10
priorweight = 1.0

random.seed(12345)
np.random.seed(12345)

groundgraph = generateTree(N, comps)
data = generateData(groundgraph,numdata)
template = sampleTemplate(groundgraph, numtemplate)

random.seed()
np.random.seed()

ground = TreeNet(N, data=data, graph=groundgraph)

if 'WORKHASH' in os.environ:
    jobhash = os.environ['WORKHASH']
    if not r.hexists('jobs:grounds', jobhash):
        r.hset('jobs:grounds', jobhash, zlib.compress(cPickle.dumps(ground)))

b = TreeNet(N, data=data, ground=ground, priorweight=priorweight,
        template=template, verbose=True)
s = SAMCRun(b,burn,stepscale,refden,thin,verbose=True)
s.sample(iters, temperature)


s.compute_means()

if 'WORKHASH' in os.environ:
    r.lpush('jobs:done:' + jobhash, s.read_db())
