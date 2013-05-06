import sys, os, random
import numpy as np
import scipy as sp
import networkx as nx
import tables as t
import zlib

from samcnet.samc import SAMCRun
from samcnet.bayesnet import BayesNet
from samcnet.bayesnetcpd import BayesNetCPD, BayesNetSampler
from samcnet.generator import *
from samcnet import utils

if 'WORKHASH' in os.environ:
    try:
        redis_server = os.environ['REDIS']
        import redis
        r = redis.StrictRedis(redis_server)
    except:
        sys.exit("ERROR in worker: Need REDIS environment variable defined.")

N = 4
iters = 1e2
numdata = 20
priorweight = 5
numtemplate = 5
burn = 10
stepscale=100000
temperature = 100.0
thin = 2
refden = 0.0

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

ground = BayesNetCPD(states, data)
ground.set_cpds(joint)

obj = BayesNetCPD(states, data)

b = BayesNetSampler(obj, template, ground, priorweight)
s = SAMCRun(b,burn,stepscale,refden,thin)
s.sample(iters, temperature)

s.compute_means()

fname = '/tmp/test.h5'
fid = open(fname, 'w')
fid.write(zlib.decompress(s.read_db()))
fid.close()

db = t.openFile(fname, 'r')

if 'WORKHASH' in os.environ:
    r.lpush('jobs:done:'+os.environ['WORKHASH'], s.read_db())
