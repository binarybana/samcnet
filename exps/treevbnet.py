import sys, os, random
import numpy as np
import scipy as sp
import networkx as nx
import json as js
import tables as t
import zlib
import cPickle
import time as gtime
import pylab as p

from samcnet.samc import SAMCRun
from samcnet.treenet import TreeNet, generateTree, generateData
from samcnet.bayesnetcpd import BayesNetSampler, BayesNetCPD
from samcnet import utils
from samcnet.generator import sampleTemplate
import samcnet.generator as gen

start = None
def time():
    global start
    if start is None:
        start = gtime.time()
    else:
        t = gtime.time()
        print("Time taken: {} seconds".format(t-start))
        start = None

if 'WORKHASH' in os.environ:
    try:
        redis_server = os.environ['REDIS']
        import redis
        r = redis.StrictRedis(redis_server)
    except:
        sys.exit("ERROR in worker: Need REDIS environment variable defined.")

N = 5
comps = 2
iters = 3e5
numdata = 30
burn = 1000
stepscale = 30000
temperature = 1.0
thin = 50
refden = 0.0
numtemplate = 10
priorweight = 0.0

random.seed(12345)
np.random.seed(12345)

groundgraph = generateTree(N, comps)
data = generateData(groundgraph,numdata)
template = sampleTemplate(groundgraph, numtemplate)

if 'WORKHASH' in os.environ:
    jobhash = os.environ['WORKHASH']
    if not r.hexists('jobs:grounds', jobhash):
        r.hset('jobs:grounds', jobhash, zlib.compress(cPickle.dumps(groundgraph)))

random.seed()
np.random.seed()

ground = TreeNet(N, data=data, graph=groundgraph)
############### TreeNet ##############

#b1 = TreeNet(N, data, template, priorweight, ground)
#s1 = SAMCRun(b1,burn,stepscale,refden,thin)
#time()
#s1.sample(iters, temperature)
#time()

#s1.compute_means()
#if 'WORKHASH' in os.environ:
    #r.lpush('jobs:done:' + jobhash, s1.read_db())

############## bayesnetcpd ############

joint = utils.graph_to_joint(groundgraph)
states = np.ones(len(joint.dists),dtype=np.int32)*2
ground = BayesNetCPD(states, data)
ground.set_cpds(joint)

obj = BayesNetCPD(states, data)
b2 = BayesNetSampler(obj, template, ground, priorweight)
s2 = SAMCRun(b2,burn,stepscale,refden,thin)
time()
s2.sample(iters, temperature)
time()
s2.compute_means()
if 'WORKHASH' in os.environ:
    r.lpush('jobs:done:' + jobhash, s2.read_db())
    
#######################################
#

