import sys, os, io
import sha
import numpy as np
import networkx as nx
try:
    import simplejson as js
except:
    import json as js

from time import time, sleep

sys.path.append('../build')
sys.path.append('./build') # Yuck!

try:
    from samc import SAMCRun
    from bayesnet import BayesNet
    from generator import *
except ImportError as e:
    print(e)
    print("Make sure LD_LIBRARY_PATH is set correctly and that the build"+\
            " directory is populated by waf.")
    sys.exit()

def sample(states, traindata, template=None, iters=1e4, priorweight=1.0, burn=100000):
    nodes = np.arange(traindata.shape[1])
    tmat = np.asarray(nx.to_numpy_matrix(template)) if template != None else None

    b = BayesNet(nodes, states, traindata, tmat, priorweight)
    s = SAMCRun(b,burn=burn)

    t1 = time()
    s.sample(iters)
    t2 = time()
    print("SAMC run took %f seconds." % (t2-t1))
    return b,s

def estimateMean(samc, graph):
    gmat = np.asarray(nx.to_numpy_matrix(graph))

    def global_edge_presence(net):
        s = net['x'].argsort()
        ordmat = net['matrix'][s].T[s].T
        return np.abs(gmat - ordmat).sum() / net['x'].shape[0]**2

    t2 = time()
    func_mean = samc.estimate_func_mean(global_edge_presence)
    t3 = time()
    print("Mean estimation run took %f seconds." % (t3-t2))
    return func_mean

if 'SAMC_JOB' in os.environ and 'WORKHASH' in os.environ:
    import redis
    r = redis.StrictRedis('knight-server.dyndns.org')

    ########## Read config from driver.py ########
    config = js.loads(os.environ['SAMC_JOB'])

    N = int(config['nodes'])
    iters = int(config['samc_iters'])
    numdata = int(config['numdata'])
    priorweight = float(config['priorweight'])
    numtemplate = int(config['numtemplate'])
    try:
        burn = int(config['burn'])
    except:
        burn = 100000

    graph = generateHourGlassGraph(nodes=N)
    traindata, states, cpds = generateData(graph,numdata)
    template = sampleTemplate(graph, numtemplate)
    b,s = sample(states, traindata, template, iters, burn=burn)
    mean1 = estimateMean(s,graph)
    b,s = sample(states, traindata, iters=iters, burn=burn)
    mean2 = estimateMean(s,graph)
    
    # Send back func_mean to store
    r.lpush(os.environ['WORKHASH'], mean2 - mean1)
    print('Function difference: %f' % (mean2 - mean1))

elif __name__ == '__main__':
    from utils import *
    if False: #WBCD Data
        iters = 3e5
        traindata = np.loadtxt('data/WBCD2.dat', np.int)
        traindata[:,:-1] -= 1
        states = np.array([10]*9 + [2],dtype=np.int)

        b,s = sample(states, traindata, iters=iters)
        plotHist(s)

    if True:
        N = 10
        iters = 3e5
        numdata = 20
        priorweight = 20
        numtemplate = 20

        graph = generateHourGlassGraph(nodes=N)
        traindata, states, cpds = generateData(graph,numdata)


        template = sampleTemplate(graph, numtemplate)
        b,s = sample(states, traindata, template, iters, priorweight)
        b2,s2 = sample(states, traindata, template=None, iters=iters, priorweight=priorweight)
        mean1 = estimateMean(s,graph)
        mean2 = estimateMean(s2,graph)


