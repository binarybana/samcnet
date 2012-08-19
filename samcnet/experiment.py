import sys, os, io
import sha
import numpy as np
import networkx as nx
import functools
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

def sample(states, data, template=None, iters=1e4, priorweight=1.0, burn=100000):
    nodes = np.arange(data.shape[1])
    tmat = np.asarray(nx.to_numpy_matrix(template)) if template != None else None

    b = BayesNet(nodes, states, data, tmat, priorweight)
    s = SAMCRun(b,burn=burn)

    t1 = time()
    s.sample(iters)
    t2 = time()
    print("SAMC run took %f seconds." % (t2-t1))
    return b,s

def global_edge_presence(gmat, net):
    s = net['x'].argsort()
    ordmat = net['matrix'][s].T[s].T
    return np.abs(gmat - ordmat).sum() / net['x'].shape[0]**2

def estimateMean(samc, graph):
    gmat = np.asarray(nx.to_numpy_matrix(graph))
    h = functools.partial(global_edge_presence, gmat)

    t2 = time()
    func_mean = samc.estimate_func_mean(h)
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
    data, states, cpds = generateData(graph,numdata)
    template = sampleTemplate(graph, numtemplate)
    b,s = sample(states, data, template, iters, burn=burn)
    mean1 = estimateMean(s,graph)
    b,s = sample(states, data, iters=iters, burn=burn)
    mean2 = estimateMean(s,graph)
    
    # Send back func_mean to store
    r.lpush(os.environ['WORKHASH'], mean2 - mean1)
    print('Function difference: %f' % (mean2 - mean1))

elif __name__ == '__main__':
    from utils import *
    if False: #WBCD Data
        iters = 3e5
        data = np.loadtxt('data/WBCD2.dat', np.int)
        data[:,:-1] -= 1
        states = np.array([10]*9 + [2],dtype=np.int)

        b,s = sample(states, data, iters=iters)
        plotHist(s)

    if True:
        N = 10
        iters = 1e5
        numdata = 200
        priorweight = 2000
        numtemplate = 20

        graph = generateHourGlassGraph(nodes=N)
        data, states, cpds = generateData(graph,numdata)
        logicdata, logicstates, logiccpds = generateData(graph,numdata,method='noisylogic')

        template = sampleTemplate(graph, numtemplate)
        b,s   = sample(logicstates, logicdata, template=None, iters=iters, priorweight=priorweight, burn=0)
        b2,s2 = sample(logicstates, logicdata, template=template, iters=iters, priorweight=priorweight, burn=0)
        mean1 = estimateMean(s,graph)
        mean2 = estimateMean(s2,graph)

        print "##### wotemp - wtemp = %f - %f = %f #######" % (mean1, mean2, mean1-mean2)

        b.update_graph(s.mapvalue)
        b2.update_graph(s2.mapvalue)
        drawGraphs(graph, template, b.graph, b2.graph)

        val1 = global_edge_presence(np.array(nx.to_numpy_matrix(graph)), {'x': s.mapvalue[1], 'matrix':s.mapvalue[0]})
        val2 = global_edge_presence(np.array(nx.to_numpy_matrix(graph)), {'x': s2.mapvalue[1], 'matrix':s2.mapvalue[0]})


