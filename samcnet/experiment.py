import sys, os, io
import sha
import numpy as np
import networkx as nx
import traceback
import functools
try:
    import simplejson as js
except:
    import json as js

from time import time, sleep
import random
import logging
import logging.handlers

formatter = logging.Formatter('%(name)s: samc %(levelname)s %(message)s')

h = logging.handlers.SysLogHandler(('knight-server.dyndns.org',10514))
h.setLevel(logging.DEBUG)
h.setFormatter(formatter)

logger = logging.getLogger('hbclient')

hstream = logging.StreamHandler()
hstream.setLevel(logging.DEBUG)
hstream.setFormatter(formatter)

logger.addHandler(hstream)
logger.addHandler(h)
logger.setLevel(logging.DEBUG)

sys.path.append('../build')
sys.path.append('./build') # Yuck!

def log_uncaught_exceptions(ex_cls, ex, tb):
    logger.critical(''.join(traceback.format_tb(tb)))
    logger.critical('{0}: {1}'.format(ex_cls, ex))

sys.excepthook = log_uncaught_exceptions

try:
    from samc import SAMCRun
    from bayesnet import BayesNet
    from generator import *
except ImportError as e:
    logger.critical(e)
    logger.critical("Make sure LD_LIBRARY_PATH is set correctly and that the build"+\
            " directory is populated by waf.")
    sys.exit()

def sample(states, data, graph, template=None, iters=1e4, priorweight=1.0, burn=100000, stepscale=10000):
    nodes = np.arange(data.shape[1])

    b = BayesNet(nodes, states, data, graph, template, priorweight)
    s = SAMCRun(b,burn,stepscale)

    t1 = time()
    s.sample(iters)
    t2 = time()
    print("SAMC run took %f seconds." % (t2-t1))
    return b,s

def estimateMean(samc, graph):
    t2 = time()
    func_mean = samc.estimate_func_mean()
    t3 = time()
    print("Mean estimation run took %f seconds." % (t3-t2))
    return func_mean

if 'SAMC_JOB' in os.environ and 'WORKHASH' in os.environ:
    import redis
    r = redis.StrictRedis('knight-server.dyndns.org')

    ########## Read config from driver.py ########
    config = js.loads(os.environ['SAMC_JOB'])

    N = config['nodes']
    iters = config['samc_iters']
    numdata = config['numdata']
    priorweight = config['priorweight']
    numtemplate = config['numtemplate']
    burn = config.get('burn', 100000)
    stepscale = config.get('stepscale', 10000)
    experiment = config.get('experiment_type', 'single')
    method = config.get('gen_method', 'dirichlet')
    noise = config.get('noise', 0.0)

    if 'graph' in config:
        assert 'seed' in config, "Seed not in configuration."
        ngraph = np.fromstring(config['graph'], dtype=np.int32)
        ngraph = ngraph.reshape(int(np.sqrt(ngraph.size)), int(np.sqrt(ngraph.size)))
        #logger.debug('Graph adjacency matrix: %s', str(ngraph))
        graph = nx.from_numpy_matrix(ngraph, create_using=nx.DiGraph())
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        _,_,cpds = generateData(graph, 1000, noise=noise, method=method) # 1000 because we need
        # to generate the CPDs deterministically, and they are only generated
        # when needed, hence the 1000 samples to exercise all 'pathways'.
        random.seed()
        np.random.seed()
        data,states,_ = generateData(graph, numdata, noise=noise, cpds=cpds, method=method)
    else:
        graph = generateHourGlassGraph(nodes=N)
        data, states, cpds = generateData(graph,numdata,noise=noise,method=method)

    if experiment == 'single':
        if numtemplate == 0:
            b,s = sample(states, data, graph, template=None, iters=iters, burn=burn, stepscale=stepscale)
        else:
            template = sampleTemplate(graph, numtemplate)
            b,s = sample(states, data, graph, template, iters, burn=burn, stepscale=stepscale)

        mean = estimateMean(s,graph)
        r.lpush(os.environ['WORKHASH'], mean)
        print('Mean estimation: %f' % mean)

    if experiment == 'difference':
        b,s = sample(states, data, graph, template=None, iters=iters, burn=burn)
        mean1 = estimateMean(s,graph)
        template = sampleTemplate(graph, numtemplate)
        b,s = sample(states, data, graph, template=template, iters=iters, burn=burn)
        mean2 = estimateMean(s,graph)
    
        # Send back func_mean to store
        r.lpush(os.environ['WORKHASH'], mean1 - mean2)
        print('Function difference: %f' % (mean1 - mean2))

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
        N = 15
        iters = 1e5
        numdata = 400
        priorweight = 5
        numtemplate = 15
        
        random.seed(1234567)
        np.random.seed(1234567)

        graph = generateHourGlassGraph(nodes=N)
        data, states, cpds = generateData(graph,numdata)
        logicdata, logicstates, logiccpds = generateData(graph,numdata,method='noisylogic')

        template = sampleTemplate(graph, numtemplate)

        b,s   = sample(logicstates, logicdata[:200], graph, template=template, 
                iters=iters, priorweight=priorweight, burn=0, stepscale=40000)

        b2,s2 = sample(logicstates, logicdata[200:], graph, template=template, 
                iters=iters, priorweight=priorweight, burn=0, stepscale=40000)

        mean1 = estimateMean(s,graph)
        mean2 = estimateMean(s2,graph)

        print "##### wotemp - wtemp = %f - %f = %f #######" % (mean1, mean2, mean1-mean2)
        print "Effective sample size: %d" % len(set(map(tuple,logicdata)))

        b.update_graph(s.mapvalue)
        b2.update_graph(s2.mapvalue)
        drawGraphs(graph, template, b.graph, b2.graph)

        val1 = b.global_edge_presence()
        val2 = b2.global_edge_presence()


