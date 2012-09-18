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
from probability import GroundNet, JointDistribution
from utils import getHost

formatter = logging.Formatter('%(name)s: samc %(levelname)s %(message)s')

h = logging.handlers.SysLogHandler(('knight-server.dyndns.org',10514))
h.setLevel(logging.INFO)
h.setFormatter(formatter)

#logger = logging.getLogger('hbclient')
logger = logging.getLogger(getHost() + ' child ' + str(os.getpid()))
#logger = logging.getLogger(__name__)

hstream = logging.StreamHandler()
hstream.setLevel(logging.INFO)
hstream.setFormatter(formatter)

logger.addHandler(hstream)
logger.addHandler(h)
logger.setLevel(logging.DEBUG)

sys.path.append('build') # Yuck!
sys.path.append('.')
sys.path.append('lib')

def log_uncaught_exceptions(ex_cls, ex, tb):
    logger.critical(''.join(traceback.format_tb(tb)))
    logger.critical('{0}: {1}'.format(ex_cls, ex))

sys.excepthook = log_uncaught_exceptions

try:
    from samc import SAMCRun
    from bayesnet import BayesNet
    from bayesnetcpd import BayesNetCPD
    from generator import *
except ImportError as e:
    logger.critical(e)
    logger.critical(sys.path)
    logger.critical("Make sure LD_LIBRARY_PATH is set correctly and that the build"+\
            " directory is populated by waf.")
    sys.exit()

def sample(states, data, ground, template=None, iters=1e4, priorweight=1.0, burn=100000, stepscale=10000):
    nodes = np.arange(data.shape[1])

    b = BayesNetCPD(nodes, states, data, template, priorweight)
    if isinstance(ground, nx.DiGraph):
        ground = np.asarray(nx.to_numpy_matrix(ground), dtype=np.int32)
        np.fill_diagonal(ground, 1)
    elif isinstance(ground, JointDistribution):
        ground = GroundNet(ground)
    s = SAMCRun(b,ground,burn,stepscale)

    t1 = time()
    s.sample(iters)
    t2 = time()
    logger.info("SAMC run took %f seconds." , (t2-t1))
    return b,s

logger.info("Beginning Job")

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
    truncate = config.get('truncate', 3)

    if 'graph' in config:
        assert 'seed' in config, "Seed not in configuration."
        ngraph = np.fromstring(config['graph'], dtype=np.int32)
        ngraph = ngraph.reshape(int(np.sqrt(ngraph.size)), int(np.sqrt(ngraph.size)))
        #logger.debug('Graph adjacency matrix: %s', str(ngraph))
        graph = nx.from_numpy_matrix(ngraph, create_using=nx.DiGraph())
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        _,_,joint = generateData(graph, 10, noise=noise, method=method) # 1000 because we need
        # to generate the CPDs deterministically
        random.seed()
        np.random.seed()
        data,states,_ = generateData(graph, numdata, noise=noise, joint=joint, method=method)
    else:
        graph = generateHourGlassGraph(nodes=N)
        data, states, joint = generateData(graph,numdata,noise=noise,method=method)

    if experiment == 'single':
        if numtemplate == 0:
            b,s = sample(states, data, joint, template=None, iters=iters, burn=burn, stepscale=stepscale)
        else:
            template = sampleTemplate(graph, numtemplate)
            b,s = sample(states, data, joint, template, iters, burn=burn, stepscale=stepscale)

        mean = s.estimate_func_mean(truncate)
        r.lpush(os.environ['WORKHASH'], mean)
        print('Mean estimation: %f' % mean)

    if experiment == 'difference':
        b,s = sample(states, data, joint, template=None, iters=iters, burn=burn)
        mean1 = s.estimate_func_mean(truncate)
        template = sampleTemplate(graph, numtemplate)
        b,s = sample(states, data, joint, template=template, iters=iters, burn=burn)
        mean2 = s.estimate_func_mean(truncate)
    
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

        #import pstats, cProfile, statprof
        #logging.critical('test %d', np.random.randint(1000))
        #x = 1/0
        #sys.exit()
        N = 5
        iters = 1e4
        numdata = 20
        priorweight = 5
        numtemplate = 5
        
        random.seed(1234567)
        np.random.seed(1234567)

        g1 = generateHourGlassGraph(nodes=N)

        #d1, states1, joint1 = generateData(g1,numdata,method='noisylogic')
        d1, states1, joint1 = generateData(g1,numdata,method='dirichlet')
        print "Ground joint: \n", joint1
        print "Effective sample size: %d" % len(set(map(tuple,d1)))

        t1 = sampleTemplate(g1, numtemplate)

        ###############################################
        #cProfile.runctx("sample(states1, d1, joint1, template=t1,iters=iters, priorweight=priorweight, burn=0, stepscale=40000)", globals(),locals(), "prof.prof")

        #s = pstats.Stats('prof.prof')
        #s.strip_dirs().sort_stats("time").print_stats()

        #with statprof.profile():
            #b,s = sample(states1, d1, g1, template=t1,
                    #iters=iters, priorweight=priorweight, burn=0, stepscale=40000)

        ###############################################
        b,s = sample(states1, d1, joint1, template=t1,
                iters=iters, priorweight=priorweight, burn=0, stepscale=40000)

        #es = []
        #newes = []
        #for i in range(20):
            #es.append(s.estimate_func_mean())
            #newes.append(s.estimate_func_mean(trunc=True))
            #s.sample(1e4)
        ###############################################

        #print "##### wotemp - wtemp = %f - %f = %f #######" % (mean1, mean2, mean1-mean2)

        #b.update_graph(s.mapvalue)
        #b2.update_graph(s2.mapvalue)
        #drawGraphs(g1, t1)

        #val1 = b.global_edge_presence()
        #val2 = b2.global_edge_presence()

        #print('Global edge error: %f' % mean1)
        #print('Map edge error: %f' % val1)


