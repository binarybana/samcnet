import sys, os, io
import sha
import numpy as np
import scipy as sp
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
from utils import getHost

sys.path.append('build') # Yuck!
sys.path.append('.')
sys.path.append('lib')

def sample(states, 
        data, 
        ground, 
        template=None, 
        iters=1e4, 
        priorweight=1.0, 
        burn=100000, 
        stepscale=10000, 
        temperature = 1.0):

    nodes = np.arange(data.shape[1])

    b = BayesNetCPD(states, data, template, priorweight)
    ground = BayesNetCPD(states, np.array([]))
    s = SAMCRun(b,ground,burn,stepscale)

    t1 = time()
    detail = s.sample(iters, temperature)
    t2 = time()
    #logger.info("SAMC run took %f seconds." , (t2-t1))
    return b,s


if __name__ == '__main__':
    try:
        syslog_server = os.environ['SYSLOG']
    except:
        print "ERROR in worker: Need SYSLOG environment variable defined."
        sys.exit(1)
    try:
        if 'WORKHASH' in os.environ:
            redis_server = os.environ['REDIS']
    except:
        print "ERROR in worker: Need REDIS environment variable defined."
        sys.exit(1)

    formatter = logging.Formatter('%(name)s: samc %(levelname)s %(message)s')

    h = logging.handlers.SysLogHandler((syslog_server,10514))
    h.setLevel(logging.INFO)
    h.setFormatter(formatter)

    logger = logging.getLogger(getHost() + '-worker-' + str(os.getpid()))

    hstream = logging.StreamHandler()
    hstream.setLevel(logging.INFO)
    hstream.setFormatter(formatter)

    logger.addHandler(hstream)
    logger.addHandler(h)
    logger.setLevel(logging.DEBUG)

    def log_uncaught_exceptions(ex_cls, ex, tb):
        logger.critical(''.join(traceback.format_tb(tb)))
        logger.critical('{0}: {1}'.format(ex_cls, ex))

    sys.excepthook = log_uncaught_exceptions
    logger.info("Beginning Job")

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

    if 'WORKHASH' in os.environ:
        import redis
        r = redis.StrictRedis(redis_server)

    if False: #WBCD Data
        iters = 3e5
        data = np.loadtxt('data/WBCD2.dat', np.int)
        data[:,:-1] -= 1
        states = np.array([10]*9 + [2],dtype=np.int)

        b,s = sample(states, data, iters=iters)
        plotHist(s)

    if True:

        N = 5
        iters = 1e4
        numdata = 50
        priorweight = 5
        numtemplate = 5
        
        random.seed(123456)
        np.random.seed(123456)

        g1 = generateHourGlassGraph(nodes=N)
        #print sp.version.version

        #d1, states1, joint1 = generateData(g1,numdata,method='noisylogic')
        d1, states1, joint1 = generateData(g1,numdata,method='dirichlet')
        #print "Ground joint: \n", joint1
        #print "Effective sample size: %d" % len(set(map(tuple,d1)))

        t1 = sampleTemplate(g1, numtemplate)

        #drawGraphs(g1)

        ###############################################
        #cProfile.runctx("sample(states1, d1, joint1, template=t1,iters=iters, priorweight=priorweight, burn=0, stepscale=40000)", globals(),locals(), "prof.prof")

        #s = pstats.Stats('prof.prof')
        #s.strip_dirs().sort_stats("time").print_stats()

        #with statprof.profile():
            #b,s = sample(states1, d1, g1, template=t1,
                    #iters=iters, priorweight=priorweight, burn=0, stepscale=40000)

        ###############################################
        b,s = sample(states1, d1, joint1, template=t1,
                iters=iters, priorweight=priorweight, burn=0, stepscale=30000, temperature=10)
    if 'WORKHASH' in os.environ:
        mean = s.estimate_func_mean()
        r.lpush('jobs:done:'+os.environ['WORKHASH'], mean)


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

