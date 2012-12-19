######### Standard Experiment Setup #############
import sys, os, io, sha, random, logging, logging.handlers, traceback
from time import time, sleep

from utils import getHost

#if __name__ == "__main__":
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

level = logging.WARNING

h = logging.handlers.SysLogHandler((syslog_server,514))
h.setLevel(level)
h.setFormatter(formatter)

hstream = logging.StreamHandler()
hstream.setLevel(level)
hstream.setFormatter(formatter)

logger = logging.getLogger(getHost() + '-worker-' + str(os.getpid()))
logger.addHandler(hstream)
logger.addHandler(h)
logger.setLevel(level)

def log_uncaught_exceptions(ex_cls, ex, tb):
    logger.critical(''.join(traceback.format_tb(tb)))
    logger.critical('{0}: {1}'.format(ex_cls, ex))

sys.excepthook = log_uncaught_exceptions
if 'WORKHASH' in os.environ:
    import redis
    r = redis.StrictRedis(redis_server)
logger.info("Beginning Job")
######### /Standard Experiment Setup #############

sys.path.append('build') # Yuck!
sys.path.append('.')
sys.path.append('lib')

import numpy as np
import scipy as sp
import networkx as nx
import simplejson as js

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

N = 5
iters = 5e4
numdata = 50
priorweight = 5
numtemplate = 5
burn = 1000
stepscale=30000
temperature = 1.0

random.seed(123456)
np.random.seed(123456)

g1 = generateHourGlassGraph(nodes=N)
data, states, joint = generateData(g1,numdata,method='noisylogic')
#data, states, joint = generateData(g1,numdata,method='dirichlet')
template = sampleTemplate(g1, numtemplate)

print joint

random.seed()
np.random.seed()

ground = BayesNetCPD(states, data, template, ground, priorweight)

b = BayesNetCPD(states, data, template, priorweight)
s = SAMCRun(b,burn,stepscale)
s.sample(iters, temperature)

mean = s.estimate_func_mean()

print("Mean is: ", mean)
if 'WORKHASH' in os.environ:
    r.lpush('jobs:done:'+os.environ['WORKHASH'], mean)
