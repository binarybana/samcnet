import sys, os, random
import zlib, cPickle
############### SAMC Setup ############### 
import numpy as np
import scipy as sp
import networkx as nx

from samcnet.samc import SAMCRun
from samcnet.bayesnetcpd import BayesNetSampler, BayesNetCPD
from samcnet import utils
from samcnet.generator import *

if 'WORKHASH' in os.environ:
    try:
        redis_server = os.environ['REDIS']
        import redis
        r = redis.StrictRedis(redis_server)
    except:
        sys.exit("ERROR in worker: Need REDIS environment variable defined.")
############### /SAMC Setup ############### 

N = 9
iters = 3e5
numdata = 0 #NEED TO ADD NOISE FIRST

temperature = 1.0
burn = 1000
stepscale = 10000
thin = 10
refden = 0.0

random.seed(12345)
np.random.seed(12345)

groundgraph = generateHourGlassGraph(nodes=N)
#joint, states = generateJoint(groundgraph, method='dirichlet')
joint, states = generateJoint(groundgraph, method='noisylogic')
data = generateData(groundgraph, joint, numdata)
groundbnet = BayesNetCPD(states, data, limparent=3)
groundbnet.set_cpds(joint)

if 'WORKHASH' in os.environ:
    jobhash = os.environ['WORKHASH']
    if not r.hexists('jobs:grounds', jobhash):
        r.hset('jobs:grounds', jobhash, zlib.compress(cPickle.dumps(groundgraph)))

random.seed()
np.random.seed()

#p_struct = float(sys.argv[1])
p_struct = 30.0
for numtemplate in [4,8]:
    for cpd in [True, False]:
        if cpd:
            p_cpd = p_struct
        else:
            p_cpd = 0.0

        random.seed(12345)
        np.random.seed(12345)

        obj = BayesNetCPD(states, data, limparent=3)
        template = sampleTemplate(groundgraph, numtemplate)

        random.seed()
        np.random.seed()

        b = BayesNetSampler(obj, 
                template, 
                groundbnet, 
                p_struct=p_struct,
                p_cpd=p_cpd)
        s = SAMCRun(b,burn,stepscale,refden,thin)
        s.sample(iters, temperature)
        s.compute_means(cummeans=False)

        if 'WORKHASH' in os.environ:
            r.lpush('jobs:done:' + jobhash, s.read_db())
            r.lpush('custom:%s:p_struct=%d:ntemplate=%d:p_cpd=%d' % 
                    (jobhash, int(p_struct*10), numtemplate, int(p_cpd*10)), 
                    s.db.root.computed.means._v_attrs['kld'] )
        s.db.close()

