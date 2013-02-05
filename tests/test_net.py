import sys, os, random
############### SAMC Setup ############### 

import numpy as np
import scipy as sp
import networkx as nx
import simplejson as js
import tables as t
import zlib

try:
    from samcnet.samc import SAMCRun
    from samcnet.bayesnet import BayesNet
    from samcnet.bayesnetcpd import BayesNetCPD
    from samcnet.generator import *
    from samcnet import utils
except ImportError as e:
    sys.exit("Make sure LD_LIBRARY_PATH is set correctly and that the build"+\
            " directory is populated by waf.\n\n %s" % str(e))

if 'WORKHASH' in os.environ:
    try:
        redis_server = os.environ['REDIS']
        import redis
        r = redis.StrictRedis(redis_server)
    except:
        sys.exit("ERROR in worker: Need REDIS environment variable defined.")
############### /SAMC Setup ############### 

N = 7
iters = 1e4
numdata = 20
priorweight = 5
numtemplate = 5
burn = 100
stepscale=100000
temperature = 100.0
thin = 10
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

ground = BayesNetCPD(states, data, template, ground=joint, priorweight=priorweight, gold=True)

b = BayesNetCPD(states, data, template, ground=ground, priorweight=priorweight,verbose=True)
s = SAMCRun(b,burn,stepscale,refden,thin,verbose=True)
s.sample(iters, temperature)

fname = '/tmp/test.h5'
fid = open(fname, 'w')
fid.write(zlib.decompress(s.read_db()))
fid.close()

db = t.openFile(fname, 'r')

#res = []
#for acc in [lambda x: x[0], lambda x: x[1], lambda x: x[2]]:
    #for get in [s.func_mean, s.func_cummean]:
        #res.append(get(acc))

#res_wire = utils.prepare_data([utils.encode_entry(x) for x in res])

#print("KLD Mean is: ", res[0])
#print("Entropy Mean is: ", res[2])
#print("Edge distance Mean is: ", res[4])

#print("Entropy Cummean len/min/max:")
#print res[1].size, res[1].min(), res[1].max()

#print("KLD Cummean len/min/max:")
#print res[3].size, res[3].min(), res[3].max()

#print("Edge Cummean len/min/max:")
#print res[5].size, res[5].min(), res[5].max()

if 'WORKHASH' in os.environ:
    r.lpush('jobs:done:'+os.environ['WORKHASH'], res_wire)
