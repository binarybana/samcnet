import os
import sys
import redis
import random
import numpy as np
from samcnet import samc,utils,simple

if 'WORKHASH' in os.environ:
    try:
        import redis
        r = redis.StrictRedis(os.environ['REDIS'])
    except:
        sys.exit("ERROR in worker: Need REDIS environment variable defined.")

random.seed(123)
np.random.seed(123)

o = simple.Simple(truemu=0.0, mu0=0.0)

random.seed()
np.random.seed()

s = samc.SAMCRun(o, burn=100, 
                    stepscale=10000, 
                    refden=0.0, 
                    thin=100)
s.sample(1e5)

res = []
res.append(s.func_mean())
fm = s.func_cummean().copy()
res.append(fm)

res_wire = utils.prepare_data([utils.encode_element(x) for x in res])
if 'WORKHASH' in os.environ:
    r.lpush('jobs:done:'+os.environ['WORKHASH'], res_wire)

