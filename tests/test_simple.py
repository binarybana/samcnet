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

random.seed(123456)
np.random.seed(123456)

o = simple.Simple()

random.seed()
np.random.seed()

s = samc.SAMCRun(o, burn=10, stepscale=10, refden=0, thin=1)
s.sample(100)

res = []
res.append(s.func_mean())
res.append(s.func_cummean())


res_wire = utils.prepare_data([utils.encode_element(x) for x in res])

sys.stderr.write("Writing res_wire, first 5 bytes: %s\n" % res_wire[:5])

if 'WORKHASH' in os.environ:
    r.lpush('jobs:done:'+os.environ['WORKHASH'], res_wire)

