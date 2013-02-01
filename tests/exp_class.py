import os
import sys
import redis
import random
import numpy as np
try:
    from samcnet import samc,lori,utils
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

random.seed(123456)
np.random.seed(123456)

c = lori.Classification()

random.seed()
np.random.seed()

s = samc.SAMCRun(c, burn=10000, stepscale=100, refden=0, thin=100)
s.sample(5e5)

res = []
def make_acc(n):
    return eval ("lambda x: x%s" %n)
for acc in [lambda x: x[0][0], lambda x: x[1][1], lambda x: x[2][0,0], make_acc("[-1]")]:
    for get in [s.func_mean, s.func_cummean]:
        res.append(get(acc))

res_wire = utils.prepare_data([utils.encode_element(x) for x in res])

if 'WORKHASH' in os.environ:
    r.lpush('jobs:done:'+os.environ['WORKHASH'], res_wire)

