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

c = lori.Johnson()
#c = lori.Classification()
s = samc.SAMCRun(c, burn=1e3, stepscale=10000, refden=2, thin=10)
s.sample(2e4, temperature=10)

s.compute_means()

if 'WORKHASH' in os.environ:
    r.lpush('jobs:done:'+os.environ['WORKHASH'], s.read_db())
