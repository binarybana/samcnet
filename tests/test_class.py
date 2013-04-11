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
s = samc.SAMCRun(c, burn=10, stepscale=10, refden=0, thin=10)
s.sample(1000)

s.compute_means()

if 'WORKHASH' in os.environ:
    r.lpush('jobs:done:'+os.environ['WORKHASH'], s.read_db())
