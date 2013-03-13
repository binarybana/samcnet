import os
import sys
import redis
import random
import numpy as np
import zlib
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

s = samc.SAMCRun(c, burn=10, stepscale=100, refden=0, thin=10)
s.sample(4e4)

s.compute_means()

if 'WORKHASH' in os.environ:
    r.lpush('jobs:done:'+os.environ['WORKHASH'], s.read_db())
else: 
    with open('/tmp/data.h5', 'w') as fid:
        fid.write(zlib.decompress(s.read_db()))

