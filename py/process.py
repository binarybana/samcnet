import redis, os, glob, sha

import numpy as np
import pandas as pa

r = redis.StrictRedis()

for key,val in r.hgetall('key-reverse').iteritems():
    vals = r.lrange(val, 0, -1)
