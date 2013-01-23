import pylab as p
import numpy as np
import scipy as sp
import networkx as nx
import simplejson as js
import base64
import zlib
import redis

def decode_array(s):
    return np.fromstring(base64.b64decode(s))
def load_data(s):
    return js.loads(zlib.decompress(s))

r = redis.StrictRedis('localhost')

done = r.keys('jobs:done:*')
datastrings = r.lrange(done[0], 0, -1)

for d in datastrings:
    res = load_data(d)
    entropy_mean = res[0]
    kld_mean = res[1]
    entropy_cummean = decode_array(res[2])
    kld_cummean = decode_array(res[3])

    print("KLD Mean is: %s" % kld_mean)
    print("Entropy Mean is: %s" % entropy_mean)

    p.plot(entropy_cummean)

p.ylim(0, 7)
p.show()
