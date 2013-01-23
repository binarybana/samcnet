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

done_hashes = sorted(r.keys('jobs:done:*'), key=lambda x: r.hget('jobs:times', x[10:]))

for i, d in enumerate(done_hashes):
    desc = r.hget('jobs:descs', d[10:]) or ''
    num = r.llen(d)
    print "%4d. (%3s) %s %s" % (i, num, d[10:18], desc)

sel = int(input("Choose a dataset: "))
assert sel in range(i+1), "Not a valid dataset selector"

datastrings = r.lrange(done_hashes[sel], 0, -1)

for d in datastrings:
    res = load_data(d)
    entropy_mean = res[0]
    kld_mean = res[1]
    entropy_cummean = decode_array(res[2])
    kld_cummean = decode_array(res[3])

    #print("KLD Mean is: %s" % kld_mean)
    #print("Entropy Mean is: %s" % entropy_mean)

    ent = p.plot(entropy_cummean, 'r', alpha=0.3, label='Entropy')
    kld = p.plot(kld_cummean, 'g', alpha=0.3, label='KLD from true')

p.ylim(0, 7)
p.xlabel('Iterations after burnin')
p.ylabel('nats')
p.grid(True)
p.legend([ent[0], kld[0]], ["Entropy", "KLD from true"], loc='best')

p.show()
