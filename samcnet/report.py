import sys
import pylab as p
import numpy as np
import scipy as sp
import networkx as nx
import json as js
import base64
import zlib
import redis

from samcnet.utils import load_data, decode_element

r = redis.StrictRedis('localhost')

done_hashes = sorted(r.keys('jobs:done:*'), key=lambda x: int(r.hget('jobs:times', x[10:]) or '0'))

for i, d in enumerate(done_hashes):
    desc = r.hget('jobs:descs', d[10:]) or ''
    num = r.llen(d)
    print "%4d. (%3s) %s %s" % (i, num, d[10:15], desc)

sel = raw_input("Choose a dataset or 'q' to exit: ")
if not sel.isdigit() or int(sel) not in range(i+1):
    sys.exit()

sel = int(sel)

datastrings = r.lrange(done_hashes[sel], 0, -1)
print "Got %d datasets from hash %s" % (len(datastrings), done_hashes[sel][:5])

for d in datastrings:
    res = [decode_element(x) for x in load_data(d)]

    n = len(res)
    indices = np.linspace(0,1,n)
    for i,val in enumerate(res):
        if type(val) == np.ndarray:
            p.plot(val, alpha=0.4, color='blue')#color=p.cm.jet(indices[i]))
            #print indices[i], p.cm.jet(indices[i])
        else:
            print val

#p.ylim(0, 7)
p.xlabel('Iterations after burnin')
p.title(r.hget('jobs:descs', done_hashes[sel][10:]))
p.grid(True)

p.show()
