import redis, os, glob, sha, io, sys

import numpy as np
import pandas as pa
import pylab as p
import simplejson as js
import sha

r = redis.StrictRedis()

# sweep-configs: hashmap from hashedjson to json
# sweep-<hash>: hashmap from paramvalue to hash
# <hash>: list of values

count = 0
valid_selections = []
sweeps = zip(range(10000), r.hgetall('sweep-configs').iteritems())
#sweeps = zip(range(10000), r.hgetall('configs').iteritems())
for i,(k,v) in sweeps:
    sweepkey = 'sweep-' + k
    for num,h in r.hgetall(sweepkey).iteritems():
        count += r.llen(h)
    if count > 0:
        valid_selections.append(i)
        print "[%d] with %d samples: " % (i,count)
        for kk,vv in js.loads(v).iteritems():
            if kk == 'graph':
                print '\t'+kk+':\t'+str(sha.sha(vv).hexdigest())[:8]
            else:
                print "\t{0}: {1}".format(kk,vv)
    count = 0

if len(valid_selections) == 0:
    print("No selections currently available.")
    sys.exit()
sel = raw_input("Select which sweep you want to plot: ")
try:
    sel = int(sel)
except:
    sys.exit()
if int(sel) not in valid_selections:
    print("Incorrect choice, please try again.")
    sys.exit()

sweepkey = 'sweep-' + sweeps[int(sel)][1][0]
hashes = r.hgetall(sweepkey)
data = {}
quants = []

for num,h in hashes.iteritems():
    if r.llen(h) > 0:
        n = int(float(num))
        quants.append(n)
        data[n] = pa.Series(r.lrange(h, 0, -1), dtype=np.float)
        print "\n##### %d #####" % n
        print data[n].describe()

#p.hold(True)
means = [data[x].mean() for x in quants]
errs = [data[x].std() for x in quants]
quants = np.array(quants) + np.random.randint(0,5)
p.errorbar(quants, means, yerr=errs, fmt='o', markersize=10)
p.grid(True)
p.show()

