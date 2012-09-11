import redis, os, glob, sha, io, sys

import numpy as np
import pandas as pa
import pylab as p
import simplejson as js

r = redis.StrictRedis()

# sweep-configs: hashmap from hashedjson to json
# sweep-<hash>: hashmap from paramvalue to hash
# <hash>: list of values

count = 0
valid_selections = []
sweeps = zip(range(10000), r.hgetall('sweep-configs').iteritems())
for i,(k,v) in sweeps:
    sweepkey = 'sweep-' + k
    for num,h in r.hgetall(sweepkey).iteritems():
        count += r.llen(h)
    if count > 0:
        valid_selections.append(i)
        print "[%d] with %d samples: " % (i,count)
        for kk,vv in js.loads(v).iteritems():
            if kk == 'graph':
                print '\t'+kk+':\t'+str(hash(v))
            else:
                print "\t{0}: {1}".format(kk,vv)
    count = 0

if len(valid_selections) == 0:
    print("No selections currently available.")
    sys.exit()
sel = raw_input("Select which sweep you want to plot: ")
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

p.errorbar(quants, means, yerr=errs, fmt='o')
p.grid(True)
p.show()

