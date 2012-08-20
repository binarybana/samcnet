import redis, os, glob, sha, io

import numpy as np
import pandas as pa
import pylab as p
import simplejson as js

r = redis.StrictRedis()

# sweep-configs: hashmap from hashedjson to json
# sweep-<hash>: hashmap from paramvalue to hash
# <hash>: list of values

sweeps = zip(range(10000), r.hgetall('sweep-configs').iteritems())
for i,(k,v) in sweeps:
    print "[%d]: " % i
    for kk,vv in js.loads(v).iteritems():
        if kk == 'graph':
            print '\t'+kk+':\t'+str(hash(v))
        else:
            print "\t{0}: {1}".format(kk,vv)

sel = raw_input("Select which sweep you want to plot: ")

sweepkey = 'sweep-' + sweeps[int(sel)][1][0]
hashes = r.hgetall(sweepkey)
data = {}

for num,h in hashes.iteritems():
    data[int(num)] = pa.Series(r.lrange(h, 0, -1), dtype=np.float)
    print num
    print data[int(num)].describe()

quants = map(int,hashes.keys())

#p.hold(True)
means = [data[x].mean() for x in quants]
errs = [data[x].std() for x in quants]

p.errorbar(quants, means, yerr=errs, fmt='o')
p.grid(True)
#p.figure()
#p.boxplot(map(lambda x: x.values, data.values()), positions=quants)
p.show()

