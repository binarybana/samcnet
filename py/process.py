import redis, os, glob, sha

import numpy as np
import pandas as pa
import pylab as p
import ConfigParser as cp

r = redis.StrictRedis()

data = r.hgetall('key-reverse')

for key,val in data.iteritems():
    vals = r.lrange(val, 0, -1)
    data[key] = pa.Series(vals, dtype=np.double)

print [(i,x.describe()) for i,x in data.iteritems()]

quants = [50,100,200,500]

def getwtempname(x):
  return "sample-%d-wtemp.cfg" % x

def getwotempname(x):
  return "sample-%d-wout-temp.cfg" % x

p.hold(True)
means = [data[getwtempname(x)].mean() for x in quants]
errs = [data[getwtempname(x)].std() for x in quants]

p.errorbar(quants, means, yerr=errs, fmt='o')

means = [data[getwotempname(x)].mean() for x in quants]
errs = [data[getwotempname(x)].std() for x in quants]

p.errorbar(quants, means, yerr=errs, fmt='o')
p.legend(('wtemp','wotemp'))
