import redis, os, glob, sha, io

import numpy as np
import pandas as pa
import pylab as p
import ConfigParser as cp

r = redis.StrictRedis()
cfg = cp.RawConfigParser()

data = r.hgetall('configs')
tdata = []

for key,val in data.iteritems():
    vals = r.lrange(key, 0, -1)
    tdata.append((key, r.hget('configs', key), pa.Series(vals, dtype=np.double)))

for key,cfg,data in tdata:
  print key
  print cfg
  print data.describe()

#quants = [50,100,200,500]

#def getwtempname(x):
  #return "sample-%d-wtemp.cfg" % x

#def getwotempname(x):
  #return "sample-%d-wout-temp.cfg" % x

#p.hold(True)
#means = [data[getwtempname(x)].mean() for x in quants]
#errs = [data[getwtempname(x)].std() for x in quants]

#p.errorbar(quants, means, yerr=errs, fmt='o')

#means = [data[getwotempname(x)].mean() for x in quants]
#errs = [data[getwotempname(x)].std() for x in quants]

#p.errorbar(quants, means, yerr=errs, fmt='o')
#p.legend(('wtemp','wotemp'))
