import os
import numpy as np
import pylab as p
import simplejson as js
import pandas as pa

from collections import defaultdict

from jobmon import redisbackend as rb

db = rb.RedisDataStore('localhost')
jobhash = db.select_jobfile()

resdir = os.path.join('/home/bana/largeresearch/results', jobhash)
resdir = os.path.join(resdir, os.listdir(resdir)[0])
output = defaultdict(list)

for fname in os.listdir(resdir):
    data = js.loads(open(os.path.join(resdir,fname)).read())
    for k,v in data.iteritems():
        output[k].append(v)

df = pa.DataFrame(output)

print(df.summarize())
