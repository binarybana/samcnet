import os
import yaml
import tables as t
import numpy as np
import pylab as p
import pandas as pa

from jobmon import redisbackend as rb

def sumvals(x):
    vec = np.array(x)
    return vec.mean(), vec.std()

db = rb.RedisDataStore('localhost')
jobhash = db.select_jobfile()

resdir = os.path.join('/home/bana/largeresearch/results', jobhash)

res = pa.DataFrame()
#res = pa.DataFrame(columns=('iters', 'bayes', 'post', 'sampler_full', 'sampler_20'))

for exp in os.listdir(resdir):
    # Get param value
    iters = yaml.load(db.get_params(exp))['iters']
    for fname in os.listdir(os.path.join(resdir,exp)):
        fid = t.openFile(os.path.join(resdir,exp,fname))
        v = {}
        for item in fid.root.object.objfxn._v_attrs._f_list():
            v[item] = fid.root.object.objfxn._v_attrs[item] # Ugly
        v['iters'] = iters
        row = pa.DataFrame(v, index=[1])
        res = res.append(row, ignore_index=True)
        fid.close()

groups = res.groupby('iters')
means = groups.mean()
std = groups.std()

print means.columns
p.errorbar(means.index, means['bayes_error'], yerr=std['bayes_error'])
p.errorbar(means.index, means['posterior_error'], yerr=std['posterior_error'])
p.errorbar(means.index, means['sample_error_full'], yerr=std['sample_error_full'], label='Sampler full')
p.errorbar(means.index, means['sample_error_20'], yerr=std['sample_error_20'], label='Sampler20')
p.legend()
p.show()

