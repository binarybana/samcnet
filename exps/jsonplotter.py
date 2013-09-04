import os
import numpy as np
import pylab as p
import simplejson as js
import pandas as pa
import yaml

from collections import defaultdict

from jobmon import redisbackend as rb

db = rb.RedisDataStore('localhost')
jobhash = db.select_jobfile()

resdir = os.path.join('/home/bana/largeresearch/results', jobhash)
num_params = len(os.listdir(resdir))
comps = 6
med_mat = np.zeros((num_params, comps))
med_diff_mat = np.zeros((num_params, comps-1))
std_diff_mat = np.zeros((num_params, comps-1))
std_mat = np.zeros((num_params, comps))
p_vec = np.zeros(num_params)
for i,paramdir in enumerate(os.listdir(resdir)):
    output = defaultdict(list)
    diffs = defaultdict(list)
    other = defaultdict(list)
    params = yaml.load(db.get_params(paramdir))
    assert len(params.values()) == 1
    val = params.values()[0]
    for fname in os.listdir(os.path.join(resdir,paramdir)):
        data = js.loads(open(os.path.join(resdir,paramdir,fname)).read())
        for k,v in data.iteritems():
            if k == 'errors':
                mpmerr = data['errors']['mpm'] 
                for kk,vv in data['errors'].iteritems():
                    output[kk].append(vv)
                    if kk != 'mpm':
                        diffs[kk].append((vv-mpmerr)/mpmerr)
            else: 
                other[k].append(v)

    df = pa.DataFrame(output)
    otherdf = pa.DataFrame(other)
    diffdf = pa.DataFrame(diffs)

    print(otherdf.describe())
    #print(df.describe())
    #p.figure()
    #df.boxplot()
    #p.ylabel('True error')
    #p.title(jobhash[:6] + ' ' + db.get_description(jobhash) + ' ' + str(params))

    p_vec[i] = val
    #med_mat[i,:] = df.quantile()
    med_mat[i,:] = df.mean()
    std_mat[i,:] = df.std()
    med_diff_mat[i,:] = diffdf.quantile()
    std_diff_mat[i,:] = diffdf.std()

ind = np.argsort(p_vec)
p.figure()
p.plot(p_vec[ind], med_mat[ind,:], marker='o')
#p.plot(p_vec[ind], med_diff_mat[ind,:], marker='o')

#for i in xrange(comps):
    #p.errorbar(p_vec[ind]+np.random.randn(p_vec.size)/4., med_mat[ind,i], yerr=std_mat[ind,i], marker='o')
    
#for i in xrange(comps-1):
    #p.errorbar(p_vec[ind], med_diff_mat[ind,i], yerr=std_diff_mat[ind,i], marker='o')

p.legend(tuple(df.columns))
p.title(jobhash[:6] + ' ' + db.get_description(jobhash))
p.ylabel('True error expectation')
#p.xlabel('Number of final features')
p.xlabel('Training samples per class')
p.grid(True)
lx,ux,ly,uy = p.axis()
xl = ux - lx
yl = uy - ly
p.axis((lx-xl*0.1, ux+xl*0.1, ly-yl*0.1, uy+yl*0.1))

p.show()
