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
dependent_var = []
alldata = {}

for i,paramdir in enumerate(os.listdir(resdir)):
    output = defaultdict(list)
    diffs = defaultdict(list)
    other = defaultdict(list)
    params = yaml.load(db.get_params(paramdir))
    if len(params.values()) != 1: # ie: make sure this is a univariate sweep
        continue
    val = params.values()[0]
    dependent_var.append(val)
    for fname in os.listdir(os.path.join(resdir,paramdir)):
        data = js.loads(open(os.path.join(resdir,paramdir,fname)).read())
        for k,v in data.iteritems():
            if k == 'errors':
                for kk,vv in data['errors'].iteritems():
                    output[kk].append(vv)
            else: 
                other[k].append(v)

    df = pa.DataFrame(output)
    alldata[val] = df
    otherdf = pa.DataFrame(other)
    print(otherdf.describe())

dependent_var = np.array(dependent_var)
ind = np.argsort(dependent_var)
markers = list('xD*o>s^<+')
offset = 0
key = {'gauss':'Normal OBC', 'svm':'SVM', 'knn':'3NN', 'lda':'LDA', 'mpm':'MP OBC',
        'mpm_prior':'MP OBC Prior', 'mpmc_calib':'MP OBC Calibrated'}

def adjust_plot():
    lx,ux,ly,uy = p.axis()
    xl = ux - lx
    yl = uy - ly
    p.axis((lx-xl*0.1, ux+xl*0.1, ly-yl*0.1, uy+yl*0.1))

#p.close('all')

offset = 0
p.figure()
for i in range(len(df.columns)):
    means = [alldata[j].mean()[i] for j in dependent_var[ind]]
    stds = [alldata[j].std()[i] for j in dependent_var[ind]]
    p.plot(dependent_var[ind], means, marker=markers[i], markersize=10, label=key[df.columns[i]])
    #p.errorbar(dependent_var[ind]+offset, means, yerr=stds, marker='o',
            #label=key[df.columns[i]])
    #offset += 0.1
p.legend()
p.ylabel('Mean holdout error')
p.xlabel('Training samples per class')
#p.xlabel(r'$\mu_1$')
#p.xlabel('Gene expression \"strength\"')
p.grid(True)
adjust_plot()

#p.figure()
#for i in range(len(df.columns)):
    #if df.columns[i] == 'mpm':
        #continue
    #diffs = [alldata[j].iloc[:,i]-alldata[j].loc[:,'mpm'] for j in dependent_var[ind]]
    #diffdf = pa.concat(diffs, keys=dependent_var[ind], axis=1)
    #p.errorbar(dependent_var[ind]+offset, diffdf.mean(), yerr=diffdf.std(), marker='o',
            #label=key[df.columns[i]])
    #offset += 0.01
#p.legend()
#p.ylabel('Holdout error - Holdout error for MP OBC')
#p.xlabel('Training samples per class')
#p.grid(True)
#adjust_plot()

p.title(jobhash[:6] + ' ' + db.get_description(jobhash))
#p.xlabel('Number of final features')
#p.xlabel('mu1')


p.show()
