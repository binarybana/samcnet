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

p.close('all')

output = defaultdict(list)
other = defaultdict(list)
diffs = defaultdict(list)

for mydir in os.listdir(resdir):
    params = yaml.load(db.get_params(mydir))
    if len(params.values()) != 0: 
        continue
    for fname in os.listdir(os.path.join(resdir,mydir)):
        data = js.loads(open(os.path.join(resdir,mydir,fname)).read())
        print(data['errors'])
        for k,v in data.iteritems():
            if k == 'errors':
                mpmerr = data['errors']['mpm'] 
                for kk,vv in data['errors'].iteritems():
                    output[kk].append(vv)
                    if kk != 'mpm':
                        diffs[kk].append((vv-mpmerr))
            else: 
                other[k].append(v)

df = pa.DataFrame(output)
otherdf = pa.DataFrame(other)
diffdf = pa.DataFrame(diffs)

print(otherdf.describe())
df.boxplot()
p.figure()
#diffdf.boxplot()
key = {'gauss':'Normal OBC', 'svm':'SVM', 'knn':'3NN', 'lda':'LDA', 'mpm':'MP OBC',
        'mpm_prior':'MP OBC Prior'}

ind = np.arange(len(df.columns))
width = 0.7
p.grid(True)
p.bar(ind, df.mean(), width=width)
meanmin, meanmax = np.min(df.mean()), np.max(df.mean())
spread = meanmax-meanmin
p.ylim(meanmin-0.8*spread, meanmax+0.8*spread)
#p.bar(ind, df.mean(), width=width, yerr=df.std())
p.gca().set_xticks(ind+width/2.)
p.gca().set_xticklabels([key[x] for x in df.columns])
p.ylabel('Mean True error')
#p.title(jobhash[:6] + ' ' + db.get_description(jobhash) + ' ' + str(params))

p.show()
