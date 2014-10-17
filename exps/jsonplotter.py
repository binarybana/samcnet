import os, sys
import numpy as np
import simplejson as js
import pandas as pa
import yaml

import matplotlib as mpl
mpl.use('Agg')
import pylab as p

from collections import defaultdict

from jobmon import redisbackend as rb

p.rc('font', size=11)

db = rb.RedisDataStore('localhost')
jobhash = db.select_jobfile()
name = raw_input("Name: ")

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
ind = np.argsort(dependent_var)#[1:]
markers = list('xD*o>s^<+')
colors = list('bgrcmy')
offset = 0
key = {'gauss':'Normal OBC', 'svm':'RBF SVM', 'knn':'3NN', 'lda':'LDA', 'mpm':'MP OBC',
        'mpm_prior':'MP OBC Prior', 'mpmc_calib':'MP OBC Calibrated',
        'nngauss': 'NN Normal OBC', 'nnsvm': 'NN RBF SVM', 'nnlda': 'NN LDA', 'nnknn': 'NN 3NN'}
colorkey = {'gauss':'b', 'svm':'y', 'knn':'g', 'lda':'r', 'mpm':'c',
        'mpm_prior':'k', 'mpmc_calib':'m',
        'nngauss':'b', 'nnsvm': 'y', 'nnlda': 'r', 'nnknn': 'g'}
symkey = {'gauss':'x', 'svm':'^', 'knn':'D', 'lda':'*', 'mpm':'o',
        'mpm_prior':'+', 'mpmc_calib':'s',
        'nngauss':'o', 'nnsvm': 'x', 'nnlda': '^', 'nnknn': '+'}

def adjust_plot():
    lx,ux,ly,uy = p.axis()
    xl = ux - lx
    yl = uy - ly
    p.axis((lx-xl*0.1, ux+xl*0.1, ly-yl*0.1, uy+yl*0.1))

#p.close('all')
df.sort_index(axis=1)

offset = 0
p.figure()
for i in range(len(df.columns)):
    if df.columns[i] == 'mpm_prior':# or df.columns[i] == 'gauss':
        continue
    means = [alldata[j].mean()[i] for j in dependent_var[ind]]
    stds = [alldata[j].std()[i] for j in dependent_var[ind]]
    p.plot(dependent_var[ind], means, marker=symkey[df.columns[i]], color=colorkey[df.columns[i]], 
            markersize=7, label=key[df.columns[i]])
    #p.errorbar(dependent_var[ind]+offset, means, yerr=stds, marker='o',
            #label=key[df.columns[i]])
    #offset += 0.1
p.legend()
#p.ylim(0.30, 0.40) # For TCGA
#p.ylim(0.40, 0.46) # For HC
p.ylabel('Mean separate sampling holdout error')
#p.ylabel('Mean estimated true error')
p.xlabel('Training samples per class')
#p.xlabel('Total training samples')
#p.xlabel(r'$\mu_1$')
#p.xlabel('Gene expression \"strength\"')
p.grid(True)
adjust_plot()

savename = '../class-paper/pdf/{}-{}.pdf'.format(name, jobhash[:4])
print('Saving to {}'.format(savename))
p.savefig(savename, bbox_inches='tight')

#p.show()
sys.exit()

p.figure()
for i in range(len(df.columns)):
    if df.columns[i] == 'mpm' or df.columns[i] == 'mpm_prior':
        continue
    diffs = [alldata[j].iloc[:,i]-alldata[j].loc[:,'mpm'] for j in dependent_var[ind]]
    diffdf = pa.concat(diffs, keys=dependent_var[ind], axis=1)
    p.errorbar(dependent_var[ind]+offset, diffdf.mean(), yerr=diffdf.std(), marker='o',
            label=key[df.columns[i]])
    offset += 0.01
p.legend()
p.ylabel('Holdout error - Holdout error for MP OBC')
p.xlabel('Total training samples per class')
p.grid(True)
adjust_plot()

#p.title(jobhash[:6] + ' ' + db.get_description(jobhash))
#p.xlabel('Number of final features')
#p.xlabel('mu1')


p.show()
