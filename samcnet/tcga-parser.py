import os
import sys
import glob
import urllib2

import pandas as pa
import numpy as np
import simplejson as js

from sklearn.feature_selection import SelectKBest, f_classif

def getuuid(fname):
    return os.path.basename(fname).split('.')[2]

def istumor(legend, fname):
    return int(legend[getuuid(fname)].split('-')[3][:-1]) < 10

def build_legend(path, globstring):
    uuids = []
    for fname in glob.glob(os.path.join(path,globstring)):
        uuids.append(os.path.basename(fname).split('.')[2])
    url = 'https://tcga-data.nci.nih.gov/uuid/uuidws/mapping/json/uuid/batch'
    req = urllib2.Request(url=url, headers={'Content-Type': 'text/plain'},data=','.join(uuids))
    data = js.loads(urllib2.urlopen(req).read())
    legend = {}
    for d in data['uuidMapping']:
        legend[d['uuid']] = d['barcode']
    return legend

def load_df(path, dtype):
    if dtype == 'norm':
        colname = 'normalized_count'
        globstring = '*_results'
    elif dtype == 'raw':
        colname = 'raw_count'
        globstring = '*.results'
    else: 
        raise Exception("Invalid data type requested")
    legend = build_legend(path,globstring)
    accum = []
    for fname in glob.glob(os.path.join(path,globstring)):
        if istumor(legend, fname):
            df = pa.read_csv(fname, sep='\t', index_col=0, usecols=['gene_id',colname])
            df.rename(columns={colname: getuuid(fname)}, inplace=True)
            accum.append(df)
    return pa.concat(accum, axis=1)


#### WRITE ####
#store = pa.HDFStore('store.h5', complib='blosc', complevel=6)

#store['lusc_norm'] = load_df('tcga-lusc','norm')
#store['lusc_raw'] = load_df('tcga-lusc','raw')

#store['luad_norm'] = load_df('tcga-luad','norm')
#store['luad_raw'] = load_df('tcga-luad','raw')

#store.close()
#sys.exit()
#### WRITE ####


#### READ ####
store = pa.HDFStore('store.h5')
#brca = store['brca_norm'] 
#paad = store['paad_norm'] 
luad = store['luad_norm'] 
lusc = store['lusc_norm'] 

#alldata = np.hstack(( paad_res, brca_all  )).T
#alllabels = np.hstack(( np.ones(paad_res.shape[1]), np.zeros(brca_all.shape[1]) ))

#somedata = np.hstack(( paad_res, brca_some  )).T
#somelabels = np.hstack(( np.ones(paad_res.shape[1]), np.zeros(brca_some.shape[1]) ))

#selector = SelectKBest(f_classif, k=4)
#selector.fit(somedata,somelabels)

##td = selector.transform(somedata)
#inds = selector.pvalues_.argsort()
#start = 8004 + np.random.randint(0,1000)
#td = alldata[:, inds[start:start+2]]

#import pylab as p
#p.figure()

#p.plot(td[40:,0], td[40:,1], 'r.')
#p.plot(td[:40,0], td[:40,1], 'g.')
#p.show()
#### READ ####

