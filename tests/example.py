import pandas as pa

import samcnet.mh as mh
from samcnet.mixturepoisson import *

trn_data0 = pa.read_csv('tests/ex_data_0.csv', header=None)
trn_data1 = pa.read_csv('tests/ex_data_1.csv', header=None)
predict_samples = pa.read_csv('tests/ex_data_predict.csv', header=None)

dist0 = MPMDist(trn_data0)
dist1 = MPMDist(trn_data1)
mpm = MPMCls(dist0, dist1) 
mh = mh.MHRun(mpm, burn=1000, thin=50, verbose=True)
mh.sample(1e4)

print(mpm.predict(mh.db, predict_samples))
mh.db.close()
