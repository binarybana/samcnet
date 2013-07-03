import os
import sys
import redis
import random
import numpy as np
import yaml

try:
    from samcnet import samc,lori,utils
    from samcnet.lori import *
except ImportError as e:
    sys.exit("Make sure LD_LIBRARY_PATH is set correctly and that the build"+\
            " directory is populated by waf.\n\n %s" % str(e))

if 'WORKHASH' in os.environ:
    try:
        server = os.environ['SERVER']
    except:
        sys.exit("ERROR in worker: Need SERVER environment variable defined.")

if 'PARAM' in os.environ:
    params = yaml.load(os.environ['PARAM'])
    iters = int(params['iters'])

## First generate true distributions and data
seed = 40767
np.random.seed(seed)
cval, dist0, dist1 = gen_dists()
np.random.seed()

## Now test Gaussian Analytic calculation
gc = GaussianCls(dist0, dist1)

c = GaussianSampler(dist0,dist1)
s = samc.SAMCRun(c, burn=0, stepscale=1000, refden=1, thin=10)
s.sample(iters, temperature=1)

# Now save extra info to the database
s.db.root.object.objfxn._v_attrs['seed'] = seed
s.db.root.object.objfxn._v_attrs['bayes_error'] = gc.approx_error("bayes", cval)
s.db.root.object.objfxn._v_attrs['posterior_error'] = gc.approx_error("true", cval)
s.db.root.object.objfxn._v_attrs['sample_error_full'] = c.approx_error(s.db, cval)
s.db.root.object.objfxn._v_attrs['sample_error_20'] = c.approx_error(s.db, cval, partial=20)

if 'WORKHASH' in os.environ:
    import zmq,time
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.connect('tcp://'+server+':7000')

    data = s.read_db()
    socket.send(os.environ['WORKHASH'], zmq.SNDMORE)
    socket.send(data)
    socket.recv()
    socket.close()
    ctx.term()

s.db.close()
