import os
import sys
import redis
import random
import numpy as np

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

## First generate true distributions and data
cval, dist0, dist1 = gen_dists()

## Now test Gaussian Analytic calculation
gc = GaussianCls(dist0, dist1)

c = GaussianSampler(dist0,dist1)
s = samc.SAMCRun(c, burn=0, stepscale=1000, refden=1, thin=10)
s.sample(1e2, temperature=1)

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
