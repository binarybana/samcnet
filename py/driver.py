#!/share/apps/bin/python
from mpi4py import MPI
import sys, shutil, os, sha
import ConfigParser as cp

sys.path.append('./build')

from samc import SAMCRun
from bayesnet import BayesNet
from generator import generateHourGlassGraph, generateData, sampleTemplate
from utils import *

import random
from time import time
import numpy as np
import redis

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if len(sys.argv) != 3:
    sys.exit("Usage: driver.py <configfile.cfg> <num of runs>")

config = cp.RawConfigParser()
print("Reading configuration information from %s." % sys.argv[1])
config.read(sys.argv[1])

N = config.getfloat('General', 'nodes')
iters = config.getfloat('General', 'samc-iters')
numdata = config.getint('General', 'numdata')
priorweight = config.getfloat('General', 'priorweight')
numtemplate = config.getint('General', 'numtemplate')
db = config.get('General', 'db')

#def indiv_edge_presence(net):
  #return net['matrix'][net['x'][0],net['x'][3]]

def calculate_mean():
    graph = generateHourGlassGraph(nodes=N)
    gmat = np.asarray(nx.to_numpy_matrix(graph))

    def global_edge_presence(net):
        s = net['x'].argsort()
        ordmat = net['matrix'][s].T[s].T
        return np.abs(gmat - ordmat).sum() / net['x'].shape[0]**2

    template = sampleTemplate(graph, numtemplate)
    tmat = np.asarray(nx.to_numpy_matrix(template))
    traindata, states, cpds = generateData(graph,numdata)
    nodes = np.arange(graph.number_of_nodes())

    nodes = np.arange(traindata.shape[1])
    b = BayesNet(nodes,states,traindata,template=tmat)
    s = SAMCRun(b,db)

    t1 = time()
    s.sample(iters)
    t2 = time()

    print("SAMC run took %f seconds." % (t2-t1))

    func_mean = s.estimate_func_mean(global_edge_presence)
    t3 = time()
    print("Mean estimation run took %f seconds." % (t3-t2))

    # Send back func_mean to store
    return func_mean

#meta_iters = config.getint('General', 'meta_iters')
meta_iters = int(sys.argv[2])

configtext = open(sys.argv[1], 'r').read()
hash = sha.sha(configtext).hexdigest()
basename = os.path.basename(sys.argv[1])
#runtop = 'runs/'+basename
#runsafe = 'runs/'+basename+'/'+hash

#try:
    #os.mkdir(runtop)
    #os.mkdir(runsafe)
#except:
    #pass

#shutil.copy(sys.argv[1], runsafe)

r = redis.StrictRedis(host='kubera.local')

if rank == 0:
    r.hset('key-fname', hash, basename)
    r.hset('key-config', hash, configtext)
    r.hset('key-reverse', basename, hash)
   
#Calculate how many meta_iters we need
if rank < (meta_iters % size):
    local_meta_iters = meta_iters / size + 1
else:
    local_meta_iters = meta_iters / size


for i in range(local_meta_iters):
    r.lpush(hash, calculate_mean())

MPI.Finalize()
