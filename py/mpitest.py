#!/share/apps/bin/python
from mpi4py import MPI
import sys, shutil, os

sys.path.append('../build')

from samc import SAMCRun
from bayesnet import BayesNet
from generator import generateHourGlassGraph, generateData, sampleTemplate
from utils import *

import random
from time import time
import numpy as np

def runSAMC(states, data, template, db):
  nodes = np.arange(data.shape[1])
  b = BayesNet(nodes,states,data,template=template)
  s = SAMCRun(b,db)
  t1 = time()
  s.sample(iters)
  t2 = time()
  print("SAMC run took %f seconds." % (t2-t1))
  
  print s.estimate_func_mean(edge_presence)
  t3 = time()
  print("Mean estimation run took %f seconds." % (t3-t2))
  return b, s

def edge_presence(net):
    return net['matrix'][net['x'][0],net['x'][3]]

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

iter = int(sys.argv[1])
print("%d total iterations split %d ways so I'm doing %d iters." % 
        (iter, size, iter/size))

np.random.seed(1234)
random.seed(1234)

N = 5
iters = iter/size
numdata = 500
priorweight = 10
numtemplate = 3
#db = 'db.' + sys.argv[1] + '.h5'
db = 'memory'

graph = generateHourGlassGraph(nodes=N)

template = sampleTemplate(graph, numtemplate)
tmat = np.asarray(nx.to_numpy_matrix(template))
traindata, states, cpds = generateData(graph,numdata)
nodes = np.arange(graph.number_of_nodes())

b,s = runSAMC(states, traindata, tmat, db)

MPI.Finalize()
