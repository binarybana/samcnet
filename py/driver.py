import sys, shutil, os

sys.path.append('../build')

from samc import SAMCRun
from bayesnet import BayesNet
from generator import generateHourGlassGraph, generateData, sampleTemplate
from utils import *

import random
from time import time
import pygraphviz as pgv
import numpy as np
import pebl as pb
#import pstats, cProfile

#cProfile.runctx("b,s=test(); s.sample(10000)", globals(), locals(), "prof.stat")

#s = pstats.Stats("prof.stat")
#s.strip_dirs().sort_stats("time").print_stats()

def test():
    traindata = np.loadtxt('../data/WBCD2.dat', dtype=np.int32)
    cols = traindata.shape[1]
    states = np.ones((cols,), dtype=np.int32)
    nodes = np.arange(cols)

    traindata[:,:-1] -= 1
    traindata[:,-1] += 1

    states[:-1] = 10
    states[-1] = 2
    b = BayesNet(nodes,states,traindata)

    return b, SAMCRun(b)#'db.h5')

def edge_presence(net):
    return net['matrix'][net['x'][0],net['x'][3]]

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

def runFullSAMC(graph, iters, numtemplate, numdata, priorweight, db):
  #tmat = (np.asarray(nx.to_numpy_matrix(template))+0.5).clip(0,1)
  template = sampleTemplate(graph, numtemplate)
  tmat = np.asarray(nx.to_numpy_matrix(template))
  traindata, states, cpds = generateData(graph,numdata)
  nodes = np.arange(graph.number_of_nodes())
  b,s = runSAMC(states, traindata, tmat, db)
  return template, cpds, b, s

np.random.seed(1234)
random.seed(1234)

N = 5
iters = 3e5
numdata = 500
priorweight = 10
numtemplate = 3
db = 'memory'

graph = generateHourGlassGraph(nodes=N)

fname = 'trash.h5'
if os.path.exists(fname):
  os.remove(fname)

template = sampleTemplate(graph, numtemplate)
tmat = np.asarray(nx.to_numpy_matrix(template))
traindata, states, cpds = generateData(graph,numdata)
nodes = np.arange(graph.number_of_nodes())

b,s = runSAMC(states, traindata, tmat, db)
b2,s2 = runSAMC(states, traindata, tmat, db)

#temp2,_,b2,s2 = runFullSAMC(graph, iters, numtemplate, numdata, priorweight, 'memory')
#temp,_,b,s = runFullSAMC(graph, iters, numtemplate, numdata, priorweight, 'trash.h5')

#sys.exit()

b.update_graph(s.mapvalue)
b2.update_graph(s2.mapvalue)
drawGraphs(graph, template, b.graph, b2.graph)

#dataset = to_pebl(states, traindata)

#t1 = time()
#learner = pb.learner.simanneal.SimulatedAnnealingLearner(dataset, 
        #start_temp=10000, delta_temp=0.8)
##learner = pb.learner.simanneal.SimulatedAnnealingLearner(dataset, 
        ##pb.prior.UniformPrior(N), start_temp=10000, delta_temp=0.9)
#ex1result = learner.run()
#t2 = time()

#try:
    #shutil.rmtree('example-result')
#except:
    #pass
#ex1result.tohtml('example-result')

#print("Simulated Annealing took %f seconds" % (t2-t1))


#os.popen('xdg-open example-result/index.html > /dev/null')

