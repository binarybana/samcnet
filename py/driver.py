import sys, shutil

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

np.random.seed(1234)
random.seed(1234)

N = 15
graph = generateHourGlassGraph(nodes=N)
traindata, states, cpds = generateData(graph, 50)
nodes = np.arange(graph.number_of_nodes())


b = BayesNet(nodes,states,traindata)
t1 = time()
s = SAMCRun(b)
s.sample(6e4)
t2 = time()

print("SAMC took %f seconds" % (t2-t1))

dataset = to_pebl(states, traindata)

t1 = time()
learner = pb.learner.simanneal.SimulatedAnnealingLearner(dataset, 
        start_temp=10000, delta_temp=0.9)
#learner = pb.learner.simanneal.SimulatedAnnealingLearner(dataset, 
        #pb.prior.UniformPrior(N), start_temp=10000, delta_temp=0.9)
ex1result = learner.run()
t2 = time()

try:
    shutil.rmtree('example-result')
except:
    pass
ex1result.tohtml('example-result')

print("Simulated Annealing took %f seconds" % (t2-t1))

b.update_graph(s.mapvalue)
before_and_after(graph, b.graph)

