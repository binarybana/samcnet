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

np.random.seed(1234)
random.seed(1234)

N = 15
iters = 9e4
numdata = 100
priorweight=1

graph = generateHourGlassGraph(nodes=N)
template = sampleTemplate(graph, 5)
tmat = (np.asarray(nx.to_numpy_matrix(template))+0.5).clip(0,1)
tmat2 = np.asarray(nx.to_numpy_matrix(template))
traindata, states, cpds = generateData(graph,numdata)
nodes = np.arange(graph.number_of_nodes())

b = BayesNet(nodes,states,traindata)
b2 = BayesNet(nodes,states,traindata,tmat,priorweight=priorweight)
b3 = BayesNet(nodes,states,traindata,tmat2,priorweight=priorweight)

s = SAMCRun(b)
s2 = SAMCRun(b2)
s3 = SAMCRun(b3)

t1 = time()
s.sample(iters)
t2 = time()
s2.sample(iters)
t3 = time()
s3.sample(iters)

print("SAMC 1 took %f seconds" % (t2-t1))
print("SAMC 2 took %f seconds" % (t3-t2))

b.update_graph(s.mapvalue)
b2.update_graph(s2.mapvalue)
b3.update_graph(s3.mapvalue)
drawGraphs(graph, template, b.graph, b2.graph, b3.graph, show=True)

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

