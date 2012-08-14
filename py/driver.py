import sys, os, io
import redis
import sha
import atexit
import numpy as np
import networkx as nx
import ConfigParser as cp

from time import time, sleep

sys.path.append('../build')
sys.path.append('./build') # Yuck!

try:
  from samc import SAMCRun
  from bayesnet import BayesNet
  from generator import *
except ImportError as e:
  print(e)
  print("Make sure LD_LIBRARY_PATH is set correctly and that the build"+\
      " directory is populated by waf.")
  sys.exit()

r = redis.StrictRedis('knight-server.dyndns.org')

def getHost():
  return os.uname()[1].split('.')[0]

def recordDeath():
  r.lrem('clients-alive', 1, getHost())

atexit.register(recordDeath)
r.rpush('clients-alive', getHost())
print("Registered with db.")

while True:
  cmd = r.get('die')
  if cmd == 'all' or cmd == getHost():
    print("Received die command, shutting down.")
    break

  with r.pipeline() as pipe:
    queue = r.hgetall('desired-samples')
    workhash = None
    for h,num in queue.iteritems():
      if int(num) > 0:
        print("\nFound %s samples left on hash %s" % (num, h))
        workhash = h
    if workhash != None:
      r.hincrby('desired-samples', workhash, -1)

  if workhash == None:
    print 'sleep... '
    sleep(2)
    continue

  job = r.hget('configs', workhash)

  ########## Read config from Redis ########
  config = cp.RawConfigParser()
  config.readfp(io.BytesIO(job))

  N = config.getfloat('General', 'nodes')
  iters = config.getfloat('General', 'samc-iters')
  numdata = config.getint('General', 'numdata')
  priorweight = config.getfloat('General', 'priorweight')
  numtemplate = config.getint('General', 'numtemplate')

  ########### Actual simulation ############
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
  s = SAMCRun(b)

  t1 = time()
  s.sample(iters)
  t2 = time()
  print("SAMC run took %f seconds." % (t2-t1))
  func_mean = s.estimate_func_mean(global_edge_presence)
  t3 = time()
  print("Mean estimation run took %f seconds." % (t3-t2))
  ########### ################# ############

  # Send back func_mean to store
  r.lpush(workhash, func_mean)
  print('Function mean: %f' % func_mean)


