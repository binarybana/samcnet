# cython: profile=True
cimport csnet
cimport cython
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.math cimport exp
from math import ceil, floor
import tables as tb

import networkx as nx
import sys
import os

import numpy as np
cimport numpy as np

#cdef class T:
  #cdef public:
    #int i
    #double f
    #object o
    #object d
  #def __cinit__(self):
    #self.i = 3
    #self.d = {}


  #def test(self):
    #print self.i
    #print "Hello"

#class Child(T):
  #def __init__(self):
    #self.age = 45
    #self.name = "richard"

  #def test2(self):
    #self.test()
    #print "hello2"

cdef class SAMCRun:
  cdef public:
    object obj, db, refden, hist, indicator, mapvalue
    int lowEnergy, highEnergy, scale, grid, accept_loc, total_loc, iteration, burn, stepscale
    double rho, tau, mapenergy, delta,
  def __init__(self, obj, db='memory'):

    self.obj = obj # Going to be a BayesNet for now, but we'll keep it general

    if db == 'memory':
      self.db = [] # Place to store the samples
    else: 
      self.db = tb.openFile(db, mode='a', title='SAMCRun', filters = tb.Filters(complevel=4, complib='blosc'))
      self.obj.init_db(self.db)

    self.lowEnergy = 8000
    self.highEnergy = 12000
    self.scale = 1
    self.grid = ceil((self.highEnergy - self.lowEnergy) * self.scale)
    self.refden = np.arange(self.grid, 0, -1, dtype=np.double)
    self.refden = self.refden**2
    self.refden /= self.refden.sum()
    self.hist = np.zeros((3,self.grid), dtype=np.double)

    self.hist[0,:] = np.arange(self.lowEnergy, self.highEnergy, 1./self.scale)
    self.indicator = np.zeros((self.grid),dtype=np.int32) # Indicator is whether we have visited a region yet

    self.rho=1.0
    self.tau=1.0;
    self.accept_loc = 0
    self.total_loc = 0

    self.mapenergy = np.inf
    self.mapvalue = None
    self.iteration = 0
    self.delta = 1.0
    self.burn = 0
    self.stepscale = 100000

  cdef find_region(self, energy):
    if energy > self.highEnergy: 
      return self.grid-1
    elif energy < self.lowEnergy:
      return 0
    else: 
      return floor((energy-self.lowEnergy)*self.scale)

  @cython.boundscheck(False) # turn of bounds-checking for entire function
  def sample(self, iters, thin=1):
    cdef int current_iter, accept, oldregion, newregion, i
    cdef double oldenergy, newenergy, r
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] indicator = \
        self.indicator
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] locfreq = \
        np.zeros((self.grid,), dtype=np.int32)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] hist = \
        self.hist[1].copy()
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] refden = \
        self.refden
    oldenergy = self.obj.energy()
    oldregion = self.find_region(oldenergy) # AKA nonempty
    self.indicator[oldregion] = 1

    print("Initial Energy: %f" % oldenergy)

    try:
      for current_iter in range(self.iteration, self.iteration + iters):
        self.iteration += 1

        self.delta = float(self.stepscale) / max(self.stepscale, self.iteration)

        self.obj.propose()
        newenergy = self.obj.energy()
    
        ####### acceptance of new moves #########

        newregion = self.find_region(newenergy)

        self.indicator[newregion] = 1

        r = hist[oldregion] - hist[newregion] + (oldenergy-newenergy) #/self.temperature
        # I'm not sure how this follows from the paper (p. 868)

        
        #print("r:%f\t oldregion:%d\t hist[old]:%f\t hist[new]:%f fold:%f, fnew:%f" %
            #(r,oldregion, hist[1,oldregion], hist[1,newregion], oldenergy, newenergy))
        if r > 0.0 or np.random.rand() < exp(r):
          accept=1
        else:
          accept=0;

        if accept == 0:
          self.hist[2,oldregion] += 1.0
          self.obj.reject()
          self.total_loc += 1
        elif accept == 1:
          self.hist[2,newregion] += 1.0
          self.accept_loc += 1
          self.total_loc += 1
          oldregion = newregion
          oldenergy = newenergy

        if newenergy < self.mapenergy: # NB: Even if not accepted
          self.mapenergy = newenergy
          self.mapvalue = self.obj.copy()
           
        locfreq[oldregion] += 1
        hist += self.delta*(locfreq-refden)
        locfreq[oldregion] -= 1

        self.obj.save_to_db(self.db, oldenergy, hist[oldregion], oldregion)

        # VV I can't see what purpose this would serve from the paper (except
        # maybe the varying truncation?)
        #if self.iteration == self.burn:
          #i = 0
          #while i<=self.grid and results.indicator[i] == 0:
            #i+=1
          #results.nonempty = i
        #elif self.iteration > self.burnin:
          #pass
          #un = hist[1,results.nonempty]
          #for i in range(self.grid):
            #if results.indicator[i] == 1:
              #hist[1,i] -= float(un)
              
        if self.iteration % 10000 == 0:
          print("Iteration: %8d, delta: %5.2f, bestenergy: %7.2f, currentenergy: %7.2f" % \
              (self.iteration, self.delta, self.mapenergy, newenergy))
    except KeyboardInterrupt: 
      pass

    self.hist[1] = hist
    self.indicator = indicator

          
    ###### Calculate summary statistics #######
    
    #print(mat)
    #print(x)
    print("Accept_loc: %d" % self.accept_loc)
    print("Total_loc: %d" % self.total_loc)

cdef class BayesNet:
  cdef public:
    object nodes,states,data,graph,x,mat,fvalue,changelist
    object oldgraph, oldmat, oldx, oldfvalue, oldchangelist, table
    int limparent, data_num, node_num, changelength, oldchangelength, lastscheme
    double prior_alpha, prior_gamma
  cdef:
    int **cmat, **cdata
  def __init__(self, nodes, states, data):
    """
    nodes: a list of strings for the nodes
    states: a list of number of states for each node
    data: a matrix with each row being a draw from the Bayesian network 
      with each entry being [0..n_i-1]
    Initializes the BayesNet as a set of independent nodes
    """
    self.nodes = nodes
    self.states = np.asarray(states)
    self.data = np.asarray(data)

    self.graph = nx.DiGraph()
    self.graph.add_nodes_from(nodes)

    self.limparent = 4
    self.prior_alpha = 1.0
    self.prior_gamma = 0.1111

    self.data_num = data.shape[0]
    self.node_num = data.shape[1]

    self.x = np.arange(self.node_num, dtype=np.int32)
    np.random.shuffle(self.x) # We're going to make this a 0-9 permutation

    cdef int cols = self.node_num

    self.mat = np.eye(cols, dtype=np.int32)
    self.fvalue = np.zeros((cols,), dtype=np.double)
    self.changelist = np.arange(cols, dtype=np.int32)

    self.changelength = self.node_num
    self.cmat = npy2c_int(self.mat)
    self.cdata = npy2c_int(self.data)

    self.oldgraph = None
    self.oldmat = None
    self.lastscheme = -1

  def init_db(self, db):
    if '/samples' in db:
      self.table = db.getNode('/samples')
      if len(db.root.samples) > 0:
        self.mat = db.root.samples[-1]['matrix']
        self.cmat = npy2c_int(self.mat)
        self.x = np.arange(self.node_num, dtype=np.int32)
        self.changelength = self.node_num
        self.changelist = np.arange(cols, dtype=np.int32)
    else:
      dtype = {'matrix': tb.IntCol(shape=(self.node_num, self.node_num)),
          'energy': tb.Float64Col(),
          'theta': tb.Float64Col(),
          'region': tb.IntCol()}
      self.table = db.createTable('/', 'samples', description=dtype)

  def save_to_db(self, db, energy, theta, region):
    s = self.x.argsort()
    if type(db) == list:
      db.append({'matrix': self.mat[s].T[s].T,
        'energy': energy,
        'theta' : theta,
        'region': region})
    else:
      self.table.row['matrix'] = self.mat[s].T[s].T
      self.table.row['energy'] = energy
      self.table.row['theta'] = theta
      self.table.row['region'] = region
      self.table.row.append()
      db.flush()

  def from_adjacency(self):
    self.graph.clear()
    s = self.x.argsort()
    self.graph = nx.from_numpy_matrix(self.mat[s].T[s].T, create_using=nx.DiGraph())
    return self.graph

  def to_dot(self):
    nx.write_dot(self.from_adjacency(), '/tmp/graph.dot')

  def to_adjacency(self):
    return nx.to_numpy_matrix(self.graph)

  def copy(self):
    """ Not sure what I should actually do here... or if this is really needed """
    return (self.mat.copy(), self.x.copy()) # FIXME

  def energy(self):
    """ Calculate the -log probability. """
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] x = \
        self.x
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] states = \
        self.states
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] fvalue = \
        self.fvalue
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] changelist = \
        self.changelist
    self.cmat = npy2c_int(self.mat)
    energy = csnet.cost(self.node_num,
                self.data_num,
                self.limparent,
                <int*> states.data,
                self.cdata,
                self.prior_alpha,
                self.prior_gamma,
                <int*> x.data,
                self.cmat,
                <double*> fvalue.data,
                <int*> changelist.data,
                self.changelength)
    return energy

  def reject(self):
    """ Revert graph, mat, x, fvalue, changelist, and changelength. """
    #assert self.oldgraph != None
    #self.graph = self.oldgraph
    self.mat = self.oldmat
    self.x = self.oldx
    self.fvalue = self.oldfvalue
    #self.changelength = self.oldchangelength
    #self.changelist = self.oldchangelist

  def propose(self):
    """ 'Propose' a new network structure by backing up the old one and then 
    changing the current one. """

    cdef int i,j,i1,j1,i2,j2
    #self.oldgraph = self.graph.copy()
    self.oldmat = self.mat.copy()
    self.oldx = self.x.copy()
    self.oldfvalue = self.fvalue.copy()
    #self.oldchangelength = self.changelength
    #self.oldchangelist = self.changelist.copy()

    scheme = np.random.randint(1,4)   
    self.lastscheme = scheme

    if scheme==1: # temporal order change 
      k = np.random.randint(self.node_num-1)
      self.x[k], self.x[k+1] = self.x[k+1], self.x[k]
      self.changelist[0], self.changelist[1] = k, k+1
      self.changelength = 2

      for j in range(k+2, self.node_num):
        if self.mat[k,j]==1 or self.mat[k+1,j]==1:
          self.changelength += 1
          self.changelist[self.changelength-1] = j
        
    if scheme==2: # skeletal change

      i = np.random.randint(self.node_num)
      j = np.random.randint(self.node_num)
      while i==j:
        j = np.random.randint(self.node_num)
      if i<j:
        self.mat[i,j] = 1-self.mat[i,j]
        self.changelength=1
        self.changelist[0]=j
      else:
        self.mat[j,i]=1-self.mat[j,i]
        self.changelength=1
        self.changelist[0]=i

    if scheme==3: # Double skeletal change 
      cand = np.arange(self.node_num)
      np.random.shuffle(cand)
      if cand[0] < cand[1]:
        i1, j1 = cand[0], cand[1] #np.min(cand[:2]), np.max(cand[:2])
      else:
        i1, j1 = cand[1], cand[0] 

      if cand[-2] < cand[-1]:
        i2, j2 = cand[-2], cand[-1] #np.min(cand[-2:]), np.max(cand[-2:])
      else:
        i2, j2 = cand[-1], cand[-2] 

      self.mat[i1,j1]=1-self.mat[i1,j1]
      self.mat[i2,j2]=1-self.mat[i2,j2]

      #not truthful to original algorithm, here j1 can never == j2
      self.changelength=2
      self.changelist[0]=j1
      self.changelist[1]=j2

def test():
    traindata = np.loadtxt('../data/WBCD2.dat', dtype=np.int32)
    cols = traindata.shape[1]
    states = np.ones((cols,), dtype=np.int32)
    nodes = np.arange(cols)

    traindata[:,-1] -= 1

    states[:-1] = 10
    states[-1] = 2
    b = BayesNet(nodes,states,traindata)

    return b, SAMCRun(b,'db.h5')

cdef double **npy2c_double(np.ndarray a):
     cdef int m = a.shape[0]
     cdef int n = a.shape[1]
     cdef int i
     cdef double **data
     data = <double **> malloc(m*sizeof(double*))
     for i in range(m):
         data[i] = &(<double *>a.data)[i*n]
     return data

cdef np.ndarray c2npy_double(double **a, int n, int m):
     cdef np.ndarray[np.double_t,ndim=2]result = np.zeros((m,n),dtype=np.double)
     cdef double *dest
     cdef int i
     dest = <double *> malloc(m*n*sizeof(double*))	
     for i in range(m):
         memcpy(dest + i*n,a[i],m*sizeof(double*))
         free(a[i])
     memcpy(result.data,dest,m*n*sizeof(double*))
     free(dest)
     free(a)
     return result

cdef int **npy2c_int(np.ndarray a):
     cdef int m = a.shape[0]
     cdef int n = a.shape[1]
     cdef int i
     cdef int **data
     data = <int **> malloc(m*sizeof(int*))
     for i in range(m):
         data[i] = &(<int *>a.data)[i*n]
     return data

cdef np.ndarray c2npy_int(int **a, int n, int m):
     cdef np.ndarray[np.int32_t,ndim=2]result = np.zeros((m,n),dtype=np.int32)
     cdef int *dest
     cdef int i
     dest = <int *> malloc(m*n*sizeof(int*))	
     for i in range(m):
         memcpy(dest + i*n,a[i],m*sizeof(int*))
         free(a[i])
     memcpy(result.data,dest,m*n*sizeof(int*))
     free(dest)
     free(a)
     return result

