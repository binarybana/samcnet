# cython: profile=True
cimport csnet
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.math cimport exp
from math import ceil, floor


import networkx as nx
import sys

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

class SAMCRun:
  def __init__(self, obj, db):
    self.obj = obj # Going to be a BayesNet for now, but we'll keep it general
    self.db = db # Place to store the samples

    self.lowEnergy = 8000
    self.highEnergy = 25000
    self.scale = 1
    self.grid = ceil((self.highEnergy - self.lowEnergy) * self.scale)
    self.refden = np.ones((self.grid),dtype=np.double)
    self.refden /= self.refden.sum()
    self.hist = np.zeros((self.grid,3), dtype=np.double)

    self.hist[:,0] = np.arange(self.lowEnergy, self.highEnergy, 1./self.scale)
    self.sze = self.grid - 1 #floor((self.maxEE + self.range - self.lowE) * self.scale)
    self.indicator = np.zeros((self.grid),dtype=np.int) # Indicator is whether we have visited a region yet

    self.rho=1.0
    self.tau=1.0;
    self.accept_loc = 0
    self.total_loc = 0

    self.mapenergy = np.inf
    self.mapvalue = None
    self.iteration = 0
    self.delta = 0.0
    self.burn = 1000
    self.stepscale = 2000

  def find_region(self, energy):
    region = np.searchsorted(self.hist[:,0],energy)
    if region == self.hist.shape[0]:
      return region-1
    return region

  def sample(self, iters, thin=1):

    self.mapenergy = self.obj.energy()
    self.mapvalue = self.obj.copy()
    oldenergy = self.mapenergy
    oldregion = self.find_region(oldenergy) # AKA nonempty
    self.indicator[oldregion] = 1

    print("Initial Energy: %f" % self.mapenergy)

    try:
      for current_iter in range(self.iteration, self.iteration + iters):
        self.iteration += 1

        delta = float(self.stepscale) / max(self.stepscale, self.iteration)

        self.obj.propose()
        newenergy = self.obj.energy()
    
        ####### acceptance of new moves #########

        newregion = self.find_region(newenergy)

        self.indicator[newregion] = 1

        r = self.hist[oldregion,1] - self.hist[newregion,1] + (oldenergy-newenergy) #/self.temperature
        # I'm not sure how this follows from the paper (p. 868)

        if r > 0.0 or np.random.rand() < exp(r):
          accept=1
        else:
          accept=0;

        if accept == 0:
          self.hist[oldregion][2] += 1.0
          self.obj.reject()
          self.total_loc += 1
        elif accept == 1:
          self.hist[newregion][2] += 1.0
          self.accept_loc += 1
          self.total_loc += 1
          oldregion = newregion

        if newenergy < self.mapenergy: # NB: Even if not accepted
          # Update self.mapenergy and self.mapvalue
          self.mapenergy = newenergy
          self.mapvalue = self.obj.copy()
          print("Best energy result: %f" % newenergy)
           
        locfreq = np.zeros((self.grid),dtype=np.double)
        locfreq[oldregion] += 1

        self.hist[:,1] += self.delta*(locfreq-self.refden)
     

        # VV I can't see what purpose this would serve from the paper (except
        # maybe the varying truncation?)
        #if current_iter == self.burn:
          #i = 0
          #while i<=self.grid and results.indicator[i] == 0:
            #i+=1
          #results.nonempty = i
        #elif iteration > self.burnin:
          #pass
          #un = hist[results.nonempty,1]
          #for i in range(self.grid):
            #if results.indicator[i] == 1:
              #hist[i,1] -= float(un)
              
        if current_iter % 10000 == 0:
          print("Iteration: %8d, delta: %5.2g, bestenergy: %7.2g, currentenergy: %7.2g" % \
              (current_iter, self.delta, self.mapenergy, newenergy))

    except KeyboardInterrupt: 
      pass

          
    ###### Calculate summary statistics #######
    
    #print(mat)
    #print(x)
    print("Accept_loc: %d" % self.accept_loc)
    print("Total_loc: %d" % self.total_loc)
    #print("Calcs: %d" % results.calcs)

#cdef class CBayesNet:
  #cdef public:
    #int **cdata, **cmat
    #np.ndarray[np.int32_t, ndim=2, mode="c"] mat 
    #np.ndarray[np.int32_t, ndim=2, mode="c"] data
    #np.ndarray[np.double_t, ndim=1, mode="c"] fvalue 
    #np.ndarray[np.int32_t, ndim=1, mode="c"] changelist 

cdef class BayesNet:#(CBayesNet):
  cdef public:
    object nodes,states,data,graph,x,mat,fvalue,changelist
    object oldgraph, oldmat, oldx, oldfvalue, oldchangelist
    int limparent, data_num, node_num, changelength, oldchangelength
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
    self.states = states
    self.data = data

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

  def from_adjacency(self, mat):
    self.graph.clear()
    self.graph = nx.from_numpy_matrix(mat)
    assert len(self.nodes) == self.graph.number_of_nodes()

  def to_adjacency(self):
    return nx.to_numpy_matrix(self.graph)

  def copy(self):
    """ Not sure what I should actually do here... or if this is really needed """
    return self.graph # FIXME

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
    assert self.oldgraph != None
    self.graph = self.oldgraph
    self.mat = self.oldmat
    self.x = self.oldx
    self.fvalue = self.oldfvalue
    self.changelength = self.oldchangelength
    self.changelist = self.oldchangelist

  def propose(self):
    """ 'Propose' a new network structure by backing up the old one and then 
    changing the current one. """

    self.oldgraph = self.graph.copy()
    self.oldmat = self.mat.copy()
    self.oldx = self.x.copy()
    self.oldfvalue = self.fvalue.copy()
    self.oldchangelength = self.changelength
    self.oldchangelist = self.changelist.copy()

    scheme = np.random.randint(4)   

    if scheme==1: # temporal order change 
      k = np.random.randint(self.node_num)
      self.x[k], self.x[k+1] = self.x[k+1], self.x[k]
      self.changelist[0], self.changelist[1] = k, k+1
      self.changelength = 2

      for j in range(k+2, self.node_num):
        if self.mat[k,j]==1 or self.mat[k+1,j]==1:
          self.changelength += 1
          self.changelist[self.changelength-1] = j
        
    if scheme==2: # skeletal change

      i = np.random.randint(self.node_num)
      candidates = np.delete(np.arange(self.node_num),i)
      np.random.shuffle(candidates)
      j = candidates[0]
      if i<j:
        self.mat[i,j] = 1-self.mat[i,j]
        self.changelength=1
        self.changelist[0]=j
       
      else:
        self.mat[j,i]=1-self.mat[j,i]
        self.changelength=1
        self.changelist[0]=i

    if scheme==3: # Double skeletal change 
      candidates = np.arange(self.node_num)
      np.random.shuffle(candidates)
      i1, j1 = np.sort(candidates[:2])
      i2, j2 = np.sort(candidates[-2:])

      self.mat[i1,j1]=1-self.mat[i1,j1]
      self.mat[i2,j2]=1-self.mat[i2,j2]

      #not truthful to original algorithm, here j1 can never == j2
      self.changelength=2
      self.changelist[0]=j1
      self.changelist[1]=j2

def test():
    traindata = np.loadtxt('../data/WBCD2.dat', dtype=np.int32)
        #cdef int rows = traindata.shape[0]
    #cdef int cols = traindata.shape[1]
    #cdef np.ndarray[np.int32_t, ndim=1, mode="c"] states = \
            #np.ones((cols,), dtype=np.int32)

    #traindata[:,-1] -= 1

    #states[:-1] = 10
    #states[-1] = 2

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

