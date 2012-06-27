# cython: profile=True
cimport csnet
from csnet cimport simParams, simResults
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.math cimport exp
from math import ceil, floor


import networkx as nx
import sys

import numpy as np
cimport numpy as np

cdef class BayesGraph:
  def __init__(self, nodes, states, data):
    """
    nodes: a list of strings for the nodes
    states: a list of number of states for each node
    data: a matrix with each row being a draw from the Bayesian network 
      with each entry being [0..n_i-1]
    Initializes the BayesGraph as a set of independent nodes
    """
    self.nodes = nodes
    self.states = states
    self.graph = nx.DiGraph()
    self.graph.add_nodes_from(nodes)

  def from_adjacency(self, mat):
    self.graph.clear()
    self.graph = nx.from_numpy_matrix(mat)
    assert len(self.nodes) == self.graph.number_of_nodes()

  def to_adjacency(self):
    return nx.to_numpy_matrix(self.graph)


def test():
    #/************** Data input  *********/
    #/*ins=fopen("data/WBCD2.dat","r");*/
    #/*if(ins==NULL){ printf("can't open datafile\n"); return 1; }*/
    #/*for(i=1; i<=params->data_num; i++){*/
        #/*for(j=1; j<=params->node_num-1; j++){ */
          #/*l = fscanf(ins," %d",&results->datax[i][j]); */
          #/*if(l!=1){ printf("Error reading data"); return 1; }*/
          #/*results->datax[i][j]-=1;*/
          #/*// subtract one to make the 9 features range from 0-9*/
        #/*}*/
        #/*l = fscanf(ins, " %d", &results->datax[i][params->node_num]);*/
        #/*if(l!=1){ printf("Error reading data"); return 1; }*/
        #/*results->datax[i][params->node_num]+=1; // I Assume this last entry is the label {1,2}*/
       #/*}*/
    #/*fclose(ins);*/
    #/*// print the data*/
    #/*[>for(i=1; i<=params->data_num; i++){ <]*/
        #/*[>for(j=1; j<=params->node_num; j++) printf(" %d",results->datax[i][j]);<]*/
        #/*[>printf("\n");<]*/
       #/*[>}<]*/
    #/***************************/

    cdef int i,iteration = 0

    cdef np.ndarray[np.int32_t, ndim=2, mode="c"] traindata = \
            np.loadtxt('../data/WBCD2.dat', dtype=np.int32)

    cdef int rows = traindata.shape[0]
    cdef int cols = traindata.shape[1]
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] states = \
            np.ones((cols,), dtype=np.int32)

    traindata[:,-1] -= 1

    states[:-1] = 10
    states[-1] = 2

    cdef simParams *params = <simParams *> malloc(sizeof(simParams))

    params.prior_alpha = 1.0
    params.prior_gamma=0.1111
    params.limparent=4

    params.maxEE = 25000.0
    params.lowE=8000.0
    params.scale=1.0
    params.range=0.0
    params.rho=1.0
    params.tau=1.0;
    params.temperature=1.0

    params.grid = ceil((params.maxEE + params.range - params.lowE) * params.scale)
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] refden = \
        np.ones((params.grid),dtype=np.double)
    refden /= refden.sum()

    params.data_num=rows
    params.node_num=cols

    params.total_iteration=50000
    params.burnin=1
    params.stepscale=25000 # t_0 in the papers

    csnet.copyParamData(params, <double*>refden.data, <int*>states.data, traindata.data)
    cdef simResults *results = <simResults*> csnet.pyInitSimResults(params)
    results.sze = floor((params.maxEE + params.range - params.lowE) * params.scale)

    cdef np.ndarray[np.double_t, ndim=2, mode="c"] hist = \
        np.zeros((params.grid,3), dtype=np.double)

    hist[:,0] = np.arange(params.lowE, params.maxEE, 1./params.scale)

    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] x = \
            np.arange(cols, dtype=np.int32)
    np.random.shuffle(x) # We're going to make this a 0-9 permutation

    cdef np.ndarray[np.int32_t, ndim=2, mode="c"] mat = \
        np.eye(cols, dtype=np.int32)
    #cdef np.ndarray[np.double_t, ndim=2, mode="c"] net = \
        #np.zeros((cols,cols), dtype=np.double)

    cdef np.ndarray[np.double_t, ndim=1, mode="c"] fvalue = \
            np.zeros((cols,), dtype=np.double)
    cdef np.ndarray[np.int32_t, ndim=1, mode="c"] changelist = \
            np.arange(cols, dtype=np.int32)
    cdef int changelength = cols

    cdef double** chist = npy2c_double(hist)
    cdef int** cmat = npy2c_int(mat)

    results.bestenergy = csnet.cost(params, results, <int*>x.data, cmat, <double*>fvalue.data, <int*>changelist.data, changelength)

    cdef int region = 0
    cdef double delta, totalweight = 0, sinw = 0.0

    for i in range(params.grid):
      results.indicator[i] = 0 # Indicator is whether we have visited a region yet

    results.accept_loc = results.total_loc = 0.0

    try:
      for iteration in range(params.total_iteration):
        delta = float(params.stepscale) / max(params.stepscale, iteration)
        csnet.metmove(params, results, <int*>x.data, cmat, <double*>fvalue.data, chist, delta, &region)

        if fvalue.sum() < results.bestenergy:
          results.bestenergy = fvalue.sum()
          #Here's we would also save bestx and bestmat if we cared

        if iteration == params.burnin:
          i = 0
          while i<=params.grid and results.indicator[i] == 0:
            i+=1
          results.nonempty = i

        elif iteration > params.burnin:
          pass
          un = hist[results.nonempty,1]
          for i in range(params.grid):
            if results.indicator[i] == 1:
              hist[i,1] -= float(un)
          sinw = exp(hist[region,1])
          totalweight += sinw
        if iteration%10000 == 0:
          print("Iteration: %8d, delta: %g, bestenergy: %g, weight: %g, totalweight: %g\n" % \
              (iteration, delta, results.bestenergy, sinw, totalweight))
          print(fvalue)
          print(changelist)
          print(mat)
          print(x)

    except KeyboardInterrupt: 
      pass

          
    # Calculate summary statistics

    #print(params.grid)
    #print(results.sze)
    #sys.exit(0)
    #ret = csnet.run(params, results)
    #print("return code: ", ret)
    
    print("results.accept_loc: ", results.accept_loc)
    print("results.total_loc: ", results.total_loc)
    print("results.calcs: ", results.calcs)

    csnet.freeSimResults(params, results)
    csnet.freeSimParams(params)


#cdef convert( numpy.ndarray[numpy.int32_t, ndim = 2] incomingMatrix ):
    ## Get the size of the matrix.
    #cdef unsigned int rows = <unsigned int>incomingMatrix.shape[0]
    #cdef unsigned int cols = <unsigned int>incomingMatrix.shape[1]

    ## Define a C-level variable to act as a 'matrix', and allocate some memory
    ## for it.
    #cdef float **data
    #data = <float **>malloc(rows*sizeof(int *))

    ## Go through the rows and pull out each one as an array of ints which can
    ## be stored in our C-level 'matrix'.
    #cdef unsigned int i
    #for i in range(rows):
        #data[i] = &(<int *>incomingMatrix.data)[i * cols]

    ## Call the C function to calculate the result.
    ##cdef float result
    ##result= cSumMatrix(rows, cols, data)

    ## Free the memory we allocated for the C-level 'matrix', and we are done.
    ##free(data)
    #return data

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

