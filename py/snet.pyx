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

cdef class SAMCRun:
  pass
  #def __cinit__(self, 
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
    params.stepscale=2000 # t_0 in the papers

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
        #csnet.metmove(params, results, <int*>x.data, cmat, <double*>fvalue.data, chist, delta, &region)
        csnet.metmove(params, results, x, mat, fvalue, hist, delta, &region)

        if fvalue.sum() < results.bestenergy:
          results.bestenergy = fvalue.sum()
          print("Best energy result: %f" % fvalue.sum())
          #Here's we would also save bestx and bestmat if we cared

        if iteration == params.burnin:
          i = 0
          while i<=params.grid and results.indicator[i] == 0:
            i+=1
          results.nonempty = i
        elif iteration > params.burnin:
          pass
          #un = hist[results.nonempty,1]
          #for i in range(params.grid):
            #if results.indicator[i] == 1:
              #hist[i,1] -= float(un)
          #sinw = exp(hist[region,1])
          #totalweight += sinw
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

        #csnet.metmove(params, results, x, mat, fvalue, hist, delta, &region)
cdef metmove (simParams *params, 
              simResults *results, 
              int x,
              int **mat,
              double *fvalue,
              double **hist,
              double delta,
              int *region):
#cdef metmove (simParams *params, 
              #simResults *results, 
              #int *x,
              #int **mat,
              #double *fvalue,
              #double **hist,
              #double delta,
              #int *region):

  cdef int k1, k2
  cdef double fnew

  cdef double fold = fvalue.sum()
  cdef double *newfvalue = fvalue[:]
   newfvalue=dvector(0,params->node_num-1);
  cdef int *newx = x[:]
   newx=ivector(0,params->node_num-1);
  cdef int **newmat = mat[:]
   newmat=imatrix(0,params->node_num-1,0,params->node_num-1);

  cdef int *changelist = <int *> malloc(params.node_num*sizeof(int *))	
   changelist=ivector(0,params->node_num-1);


  if fold > params.maxEE:
    k1 = results.sze
  elif fold < params.lowE:
    k1 = 0
  else:
    k1 = floor((fold-params.lowE)*params.scale)

  scheme = np.random.randint(4)   

  if scheme==1: # temporal order change 
    k = np.random.randint(params.node_num)
    newx[k], newx[k+1] = x[k+1], x[k]
    changelist[0], changelist[1] = k, k+1
    changelength = 2

    for j in range(k+2, params.node_num):
      if newmat[k][j]==1 or newmat[k+1][j]==1:
        changelength += 1
        changelist[changelength-1] = j
      
    fnew = csnet.cost(params,results,newx,newmat,newfvalue,changelist,changelength);

  if scheme==2: # skeletal change

    i = np.random.randint(params.node_num)
    candidates = np.arange(params.node_num).remove(i)
    np.random.shuffle(candidates)
    j = candidates[0]
    if i<j:
      newmat[i][j] = 1-newmat[i][j]
      changelength=1
      changelist[0]=j
     
    else:
      newmat[j][i]=1-newmat[j][i]
      changelength=1
      changelist[0]=i

    fnew = csnet.cost(params,results,newx,newmat,newfvalue,changelist,changelength);
   
    if scheme==3: # Double skeletal change 
      candidates = np.arange(params.node_num)
      np.random.shuffle(candidates)
      i1, j1 = np.sort(candidates[:2])
      i2, j2 = np.sort(candidates[-2:])

      newmat[i1][j1]=1-newmat[i1][j1]
      newmat[i2][j2]=1-newmat[i2][j2]

      #not truthful to original algorithm, here j1 can never == j2
      changelength=2
      changelist[0]=j1
      changelist[1]=j2
    
      fnew = csnet.cost(params,results,newx,newmat,newfvalue,changelist,changelength);
 
    
    # acceptance of new moves 

    if fnew > params.maxEE + params.range:
      k2 = results.sze-1
    elif fnew<params.lowE:
      k2 = 0
    else:
      k2 = floor((fnew-params.lowE)*params.scale)

    results.indicator[k2]=1

    r = hist[k1,1]-hist[k2,1]+(fold-fnew)/params.temperature

    if r > 0.0 or np.random.rand() < exp(r):
      accept=1
    else:
      accept=0;

    if accept==1:
      k1=k2
      fold=fnew # Codesmell FIXME: fold not used after this point
      x = newx
      fvalue = newfvalue
      mat = newmat
      hist[k2][2] += 1.0
      results.accept_loc+=1.0
      results.total_loc+=1.0
    else:
      hist[k1][2] += 1.0

      results.total_loc+=1.0

    if accept == 1 and fnew < results.bestenergy:
       results.bestenergy = fnew
       for i in range(params.node_num):
         results.bestx = x
         results.bestfvalue = fvalue
         results.bestmat = mat
       
    region[0]=k1


    for i in range(results.sze):
      if results.indicator[i] == 1:
        hist[i][1]+=delta*(hist[i][2]-params.refden[i])

    free(changelist)
         
