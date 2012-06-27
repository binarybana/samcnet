cimport csnet
from csnet cimport simParams, simResults
from libc.stdlib cimport malloc, free
from math import ceil

import networkx as nx

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
    params.limparent=5

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

    params.data_num=rows
    params.node_num=cols

    params.total_iteration=500000
    params.burnin=30000
    params.stepscale=100000


    #csnet.pyInitSimParams(rows, cols, <double*>refden.data, <int*>states.data, traindata.data)
    csnet.copyParamData(params, <double*>refden.data, <int*>states.data, traindata.data)
    cdef simResults *results = <simResults*> csnet.pyInitSimResults(params)


    ret = csnet.run(params, results)

    print("return code: ", ret)
    print("results.accept_loc: ", results.accept_loc)
    print("results.total_loc: ", results.total_loc)

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
