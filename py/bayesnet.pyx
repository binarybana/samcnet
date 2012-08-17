cimport csnet
cimport cython
from libc.stdlib cimport malloc, free, rand, RAND_MAX
from libc.string cimport memcpy
from libc.math cimport exp
from math import ceil, floor
import tables as tb

import networkx as nx
import sys
import os

import numpy as np
cimport numpy as np

cdef class BayesNet:
    cdef public:
        object nodes,states,data,graph,x,mat,fvalue,changelist,table
        object oldmat, oldx, oldfvalue
        object template
        int limparent, data_num, node_num, changelength, lastscheme
        double prior_alpha, prior_gamma
    cdef:
        int **cmat, **cdata
        double **ctemplate
    def __init__(self, nodes, states, data, template=None, priorweight=1.0):
        """
        nodes: a list of strings for the nodes
        states: a list of number of states for each node
        data: a matrix with each row being a draw from the Bayesian network 
            with each entry being [0..n_i-1]
        template: A matrix of doubles \in[0,1.0] giving strength to the various connections
        Initializes the BayesNet as a set of independent nodes
        """
        self.nodes = nodes
        self.states = np.asarray(states,dtype=np.int32)
        self.data = np.asarray(data,dtype=np.int32)

        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(nodes)

        self.limparent = 4
        self.prior_alpha = 1.0
        self.prior_gamma = priorweight

        self.data_num = data.shape[0]
        self.node_num = data.shape[1]

        if template == None:
            self.template = np.zeros((self.node_num, self.node_num), dtype=np.double)
        else:
            self.template = np.asarray(template.copy(), dtype=np.double)
            #self.template = np.asarray(nx.to_numpy_matrix(template.copy()), dtype=np.double) 

        np.fill_diagonal(self.template, 1.0) 
        self.ctemplate = npy2c_double(self.template)

        self.x = np.arange(self.node_num, dtype=np.int32)
        np.random.shuffle(self.x) # We're going to make this a 0-9 permutation

        cdef int cols = self.node_num

        self.mat = np.eye(cols, dtype=np.int32)
        self.fvalue = np.zeros((cols,), dtype=np.double)
        self.changelist = self.x.copy()

        self.changelength = self.node_num
        self.cmat = npy2c_int(self.mat)
        self.cdata = npy2c_int(self.data)

        self.lastscheme = -1

    #def __del__(self):
        #""" This should not be expected to run (google python's behavior
        #in __del__ to see why). But we really don't need it to. """
        #self.table.flush()

    #def __dealloc__(self):
        #""" Should deallocate self.cmat here, but since it is tied to 
        #c.mat, I don't really want to mess with it right now.
        #"""
        #pass

    def init_db(self, db):
        if '/samples' in db:
            self.table = db.getNode('/samples')
            if len(db.root.samples) > 0:
                self.mat = db.root.samples[-1]['matrix']
                self.cmat = npy2c_int(self.mat)
                self.x = db.root.samples[-1]['x']
                self.changelength = self.node_num
                self.changelist = self.x.copy()
        else:
            dtype = {'matrix': tb.IntCol(shape=(self.node_num, self.node_num)),
                    'x': tb.IntCol(shape=(self.node_num,)),
                    'energy': tb.Float64Col(),
                    'theta': tb.Float64Col(),
                    'region': tb.IntCol()}
            self.table = db.createTable('/', 'samples', description=dtype)

    def save_to_db(self, db, energy, theta, region):
        if type(db) == list:
            db.append({'matrix': self.mat,
                'x': self.x,
                'energy': energy,
                'theta' : theta,
                'region': region})
        else:
            self.table.row['matrix'] = self.mat#[s].T[s].T
            self.table.row['energy'] = energy
            self.table.row['theta'] = theta
            self.table.row['region'] = region
            self.table.row['x'] = self.x
            self.table.row.append()

    def update_graph(self, matx=None):
        """
        Update the networkx graph from either the current state, or pass 
        in a 2-tuple of (matrix,vector) with the adjacency matrix and the 
        node values.

        See self.update_matrix as well.
        """
        if matx:
            assert len(matx) == 2
            assert matx[0].shape == (self.node_num,self.node_num)
            assert matx[1].shape == (self.node_num,)
            assert matx[0].dtype == np.int32
            assert matx[1].dtype == np.int32
            self.mat = matx[0].copy()
            self.x = matx[1].copy()

        self.cmat = npy2c_int(self.mat)
        self.fvalue = np.zeros_like(self.fvalue)
        self.changelength = self.node_num
        self.changelist = self.x.copy()

        self.graph.clear()
        s = self.x.argsort()
        ordered = self.mat[s].T[s].T
        self.graph = nx.from_numpy_matrix(ordered - np.eye(self.node_num), create_using=nx.DiGraph())

    def update_matrix(self, graph):
        """ 
        From a networkx graph, update the internal representation of the graph
        (an adjacency matrix and node list).

        Also see self.update_graph.
        """
        assert graph.number_of_nodes() == self.node_num
        mat = np.array(nx.to_numpy_matrix(graph),dtype=np.int32)
        np.fill_diagonal(mat, 1)
        
        self.mat = mat.copy()
        self.x = np.arange(self.node_num, dtype=np.int32)
        self.cmat = npy2c_int(self.mat)
        self.fvalue = np.zeros_like(self.fvalue)
        self.changelength = self.node_num
        self.changelist = self.x.copy()
        self.graph = graph.copy()

    def to_dot(self):
        self.update_graph()
        nx.write_dot(self.graph, '/tmp/graph.dot')

    def to_adjacency(self):
        return nx.to_numpy_matrix(self.graph)

    def copy(self):
        return (self.mat.copy(), self.x.copy())

    def energy(self):
        """ 
        Calculate the -log probability. 
        """
        cdef double prior
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] x = \
                self.x
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] states = \
                self.states
        cdef np.ndarray[np.double_t, ndim=1, mode="c"] fvalue = \
                self.fvalue
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] changelist = \
                self.changelist
        self.cmat = npy2c_int(self.mat)
        energy = csnet.cost(
                self.node_num,
                self.data_num,
                self.limparent,
                <int*> states.data,
                self.cdata,
                self.prior_alpha,
                self.prior_gamma,
                self.ctemplate,
                <int*> x.data,
                self.cmat,
                <double*> fvalue.data,
                <int*> changelist.data,
                self.changelength)

        return energy

    def reject(self):
        """ Revert graph, mat, x, fvalue, changelist, and changelength. """
        self.mat = self.oldmat
        self.x = self.oldx
        self.fvalue = self.oldfvalue
        # IF I really wanted to be safe I would set changelngeth=10 and maybe
        # changelist

    def propose(self):
        """ 'Propose' a new network structure by backing up the old one and then 
        changing the current one. """

        cdef int i,j,i1,j1,i2,j2
        self.oldmat = self.mat.copy()
        self.oldx = self.x.copy()
        self.oldfvalue = self.fvalue.copy()

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
            i1=i2=0
            j1=j2=0
            while(i1==i2 and j1==j2):
                i1=0
                while(i1<0 or i1>self.node_num-1):
                    i1=floor(rand()*1.0/RAND_MAX*self.node_num)+1
                j1=i1
                while(j1<0 or j1>self.node_num-1 or j1==i1):
                    j1=floor(rand()*1.0/RAND_MAX*self.node_num)+1
                if(i1>j1):
                    k=i1
                    i1=j1
                    j1=k
                i2=0
                while(i2<0 or i2>self.node_num-1):
                    i2=floor(rand()*1.0/RAND_MAX*self.node_num)+1
                j2=i2
                while(j2<0 or j2>self.node_num-1 or j2==i2):
                    j2=floor(rand()*1.0/RAND_MAX*self.node_num)+1
                if(i2>j2):
                    k=i2
                    i2=j2
                    j2=k
              
            if(j1==j2):
                self.changelength=1
                self.changelist[1]=j1
            else:
                self.changelength=2
                self.changelist[1]=j1
                self.changelist[2]=j2

            self.mat[i1,j1]=1-self.mat[i1,j1]
            self.mat[i2,j2]=1-self.mat[i2,j2]

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

