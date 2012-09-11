cimport bayesnet
cimport cython

import numpy as np
cimport numpy as np

from probability import JointDistribution, CPD

cdef class BayesNetCPD(BayesNet):
    """To initialize a BayesNetCPD we need:
    
    ==Required==
    nodes: Node names.
    states: Arities.
    data: For -logP calculations.

    ==Optional prior information==
    priorweight: A float used in the -logP calculations.
    template: An adjacency matrix over the nodes with float values from [0,1]

    With all of these parameters, the network will be initialized with no
    interconnections and all nodes assigned a uniform probability over their 
    arities.

    """
    cdef public:
      object joint
    def __init__(self, nodes, states, data, template=None, priorweight=1.0):
        BayesNet.init(self, nodes, states, data, template, priorweight)

        self.joint = JointDistribution()
        for name,arity in zip(nodes,states):
            self.joint.add_distribution(CPD(name, arity, {(), np.ones(arity-1)/float(arity)))
        
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

        TODO: What should we do about CPDs here? Uniform again?

        Also see self.update_graph
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

    cpdef energy(self):
        """ 
        Calculate the -log probability. 
        """
        cdef float sum = 0.0
        cdef int i,node
        
        for i in range(self.changelength):
            node = changelist[i]


        #priordiff = 0.0;
        #for(i=0; i<node_num; i++){
            #for(j=0; j<node_num; j++){
                #if(j!=i){
                    #priordiff += abs((double)mat[j][i] - priormat[x[j]][x[i]]);
                #}
            #}
        #}
        #sum += (priordiff)*prior_gamma;


        #energy = csnet.cost(
                #self.node_num,
                #self.data_num,
                #self.limparent,
                #<int*> states.data,
                #self.cdata,
                #self.prior_alpha,
                #self.prior_gamma,
                #self.ctemplate,
                #<int*> x.data,
                #self.cmat,
                #<double*> fvalue.data,
                #<int*> changelist.data,
                #self.changelength)

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

    #def add_edge(self,n1,n2):
        ##check order
        ##add edge
        ##adjust parents
        #pass
    #def rem_edge(self,n1,n2):
        ##remove edge
        ##adjust parents
        #pass
        if scheme==1: # temporal order change 
        if scheme==2: # skeletal change
        if scheme==3: # Double skeletal change 

