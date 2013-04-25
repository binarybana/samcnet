from bayesnet cimport BayesNet
from dai_bind cimport FactorGraph, Var, VarSet, JTree
from libcpp.vector cimport vector

cdef class BayesNetCPD:
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
        # From Bayesnet
        object states,data,x,mat,changelist,fvalue
        object oldmat, oldx, oldfvalue
        int limparent, node_num, changelength

        double logqfactor # For RJMCMC weighting of the acceptance probability
        double memo_entropy
        object dirty
    cdef:
        vector[Var] pnodes
        vector[vector[ulong]] pdata
        FactorGraph fg
        JTree jtree
        int convert(BayesNetCPD self, int node, vector[ulong] state)
        int convert_separate(BayesNetCPD self, int node, int state, int parstate)
