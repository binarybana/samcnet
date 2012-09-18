# cython: profile=False
cimport cython

from bayesnet cimport BayesNet
from probability cimport JointDistribution,CPD,fast_space_iterator,GroundNet

import numpy as np
cimport numpy as np

from collections import Counter
from math import lgamma, log

cdef class MemoCounter:
    def __init__(self, data):
        self.memo_table = {}
        self.data = data

    def lookup(self, config):
        if config in self.memo_table:
            return self.memo_table[config]
        else:
            c = Counter(tuple(x[np.array(config,dtype=np.int)]) for x in self.data)
            self.memo_table[config] = c
            return c


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
    def __cinit__(self, *args, **kwargs):
        pass
    
    def __init__(self, nodes, states, data, intemplate=None, priorweight=1.0):
        BayesNet.__init__(self, nodes, states, data, intemplate, priorweight)

        self.joint = JointDistribution()
        for name,arity in zip(nodes,states):
            self.joint.add_distribution(CPD(name, arity, {(): np.ones(arity-1)/float(arity)}))

        self.logqfactor = 0.0
        self.memo_table = MemoCounter(data)
        
    def update_graph(self, matx=None):
        """
        Update the networkx graph from either the current state, or pass 
        in a 2-tuple of (matrix,vector) with the adjacency matrix and the 
        node values.

        See self.update_matrix as well.
        """
        raise Exception("Not Implemented")

    def update_matrix(self, graph):
        """ 
        From a networkx graph, update the internal representation of the graph
        (an adjacency matrix and node list).

        TODO: What should we do about CPDs here? Uniform again?

        Also see self.update_graph
        """

        raise Exception("Not Implemented")

    def copy(self):
        #return (self.mat.copy(), self.x.copy(), {k:v.copy() for k,v in self.joint.dists.iteritems()})
        return (self.mat.copy(), self.x.copy(), self.joint.copy())

    def save_to_db(self, object db, double theta, double energy, int i, GroundNet ground_truth):
        func = ground_truth.kld(self.joint)
        assert db is not None, 'DB None when trying to save sample.'
        db[i] = np.array([theta, energy, func])

    @cython.boundscheck(False)
    def energy(self):
        """ 
        Calculate the -log probability. 
        """
        cdef float alphaik,alphaijk,sum,priordiff,accum = 0.0
        cdef int i,node,j,l,parstate
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] x = \
                self.x
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] states = \
                self.states
        cdef np.ndarray[np.double_t, ndim=2, mode="c"] ntemplate = \
                self.ntemplate
        cdef np.ndarray[np.int32_t, ndim=2, mode="c"] mat = \
                self.mat

        lut = self.x.argsort()
        
        for i in range(self.changelength):
            node = self.changelist[i]
            cpd = self.joint.dists[x[node]]
            pdomain = cpd.parent_domain

            #print("NEW ENERGY LOOP:")
            #print self.joint
            #print(node, self.x[node], cpd.name, cpd.params, self.changelist, pdomain, id(pdomain))

            par_node_cols = lut[pdomain.keys() + [x[node]]]
            node_par_counts = self.memo_table.lookup(tuple(par_node_cols))

            parstate = 1
            for j in range(0,node):
                if mat[j,node]==1:
                   parstate *= states[x[j]]

            alphaijk=self.prior_alpha/parstate/states[x[node]]
            alphaik=self.prior_alpha/parstate
            
            accum = 0.0
            accum -= states[x[node]]*parstate*lgamma(alphaijk)
            accum += parstate*lgamma(alphaik)

            for pval in fast_space_iterator(pdomain):
                fastp = cpd.fastp(pval)
                for j in range(0,cpd.arity-1):
                    #print j, fastp, pval, pdomain, alphaijk, node_par_counts
                    accum += (node_par_counts[pval + (j,)] + alphaijk -1) * log(fastp[j])
                accum += (node_par_counts[pval + (cpd.arity-1,)] + alphaijk -1) * log((1 - fastp.sum()))

            self.fvalue[i] = accum;
            self.fvalue[i] *= -1.0;

        sum = self.fvalue.sum()

        priordiff = 0.0
        for i in range(self.node_num):
            for j in range(self.node_num):
                if(j!=i):
                    priordiff += abs(mat[j,i] - ntemplate[x[j],x[i]])
        #print "energy: marginal likelihood %f; struct prior %f; qfactor %f" % (sum, priordiff*self.prior_gamma, self.logqfactor)
        sum += (priordiff)*self.prior_gamma
        sum -= self.logqfactor #TODO Check this negative sign

        return sum

    def reject(self):
        """ Revert graph, mat, x, fvalue, changelist, and changelength. """
        #print("REJECTING")
        self.mat = self.oldmat
        self.x = self.oldx
        self.fvalue = self.oldfvalue
        self.joint.dists.update(self.jointchanges)
        # TODO Do I need to reset logqfactor here?
        # If I really wanted to be safe I would set changelngeth=10 and maybe
        # changelist

    def propose(self):
        """ 'Propose' a new network structure by backing up the old one and then 
        changing the current one. """
        #print "#############################################"
        #print("PROPOSING")
        #print self.joint

        cdef int i,j,i1,j1,i2,j2,edgedel
        self.oldmat = self.mat.copy()
        self.oldx = self.x.copy()
        self.oldfvalue = self.fvalue.copy()
        self.jointchanges = {}

        scheme = np.random.randint(1,4)   

        if scheme==1: # temporal order change 
            k = np.random.randint(self.node_num-1)
            
            #print "k: %d" % k
            #print np.arange(self.node_num)
            #print self.x
            #print ""
            #print self.mat
            #s = self.x.argsort()
            #print ""
            #print self.mat[s].T[s].T
            
            # Change parameters:
            self.logqfactor = 0.0

            #For any parents of node k not shared with k+1 (and vice versa):
            for i in range(0, k):
                if self.mat[i,k] and not self.mat[i,k+1]:
                    if self.nodes[self.x[k]] not in self.jointchanges:
                        self.jointchanges[self.nodes[self.x[k]]] = self.joint.dists[self.nodes[self.x[k]]].copy()
                    if self.nodes[self.x[k+1]] not in self.jointchanges:
                        self.jointchanges[self.nodes[self.x[k+1]]] = self.joint.dists[self.nodes[self.x[k+1]]].copy()
                    self.logqfactor += self.joint.dists[self.nodes[self.x[k]]].remove_parent(
                            self.nodes[self.x[i]])
                    self.logqfactor += self.joint.dists[self.nodes[self.x[k+1]]].add_parent(
                            self.nodes[self.x[i]], self.states[self.x[i]])
                elif not self.mat[i,k] and self.mat[i,k+1]:
                    if self.nodes[self.x[k]] not in self.jointchanges:
                        self.jointchanges[self.nodes[self.x[k]]] = self.joint.dists[self.nodes[self.x[k]]].copy()
                    if self.nodes[self.x[k+1]] not in self.jointchanges:
                        self.jointchanges[self.nodes[self.x[k+1]]] = self.joint.dists[self.nodes[self.x[k+1]]].copy()
                    self.logqfactor += self.joint.dists[self.nodes[self.x[k]]].add_parent(
                            self.nodes[self.x[i]], self.states[self.x[i]])
                    self.logqfactor += self.joint.dists[self.nodes[self.x[k+1]]].remove_parent(
                            self.nodes[self.x[i]])

            self.changelength = 2

            #For any children of node k not shared with k+1 (and vice versa):
            for j in range(k+2, self.node_num):
                #print self.mat[k,j], self.mat[k+1,j]
                if self.mat[k,j] and not self.mat[k+1,j]:
                    if self.nodes[self.x[j]] not in self.jointchanges:
                        self.jointchanges[self.nodes[self.x[j]]] = self.joint.dists[self.nodes[self.x[j]]].copy()
                    self.logqfactor += self.joint.dists[self.nodes[self.x[j]]].remove_parent(
                            self.nodes[self.x[k]])
                    self.logqfactor += self.joint.dists[self.nodes[self.x[j]]].add_parent(
                            self.nodes[self.x[k+1]], self.states[self.x[k+1]])
                    self.changelength += 1
                    self.changelist[self.changelength-1] = j
                elif not self.mat[k,j] and self.mat[k+1,j]:
                    if self.nodes[self.x[j]] not in self.jointchanges:
                        self.jointchanges[self.nodes[self.x[j]]] = self.joint.dists[self.nodes[self.x[j]]].copy()
                    #print "going here"
                    #print "base node %d, parent node %d" % (self.nodes[self.x[j]], self.nodes[self.x[k+1]])
                    self.logqfactor += self.joint.dists[self.nodes[self.x[j]]].remove_parent(
                            self.nodes[self.x[k+1]])
                    self.logqfactor += self.joint.dists[self.nodes[self.x[j]]].add_parent(
                            self.nodes[self.x[k]], self.states[self.x[k]])
                    self.changelength += 1
                    self.changelist[self.changelength-1] = j
                #if self.mat[k,j]==1 or self.mat[k+1,j]==1:
            
            if self.mat[k,k+1]:
                if self.nodes[self.x[k]] not in self.jointchanges:
                    self.jointchanges[self.nodes[self.x[k]]] = self.joint.dists[self.nodes[self.x[k]]].copy()
                if self.nodes[self.x[k+1]] not in self.jointchanges:
                    self.jointchanges[self.nodes[self.x[k+1]]] = self.joint.dists[self.nodes[self.x[k+1]]].copy()

                self.logqfactor += self.joint.dists[self.nodes[self.x[k+1]]].remove_parent(
                        self.nodes[self.x[k]])
                self.logqfactor += self.joint.dists[self.nodes[self.x[k]]].add_parent(
                        self.nodes[self.x[k+1]], self.states[self.x[k+1]])


            self.x[k], self.x[k+1] = self.x[k+1], self.x[k]
            self.changelist[0], self.changelist[1] = k, k+1
                
        if scheme==2: # skeletal change

            i = np.random.randint(self.node_num)
            j = np.random.randint(self.node_num)
            while i==j:
                j = np.random.randint(self.node_num)
            if i>j:
                i,j = j,i

            #print np.arange(self.node_num)
            #print self.x
            #print ""
            #print self.mat
            #s = self.x.argsort()
            #print ""
            #print self.mat[s].T[s].T

            edgedel = self.mat[i,j]
            self.mat[i,j] = 1-self.mat[i,j]
            self.changelength=1
            self.changelist[0]=j

            #Remember old parameters
            self.jointchanges[self.nodes[self.x[j]]] = self.joint.dists[self.nodes[self.x[j]]].copy()
            # Change parameters:
            #print "adding edge: base node %d, parent node %d" % (self.nodes[self.x[j]], self.nodes[self.x[i]])
            if edgedel:
                self.logqfactor = self.joint.dists[self.nodes[self.x[j]]]\
                        .remove_parent(self.nodes[self.x[i]])
            else:
                self.logqfactor = self.joint.dists[self.nodes[self.x[j]]]\
                        .add_parent(self.nodes[self.x[i]], self.states[self.x[i]])



        if scheme==3: # Null move (parameters only)
            k = np.random.randint(self.node_num-1)

            self.jointchanges[self.nodes[self.x[k]]] = self.joint.dists[self.nodes[self.x[k]]].copy()

            self.logqfactor = self.joint.dists[self.nodes[self.x[k]]]\
                    .move_params()

            self.changelength=1
            self.changelist[0]=k

        #print("DONE PROPOSING. Scheme: %d" % scheme)
        #print self.mat
        #s = self.x.argsort()
        #print ""
        #print self.mat[s].T[s].T
        #print ''

        #print self.joint
        #print ''

        #for k,v in self.jointchanges.iteritems():
            #print k, v.parent_domain, v.params

        #print "#############################################"

