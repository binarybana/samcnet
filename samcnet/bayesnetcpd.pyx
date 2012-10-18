# cython: profile=False
cimport cython

from bayesnet cimport BayesNet
from dai_bind cimport FactorGraph, Factor, VarSet, Var
cimport dai_bind as dai
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map

import numpy as np
cimport numpy as np

import scipy.stats as st

from collections import Counter
from math import lgamma, log

cdef extern from "utils.h":
    string crepr(FactorGraph &)
    string crepr(Factor &)
    string crepr(VarSet &)

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
    
    def __init__(self, states, data, intemplate=None, priorweight=1.0):
        cdef int i, j
        nodes = np.arange(states.shape[0])
        BayesNet.__init__(self, states, data, intemplate, priorweight)

        cdef vector[Factor] facvector

        for name,arity in zip(nodes,states):
            self.pnodes.push_back(Var(name, arity))
            facvector.push_back(Factor(self.pnodes.back()))

        self.fg = FactorGraph(facvector)

        self.logqfactor = 0.0
        #self.memo_table = MemoCounter(data)
        for i in range(data.shape[0]): 
            self.pdata.push_back(vector[ulong]())
            for j in range(data.shape[1]):
                self.pdata[i].push_back(data[i,j])
        
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
        return (self.mat.copy(), self.x.copy(), 0 )# self.fg.clone()) # need to wrap this

    def save_to_db(self, object db, double theta, double energy, int i, BayesNetCPD ground_truth):
        func = 0 #ground_truth.kld(self.fg)
        assert db is not None, 'DB None when trying to save sample.'
        db[i] = np.array([theta, energy, func])

    @cython.boundscheck(False)
    def energy(self):
        """ 
        Calculate the -log probability. 
        """
        print "Energy top"
        #cdef float alphaik,alphaijk,sum,priordiff,accum = 0.0
        #cdef int i,node,j,l,parstate
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] x = \
                self.x
        #cdef np.ndarray[np.int32_t, ndim=1, mode="c"] states = \
                #self.states
        cdef np.ndarray[np.double_t, ndim=2, mode="c"] ntemplate = \
                self.ntemplate
        cdef np.ndarray[np.int32_t, ndim=2, mode="c"] mat = \
                self.mat

        #lut = self.x.argsort()
        
        #for i in range(self.changelength):
            #node = self.changelist[i]
            #cpd = self.joint.dists[x[node]]
            #pdomain = cpd.parent_domain

            ##print("NEW ENERGY LOOP:")
            ##print self.joint
            ##print(node, self.x[node], cpd.name, cpd.params, self.changelist, pdomain, id(pdomain))

            #par_node_cols = lut[pdomain.keys() + [x[node]]]
            #node_par_counts = self.memo_table.lookup(tuple(par_node_cols))

            #parstate = 1
            #for j in range(0,node):
                #if mat[j,node]==1:
                   #parstate *= states[x[j]]

            #alphaijk=self.prior_alpha/parstate/states[x[node]]
            #alphaik=self.prior_alpha/parstate
            
            #accum = 0.0
            #accum -= states[x[node]]*parstate*lgamma(alphaijk)
            #accum += parstate*lgamma(alphaik)

            #for pval in fast_space_iterator(pdomain):
                #fastp = cpd.fastp(pval)
                #for j in range(0,cpd.arity-1):
                    ##print j, fastp, pval, pdomain, alphaijk, node_par_counts
                    #accum += (node_par_counts[pval + (j,)] + alphaijk -1) * log(fastp[j])
                #accum += (node_par_counts[pval + (cpd.arity-1,)] + alphaijk -1) * log((1 - fastp.sum()))

            #self.fvalue[i] = accum;
            #self.fvalue[i] *= -1.0;

        cdef int i
        cdef double sum, priordiff, accum = 0
        
        for i in range(self.pdata.size()):
            accum -= self.fg.logScore(self.pdata[i])

        sum = accum

        print "energy bottom. energy: %f" % accum

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

        self.restore_backups()

        # TODO Do I need to reset logqfactor here?
        # If I really wanted to be safe I would set changelngeth=10 and maybe
        # changelist

    def propose(self):
        """ 'Propose' a new network structure by backing up the old one and then 
        changing the current one. """
        print "#############################################"

        cdef int i,j,i1,j1,i2,j2,edgedel,scheme
        self.oldmat = self.mat.copy()
        self.oldx = self.x.copy()
        self.oldfvalue = self.fvalue.copy()

        scheme = np.random.randint(1,4)   
        self.clear_backups()
        print("PROPOSING Scheme %d" % scheme)

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
                    self.logqfactor += self.remove_parent(
                            self.x[k],
                            self.x[i])
                    self.logqfactor += self.add_parent(
                            self.x[k+1],
                            self.x[i],
                            self.states[self.x[i]])
                elif not self.mat[i,k] and self.mat[i,k+1]:
                    self.logqfactor += self.add_parent(
                            self.x[k],
                            self.x[i],
                            self.states[self.x[i]])
                    self.logqfactor += self.remove_parent(
                            self.x[k+1],
                            self.x[i])

            self.changelength = 2

            #For any children of node k not shared with k+1 (and vice versa):
            for j in range(k+2, self.node_num):
                #print self.mat[k,j], self.mat[k+1,j]
                if self.mat[k,j] and not self.mat[k+1,j]:
                    self.logqfactor += self.remove_parent(
                            self.x[j],
                            self.x[k])
                    self.logqfactor += self.add_parent(
                            self.x[j],
                            self.x[k+1],
                            self.states[self.x[k+1]])
                    self.changelength += 1
                    self.changelist[self.changelength-1] = j
                elif not self.mat[k,j] and self.mat[k+1,j]:
                    self.logqfactor += self.remove_parent(
                            self.x[j],
                            self.x[k+1])
                    self.logqfactor += self.add_parent(
                            self.x[j],
                            self.x[k],
                            self.states[self.x[k]])
                    self.changelength += 1
                    self.changelist[self.changelength-1] = j
            
            if self.mat[k,k+1]:
                self.logqfactor += self.remove_parent(
                        self.x[k+1],
                        self.x[k])
                self.logqfactor += self.add_parent(
                        self.x[k],
                        self.x[k+1],
                        self.states[self.x[k+1]])

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

            # Change parameters:
            #print "adding edge: base node %d, parent node %d" % (self.nodes[self.x[j]], self.nodes[self.x[i]])
            if edgedel:
                self.logqfactor = self.remove_parent(
                        self.x[j],
                        self.x[i])
            else:
                self.logqfactor = self.add_parent(
                        self.x[j],
                        self.x[i],
                        self.states[self.x[i]])

        if scheme==3: # Null move (parameters only)
            k = np.random.randint(self.node_num-1)

            self.logqfactor = self.move_params(self.x[k])

            self.changelength=1
            self.changelist[0]=k

        print("DONE PROPOSING. Scheme: %d" % scheme)
        #print self.mat
        #s = self.x.argsort()
        #print ""
        #print self.mat[s].T[s].T
        #print ''

        #print ''


        print "#############################################"

        return scheme
    
    def add_parent(self, int node, int parent, int arity):
        #make a new factor 
        cdef Factor oldfac = self.fg.factor(node)
        cdef VarSet vars = oldfac.vars()
        cdef Factor newfac = oldfac.embed(vars.insert(self.pnodes[parent]))

        #print "Node %d, Added parent %d, new params: %s" % (self.name,name,str(self.params))
        
        #self.fg.setFactor(node, newfac, True)
        
        # Calculate logqfactor from arity of new parents (use
        # B of dirichlet distribution from wikipedia)
        #return new_count * lgamma(self.arity)
        return 0 # As we are not introducing new parameters right?
    
    def remove_parent(self, int node, int parent):
        print "Node %d, Removing parent %d" % (node, parent)
        print self.mat[s].T[s].T
        cdef Factor oldfac = self.fg.factor(node)

        print "### oldfac"
        print crepr(oldfac)

        cdef VarSet vars = oldfac.vars()
        cdef Factor newfac = oldfac.marginal(vars.erase(self.pnodes[parent]))

        print "### newfac"
        print crepr(newfac)

        self.fg.setFactor(node, newfac, True)

        # Calculate logqfactor from arity of dropped parents (use
        # B of dirichlet distribution from wikipedia)
        #return -rem_count * lgamma(self.arity)
        return 0 # As we are not introducing new parameters right?

    def move_params(self, int node):
        cdef int s, p, parstates, aggstate
        cdef double oldval, a, b, std
        cdef Factor fac = self.fg.factor(node)
        cdef map[Var, size_t] state

        print "### oldfac"
        print crepr(fac)

        std = 0.3
        cdef VarSet parents = fac.vars()
        parents.erase(self.pnodes[node])
        print "\tparents: "
        print crepr(parents)

        parstates = dai.BigInt_size_t(parents.nrStates())
        print "\tnumparentstates: %d" % parstates

        arity = self.states[node]

        for p in range(parstates):
            newval = np.random.dirichlet(np.ones(arity))
            state = dai.calcState(parents, p)
            # we could get this newval closer to the oldval by changing the 
            # alpha values...
            #oldval = fac.get(aggstate)
            #a,b = -oldval/std, (1-oldval)/std
            #newval = st.truncnorm.rvs(a,b,loc=oldval, scale=std)
            for s in range(arity):
                state[self.pnodes[node]] = s
                aggstate = dai.calcLinearState(fac.vars(), state)

                fac.set(aggstate, newval[s])

        print "### newfac"
        print crepr(fac)
        print ""
        print "### oldfg"
        print crepr(self.fg)
        self.fg.setFactor(node, fac, True)
        print "### newfg"
        print crepr(self.fg)

        return 0.0

    def clear_backups(self):
        self.fg.clearBackups()

    def restore_backups(self):
        self.fg.restoreFactors()

        return scheme

