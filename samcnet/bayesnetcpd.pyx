# cython: profile=False
cimport cython

from bayesnet cimport BayesNet
from dai_bind cimport FactorGraph, Factor, VarSet, Var, calcState, calcLinearState,\
        PropertySet
cimport dai_bind as dai
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map

import numpy as np
cimport numpy as np

import scipy.stats as st

from collections import Counter, defaultdict
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

DEF DEBUG=0

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
        self.memo_entropy = 0.0
        self.dirty = True
    
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
        if data.size:
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

    def save_to_db(BayesNetCPD self, object db, \
            double theta, double energy, int i, BayesNetCPD ground_truth):
        func = ground_truth.kld(self)
        assert db is not None, 'DB None when trying to save sample.'
        db[i] = np.array([theta, energy, func])

    #@cython.boundscheck(False)
    def energy(self):
        """ 
        Calculate the -log probability. 
        """
        IF DEBUG:
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
            # WAT?! How can this i index possibly be right? it should be node
            # or x[node] right?!?
            #self.fvalue[i] *= -1.0;

        cdef int i,j,node,data,index,arity,parstate,state
        cdef double sum, priordiff, accum, accumgold
        accum = 0
        accumgold = 0

        # Limit the number of parents
        for i in range(self.changelength):
            if self.fg.factor(x[self.changelist[i]]).vars().size()-1 > self.limparent:
                return 1e20

        cdef vector[vector[int]] counts = vector[vector[int]]()
        for i in range(self.changelength):
            node = x[self.changelist[i]]
            counts.push_back(vector[int](self.fg.factor(node).nrStates()))
        
        for data in range(self.pdata.size()):
            for i in range(counts.size()):
                node = x[self.changelist[i]]
                index = self.convert(node, self.pdata[data])
                counts[i][index] = counts[i][index] + 1

        accum = 0
        for i in range(counts.size()):
            node = x[self.changelist[i]]
            arity = self.pnodes[node].states()
            numparstates = self.fg.factor(node).nrStates()/arity
            alpha_ijk = self.prior_alpha/numparstates/arity
            alpha_ik = self.prior_alpha/numparstates
            for parstate in range(numparstates):
                accum -= lgamma(alpha_ijk) * arity
                accum += lgamma(alpha_ik)
                for state in range(arity):
                    index = self.convert_separate(node, state, parstate)
                    accum += log(self.fg.factor(node).get(index))*(counts[i][index] + alpha_ijk - 1)


            self.fvalue[node] = -1.0 * accum

        sum = self.fvalue.sum()

        # Gold standard
        #for i in range(self.pdata.size()):
            #accumgold -= self.fg.logScore(self.pdata[i])

        IF DEBUG:
            print "energy bottom. energy: %f" % accum
            print "energy: marginal likelihood %f; struct prior %f; qfactor %f" % \
                    (sum, priordiff*self.prior_gamma, self.logqfactor)

        priordiff = 0.0
        for i in range(self.node_num):
            for j in range(self.node_num):
                if(j!=i):
                    priordiff += abs(mat[j,i] - ntemplate[x[j],x[i]])
        sum += (priordiff)*self.prior_gamma
        sum -= self.logqfactor #TODO Check this negative sign

        return sum

    cdef int convert(BayesNetCPD self, int node, vector[ulong] state):
        cdef int i
        cdef map[Var, size_t] statemap
        for i in range(state.size()):
            statemap[self.pnodes[i]] = state[i]
        return calcLinearState(self.fg.factor(node).vars(), statemap)

    cdef int convert_separate(BayesNetCPD self, int node, int state, int parstate):
        cdef int i
        cdef VarSet vars = self.fg.factor(node).vars()
        cdef map[Var, size_t] mapstate = calcState( vars/VarSet(self.pnodes[node]), parstate)
        mapstate[self.pnodes[node]] = state
        return calcLinearState(vars, mapstate)

    def kld(self, BayesNetCPD other):
        if self.dirty:
            self.memo_entropy = self.entropy()
        # Now we know our entropy AND our JTree are correct

        cdef:
            double accum, subaccum, cpd_lookup, prob
            VarSet jointvars, parvars
            Factor marginal, joint
            map[Var, size_t] mapstate
            int index

        accum = 0.0
        for i in range(self.fg.nrFactors()):
            arity = other.pnodes[i].states()
            numparstates = other.fg.factor(i).nrStates()/arity
            # Here we use assumption that the vars are equatable from the
            # two BNs
            parvars = other.fg.factor(i).vars()/VarSet(other.pnodes[i])
            marginal = self.jtree.calcMarginal(parvars)

            jointvars = other.fg.factor(i).vars()
            joint = self.jtree.calcMarginal(jointvars)
            for parstate in range(numparstates):
                subaccum = 0.0
                mapstate = calcState(parvars, parstate)
                for state in range(arity):
                    cpd_lookup = other.fg.factor(i)[other.convert_separate(i, state, parstate)]
                    mapstate[self.pnodes[i]] = state
                    index = calcLinearState(jointvars, mapstate) 
                    # Is this actually necessary? 
                    # Or does index == other.convert_separate(i, state,
                    # parstate)?
                    prob = joint[index]/marginal[parstate]
                    subaccum += prob * log(cpd_lookup)
                accum += marginal[parstate] * subaccum
        return -self.memo_entropy - accum

    def entropy(self):
        cdef PropertySet ps = PropertySet('[updates=HUGIN]')
        cdef int i,parstate,state,arity,numparstates
        cdef double accum, accumsub, temp

        cdef Factor marginal

        self.jtree = JTree(self.fg, ps)
        self.jtree.init()
        self.jtree.run()
        accum = 0.0
        for i in range(self.fg.nrFactors()):
            arity = self.pnodes[i].states()
            numparstates = self.fg.factor(i).nrStates()/arity
            marginal = self.jtree.calcMarginal(self.fg.factor(i).vars()/VarSet(self.pnodes[i]))
            for parstate in range(numparstates):
                accumsum = 0.0
                for state in range(arity):
                    temp = self.fg.factor(i)[self.convert_separate(i, state, parstate)]
                    accumsum -= temp * log(temp)
                accum += marginal[parstate] * accumsum

        self.dirty = False
        return accum

    def naive_entropy(self):
        cdef:
            int numstates,i
            double temp,accumsum,accumprod,accum
            map[Var, size_t] statemap
            VarSet allvars = VarSet(self.pnodes.begin(), self.pnodes.end(), self.pnodes.size())

        numstates = 1
        accum = 0.0

        for i in range(self.pnodes.size()):
            numstates *= self.pnodes[i].states()

        for i in range(numstates):
            accumsum = 0.0
            accumprod = 1.0
            statemap = calcState(allvars, i)
            for j in range(self.fg.nrFactors()):
                temp = self.fg.factor(j)[calcLinearState(self.fg.factor(j).vars(), statemap)]
                accumsum += log(temp)
                accumprod *= temp
            accum += accumsum * accumprod

        # Cannot put dirty to false here because our JTree is not up to date.
        return -accum

    def reject(self):
        """ Revert graph, mat, x, fvalue, changelist, and changelength. """
        IF DEBUG:
            print("REJECTING")
        self.mat = self.oldmat
        self.x = self.oldx
        self.fvalue = self.oldfvalue
        self.dirty = True # without checking more carefully, we do 
        #the safe and simple thing here

        self.restore_backups()

        # TODO Do I need to reset logqfactor here?
        # If I really wanted to be safe I would set changelngeth=10 and maybe
        # changelist

    def propose(self):
        """ 'Propose' a new network structure by backing up the old one and then 
        changing the current one. """

        cdef int i,j,i1,j1,i2,j2,edgedel,scheme
        self.oldmat = self.mat.copy()
        self.oldx = self.x.copy()
        self.oldfvalue = self.fvalue.copy()
        self.dirty = True
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] x = \
                self.x

        scheme = np.random.randint(1,4)   
        self.clear_backups()
        addlist, dellist = defaultdict(list), defaultdict(list)

        IF DEBUG:
            print "#############################################"
            print("PROPOSING Scheme %d" % scheme)
            s = self.x.argsort()
            print "old mat:"
            print self.mat[s].T[s].T
            print " ##### "
            print "old x:"
            print x

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
            IF DEBUG:
                print "swapping k=%d and k+1=%d" % (x[k],x[k+1])

            #For any parents of node k not shared with k+1 (and vice versa):
            for i in range(0, k):
                if self.mat[i,k] and not self.mat[i,k+1]:
                    dellist[x[k]].append(x[i])
                    addlist[x[k+1]].append(x[i])
                elif not self.mat[i,k] and self.mat[i,k+1]:
                    dellist[x[k+1]].append(x[i])
                    addlist[x[k]].append(x[i])

            self.changelength = 2

            #For any children of node k not shared with k+1 (and vice versa):
            for j in range(k+2, self.node_num):
                #print self.mat[k,j], self.mat[k+1,j]
                if self.mat[k,j] and not self.mat[k+1,j]:
                    dellist[x[j]].append(x[k])
                    addlist[x[j]].append(x[k+1])
                    self.changelength += 1
                    self.changelist[self.changelength-1] = j
                elif not self.mat[k,j] and self.mat[k+1,j]:
                    dellist[x[j]].append(x[k+1])
                    addlist[x[j]].append(x[k])
                    self.changelength += 1
                    self.changelist[self.changelength-1] = j
            
            if self.mat[k,k+1]:
                dellist[x[k+1]].append(x[k])
                addlist[x[k]].append(x[k+1])

            self.x[k], self.x[k+1] = self.x[k+1], self.x[k]
            self.changelist[0], self.changelist[1] = k, k+1
                
        if scheme==2: # skeletal change

            i = np.random.randint(self.node_num)
            j = np.random.randint(self.node_num)
            while i==j:
                j = np.random.randint(self.node_num)
            if i>j:
                i,j = j,i

            IF DEBUG:
                print "toggling edge (i,j)=(%d,%d)" % (x[i],x[j])
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
                dellist[x[j]].append(x[i])
            else:
                addlist[x[j]].append(x[i])

        if scheme==3: # Null move (parameters only)

            k = np.random.randint(self.node_num-1)

            IF DEBUG:
                print "null moving node k=%d" % (x[k])

            self.logqfactor = self.move_params(x[k])

            self.changelength=1
            self.changelist[0]=k

        if scheme != 3:
            for node in set(addlist.keys() + dellist.keys()):
                self.logqfactor += self.adjust_factor(node, addlist[node], dellist[node])

        IF DEBUG:
            s = self.x.argsort()
            print "new mat:"
            print self.mat[s].T[s].T
            print " ##### "
            print "new x:"
            print self.x
            
            print("DONE PROPOSING. Scheme: %d" % scheme)
            print "#############################################"

        return scheme
    
    def adjust_factor(self, int node, object addlist, object dellist):
        cdef int s
        cdef Factor oldfac = self.fg.factor(node)

        cdef VarSet oldvars = oldfac.vars()

        cdef VarSet addvars = VarSet()
        cdef VarSet delvars = VarSet()

        for var in addlist:
            addvars.insert(self.pnodes[var])
        for var in dellist:
            delvars.insert(self.pnodes[var])

        cdef VarSet newvars = (oldvars|addvars)/delvars
        cdef Factor newfac = Factor(newvars)

        for s in range(newfac.nrStates()):
            newval = oldfac.get(calcLinearState(oldvars, calcState(newvars, s)))
            newfac.set(s, newval) 

        self.fg.setFactor(node, newfac, True)
        
        IF DEBUG:
            print "## Adjusted factors"
            print "### old fac"
            print crepr(oldfac)
            print "### new fac"
            print crepr(newfac)

        # Calculate logqfactor from arity of new parents (use
        # B of dirichlet distribution from wikipedia)
        #return new_count * lgamma(self.arity)
        #return 0 # As we are not introducing new parameters right?
        
        ## Calculate logqfactor from arity of dropped parents (use
        ## B of dirichlet distribution from wikipedia)
        ##return -rem_count * lgamma(self.arity)
        #return 0 # As we are not introducing new parameters right?
        return 0.0

    def move_params(self, int node):
        cdef int s, p, parstates, aggstate, arity
        cdef double oldval, a, b, std
        cdef Factor fac = self.fg.factor(node)
        cdef map[Var, size_t] state

        std = 0.3
        cdef VarSet parents = fac.vars()
        parents.erase(self.pnodes[node])
        parstates = dai.BigInt_size_t(parents.nrStates())


        IF DEBUG: 
            print "### oldfac"
            print crepr(fac)
            print "\tparents: "
            print crepr(parents)
            print "\tnumparentstates: %d" % parstates

        arity = self.states[node]
        alpha = np.ones(arity)

        for p in range(parstates):
            newval = np.random.dirichlet(alpha)
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

        IF DEBUG:
            print "### newfac"
            print crepr(fac)
            print ""
            print "### oldfg"
            print crepr(self.fg)
            print "### newfg"
            print crepr(self.fg)

        self.fg.setFactor(node, fac, True)

        return 0.0

    def clear_backups(self):
        self.fg.clearBackups()

    def restore_backups(self):
        self.fg.restoreFactors()

    def __repr__(self):
        return (crepr(self.fg))

