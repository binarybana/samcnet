# cython: profile=False
cimport cython

from bayesnet cimport BayesNet
from dai_bind cimport FactorGraph, Factor, VarSet, Var, calcState, calcLinearState,\
        PropertySet
cimport dai_bind as dai
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.pair cimport pair

import numpy as np
cimport numpy as np

import scipy.stats as st

from collections import defaultdict
from math import lgamma, log
import networkx as nx
import tables as t

cdef extern from "utils.h":
    string crepr(FactorGraph &)
    string crepr(Factor &)
    string crepr(VarSet &)

DEF DEBUG=0

cdef class BayesNetSampler:
    """ 
    == Optional prior information ==

    prior_structural: A float used in the -logP calculations.
    template: An adjacency matrix over the nodes with float values from [0,1]

    """
    cdef public:
        object bayesnet, ntemplate, ground, verbose
        double prior_structural
    def __init__(self, bayesnet, ntemplate=None, ground=None, prior_structural=1.0, verbose=False):
        self.bayesnet = bayesnet
        self.verbose = verbose
        if type(ntemplate) == np.array:
            self.ntemplate = ntemplate
        elif type(ntemplate) == nx.DiGraph:
            self.ntemplate = np.array(nx.to_numpy_matrix(ntemplate), dtype=np.int32)
        self.ground = ground
        self.prior_structural = prior_structural
        
    def copy(self):
        """ Create a copy of myself for suitable use later """
        return (self.bayesnet.mat.copy(), self.bayesnet.x.copy())

    def init_db(self, db, size):
        """ Takes a Pytables Group object (group) and the total number of samples expected and
        expands or creates the necessary groups.
        """
        objroot = db.root.object
        db.createEArray(objroot.objfxn, 'entropy', t.Float64Atom(), (0,), expectedrows=size)
        if self.ground:
            db.createEArray(objroot.objfxn, 'kld', t.Float64Atom(), (0,), expectedrows=size)
            db.createEArray(objroot.objfxn, 'edge_distance', t.Float64Atom(), (0,), expectedrows=size)
        if self.verbose:
            N = self.x.size
            db.createEArray(objroot.samples, 'mat', t.UInt8Atom(), shape=(0,N,N),
                    expectedrows=size)
            db.createEArray(objroot.samples, 'x', t.UInt32Atom(), shape=(0,N),
                    expectedrows=size)

    def save_iter_db(self, db):
        """ Saves objective function (and possible samples depending on verbosity) to
        Pytables db
        """ 
        root = db.root.object
        root.objfxn.entropy.append((self.bayesnet.entropy(),))
        if self.verbose:
            root.samples.mat.append((self.bayesnet.mat,))
            root.samples.x.append((self.bayesnet.x,))
        if self.ground:
            root.objfxn.kld.append((self.ground.kld(self.bayesnet),))
            root.objfxn.edge_distance.append((self.bayesnet.global_edge_presence(self.ground),))

    def energy(self):
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] x = \
                self.bayesnet.x
        cdef np.ndarray[np.int32_t, ndim=2, mode="c"] mat = \
                self.bayesnet.mat
        cdef double priordiff = 0.0
        ntemplate = self.ntemplate
        for i in range(self.bayesnet.node_num):
            for j in range(self.bayesnet.node_num):
                if(j!=i):
                    priordiff += abs(mat[j,i] - ntemplate[x[j],x[i]]) 
        return (priordiff)*self.prior_structural + self.bayesnet.energy()

    def propose(self):
        self.bayesnet.mutate()

    def reject(self):
        self.bayesnet.revert()

    def info(self):
        return self.bayesnet.num_edges()

cdef class BayesNetCPD:
    """To initialize a BayesNetCPD we need:
    
    ==Required==
    states: Arities.
    data: For -logP calculations.

    With all of these parameters, the network will be initialized with no
    interconnections and all nodes assigned a uniform probability over their 
    arities.
    """
    def __cinit__(self, *args, **kwargs):
        self.memo_entropy = 0.0
        self.dirty = True
    
    def __init__(self, states, data=None):
        cdef int i, j 

        self.logqfactor = 0.0
        self.data = data
        self.states = states
        self.node_num = states.shape[0]
        self.limparent = 1
        self.x = np.arange(self.node_num, dtype=np.int32)
        self.mat = np.eye(self.node_num, dtype=np.int32)
        self.fvalue = np.zeros((self.node_num,), dtype=np.double)
        np.random.shuffle(self.x) # We're going to make this a 0-9 permutation
        self.changelist = self.x.copy()
        self.changelength = self.node_num

        if data != None and data.size:
            for i in range(data.shape[0]): 
                self.pdata.push_back(vector[ulong]())
                for j in range(data.shape[1]):
                    self.pdata[i].push_back(data[i,j])

        cdef vector[Factor] facvector
        for name,arity in zip(np.arange(self.node_num),states):
            self.pnodes.push_back(Var(name, arity))
            facvector.push_back(Factor(self.pnodes.back()))

        self.fg = FactorGraph(facvector)

    def set_cpds(self, ground):
        self.dirty = True

        self.pnodes.clear()
        cdef vector[Factor] facvector
        cdef map[Var, size_t] mapstate
        totuple = []
        cdef int i, j, k, l
        cdef Factor tempfactor
        cdef VarSet tempvarset, parset

        for name,arity in zip(np.arange(self.node_num),self.states):
            self.pnodes.push_back(Var(name, arity))
        for i, dist in ground.dists.iteritems():
            tempvarset = VarSet(self.pnodes[i])
            parset = VarSet()
            for j in dist.sorted_parent_names:
                tempvarset.insert(self.pnodes[j])
                parset.insert(self.pnodes[j])
            
            tempfactor = Factor(tempvarset)
            for j in range(dai.BigInt_size_t(parset.nrStates())):
                mapstate = calcState(parset, j)
                totuple = []
                for l in dist.sorted_parent_names:
                    totuple.append(mapstate[self.pnodes[l]])
                for k in range(self.states[i]):
                    mapstate[self.pnodes[i]] = k
                    if k == self.states[i]-1:
                        tempfactor.set(calcLinearState(tempvarset, mapstate), 
                                1 - np.sum(ground.dists[i].fastp(tuple(totuple))))
                    else:
                        tempfactor.set(calcLinearState(tempvarset, mapstate), 
                                ground.dists[i].fastp(tuple(totuple))[k])
            facvector.push_back(tempfactor)

        self.fg = FactorGraph(facvector)

    def global_edge_presence(self, other):
        s = self.x.argsort()
        sg = other.x.argsort()
        ordmat = self.mat[s].T[s].T
        return np.abs(other.mat[sg].T[sg].T - ordmat).sum() # TODO report TN, TP, FN, FP

    #@cython.boundscheck(False)
    def energy(self):
        """ 
        Calculate the -log probability. 
        """
        IF DEBUG:
            print "Energy top"
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] x = \
                self.x
        cdef np.ndarray[np.int32_t, ndim=2, mode="c"] mat = \
                self.mat

        cdef int i,j,node,data,index,arity,parstate,state
        cdef double sum, accum
        accum = 0

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

        for i in range(counts.size()):
            accum = 0
            node = x[self.changelist[i]]
            arity = self.pnodes[node].states()
            numparstates = self.fg.factor(node).nrStates()/arity
            fac = self.fg.factor(node)
            #alpha_ijk = prior_alpha/numparstates/arity
            #alpha_ik = prior_alpha/numparstates
            for parstate in range(numparstates):
                #accum -= lgamma(alpha_ijk) * arity
                #accum += lgamma(alpha_ik)
                for state in range(arity):
                    index = self.convert_separate(node, state, parstate)
                    accum += log(fac.get(index))*(counts[i][index])
                    #accum += log(self.fg.factor(node).get(index))*(counts[i][index] + alpha_ijk - 1)

            self.fvalue[node] = -1.0 * accum

        sum = self.fvalue.sum()

        IF DEBUG:
            print "energy bottom. energy: %f" % accum
            print "energy: marginal likelihood %f; qfactor %f" % \
                    (sum, self.logqfactor)

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

    def num_edges(self):
        cdef int i
        return (self.mat-np.eye(self.mat.shape[0],dtype=np.int)).sum()

    def marginal(self, node):
        if self.dirty:
            self.memo_entropy = self.entropy()
        # Now we know our entropy AND our JTree are correct
        cdef int i
        marg = self.jtree.calcMarginal(VarSet(self.pnodes[node]))
        res = np.empty(marg.nrStates())
        for i in range(marg.nrStates()):
            res[i] = marg[i]
        return res

    def print_factor(self, node):
        cdef:
            Factor fac = self.fg.factor(node)
            map[Var, size_t] mapstate
            int i
            pair[Var,size_t] v

        for i in range(fac.nrStates()):
            mapstate = calcState(fac.vars(), i)
            print "i:%d " % i,
            for v in mapstate:
                print "%d %d" % (v.first.label(),v.second),
            print "%f" % fac[i]

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
                    subaccum += prob * log(cpd_lookup,2)
                accum += marginal[parstate] * subaccum
        return -self.memo_entropy - accum

    def entropy(self):
        if not self.dirty:
            return self.memo_entropy
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
                    accumsum -= 0.0 if temp == 0.0 else temp * log(temp,2)
                accum += marginal[parstate] * accumsum

        self.dirty = False
        self.memo_entropy = accum
        return accum

    def naive_entropy(self):
        cdef:
            int numstates,i
            double accumprod,accum
            map[Var, size_t] statemap
            VarSet allvars = VarSet(self.pnodes.begin(), self.pnodes.end(), self.pnodes.size())

        numstates = 1
        accum = 0.0

        for i in range(self.pnodes.size()):
            numstates *= self.pnodes[i].states()

        for i in range(numstates):
            accumprod = 1.0
            statemap = calcState(allvars, i)
            for j in range(self.fg.nrFactors()):
                accumprod *= self.fg.factor(j)[calcLinearState(self.fg.factor(j).vars(), statemap)]
            accum += 0.0 if accumprod == 0.0 else accumprod * log(accumprod,2)

        # Cannot put dirty to false here because our JTree is not up to date.
        return -accum

    def revert(self):
        """ Revert mat, x, fvalue, changelist, and changelength. """
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

    def mutate(self):
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
            
            # Change parameters:
            self.logqfactor = 0.0
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

            underparlimit = True
            while underparlimit:
                i = np.random.randint(self.node_num)
                j = np.random.randint(self.node_num)
                while i==j:
                    j = np.random.randint(self.node_num)
                if i>j:
                    i,j = j,i

                edgedel = self.mat[i,j]
                self.mat[i,j] = 1-self.mat[i,j]
                underparlimit = self.mat[:,j].sum() > (self.limparent+1) 
                #+1 because of unitary diagonal

            self.changelength=1
            self.changelist[0]=j

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
                self.logqfactor += self.adjust_factor(node, addlist[node], dellist[node],True)

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

    def set_factor(self, node, newfac):
        cdef int i
        cdef Factor fac = self.fg.factor(node)
        assert len(newfac) == fac.nrStates()

        for i in range(fac.nrStates()):
            fac.set(i, newfac[i])
        self.fg.setFactor(node, fac)
        self.dirty = True
    
    def adjust_factor(self, int node, object addlist, object dellist, object undo=False):
        """ Adjust the factor according to the add list and delete list 
        """
        cdef int s
        cdef Factor oldfac = self.fg.factor(node)
        cdef VarSet oldvars = oldfac.vars()
        cdef VarSet addvars = VarSet()
        cdef VarSet delvars = VarSet()

        self.dirty = True

        for var in addlist:
            addvars.insert(self.pnodes[var])
        for var in dellist:
            delvars.insert(self.pnodes[var])

        cdef VarSet newvars = (oldvars|addvars)/delvars
        cdef Factor newfac = Factor(newvars)

        arity = self.states[node]
        alpha = np.ones(arity)
        for s in range(newfac.nrStates()):
            newval = oldfac.get(calcLinearState(oldvars, calcState(newvars, s)))
            newfac.set(s, newval) 

        #TESTING if I keep this then I can delete some of the above
        cdef VarSet parents = newfac.vars()
        parents.erase(self.pnodes[node])
        parstates = dai.BigInt_size_t(parents.nrStates())
        arity = self.states[node]
        alpha = np.ones(arity)
        for p in range(parstates):
            newval = np.random.dirichlet(alpha)
            state = dai.calcState(parents, p)
            for s in range(arity):
                state[self.pnodes[node]] = s
                aggstate = dai.calcLinearState(newfac.vars(), state)
                newfac.set(aggstate, newval[s])

        if undo: #For some reason Cython does not seem to be able to pass False into setFactor
            self.fg.setFactor(node, newfac, True)
        else:
            self.fg.setFactor(node, newfac)
        
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

        self.dirty = True

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
        self.dirty = True
        self.fg.restoreFactors()

    def __repr__(self):
        return (crepr(self.fg))

def test():
    empty = BayesNetCPD(np.array([2,2,2]))
    assert empty.entropy() == 3.0
    assert empty.naive_entropy() == 3.0

    connected = BayesNetCPD(np.array([2,2,2]))
    connected.adjust_factor(0,[1],[])
    connected.set_factor(0, [0.0,1.0,0.0,1.0])
    connected.adjust_factor(1,[2],[])
    connected.set_factor(1, [0.0,1.0,0.0,1.0])

    assert connected.entropy() == 1.0
    assert connected.kld(empty) == 2.0

    #np.random.seed(1234)
    #con = TreeNet(4)
    #rcon = TreeNet(4)
    #for i in range(1000):
        #kl1 = rcon.kld(con)
        #kl2 = con.kld(rcon)
        #assert (kl1 >= 0.0)
        #assert (kl2 >= 0.0)
        #rcon.propose()
        #con.propose()

    t = BayesNetCPD(np.ones(6,dtype=np.int32)*2)
    for i in range(500):
        t.mutate()

    return empty, connected, t
