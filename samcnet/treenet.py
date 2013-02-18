from __future__ import division


import numpy as np
import networkx as nx
import pylab as p
import scipy.stats as st
import tables as t
import random

from math import log, exp, pi, lgamma
from numpy import log2

class TreeNet():
    def __init__(self, numnodes, data=None, template=None, priorweight=1.0,
            verbose=False, ground=None, graph=None):
        if graph is None:
            self.graph = nx.DiGraph()
            self.graph.add_nodes_from(range(numnodes), marginal=0.5, 
                    delta=np.nan, eta=np.nan)
        else:
            assert graph.number_of_nodes() == numnodes
            self.graph = graph.copy()

        assert data is None or data.shape[1] == numnodes
        self.data = data
        self.template = template
        self.priorweight = priorweight
        self.verbose = verbose
        self.ground = ground

        self.oldgraph = None
        self.memo_entropy = None

    def copy(self):
        return self.graph.copy()

    def global_edge_presence(self):
        mat = np.array(nx.to_numpy_matrix(self.graph),dtype=np.int32)
        groundmat = np.array(nx.to_numpy_matrix(self.ground.graph),dtype=np.int32)
        return float(np.abs(mat - groundmat).sum()) / self.graph.number_of_nodes()**2

    def propose(self):
        self.oldgraph = self.graph.copy()
        self.memo_entropy = None

        g = self.graph
        if g.edges():
            scheme = np.random.randint(3)
        else:
            scheme = np.random.randint(2)

        nodes = g.nodes()

        if scheme == 0: # Perturb probabilities
            n1 = np.random.randint(len(nodes))
            if g.predecessors(n1):
                g.node[n1]['delta'] = st.beta.rvs(0.4,0.4)
                g.node[n1]['eta'] = st.beta.rvs(0.4,0.4)
                g.node[n1]['marginal'] = np.nan
            else: 
                g.node[n1]['delta'] = np.nan
                g.node[n1]['eta'] = np.nan
                g.node[n1]['marginal'] = np.random.rand()

        if scheme == 1: # Add or change edge 'backwards'
            while True:
                random.shuffle(nodes) #inefficient
                n0 = nodes[0]
                n1 = nodes[1]
                if g.predecessors(n1):
                    g.remove_edge(g.predecessors(n1)[0], n1)
                g.add_edge(n0, n1)

                if nx.is_directed_acyclic_graph(g):
                    break
                else:
                    g = self.graph = self.oldgraph.copy()

            g.node[n1]['delta'] = st.beta.rvs(0.4,0.4)
            g.node[n1]['eta'] = st.beta.rvs(0.4,0.4)
            g.node[n1]['marginal'] = np.nan

        if scheme == 2: # Remove edge
            edges = g.edges()
            edge = edges[np.random.randint(len(edges))]
            n1 = edge[1]
            g.remove_edge(*edge)

            g.node[n1]['delta'] = np.nan
            g.node[n1]['eta'] = np.nan
            g.node[n1]['marginal'] = np.random.rand()

        #print len(g.edges())
        trav = nx.dfs_preorder_nodes(g,n1) # FIXME, this may be a problem, dfs_preorder 
        # was not what I thought it was before
        trav.next()
        for node in trav:
            g.node[node]['marginal'] = np.nan #invalidate downstream cached marginals

        #if len(g.edges()) == len(g.nodes()): # Hmm this could be a problem...
            #print(len(g.edges())) # and by this, I mean that we rarely (if ever)
            # 'saturate' the network with connections (or aka evaluate fully
            # connected networks).
    
    def clear_memory(self):
        self.memo_entropy = None
        #for n,p in self.graph.node.iteritems():
            #if not np.isnan(p['eta']): #has parents
                #p['marginal'] = np.nan

    def add_edge(self, n1, n2, delta=None, eta=None):
        self.clear_memory()
        assert np.isnan(self.graph.node[n2]['eta']), "Node %d already has a parent!" % n2
        self.graph.add_edge(n1,n2)
        if delta is not None:
            self.graph.node[n2]['delta'] = delta 
        else:
            self.graph.node[n2]['delta'] = st.beta.rvs(0.4,0.4)
        if eta is not None:
            self.graph.node[n2]['eta'] = eta
        else:
            self.graph.node[n2]['eta'] = st.beta.rvs(0.4,0.4)
        self.graph.node[n2]['marginal'] = np.nan
        assert nx.is_directed_acyclic_graph(self.graph)

    def rem_edge(self, n1, n2, marginal=None):
        self.clear_memory()
        assert (n1,n2) in self.graph.edges(), "Edge does not exist in graph"
        self.graph.remove_edge(n1,n2)
        self.graph.node[n2]['delta'] = np.nan
        self.graph.node[n2]['eta'] = np.nan
        if marginal is None:
            self.graph.node[n2]['marginal'] = marginal
        else:
            self.graph.node[n2]['marginal'] = np.random.rand()
        assert nx.is_directed_acyclic_graph(self.graph)

    def set_params(self, node, delta, eta, marginal):
        self.clear_memory()
        assert node in self.graph.nodes()
        self.graph.node[node]['delta'] = delta
        self.graph.node[node]['eta'] = eta
        self.graph.node[node]['marginal'] = marginal
        assert nx.is_directed_acyclic_graph(self.graph)

    def reject(self):
        self.graph = self.oldgraph
        self.memo_entropy = None

    def ve(self):
        g = self.graph
        for node in nx.topological_sort(g):
            params = g.node[node]
            if np.isnan(params['marginal']):
                predmarg = g.node[g.predecessors(node)[0]]['marginal']
                g.node[node]['marginal'] = (1-params['delta'])*predmarg + \
                        params['eta']*(1-predmarg)

    def energy(self):
        e = 0.0
        data = self.data
        for i,p in self.graph.node.iteritems():
            thisval = data[:,i]
            if np.isnan(p['eta']):
                marg = p['marginal']
                e -= np.log((thisval*marg + (1-thisval)*(1-marg))).sum()
            else:
                delta = p['delta']
                eta = p['eta']
                parval = data[:,self.graph.predecessors(i)[0]]
                prob = thisval*(parval*(1-delta) + (1-parval)*eta) + \
                        (1-thisval)*(parval*delta + (1-parval)*(1-eta))
                np.clip(prob, 1e-300, 1.0)
                e -= np.log(prob).sum()

        mat = np.array(nx.to_numpy_matrix(self.graph),dtype=np.int32)
        if self.template:
            tempmat = np.array(nx.to_numpy_matrix(self.template),dtype=np.int32)
        else:
            tempmat = np.zeros_like(mat)
        e += self.priorweight * float(np.abs(mat - tempmat).sum())

        return e

    def entropy(self):
        if self.memo_entropy:
            return self.memo_entropy
        self.ve()
        h = 0.0
        for i,p in self.graph.node.iteritems():
            if np.isnan(p['eta']): # No parent
                marg = p['marginal']
                h -= marg*log2(marg) + (1-marg)*log2(1-marg)
            else:
                delta = p['delta']
                eta = p['eta']
                parmarg = self.graph.node[self.graph.predecessors(i)[0]]['marginal']
                if delta != 0.0 and delta != 1.0: # FIXME: Float comparisons
                    h -= parmarg*(delta*log2(delta) + (1-delta)*log2(1-delta))
                if eta != 0.0 and eta != 1.0: # FIXME: Float comparisons
                    h -= (1-parmarg)*(eta*log2(eta) + (1-eta)*log2(1-eta))
        self.memo_entropy = h
        return h

    def kld(self, other):
        div = 0.0
        div -= self.entropy() # Assuming this will self.ve() if needed
        other.ve()
        for i,p in other.graph.node.iteritems(): # note other here
            if np.isnan(p['eta']): # No parent
                marg = p['marginal']
                othermarg = other.graph.node[i]['marginal']
                div -= marg*log2(othermarg) + (1-marg)*log2(1-othermarg)
            else:
                parent = other.graph.predecessors(i)[0]
                parmarg = other.graph.node[parent]['marginal']

                cond = self.cond_prob(i, parent) 
                div -= parmarg*(cond[1]*log2(1-p['delta']) + (1-cond[1])*log2(p['delta'])) \
                    + (1-parmarg)*(cond[0]*log2(p['eta']) + (1-cond[0])*log2(1-p['eta'])) # parent = 0
        return div

    def cond_prob(self, node, cond):
        """ Compute Pr(node=1 | cond=0) and Pr(node=1 | cond=1)
        by finding a path from cond to node and summing out the other variables
        """
        self.ve()
        path = list(nx.all_simple_paths(self.graph, node, cond))
        pathalt = list(nx.all_simple_paths(self.graph, cond, node))
        if len(path) == len(pathalt) == 0:
            temp = self.graph.node[node]["marginal"]
            return (temp,temp)
        elif len(path) > 0:
            down = True # Cond is DOWNstream from node
        else:
            down = False
            path = pathalt
        par0sum = 0.0 # Two sums, for cond = {0,1}
        par1sum = 0.0
        for state in range(2**(len(path[0][1:-1]))): #enumerate states of inter nodes
            if down: #start at node
                par0term = self.graph.node[node]["marginal"]
                par1term = self.graph.node[node]["marginal"]
                lastval0 = 1
                lastval1 = 1
            else: #start at cond 
                par0term = (1-self.graph.node[cond]["marginal"])
                par1term = self.graph.node[cond]["marginal"]
                lastval0 = 0 
                lastval1 = 1

            # Now we need to go down the chain of intermediate nodes 
            for ind, inter in enumerate(path[0][1:-1]):
                this = (state>>ind) & 1
                par0term *= (1-this)*(1-lastval0)*(1-self.graph.node[inter]['eta']) \
                        + (1-this)*(lastval0)*(self.graph.node[inter]['delta']) \
                        + this*(1-lastval0)*self.graph.node[inter]['eta'] \
                        + this*lastval0*(1-self.graph.node[inter]['delta'])
                par1term *= (1-this)*(1-lastval1)*(1-self.graph.node[inter]['eta']) \
                        + (1-this)*(lastval1)*(self.graph.node[inter]['delta']) \
                        + this*(1-lastval1)*self.graph.node[inter]['eta'] \
                        + this*lastval1*(1-self.graph.node[inter]['delta'])
                lastval0 = this
                lastval1 = this

            # Now for last node
            if down: #end at cond
                par0term *= (1-lastval0)*(1-self.graph.node[cond]['eta']) \
                        + (lastval0)*(self.graph.node[cond]['delta'])
                par1term *= (1-lastval1)*self.graph.node[cond]['eta'] \
                        + lastval1*(1-self.graph.node[cond]['delta'])
            else: #end at node
                par0term *= (1-lastval0)*self.graph.node[node]['eta'] \
                        + lastval0*(1-self.graph.node[node]['delta'])
                par1term *= (1-lastval1)*self.graph.node[node]['eta'] \
                        + lastval1*(1-self.graph.node[node]['delta'])
            par0sum += par0term
            par1sum += par1term

        par0sum /= (1-self.graph.node[cond]['marginal'])
        par1sum /= self.graph.node[cond]['marginal']

        return (par0sum, par1sum)

    def init_db(self, db, size):
        """ Takes a Pytables Group object (group) and the total number of samples expected and
        expands or creates the necessary groups.
        """
        objroot = db.root.object
        db.createEArray(objroot.objfxn, 'entropy', t.Float64Atom(), (0,), expectedrows=size)
        if self.ground:
            db.createEArray(objroot.objfxn, 'kld', t.Float64Atom(), (0,), expectedrows=size)
            db.createEArray(objroot.objfxn, 'edge_distance', t.Float64Atom(), (0,), expectedrows=size)
            objroot._v_attrs.true_entropy = self.ground.entropy()
            objroot._v_attrs.true_energy = self.ground.energy()
            objroot._v_attrs.true_numedges = self.ground.graph.number_of_edges()

        temp = {}
        temp['entropy'] = 'Entropy in bits'
        temp['kld']  = 'KL-Divergence from true network in bits'
        temp['edge_distance']  = 'Proportion of incorrect edges |M-X|/n^2'
        objroot._v_attrs.descs = temp

        #if self.verbose:
            #N = self.x.size
            #db.createEArray(objroot.samples, 'mat', t.UInt8Atom(), shape=(0,N,N),
                    #expectedrows=size)

    def save_iter_db(self, db):
        """ Saves objective function (and possible samples depending on verbosity) to
        Pytables db
        """ 
        root = db.root.object
        root.objfxn.entropy.append((self.entropy(),))
        #if self.verbose:
            #root.samples.mat.append((self.mat,))
        if self.ground:
            root.objfxn.kld.append((self.ground.kld(self),))
            root.objfxn.edge_distance.append((self.global_edge_presence(),))

def generateTree(nodes = 5, clusters = 2):
    g = nx.DiGraph()
    for i in range(nodes):
        g.add_node(i)
        if i >= clusters:
            candidates = set(g.nodes()) - set([i])
            g.add_edge(random.choice(list(candidates)), i)
            g.node[i]['delta'] = st.beta.rvs(0.4,0.4)
            g.node[i]['eta'] = st.beta.rvs(0.4,0.4)
            g.node[i]['marginal'] = np.nan
        else:
            g.node[i]['delta'] = np.nan
            g.node[i]['eta'] = np.nan
            g.node[i]['marginal'] = np.random.rand()
    return g

def generateData(g, num):
    topo = nx.topological_sort(g)
    data = np.zeros((num, g.number_of_nodes()), dtype=np.int32)
    for i in range(num):
        for j in topo:
            params = g.node[j]
            if np.isnan(params['eta']): # No parents
                data[i,j] = int(np.random.rand() < params['marginal'])
            else:
                parstate = data[i, g.predecessors(j)[0]]
                data[i,j] = int( np.random.rand() < ( (1-params['delta']) if parstate else params['eta']))

    return data

if __name__ == '__main__':
    empty = TreeNet(3)
    assert empty.entropy() == 3.0

    connected = TreeNet(3)
    connected.add_edge(0,1,0.0,0.0)
    connected.add_edge(1,2,0.0,0.0)
    assert connected.entropy() == 1.0
    assert connected.kld(empty) == 2.0

    t = TreeNet(6)
    for i in range(500):
        t.propose()
