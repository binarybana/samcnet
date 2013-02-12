from __future__ import division

import numpy as np
import pylab as p
import networkx as nx
import random
from math import log, exp, pi, lgamma

import scipy.stats as st

class TreeNet():
    def __init__(self, numnodes, data=None, template=None, priorweight=1.0,
            verbose=False, ground=None):
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(range(numnodes), marginal=0.5, 
                delta=np.nan, eta=np.nan)
        self.data = data
        self.template = template
        self.priorweight = priorweight
        self.verbose = verbose
        self.ground = ground

        self.oldgraph = None
        self.memo_entropy = None

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
        trav = nx.dfs_preorder_nodes(g,n1)
        trav.next()
        for node in trav:
            g.node[node]['marginal'] = np.nan #invalidate downstream cached marginals

        #if len(g.edges()) == len(g.nodes()): # Hmm this could be a problem...
            #print(len(g.edges())) # and by this, I mean that we rarely (if ever)
            # 'saturate' the network with connections (or aka evaluate fully
            # connected networks).

    def reject(self):
        self.graph = self.oldgraph
        self.memo_entropy = None

    def ve(self):
        g = self.graph
        for node in nx.dfs_preorder_nodes(g):
            params = g.node[node]
            if np.isnan(params['marginal']):
                predmarg = g.node[g.predecessors(node)[0]]['marginal']
                params['marginal'] = (1-params['delta'])*predmarg + \
                        params['eta']*(1-predmarg)

    def energy(self):
        e = 0.0
        for row in self.data:
            for i,p in self.g.node.iteritems():
                thisval = row[i]
                if np.isnan(p['eta']):
                    marg = p['marginal']
                    e -= log(thisval*marg + (1-thisval)*(1-marg))
                else:
                    delta = p['delta']
                    eta = p['eta']
                    parval = row[self.graph.predecessors(i)[0]]
                    e -= log(thisval*parval*(1-delta) + \
                            thisval*(1-parval)*eta + \
                            (1-thisval)*parval*delta + \
                            (1-thisval)*(1-parval)*(1-eta))

        # Here I should penalize number of edges and edges that don't 
        # match to the template (with the priorweighting too).
        return e

    def entropy(self):
        if self.memo_entropy:
            return self.memo_entropy
        self.ve()
        h = 0.0
        for i,p in self.graph.node.iteritems():
            if np.isnan(p['eta']): # No parent
                marg = p['marginal']
                h -= marg*log(marg) + (1-marg)*log(1-marg)
            else:
                delta = p['delta']
                eta = p['eta']
                parmarg = self.graph.node[self.graph.predecessors(i)[0]]['marginal']
                h -= parmarg*(delta*log(delta) + (1-delta)*log(1-delta))
                h -= (1-parmarg)*(eta*log(eta) + (1-eta)*log(1-eta))

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
                div -= marg*log(othermarg) + (1-marg)*log(1-othermarg)
            else:
                parent = self.graph.predecessors(i)[0]
                parmarg = self.graph.node[parent]['marginal']

                cond = self.cond_prob(i, parent)
                div -= parmarg*(cond[1]*log(1-p['delta']) + (1-cond[1])*log(p['delta'])) \
                    + (1-parmarg)*(cond[0]*log(p['eta']) + (1-cond[0])*log(1-p['eta'])) # parent = 0
        return div

    def cond_prob(self, node, cond):
        """ Compute Pr(node=1 | cond=0) and Pr(node=1 | cond=1)
        by finding a path from cond to node and summing out the other variables
        """
        self.ve()
        path = list(nx.all_simple_paths(self.graph, node, cond))
        pathalt = list(nx.all_simple_paths(self.graph, cond, node))
        if len(path) == len(pathalt) == 0:
            return self.graph.node[node]["marginal"]
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

if __name__ == '__main__':
    t = TreeNet(6)
    for i in range(500):
        t.propose()

