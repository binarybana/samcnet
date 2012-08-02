import networkx as nx
import numpy as np
import random as ra
from collections import defaultdict
from functools import partial
from math import ceil,floor

def generateSubGraphs(numgraphs = 2, nodespersub = 5, interconnects = 0):
    kernel = lambda x: x**0.0001
    x = temp = nx.gn_graph(nodespersub, kernel)

    for i in range(numgraphs-1):
        temp = nx.gn_graph(nodespersub, kernel)
        x = nx.disjoint_union(x, temp)
        #Now add some crosstalk
        workinglen = len(x) - len(temp)
        for i in range(interconnects):
            firstnode = ra.choice(range(workinglen))
            secondnode = ra.choice(range(workinglen,len(x)))

            newedge = [firstnode, secondnode]
            ra.shuffle(newedge)
            x.add_edge(newedge[0], newedge[1])
    return x

def generateHourGlassGraph(nodes=10, interconnects = 0):
    def flipGraph(g):
        e = g.edges()
        g.remove_edges_from(e)
        g.add_edges_from(zip(*zip(*e)[::-1]))

    kernel = lambda x: x**0.0001
    n1 , n2 = int(floor(nodes/2.)), int(ceil(nodes/2.))
    x1 = nx.gn_graph(n1, kernel)
    x2 = nx.gn_graph(n2, kernel)
    flipGraph(x2)
    x = nx.disjoint_union(x1,x2)
    x.add_edge(0,n1+1)

    for i in range(interconnects):
        firstnode = ra.choice(range(n1))
        secondnode = ra.choice(range(n1,n1+n2))
        #newedge = [firstnode, secondnode]
        #ra.shuffle(newedge)
        #x.add_edge(newedge[0], newedge[1])
        x.add_edge(firstnode,secondnode)
    return x


def generateData(graph, numPoints=50, noise=0.5, cpds=None):
    """ 
    Generate <numPoints> random draws from graph, with 
    randomly assigned CPDs and additive zero mean Gaussian 
    noise with std_dev=noise on the observations.

    TODO: add noise
    """
    
    order = nx.topological_sort(graph)
    numnodes = graph.number_of_nodes()
    adj = np.array(nx.to_numpy_matrix(graph),dtype=np.int)

    if not cpds:
        numparents = adj.sum(axis=0)
        #states = np.ones((numnodes,)) * 2 # assuming 2 states per node
        states = np.random.randint(2,4, size=numnodes)
        print "states:", states
        numparentstates = (states ** numparents)
        def cpdGen(states, parstates):
            return np.cumsum(np.random.dirichlet([1./states]*states))
        cpds = [defaultdict(partial(cpdGen,s,p)) \
                for s,p in zip(states,numparentstates)]
    
    draws = np.empty((numPoints, numnodes), dtype=np.int)
    for i in range(numPoints):
        for node in order:
            parents = adj[:,node]
            parstate = tuple(draws[i,parents==1])
            draws[i,node] = np.searchsorted(cpds[node][parstate], np.random.random())

    return draws, states, cpds

def sampleTemplate(graph, numEdges=3):
    edges = graph.edges()
    ra.shuffle(edges)
    new = graph.copy()
    new.remove_edges_from(new.edges())
    new.add_edges_from(edges[:numEdges])
    return new

#x = generateHourGlassGraph(10, 2)
#visualizeGraph(x)
#y = generateData(x)

