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

def noisylogic(s,p):
    """
    Generate a dictionary where each value is the cumulative distribution of 
    the categorical random variable given the parents configuration as the key.

    In this case, we want to pick a random logic function, and then return that.

    Assuming binary valued nodes for now.
    """
    eps = 0.1
    numparents = int(np.log2(p))
    if numparents == 0:
        if np.random.rand() < 0.5:
            cpd = lambda: np.array([ra.choice([0+eps, 1-eps]), 1.0])
        else:
            cpd = lambda: np.array([0.5, 1.0])
    else:
        cpd = lambda: np.array([ra.choice([0+eps, 1-eps]), 1.0])
    return defaultdict(cpd)

def dirichlet(s, p):
    """
    Generate a dictionary where each value is the cumulative distribution of 
    the categorical random variable given the parents configuration as the key.

    In this case, we will use a random distribution for each state of the 
    parents variables.
    """
    cpd = lambda states: np.cumsum(np.random.dirichlet([1./states]*states))
    return defaultdict(partial(cpd,s))

def generateData(graph, numPoints=50, noise=0.0, cpds=None, method='dirichlet'):
    """ 
    Generate <numPoints> random draws from graph, with 
    randomly assigned CPDs and additive zero mean Gaussian 
    noise with std_dev=noise on the observations.

    """
    
    order = nx.topological_sort(graph)
    numnodes = graph.number_of_nodes()
    adj = np.array(nx.to_numpy_matrix(graph),dtype=np.int)
    states = np.ones(numnodes)*2

    if not cpds:
        numparents = adj.sum(axis=0)
        numparentstates = (states ** numparents)
        #states = np.random.randint(2,3, size=numnodes)
        if method == 'dirichlet':
            func = dirichlet
        elif method == 'noisylogic':
            assert np.all(states == 2) # we can generalize this later
            func = noisylogic

        cpds = [func(s,p) \
            for s,p in zip(states,numparentstates)]
    
    draws = np.empty((numPoints, numnodes), dtype=np.int)
    for i in range(numPoints):
        for node in order:
            parents = adj[:,node]
            parstate = tuple(draws[i,parents==1])
            if np.random.rand() < noise:
                draws[i,node] = np.random.randint(0, states[node])
            else:
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

