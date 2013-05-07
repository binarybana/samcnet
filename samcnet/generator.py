import networkx as nx
import numpy as np
import random as ra
from collections import defaultdict
from functools import partial
from math import ceil,floor
from probability import CPD,fast_space_iterator,JointDistribution

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
    if nodes < 4:
        return nx.gn_graph(nodes,kernel)
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

def noisylogic(name, arity, pdomain):
    """
    Generate a CPD where each value is the cumulative distribution of 
    the categorical random variable given the parents configuration as the key.

    In this case, we want to pick a random logic function, and then return that.

    Assuming binary valued nodes for now.
    """
    eps = 0.1
    if pdomain == {}:
        if np.random.rand() < 0.5:
            params = {(): np.array([ra.choice([0+eps, 1-eps])])}
        else:
            params = {(): np.array([0.5])}
    else:
        params = {}
        for key in fast_space_iterator(pdomain):
            params[key] = np.array([ra.choice([0+eps, 1-eps])])
    return CPD(name, arity, params, pdomain)

def dirichlet(name, arity, pdomain):
    """
    Generate a CPD where each value is the cumulative distribution of 
    the categorical random variable given the parents configuration as the key.

    In this case, we will use a random distribution for each state of the 
    parents variables.
    """
    params = {}
    for key in fast_space_iterator(pdomain):
        params[key] = np.random.dirichlet([1.]*(arity))[:-1]
    return CPD(name, arity, params, pdomain)

def generateJoint(graph, method='dirichlet'):
    numnodes = graph.number_of_nodes()
    adj = np.array(nx.to_numpy_matrix(graph),dtype=np.int)
    states = np.ones(numnodes)*2
    names = graph.nodes() #or maybe np.arange(numnodes)

    if method == 'dirichlet':
        func = dirichlet
    elif method == 'noisylogic':
        assert np.all(states == 2) # we can generalize this later
        func = noisylogic
    cpds = [func(nd,st,{k:int(states[k]) 
        for k in graph.predecessors(nd)}) for nd,st in zip(names,states)]
    joint = JointDistribution()
    for cpd in cpds:
        joint.add_distribution(cpd)
    return joint, states

def generateData(graph, joint, numpoints=50, noise=0.0, ):
    """ 
    Generate <numpoints> random draws from graph, with 
    randomly assigned CPDs and additive zero mean Gaussian 
    noise with std_dev=noise on the observations.
    """
    numnodes = graph.number_of_nodes()
    order = nx.topological_sort(graph)
    adj = np.array(nx.to_numpy_matrix(graph),dtype=np.int)
    states = np.ones(numnodes)*2 # FIXME this is hardcoded atm
    cpds = zip(*sorted((k,v) for k,v in joint.dists.iteritems()))[1] 
    draws = np.empty((numpoints, numnodes), dtype=np.int)
    for i in range(numpoints):
        for node in order:
            parents = adj[:,node]
            parstate = tuple(draws[i,parents==1])
            if np.random.rand() < noise:
                draws[i,node] = np.random.randint(0, states[node])
            else:
                draws[i,node] = np.searchsorted(np.cumsum(cpds[node].params[parstate]), np.random.random())
    return draws

def sampleTemplate(graph, numEdges=3):
    edges = graph.edges()
    ra.shuffle(edges)
    new = graph.copy()
    new.remove_edges_from(new.edges())
    new.add_edges_from(edges[:numEdges])
    return new

