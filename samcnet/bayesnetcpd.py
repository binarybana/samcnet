import numpy as np
from itertools import product
from math import log

def KLD(n1,n2):
    """Calculates the Kullback-Leibler divergence or relative entropy between 
    two distributions n1 and n2.
    >>> d1 = Distribution(dict(x=2), {0:.1, 1:.9})
    >>> d2 = Distribution(dict(x=2), {0:.5, 1:.5})
    >>> "%.3f" % KLD(d1,d2)
    '0.368'
    >>> "%.3f" % KLD(d2,d1)
    '0.511'
    >>> "%.3f" % KLD(d1,d1)
    '0.000'
    >>> "%.3f" % KLD(d2,d2)
    '0.000'
    """
    
    if isinstance(n1, BayesNetCPD):
        n1 = n1.joint
    if isinstance(n2, BayesNetCPD):
        n2 = n2.joint

    assert n1.domain == n2.domain
    accum = 0.0
    for v in space_iterator(n1.domain):
        p1 = n1.p(v)
        p2 = n2.p(v)
        accum += p1 * log(p1/p2)

    return accum

def space_iterator(domain):
    """Will traverse the entire domain by generating a sequence of tuples.
    >>> d = {'x':2, 'y':2}
    >>> [sorted(x.items()) for x in space_iterator(d)]
    [[('x', 0), ('y', 0)], [('x', 0), ('y', 1)], [('x', 1), ('y', 0)], [('x', 1), ('y', 1)]]
    """
    names, values = zip(*sorted(domain.items()))
    for v in product(*[range(v) for v in values]):
        yield(dict(zip(names,v)))

class BayesNetCPD():
    def __init__(self):
        self.nodes = []
        self.joint = JointDistribution()
        self.graph = None # TBD
        
    def add_edge(self,n1,n2):
        #check order
        #add edge
        #adjust parents
        pass
    def rem_edge(self,n1,n2):
        #remove edge
        #adjust parents
        pass

    adjust param

class BayesNode():
    def __init__(self, name, parents, dist):
        self.name = name
        self.parents = parents
        self.dist = dist

class JointDistribution():
    """Contains a collection of distributions that when multiplied together
    are the joint distribution
    >>> node1 = Distribution(dict(x=2,y=2), {(0,0):0.25, (0,1):0.5, (1,0):0.0, 
    ... (1,1):0.25})
    >>> node2 = Distribution(dict(z=2), {0:0.25, 1:0.5}) 
    >>> j = node1*node2
    >>> isinstance(j, JointDistribution)
    True
    >>> j.p(dict(x=0, y=0, z=1))
    0.125
    >>> j = j*Distribution(dict(b=3), {0:0.5, 1:0.4, 2:0.1})
    >>> j.p(dict(x=0, y=0, z=1, b=0))
    0.0625
    >>> j2 = j * (node1 * node2)
    Traceback (most recent call last):
        ...
    Exception: Not Implemented

    """
    def __init__(self):
        self.domain = {}
        self.dists = []
    def p(self,vals):
        """Obtain the probability of this distribution given the
        discrete values in the dictionary vals: {'Name': <value>, ....}.
        Currently does not support marginalization through summing out."""
        accum = 1.0
        for d in self.dists:
            accum *= d.p(vals)
        return accum

    def add_distribution(self, other):
        self.domain = intersect_domains(self.domain, other.domain)
        self.dists.append(other)

    def __mul__(self, other):
        if isinstance(other, Distribution):
            return other.__mul__(self)
        #else isinstance(other, JointDistribution):
        raise Exception("Not Implemented")

class Distribution():
    """A probability distribution (possibly conditional) with domain
    being a dict of names and arities and values being a dict of tuples of 
    values to probabilities.

    >>> p = Distribution(dict(x=2,y=2), {(0,0):0.25, (0,1):0.5, (1,0):0.0, 
    ... (1,1):0.25})
    >>> p.p(dict(x=1, y=0))
    0.0
    """
    def __init__(self, domain, values):
        self.domain = domain
        if isinstance(values.keys()[0], tuple):
            self.vals = values
        else:
            self.vals = {(k,):v for k,v in values.iteritems()}
        self.names = set(domain.keys())
        self.sorted_names = sorted(domain.keys())
    def p(self, vals):
        """Obtain the probability of this distribution given the
        discrete values in the dictionary vals: {'Name': <value>, ....}.
        Currently does not support marginalization through summing out."""

        # Check that the index is fully specified
        nodes = set(vals.keys())
        inter = nodes.intersection(self.names)
        assert inter == self.names
        # I could also check to make sure the values are in bounds

        index = tuple([vals[k] for k in self.sorted_names])
        return self.vals[index]

    def __mul__(self, other):
        if isinstance(other, Distribution):
            j = JointDistribution()
            j.add_distribution(self)
            j.add_distribution(other)
            return j
        elif isinstance(other, JointDistribution):
            other.add_distribution(self)
            return other
        else:
            raise Exception("Trying to multiply something weird against a Distribution().")

def intersect_domains(x,y):
    """First assert that these probability spaces are compatible (in arity)
    and then combine the two dictionaries into one. The format of the 
    dictionaries is {'Name': <arity>, ...}"""
    nodes = set(x.keys())
    inter = nodes.intersection(set(y.keys()))
    for i in inter:
        assert x[i] == y[i]
    new = x.copy()
    new.update(y)
    return new

if __name__ == '__main__':
    import doctest
    doctest.testmod()

