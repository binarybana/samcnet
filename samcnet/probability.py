import numpy as np
from itertools import product
from math import log



def KLD(n1,n2):
    """Calculates the Kullback-Leibler divergence or relative entropy between 
    two distributions n1 and n2.
    >>> d1 = CPD('x', 2, {():.1})
    >>> d2 = CPD('x', 2, {():.5})
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

    if isinstance(n1, CPD):
        n1 = JointDistribution(n1)
    if isinstance(n2, CPD):
        n2 = JointDistribution(n2)

    assert n1.domain == n2.domain
    accum = 0.0
    for v in space_iterator(n1.domain):
        p1 = n1.p(v)
        p2 = n2.p(v)
        if p1 < 1e-30:
            pass
        elif p2 < 1e-30:
            raise Exception("Small Q(i) but not small P(i)")
        else:
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

class JointDistribution():
    """Contains a collection of distributions that when multiplied together
    are the joint distribution
    >>> node1 = CPD('y', 2, {0:0.25, 1:0.9}, dict(x=2))
    >>> node2 = CPD('x', 2, {(): 0.25}) 
    >>> j = node1*node2
    >>> isinstance(j, JointDistribution)
    True
    >>> j.p(dict(x=0, y=0))
    0.0625
    >>> j = j*CPD('b', 3, {():[0.5,0.4]})
    >>> j.p(dict(x=1, y=0, b=0)) - 0.3375 < 1e-8
    True
    >>> j2 = j * (node1 * node2)
    Traceback (most recent call last):
        ...
    Exception: Not Implemented

    """
    def __init__(self, cpd=None):
        self.domain = {}
        self.parent_domain = {}
        self.dists = []
        if cpd:
            assert isinstance(cpd,CPD)
            self.add_distribution(cpd)

    def p(self,vals):
        """Obtain the probability of this distribution given the
        discrete values in the dictionary vals: {'Name': <value>, ....}.
        Currently does not support marginalization through summing out."""
        accum = 1.0
        for d in self.dists:
            accum *= d.p(vals)
        return accum

    def add_distribution(self, other):
        self.dists.append(other)
        assert other.name not in self.domain
        self.domain[other.name] = other.arity
        self.parent_domain = intersect_domains(self.parent_domain, other.parent_domain)
        for x in self.domain:
            if x in self.parent_domain:
                del self.parent_domain[x]

    def __mul__(self, other):
        if isinstance(other, CPD):
            self.add_distribution(other)
            return self
        #else isinstance(other, JointDistribution):
        raise Exception("Not Implemented")

class CPD():
    """A univariate conditional probability distribution with a variable name, 
    arity, parent domain being a
    dict of names and arities and params being a dict:
    {<tuple of parent variables>: <numpy array of length arity-1} 
    or 
    {value of parent: <numpy array of length arity-1} 
    for singleton parent nodes.
    NB: Each parameter value is the density that value <name> will take i where
    i is the index of the parameter value. This is opposite of how most people
    see binary valued parameters.
    >>> p = CPD('x', 2, {(0,0):np.r_[0.1], (0,1):np.r_[0.2],
    ... (1,0):np.r_[0.01], (1,1):np.r_[0.9]}, dict(y=2, z=2))
    >>> p.p(dict(x=1, y=0, z=0)) == 0.9
    True
    >>> p.p(dict(x=1, y=1, z=0)) == 0.99
    True
    >>> p.p(dict(x=0, y=0, z=1)) == 0.2
    True
    >>> q = CPD('x', 3, {():np.r_[0.1, 0.7]})
    >>> q.p(dict(x=1)) == 0.7
    True
    >>> q.p(dict(x=2)) - 0.2 < 1e-8
    True
    """
    def __init__(self, name, arity, params, parent_domain={}):
        self.name = name
        self.arity = arity
        self.parent_domain = parent_domain
        self.parent_names = set(parent_domain.keys())
        self.sorted_parent_names = sorted(parent_domain.keys())

        # Convert the parameter values into numpy arrays whether they
        # start out as lists, floats or leave them as ndarrays
        if isinstance(params.values()[0], list):
            params = {k:np.array(v) for k,v in params.iteritems()}
        elif isinstance(params.values()[0],np.ndarray):
            pass
        elif isinstance(params.values()[0], float):
            params = {k:np.r_[v] for k,v in params.iteritems()}
        else:
            raise Exception("Weird CPD parameter value")

        # Convert parameter value keys to tuples
        if isinstance(params.keys()[0], tuple):
            self.params = params
        else:
            self.params = {(k,):v for k,v in params.iteritems()}

        # Make sure that the number of indices in the parameter keys
        # are equal to the number of parents
        assert len(self.params.keys()[0]) == len(parent_domain), \
                "Wrong parameter parent keying"

        #Make sure that the number of values in an example parameter are 
        #equal to the arity 
        assert len(self.params.values()[0]) == arity-1, \
                "Wrong parameter format"

    def p(self, vals):
        """Obtain the probability of this distribution given the
        discrete values of the parentsin the dictionary vals: 
        {'Name': <value>, ....}.
        """
        # Check that the parents are fully specified
        nodes = set(vals.keys())
        assert nodes >= self.parent_names
        # I could also check to make sure the values are in bounds

        parent_index = tuple([vals[k] for k in self.sorted_parent_names])
        if vals[self.name] == self.arity-1:
            #sum up params and subtract from 1
            count = self.params[parent_index].sum()
            return 1 - count
        else:
            return self.params[parent_index][vals[self.name]]

    def __mul__(self, other):
        j = JointDistribution()
        return j * self * other

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

