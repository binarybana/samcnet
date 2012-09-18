# cython: profile=False
cimport numpy as np
import numpy as np
from itertools import product
from math import log, lgamma
#import bayesnetcpd

from pydai cimport PyVar, PyVarSet, PyFactor, PyJTree#, Var, VarSet

cdef class GroundNet:
    def __init__(self, JointDistribution joint):
        self.entropy = 0.0
        domain = [x.name for x in joint.dists.values()]
        num = len(joint.dists)
        assert np.all(np.arange(num) == np.array(sorted(domain)))
        self.joint = joint
        self.jtree = self.makejtree(joint)
        self.mymarginal = None

    def makejtree(self, joint):
        fs = []
        #print joint
        for dist in joint.dists.values():
            f = self.makeFactor(dist)
            fs.append(f)

        jtree = PyJTree(*fs)
        jtree.init()
        jtree.run()
        return jtree

    def makeFactor(self, CPD cpd):
        """Make a PyFactor from a CPD."""
        cdef int i
        name = cpd.name
        metadomain = cpd.parent_domain.copy()
        metadomain[name] = cpd.arity
        namelist = sorted(metadomain.keys())
        varlist = [PyVar(x,metadomain[x]) for x in namelist]
        vs = PyVarSet(*varlist)
        factor = PyFactor(vs)
        for i,k in zip(range(factor.nrStates()), reverse_space(metadomain)):
            factor.set(i, cpd.p(k))
        #factor.normalize()
        return factor

    def belief_array(self, *vs):
        return self.jtree.belief_array(*vs)

    def belief(self, *vs):
        varlist = [PyVar(x, self.joint.domain[x]) for x in vs]
        return self.jtree.belief(PyVarSet(*varlist))

    def marginal(self, *vs):
        varlist = [PyVar(x, self.joint.domain[x]) for x in vs]
        return self.jtree.marginal(PyVarSet(*varlist))

    def marginal_array(self, *vs):
        return self.jtree.marginal_array(*vs)

    def kld(self, JointDistribution other):
        return self.naivekld(other)
        #return naiveKLD(self.joint, other)

    def naivekld(self, JointDistribution other):
        cdef int i = len(self.joint.domain)
        cdef int j = len(other.domain)
        assert i == j
        if self.mymarginal is not None:
            x = self.mymarginal
        else:
            x = self.marginal_array(*range(i))
            self.mymarginal = x
        ytree = self.makejtree(other)
        y = ytree.marginal_array(*range(i))
        return (x * np.log(x/y)).sum()

    cpdef int mux(self, int state, int pastate, int pos, int numpars):
        cdef int i, count = 0
        s = bin(pastate)[2:].zfill(numpars)
        for i in range(numpars):
            if i == pos:
                count *= 2
                count += state
            count *= 2
            if s[i] == '1':
                count += 1
        if pos == numpars:
            count *= 2
            count += state
        return count

    def calcEntropy(self):
        cdef int i, pos, pa, nodestate, numps, numstates, numpars, nr = len(self.joint.domain)
        cdef np.ndarray [np.double_t, ndim=1, mode="c"] marginal_pn
        cdef np.ndarray [np.double_t, ndim=1, mode="c"] marginal_p
        
        cdef double sum = 0.0
        cdef double subsum = 0.0
        cdef double temp
        for i in range(nr):
            numstates = self.joint.domain[i]
            parents = self.joint.dists[i].sorted_parent_names
            numpars = len(parents)
            parentsandnode = sorted(self.joint.dists[i].sorted_parent_names + [i])
            pos = parentsandnode.index(i)
            marginal_pn = self.marginal_array(*parentsandnode)
            marginal_p = self.marginal_array(*parents)
            parstates = [self.joint.parent_domain[x] for x in parents]
            numps = int(np.array(parstates).prod())
            for pa in range(numps):
                subsum = 0.0
                for nodestate in range(numstates):
                    temp = marginal_pn[self.mux(nodestate, pa, pos, numpars)]/marginal_p[pa]
                    subsum -= temp * log(temp)
                sum += marginal_p[pa] * subsum

            return sum

    def calcNaiveEntropy(self):
        cdef int i = len(self.joint.domain)
        x = self.marginal_array(*range(i))
        self.entropy = -(x * np.log(x)).sum()
        return self.entropy


def naiveKLD(n1,n2):
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
    
    #if isinstance(n1, BayesNetCPD):
        #n1 = n1.joint
    #if isinstance(n2, BayesNetCPD):
        #n2 = n2.joint

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

cpdef fast_space_iterator(domain):
    """Will traverse the entire domain by generating a sequence of tuples.
    >>> d = {'x':2, 'y':2}
    >>> [x for x in fast_space_iterator(d)]
    [(0, 0), (0, 1), (1, 0), (1, 1)]
    """
    if domain == {}:
        return [()]
    _, values = zip(*sorted(domain.items()))
    return [k for k in product(*[range(v) for v in values])]

def reverse_space(domain):
    """Will traverse the entire domain by generating a sequence of lists of tuples.
    Probably slow as heck, but it'll do.
    >>> d = {'x':2, 'y':2}
    >>> [sorted(x.items()) for x in space_iterator(d)]
    [[('x', 0), ('y', 0)], [('x', 1), ('y', 0)], [('x', 0), ('y', 1)], [('x', 1), ('y', 1)]]
    """
    if domain == {}:
        return [()]
    names, values = zip(*sorted(domain.items()))
    x = [k for k in product(*[range(v) for v in values])]
    return [dict(zip(names,v)) for v in zip(*zip(*x)[::-1])]

def space_iterator(domain):
    """Will traverse the entire domain by generating a sequence of lists of tuples.
    >>> d = {'x':2, 'y':2}
    >>> [sorted(x.items()) for x in space_iterator(d)]
    [[('x', 0), ('y', 0)], [('x', 0), ('y', 1)], [('x', 1), ('y', 0)], [('x', 1), ('y', 1)]]
    """
    names, values = zip(*sorted(domain.items()))
    for v in product(*[range(v) for v in values]):
        yield(dict(zip(names,v)))

cdef class JointDistribution:
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
    def __init__(self, cpd=None, cpds=None):
        # FIXME Currently the domain and parent_domain attributes get out of
        # sync when being used from BayesNetCPD
        self.domain = {}
        self.parent_domain = {}
        self.dists = {}
        if cpd:
            assert isinstance(cpd,CPD)
            self.add_distribution(cpd)
        if cpds:
            for cpd in cpds:
                assert isinstance(cpd,CPD)
                self.add_distribution(cpd)

    def p(self,vals):
        """Obtain the probability of this distribution given the
        discrete values in the dictionary vals: {'Name': <value>, ....}.
        Currently does not support marginalization through summing out."""
        accum = 1.0
        for d in self.dists.values():
            accum *= d.p(vals)
        return accum

    def add_distribution(self, other):
        self.dists[other.name] = other
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

    def copy(self):
        # Warning: potentially slow, but it should work
        j = JointDistribution()
        for k,v in self.dists.iteritems():
            j.add_distribution(v.copy())
        return j

    def __repr__(self):
        s = ''
        for name in sorted(self.domain.keys()):
            dist = self.dists[name]
            s += "%r, %r, %r, %r\n" % (dist.name, dist.parent_domain, dist.params, id(dist.parent_domain))
            #s += "%s, %s, %s\n" % (str(dist.name), str(dist.parent_domain), str(dist.params))
        return s

    def __reduce__(self):
        """For pickling and deepcopying"""
        return (JointDistribution, (None,self.dists.values()))

cdef class CPD:
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
    >>> np.all(q.fastp(()) == np.r_[0.1,0.7])
    True
    >>> np.all(p.fastp((0,0)) == np.r_[0.1])
    True
    >>> p.remove_parent('y') < 1e-8
    True
    >>> sorted(p.parent_domain.keys()) == ['z']
    True
    >>> len(p.parent_domain) == 1
    True
    >>> len(p.params) == 2
    True
    """
    def __init__(self, name, arity, params, parent_domain={}):
        self.name = name
        self.arity = arity
        if parent_domain:
            self.parent_domain = parent_domain.copy()
        else:
            self.parent_domain = dict()
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
            self.params = {k:v.copy() for k,v in params.iteritems()}
        else:
            self.params = {(k,):v.copy() for k,v in params.iteritems()}

        # Make sure that the number of indices in the parameter keys
        # are equal to the number of parents
        assert len(self.params.keys()[0]) == len(parent_domain), \
                "Wrong parameter parent keying"

        #Make sure that the number of values in an example parameter are 
        #equal to the arity 
        assert len(self.params.values()[0]) == arity-1, \
                "Wrong parameter format"

        self.parent_arity = np.array(parent_domain.values()).prod()

    def p(self, vals):
        """Obtain the probability of this distribution given the
        discrete values of the parentsin the dictionary vals: 
        {'Name': <value>, ....}.
        """
        # Check that the parents are fully specified
        nodes = set(vals.keys())
        assert nodes >= set(self.sorted_parent_names)
        # I could also check to make sure the values are in bounds

        parent_index = tuple([vals[k] for k in self.sorted_parent_names])
        if vals[self.name] == self.arity-1:
            #sum up params and subtract from 1
            count = self.params[parent_index].sum()
            return 1 - count
        else:
            return self.params[parent_index][vals[self.name]]

    def fastp(self, vals):
        """Like p() but assumes that the correct number of parents are input
        as a tuple of the parents values. Also, returns the entire parameter
        vector."""
        #print "fastp:", self.name, vals, self.parent_domain, self.params
        return self.params[vals]

    def __mul__(self, other):
        j = JointDistribution()
        return j * self * other

    def add_parent(self, name, arity):
        # Update the self.params (copy old into new slots)
        #print self, self.parent_domain, type(self.parent_domain), name, arity
        self.parent_domain[name] = arity
        self.sorted_parent_names = sorted(self.parent_domain.keys())
        self.parent_arity *= arity
        new_params = {}
        new_count = 0

        index = self.sorted_parent_names.index(name)

        for key in fast_space_iterator(self.parent_domain):
            if key[index] == 0:
                new_params[key] = self.params[key[:index] + key[index+1:]].copy()
            else:
                new_count += 1
                new_params[key] = np.random.dirichlet(np.ones(self.arity))[:-1]

        self.params = new_params
        #print "Node %d, Added parent %d, new params: %s" % (self.name,name,str(self.params))
        
        # Calculate logqfactor from arity of new parents (use
        # B of dirichlet distribution from wikipedia)
        return new_count * lgamma(self.arity)

    def remove_parent(self, name):
        # Update the self.params (collapse... averaging or dropping?)
        # dropping:

        #print "REMOVING"
        #print "Node %d, Removing parent %d" % (self.name,name)

        index = self.sorted_parent_names.index(name)
        new_params = {}
        rem_count = 0
        for key in fast_space_iterator(self.parent_domain):
            if key[index] == 0:
                new_params[key[:index] + key[index+1:]] = self.params[key].copy()
            else:
                rem_count += 1
            
        self.params = new_params
        self.parent_arity /= self.parent_domain[name]
        del self.parent_domain[name]
        self.sorted_parent_names = sorted(self.parent_domain.keys())

        # Calculate logqfactor from arity of dropped parents (use
        # B of dirichlet distribution from wikipedia)
        return -rem_count * lgamma(self.arity)

    def move_params(self):
        alpha = np.ones(self.arity)
        for k,v in self.params.iteritems():
            self.params[k] = np.random.dirichlet(alpha)[:-1]

        return 0.0

    def copy(self):
        #print self.arity, self.params, self.parent_domain
        return CPD(self.name, self.arity, self.params, self.parent_domain)

    def __reduce__(self):
        return (CPD, (self.name, self.arity, self.params, self.parent_domain))


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

