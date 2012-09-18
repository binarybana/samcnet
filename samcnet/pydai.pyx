# cython: profile=False
cimport cython 
from dai_bind cimport Var, VarSet, Factor, FactorGraph, JTree, PropertySet
from libcpp.vector cimport vector
from libcpp.string cimport string
from cython.operator cimport dereference as deref

cimport numpy as np
import numpy as np

cdef class PyVar:
    """A variable, initialized with (int, int) being the label
    and arity"""
    def __cinit__(self, *args):#int label, int states):
        if len(args) == 0:
            self.thisptr = NULL 
            return
        assert len(args) == 2
        #print "ALLOCATE PyVar"
        self.thisptr = new Var(<int>args[0], <int>args[1])
    def __dealloc__(self):
        if self.thisptr is not NULL:
            #print "DELETING PyVar"
            del self.thisptr
        else:
            print "Possible memory leak PyVar"
    def states(self):
        if self.thisptr is not NULL:
            return self.thisptr.states()
        else:
            raise MemoryError()
    def label(self):
        if self.thisptr is not NULL:
            return self.thisptr.label()
        else:
            raise MemoryError()

cdef class PyVarSet:
    """A set of variables initialized with a variable
    number of PyVars."""
    def __cinit__(self, *args):
        cdef int i 
        cdef vector[Var] vargs
        if len(args) == 0:
            self.thisptr = NULL 
            return
        for i in range(len(args)):
            assert isinstance(args[i], PyVar)
            vargs.push_back(Var(args[i].label(), args[i].states()))
        self.thisptr = new VarSet(vargs.begin(), vargs.end(), vargs.size())
        #print "ALLOCATE PyVarSet"
        if self.thisptr is NULL:
            raise MemoryError()
    def __dealloc__(self):
        if self.thisptr is not NULL:
            #print "DELETING PyVarSet"
            del self.thisptr
        else:
            print "Possible memory leak PyVarSet"
    def __len__(self):
        if self.thisptr is not NULL:
            return self.thisptr.size()
        else:
            raise MemoryError()
    #cdef copy(self):
        #if self.thisptr is not NULL:

            #return self.thisptr.size()
        #else:
            #raise MemoryError()

cdef class PyFactor:
    """A factor initialized with either a PyVar or a PyVarSet."""
    def __cinit__(self, *arg):
        if len(arg) == 0:
            self.thisptr = NULL
            #print "Created Empty PyFactor"
            return 
        assert len(arg) == 1
        arg = arg[0]
        if isinstance(arg,PyVar):
            self.thisptr = new Factor(deref((<PyVar>arg).thisptr))
        elif isinstance(arg,PyVarSet):
            self.thisptr = new Factor(deref((<PyVarSet>arg).thisptr))
        #print "ALLOCATED new PyFactor"
        if self.thisptr is NULL:
            raise MemoryError()
    def __dealloc__(self):
        if self.thisptr is not NULL:
            #print "DELETING PyFactor"
            del self.thisptr
        else:
            print "Possible memory leak PyFactor"

    def nrStates(self):
        if self.thisptr is not NULL:
            return self.thisptr.nrStates()
        else:
            raise MemoryError()
    def entropy(self):
        if self.thisptr is not NULL:
            return self.thisptr.entropy()
        else:
            raise MemoryError()
    def get(self, int i):
        if self.thisptr is not NULL:
            return self.thisptr.get(i)
        else:
            raise MemoryError()
    def set(self, int i, double value):
        if self.thisptr is not NULL:
            self.thisptr.set(i,value)
        else:
            raise MemoryError()
    def vars(self):
        cdef VarSet *copy = new VarSet()
        if self.thisptr is not NULL:
            copy[0] = self.thisptr.vars()

            pv = PyVarSet()
            pv.thisptr = copy

            return pv
        else:
            raise MemoryError()
    def marginal(self, PyVarSet vs):
        cdef Factor *copy = new Factor()
        if self.thisptr is not NULL:
            copy[0] = self.thisptr.marginal(deref(vs.thisptr))

            pv = PyFactor()
            pv.thisptr = copy
            return pv
        else:
            raise MemoryError()
    def normalize(self):
        if self.thisptr is not NULL:
            self.thisptr.normalize()
        else:
            raise MemoryError()
    def __repr__(self):
        if self.thisptr is NULL:
            raise MemoryError()

        cdef int states = self.thisptr.nrStates()
        cdef VarSet vs = self.thisptr.vars()
        cdef vector[Var] celements = vs.elements()
        s = ''
        elements = []
        for i in range(vs.size()):
            elements.append((celements[i].label(), celements[i].states()))
        s += '[(State, Arity)]:\n'
        s += str(elements)
        s += '\nState: value\n'
        for i in range(states):
            s += '  %3d: %3f\n' % (i,self.thisptr.get(i))
        return s
    def __array__(self):
        cdef int nr = self.thisptr.nrStates()
        cdef np.ndarray[np.double_t, ndim=1, mode="c"] x = np.empty(nr, dtype=np.double)
        cdef int i
        for i in range(nr):
            x[i] = self.thisptr.get(i)
        return x

cdef np.ndarray [np.double_t, ndim=1, mode="c"] factor_to_array(Factor f):
    cdef int nr = f.nrStates()
    cdef np.ndarray[np.double_t, ndim=1, mode="c"] x = np.empty(nr, dtype=np.double)
    cdef int i
    for i in range(nr):
        x[i] = f.get(i)
    return x

cdef class PyJTree:
    """A junction tree for efficient inference, initialized with a varargs number of 
    PyFactors."""
    def __cinit__(self, *facs):
        cdef int i 
        cdef vector[Factor] vargs
        for i in range(len(facs)):
            assert isinstance(facs[i], PyFactor)
            vargs.push_back(deref((<PyFactor?>(facs[i])).thisptr)) # Should we be copying here instead??
        self.fg = new FactorGraph(vargs)
        if self.fg is NULL:
            raise MemoryError()
        cdef PropertySet ps = PropertySet('[updates=HUGIN]')
        self.thisptr = new JTree(deref(self.fg), ps)
        #print "ALLOCATED Jtree"
        if self.thisptr is NULL:
            raise MemoryError()

    def __dealloc__(self):
        if self.thisptr is not NULL and self.fg is not NULL:
            #print "DELETING PyJtree"
            del self.thisptr
            del self.fg
        else:
            print "Possible memory leak PyFactor"

    def init(self):
        if self.thisptr is not NULL:
            self.thisptr.init()
        else:
            raise MemoryError()
    def run(self):
        if self.thisptr is not NULL:
            self.thisptr.run()
        else:
            raise MemoryError()
    def iterations(self):
        if self.thisptr is not NULL:
            return self.thisptr.Iterations()
        else:
            raise MemoryError()

    def belief_array(self, *vs):
        cdef vector[Var] vec
        if self.fg is NULL or self.thisptr is NULL:
            raise MemoryError()
        if len(vs)==1:
            return factor_to_array(self.thisptr.belief(VarSet(self.fg.var(vs[0]))))
        elif len(vs)==2:
            return factor_to_array(self.thisptr.belief(VarSet(self.fg.var(vs[0]), self.fg.var(vs[1]))))
        else:
            for i in vs:
                vec.push_back(self.fg.var(i))
            return factor_to_array(self.thisptr.belief(VarSet(vec.begin(), vec.end(), vec.size())))

    def belief(self, PyVarSet vs):
        cdef Factor *copy = new Factor()
        if self.thisptr is not NULL:
            copy[0] = self.thisptr.belief(deref(vs.thisptr))

            pv = PyFactor()
            pv.thisptr = copy
            return pv
        else:
            raise MemoryError()

    def marginal_array(self, *vs):
        cdef vector[Var] vec
        if self.fg is NULL or self.thisptr is NULL:
            raise MemoryError()
        if len(vs)==1:
            return factor_to_array(self.thisptr.calcMarginal(VarSet(self.fg.var(vs[0]))))
        elif len(vs)==2:
            return factor_to_array(self.thisptr.calcMarginal(VarSet(self.fg.var(vs[0]), self.fg.var(vs[1]))))
        else:
            for i in vs:
                vec.push_back(self.fg.var(i))
            return factor_to_array(self.thisptr.calcMarginal(VarSet(vec.begin(), vec.end(), vec.size())))

    def marginal(self, PyVarSet vs):
        cdef Factor *copy = new Factor()
        if self.thisptr is not NULL:
            copy[0] = self.thisptr.calcMarginal(deref(vs.thisptr))

            pv = PyFactor()
            pv.thisptr = copy
            return pv
        else:
            raise MemoryError()

if __name__ == '__main__':
    import doctest
    doctest.testmod()

