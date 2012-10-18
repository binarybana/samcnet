from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.map cimport map

cdef extern from "dai/var.h" namespace "dai":
    cdef cppclass Var:
        Var(size_t, size_t)
        size_t states()
        size_t label()

#cdef extern from "dai/smallset.h" namespace "dai":
    #cdef cppclass SmallSet[T]:
        #SmallSet()
        #SmallSet(T)

cdef extern from "dai/util.h" namespace "dai":
    ctypedef void* BigInt
    size_t BigInt_size_t(BigInt &)

cdef extern from "dai/varset.h" namespace "dai":
    size_t calcLinearState( VarSet &, map[Var, size_t] &)
    map[Var, size_t] calcState( VarSet &, size_t)
    cdef cppclass VarSet:
        VarSet()
        #VarSet(SmallSet[Var] &)
        VarSet(Var &)
        VarSet(Var &, Var &)
        VarSet(vector[Var].iterator, vector[Var].iterator, size_t)
        BigInt nrStates()
        #SmallSet[Var] & operator|(SmallSet[Var] &)
        size_t size()
        vector[Var] & elements()
        vector[Var].iterator begin()
        vector[Var].iterator end()
        VarSet& insert(Var &)
        VarSet& erase(Var &)
        #VarSet& remove(VarSet &)
        #VarSet& add(VarSet &)

cdef extern from "dai/properties.h" namespace "dai":
    cdef cppclass PropertySet:
        PropertySet()
        PropertySet(string)

cdef extern from "dai/factor.h" namespace "dai":
    cdef cppclass TFactor[T]:
        TFactor()
        TFactor(Var &)
        TFactor(VarSet &)
        void set(size_t, T)
        T get(size_t)
        VarSet & vars()
        size_t nrStates()
        T entropy()
        TFactor[T] marginal(VarSet &)
        TFactor[T] embed(VarSet &)
        T normalize()

ctypedef TFactor[double] Factor

cdef extern from "dai/factorgraph.h" namespace "dai":
    cdef cppclass FactorGraph:
        FactorGraph()
        FactorGraph(vector[Factor] &)
        Var & var(size_t)
        FactorGraph* clone()
        double logScore(vector[long unsigned int] &)
        Factor factor(int)
        void setFactor(int, Factor, bool)
        void clearBackups()
        void restoreFactors()

cdef extern from "dai/jtree.h" namespace "dai":
    cdef cppclass JTree:
        JTree(FactorGraph &, PropertySet &)
        void init()
        void run()
        size_t Iterations()
        string printProperties()
        Factor calcMarginal(VarSet &)
        Factor belief(VarSet &)
