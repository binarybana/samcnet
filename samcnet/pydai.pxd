from dai_bind cimport Var, VarSet, Factor, FactorGraph, JTree
cimport numpy as np

cdef np.ndarray [np.double_t, ndim=1, mode="c"] factor_to_array(Factor f)

cdef class PyVar:
    cdef Var *thisptr
cdef class PyVarSet:
    cdef VarSet *thisptr
cdef class PyFactor:
    cdef Factor *thisptr
cdef class PyJTree:
    cdef JTree *thisptr
    cdef FactorGraph *fg
