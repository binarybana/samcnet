from pydai cimport PyJTree

cpdef object fast_space_iterator(object domain)

cdef class GroundNet:
    cdef public:
        object joint
        object mymarginal

        #object factors
        PyJTree jtree
        double entropy
    cpdef int mux(self, int state, int pastate, int pos, int numpars)

cdef class JointDistribution:
    cdef public:
        object domain, parent_domain, dists

cdef class CPD:
    cdef public:
        object name, parent_domain, sorted_parent_names, params
        int arity, parent_arity

