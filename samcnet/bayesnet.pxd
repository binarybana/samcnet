cimport numpy as np

cdef class BayesNet:
    cdef public:
        object nodes,states,data,graph,x,mat,fvalue,changelist,table
        object oldmat, oldx, oldfvalue
        object gtemplate, ntemplate, ground
        object verbose
        int limparent, data_num, node_num, changelength
        double prior_alpha, prior_gamma
    cdef:
        int **cmat, **cdata
        double **ctemplate

cdef double **npy2c_double(np.ndarray a)
cdef np.ndarray c2npy_double(double **a, int n, int m)
cdef int **npy2c_int(np.ndarray a)
cdef np.ndarray c2npy_int(int **a, int n, int m)
