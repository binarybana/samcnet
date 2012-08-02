cdef extern from "cost.h":
    double cost(int node_num,
                int data_num,
                int limparent,
                int *state,
                int **datax,
                double prior_alpha,
                double prior_gamma,
                int *x, 
                int **mat, 
                double *fvalue, 
                int *changelist, 
                int changelength)

