cdef extern from "snet.h":
    ctypedef struct simParams:
      int data_num
      int node_num
      int total_iteration
      int burnin
      int stepscale
      int grid

      double prior_alpha
      double prior_gamma
      int limparent

      double maxEE,lowE,scale,range
      double rho,tau,temperature
      int *state, **datax
      double *refden
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

