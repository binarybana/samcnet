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

    ctypedef struct simResults:
      int sze, *bestx, **bestmat, *indicator, nonempty
      double *bestfvalue, bestenergy
      double accept_loc,total_loc

    void test(int *data)
    void test2(int **data)
    simParams* initSimParams(int rows, int cols, int *states, int **data)
    int freeSimParams(simParams *params)
    int run(simParams* params, simResults* results)
