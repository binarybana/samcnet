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
      int sze, *bestx, **bestmat, *indicator, nonempty, calcs
      double *bestfvalue, bestenergy
      double accept_loc,total_loc

    void copyParamData(simParams* params, double *refden, int *states, char *data)
    simResults* pyInitSimResults(simParams*)

    int freeSimParams(simParams *params)
    void freeSimResults(simParams*, simResults* results)

    #int run(simParams* params, simResults* results)
    #simParams* initSimParams(int rows, int cols, int *states, int **data)
    #simParams* pyInitSimParams(int rows, int cols, double *refden, int *states, char *data)


cdef extern from "metmove.h":
    int metmove(simParams *params, 
                simResults *results, 
                int *x,
                int **mat,
                double *fvalue,
                double **hist,
                double delta,
                int *region)

cdef extern from "cost.h":
    double cost(simParams *params,
                simResults *results,
                int *x, 
                int **mat, 
                double *fvalue, 
                int *changelist, 
                int changelength)

