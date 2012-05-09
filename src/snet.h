#ifndef _SNET_H_
#define _SNET_H_

typedef struct {
  int data_num; //number of data records
  int node_num; //variables in bayesian network
  int total_iteration; 
  int burnin;
  int stepscale;
  int grid; //number of regions

  double prior_alpha;
  double prior_gamma;
  int limparent;

  double maxEE,lowE,scale,range;
  double rho,tau,temperature;
  int *state, **datax;
  double *refden;
} simParams;

typedef struct {
  int sze, *bestx, **bestmat, 
         *indicator, nonempty;
  double *bestfvalue, bestenergy;
  double accept_loc,total_loc;
} simResults;

void test(int *data);
void test2(int **data);
simParams* initSimParams(int rows, int cols, int *states, int **data);
int freeSimParams(simParams *params);
int run(simParams* params, simResults* results);

#endif // _SNET_H_
