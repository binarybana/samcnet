#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define pi 3.14159265
#define data_num       683 
#define node_num        10

#define total_iteration     500000
#define burnin              300000
#define stepscale           100000
#define WARM                     1
//#define warm_iteration      10000

static double prior_alpha=1.0, prior_gamma=0.1111;
static int    limparent=5;

static int **datax, sze, *bestx, **bestmat, *state, *indicator, nonempty;
static double *bestfvalue, bestenergy;
static double maxE,lowE=8000.0, maxEE=25000.0, scale=1.0, range=0.0;
static double rho=1.0, tau=1.0, temperature=1.0;
static double *refden, accept_loc,total_loc;
