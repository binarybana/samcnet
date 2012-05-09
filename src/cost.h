#ifndef _COST_H_
#define _COST_H_

#include "snet.h"

double cost(simParams *params,
            simResults *results,
            int *x, 
            int **mat, 
            double *fvalue, 
            int *changelist, 
            int changelength);

#endif // _COST_H_
