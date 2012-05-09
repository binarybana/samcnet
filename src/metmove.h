#ifndef _METMOVE_H_
#define _METMOVE_H_

#include "snet.h"

int metmove(simParams *params, 
            simResults *results, 
            int *x,
            int **mat,
            double *fvalue,
            double **hist,
            double delta,
            int *region);

#endif // _METMOVE_H_
