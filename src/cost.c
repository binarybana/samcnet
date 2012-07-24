#include <Judy.h>
#include <math.h>

#include "lib.h"

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
{
  int count, count00, num, num00, i, j, k, m, s, tep;
  int numparent, parstate, parlist[node_num];
  double sum;
  double accum = 0.0;
  double alphaijk = 0.0;
  double alphaik = 0.0;

  for(m=0; m<changelength; m++){
    
    i = changelist[m]; 

    parstate = 1; 
    s = 0;

    for(j=0; j<i-1; j++){
        if(mat[j][i]==1){ 
           parstate*=state[x[j]]; //accumulating the total number of parent sattes
           s++;
           parlist[s]=x[j]; //parent list
         }
       }

    numparent = s; // tot num of parents

    // Structure Prior, in log form:
    fvalue[i] = numparent*log(prior_gamma);

    // Number of parents limited to limparent
    if(numparent>limparent){ 
       for(j=0; j<node_num; j++) fvalue[j]=1.0e+100;
       goto ABC;
      }
    
    /* 
    printf("cost1: %g %d\n", -1.0*fvalue[i], parstate);
    */

    /* parent state table */
    Pvoid_t  Parray = (Pvoid_t) NULL;		// empty JudyL array.
    Pvoid_t  Parray00 = (Pvoid_t) NULL;		// empty JudyL array.
    Word_t * Pvalue;				// value for one index.
    Word_t   index;				// in JudyL array.

    /* data summary: count N_{ijk} */
    count=count00=0;
    for(k=0; k<data_num; k++){ 
       
        for(num00=0,s=numparent; s>=1; s--){
            tep=1;
            for(j=1; j<s; j++) tep*=10; 
            //I'm not so sure...anymore, this may be due to
            //decimal encoding than because of the 10 states
            //in the breast cancer data.
            //FIXME Shouldn't this be state[x[i]] instead of 10?
            num00+=datax[k][parlist[s]]*tep; 
           }
        num=num00*10+datax[k][x[i]]; 
        // ^^ Encode the current data row's state and parent values as a
        // numparents+1 digit decimal number. For data sets with too many
        // nodes >32 or >64, we should worry about overflow.
        //
        // Also, encode just the parents values into decimal number num00
        

        index = (Word_t) num;
        JLI(Pvalue, Parray, index);
        ++(*Pvalue);

        index = (Word_t) num00;
        JLI(Pvalue, Parray00, index);
        ++(*Pvalue);
       } /* end data summary */
          
    if(numparent>8){
      printf("p %d ", numparent);
    }
#ifdef DEBUG
     if(numparent==1) { 
       printf("count: %d, count00: %d, parstate*state: %d\n",
           count,count00,(parstate+1)*state[x[i]]);
       printf("numparent=%d parstate=%d\n", numparent, parstate);

     index = 0;
     JLF(Pvalue, Parray, index);
     while (Pvalue != NULL)
     {
       printf("%lu %lu\n",index, *Pvalue);
       JLN(Pvalue, Parray, index);
     }

     index = 0;
     JLF(Pvalue, Parray00, index);
     while (Pvalue != NULL)
     {
       printf("%lu %lu\n",index, *Pvalue);
       JLN(Pvalue, Parray00, index);
     }

     }
#endif //debug

    JLC(count, Parray, 0,-1);
    JLC(count00, Parray00, 0,-1);
     
    alphaijk=prior_alpha/parstate/state[x[i]];
    alphaik=prior_alpha/parstate;

    accum = 0.0;
    accum-=count*gammln(alphaijk);
    accum+=count00*gammln(alphaik);

    index = 0;
    JLF(Pvalue, Parray, index);
    while (Pvalue != NULL)
    {
     accum+=gammln((*Pvalue) + alphaijk);
     JLN(Pvalue, Parray, index);
    }

    index = 0;
    JLF(Pvalue, Parray00, index);
    while (Pvalue != NULL)
    {
     accum-=gammln((*Pvalue) + alphaik);
     JLN(Pvalue, Parray00, index);
    }

    JLFA(count, Parray);
    JLFA(count00, Parray00);

   fvalue[i] += accum;
   fvalue[i] *= -1.0;

  }
ABC:
  for(sum=0.0,m=0; m<node_num; m++){
      sum+=fvalue[m];
     }

  return sum;
}
