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
        double **priormat,
        int *x, 
        int **mat, 
        double *fvalue, 
        int *changelist, 
        int changelength)
{
  int num_obs_p_states, num_obs_pn_states, node_state, parent_state, i, j, k, m, s, tep;
  int numparent, parstate, parlist[node_num];
  double sum, priordiff;
  double accum = 0.0;
  double alphaijk = 0.0;
  double alphaik = 0.0;

  for(m=0; m<changelength; m++){
    
    i = changelist[m]; 

    parstate = 1; 
    s = 0;

    for(j=0; j<i; j++){
        if(mat[j][i]==1){ 
           parstate*=state[x[j]]; //accumulating the total number of parent states
           parlist[s]=x[j]; //parent list
           s++;
         }
       }

    numparent = s; // tot num of parents

    // Structure Prior, in log form:
    priordiff = 0.0;
    for(j=0; j<node_num; j++){
        if(j!=i){
            priordiff += abs((double)mat[j][i] - priormat[x[j]][x[i]]);
        }
    }
    fvalue[i] = -(priordiff+numparent)*prior_gamma;
    // FIXME Need to move this to the end, as it could cause small errors.
    /*fvalue[i] = numparent*log(0.1111);*/

    // Number of parents limited to limparent
    if(numparent>limparent){ 
       /*for(j=0; j<node_num; j++) fvalue[j]=1.0e+100;*/
       fvalue[i]=1.0e+100;
       goto ABC;
      }
    
    /* 
    printf("cost1: %g %d\n", -1.0*fvalue[i], parstate);
    */

    /* parent state table */
    Pvoid_t  node_p_array = (Pvoid_t) NULL;		// empty JudyL array.
    Pvoid_t  parent_array = (Pvoid_t) NULL;		// empty JudyL array.
    Word_t * Pvalue;				// value for one index.
    Word_t   index;				// in JudyL array.

    /* data summary: count N_{ijk} */
    num_obs_p_states=num_obs_pn_states=0;
    for(k=0; k<data_num; k++){ 
       
        for(parent_state=0,s=numparent-1; s>=0; s--){
            tep=1;
            for(j=0; j<s; j++) tep*=10; 
            // ^^ This assumes that all the nodes have 10 or less
            // states.
            parent_state += datax[k][parlist[s]]*tep; 
           }
        node_state = parent_state*10+datax[k][x[i]]; 
        // ^^ Encode the current data row's state and parent values as a
        // numparents+1 digit decimal number. For data sets with too many
        // nodes >32 or >64, we should worry about overflow.
        //
        // Also, encode just the parents values into decimal number parent_state

        index = (Word_t) node_state;
        JLI(Pvalue, node_p_array, index);
        ++(*Pvalue);

        index = (Word_t) parent_state;
        JLI(Pvalue, parent_array, index);
        ++(*Pvalue);
       } /* end data summary */
          
    JLC(num_obs_pn_states, node_p_array, 0,-1);
    JLC(num_obs_p_states, parent_array, 0,-1);
     
#ifdef DEBUG
     if(x[i] == 0) { 

       printf("\ni: %d \nm: %d \nx[i]: %d\nnumparents: %d \nparent: %d\ncount_parent: %d \
           \ncount_parent_node: %d \nparstates: %d \nstates: %d \
           \nparstate*state: %d\n", 
            i, m, x[i], numparent, parlist[0], num_obs_p_states,num_obs_pn_states,parstate, 
            state[x[i]], parstate*state[x[i]]);

     printf("node_p_array:\n");
     index = 0;
     JLF(Pvalue, node_p_array, index);
     while (Pvalue != NULL)
     {
       printf("%lu %lu\n",index, *Pvalue);
       JLN(Pvalue, node_p_array, index);
     }

     printf("parent_array:\n");
     index = 0;
     JLF(Pvalue, parent_array, index);
     while (Pvalue != NULL)
     {
       printf("%lu %lu\n",index, *Pvalue);
       JLN(Pvalue, parent_array, index);
     }

     }
#endif //debug

    alphaijk=prior_alpha/parstate/state[x[i]];
    alphaik=prior_alpha/parstate;

    accum = 0.0;
    accum -= num_obs_pn_states*gammln(alphaijk);
    accum += num_obs_p_states*gammln(alphaik);

    index = 0;
    JLF(Pvalue, node_p_array, index);
    while (Pvalue != NULL)
    {
     accum += gammln((*Pvalue) + alphaijk);
     JLN(Pvalue, node_p_array, index);
    }

    index = 0;
    JLF(Pvalue, parent_array, index);
    while (Pvalue != NULL)
    {
     accum -= gammln((*Pvalue) + alphaik);
     JLN(Pvalue, parent_array, index);
    }

    JLFA(num_obs_p_states, node_p_array);
    JLFA(num_obs_pn_states, parent_array);

   fvalue[i] += accum;
   fvalue[i] *= -1.0;

#ifdef DEBUG
   printf("i: %d\t x[i]: %d\t fvalue[i]: %f\t fvalue[x[i]]: %f\n",i, x[i], fvalue[i], fvalue[x[i]]);
#endif

  }
ABC:
  for(sum=0.0,m=0; m<node_num; m++){
      sum+=fvalue[m];
     }

  return sum;
}
