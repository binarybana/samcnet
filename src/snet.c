#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <malloc.h>

#include "nrutil.h"
#include "cost.h"
#include "metmove.h"
#include "lib.h"

int run(simParams* params, simResults* results) {
int stime;
long ltime;
int i, j, k0, iter, Repeat; //s,k1,k2
int region,changelength, *changelist, *x, **mat;
double delta,**hist,*fvalue,sum,ave,max,un,maxE;
double **net, sinw, totalweight;
FILE *ins, *ins2;
/*char buf[255];*/

ltime=time(NULL);
stime=(unsigned int)ltime/2;
/*srand(stime);*/
/*srand(912345);*/


hist=dmatrix(0,params->grid,1,3);
changelist=ivector(1,params->node_num);
fvalue=dvector(1,params->node_num); 
x=ivector(1,params->node_num);
mat=imatrix(1,params->node_num,1,params->node_num);
net=dmatrix(1,params->node_num,1,params->node_num);


ins=fopen("ee.log","a");
fprintf(ins, "random seed=%d  t0=%d tau=%g rho=%g scale=%g iteration=%d\n",
    stime,params->stepscale,params->tau,params->rho,params->scale,params->total_iteration);
fclose(ins);

ins=fopen("ee.sum","a");
fprintf(ins, "random seed=%d  t0=%d tau=%g rho=%g  scale=%g iteration=%d\n",
    stime,params->stepscale,params->tau,params->rho,params->scale,params->total_iteration);
fclose(ins);


maxE=params->maxEE; 
params->grid=ceil((params->maxEE+params->range-params->lowE)*params->scale);

/* Initialization */
for(sum=0.0, i=0; i<=params->grid; i++){ 
  params->refden[i]=(params->grid-i+1)*(params->grid-i+1); 
  sum+=params->refden[i]; 
  }
for(i=0; i<=params->grid; i++) params->refden[i]/=sum;
for(i=0; i<=params->grid; i++){
    hist[i][1]=params->lowE+i*1.0/params->scale;
    hist[i][2]=0.0;
    hist[i][3]=0.0;
   }


/* initialization of the network */
permut_sample(x,params->node_num); //get a 1-10 permutation
for(i=1; i<=params->node_num; i++)
   for(j=1; j<=params->node_num; j++) mat[i][j]=0;
for(i=1; i<=params->node_num; i++) mat[i][i]=1; 
//^^ initialize mat to identity
 
changelength=params->node_num;
for(i=1; i<=changelength; i++) changelist[i]=i; //changelist== 1,2,3...
results->bestenergy = cost(params,results,x,mat,fvalue,changelist,changelength);
printf("initial energy=%g\n",results->bestenergy);

for(i=1; i<=params->node_num; i++){
    maxE=results->bestenergy;
    results->bestx[i] = x[i];
    // bestx[i] and x[i] and are the node labels for the columns of the
    // adjacency matrix
    results->bestfvalue[i] = fvalue[i]; 
    // ^^ fvalue is the additive fit of the data for each node
    for(j=1; j<=params->node_num; j++) results->bestmat[i][j]=mat[i][j];
    //^^ copy mat to bestmat
   }

if(maxE > params->maxEE || maxE < params->lowE){
   ins=fopen("ee.log","a");
   fprintf(ins, "initial maxE=%g\n", maxE);
   fprintf(ins,"The choice of maxEE is too small or the choice of lowE is too large.\n");
   fclose(ins);
  }

for(i=0; i<=params->grid; i++) results->indicator[i]=0;
//^^ indicator[i] is whether we have visited a region yet.
for(sum=0.0,i=1; i<=params->node_num; i++) sum+=fvalue[i];
if(sum > params->maxEE+params->range) results->nonempty=results->sze; //higher than maxEE
  else if(sum<params->lowE) results->nonempty=0; //lower than lowE
     else results->nonempty=floor((sum-params->lowE)*params->scale); 
results->indicator[results->nonempty]=1; //set bin to nonempty
// nonempty is the currently lowest energy visited grid region

for(i=1; i<=params->node_num; i++)
  for(j=1; j<=params->node_num; j++) net[i][j]=0.0;
// net is something like the the energy for each link

ins=fopen("ee.log","a"); 
iter = 1; 
results->accept_loc = results->total_loc = 0.1; 
// accept_loc and total_loc are the accepted moves and total move proposals
totalweight = 0.0;

while(iter<=params->total_iteration){

   if(iter <= params->stepscale) delta = params->rho;
      else delta = params->rho*exp(-params->tau*log(iter/params->stepscale));
      // delta = rho * exp(-tau*log(iter/stepscale))

   metmove(params,results,x,mat,fvalue,hist,delta, &region);
   for(sum=0.0, i=1; i<=params->node_num; i++) sum+=fvalue[i]; 

   if(iter==params->burnin){
      i=0;
 printf("s2.8\n");
      while(i<=params->grid && results->indicator[i]==0) i++;
      results->nonempty=i;
      //^^ nonempty is the lowest nonempty region
     }

   sinw=1;
   if(iter>params->burnin){ 
    
      un=hist[results->nonempty][2];
      for(i=0; i<=params->grid; i++){
          if(results->indicator[i]==1) hist[i][2]-=un;
         }
      
      sinw=exp(hist[region][2]);
      for(i=1; i<=params->node_num; i++)
         for(j=i+1; j<=params->node_num; j++)
           if(mat[i][j]==1) net[x[i]][x[j]]+=sinw;
      totalweight+=sinw;
     }

      
   if(iter%10000==0){
      fprintf(ins,"%g %d %g %g\n",delta,iter,results->bestenergy,sum);
      printf("nonempty=%d \t hist=%g %g %g %g\n", 
          results->nonempty,
          hist[results->nonempty][2],
          hist[results->nonempty+1][2],
          hist[results->nonempty+2][2], 
          hist[results->nonempty+3][2]);
      printf("delta=%g iter=%d bestenergy=%g current=%g weight=%g region=%d\n",delta,iter,
             results->bestenergy,sum,sinw,region);
     }  

   iter++;

   if(iter%100000==0){
      ins2=fopen("ee.sol","a");
      for(sum=0.0, i=1; i<=params->node_num; i++) sum+=results->bestfvalue[i];
      fprintf(ins2,"total energy value=%g\n", sum);
      for(i=1; i<=params->node_num; i++) fprintf(ins2, " %d", results->bestx[i]);
      fprintf(ins2,"\n");
      for(i=1; i<=params->node_num; i++){
         for(j=1; j<=params->node_num; j++) fprintf(ins2, " %d",results->bestmat[i][j]);
         fprintf(ins2, "\n");
        }
      fprintf(ins2,"\n\n");
      fclose(ins2);
     }
 }
fclose(ins);



for(sum=0.0,k0=0,i=0; i<=results->sze; i++)
   if(hist[i][3]<=0.0){ sum+= params->refden[i]; k0++; }
if(k0>0) ave=sum/k0;
   else ave=0.0;
for(i=0; i<=results->sze; i++) hist[i][2]=hist[i][2]+log( params->refden[i]+ave);
max=hist[0][2];
for(i=1; i<=results->sze; i++)
   if(hist[i][2]>max) max=hist[i][2];
for(sum=0.0, i=0; i<=results->sze; i++){ hist[i][2]-=max; sum+=exp(hist[i][2]); }
for(i=0; i<=results->sze; i++) hist[i][2]=hist[i][2]-log(sum)+log(100.0);

// ^^ Processing the history, first finding the average target density in
// the unvisited regions, then adding the log of the target density plus
// the average to second column of the history
// Then finding the maximum of the second column and subtracting that off
// (log) and adding log(100).


ins=fopen("ee.log","a"); 
fprintf(ins, "node energy values:\n");
for(sum=0.0, i=1; i<=params->node_num; i++){
    fprintf(ins, " %g",results->bestfvalue[i]);
    sum+=results->bestfvalue[i];
   }
fprintf(ins,"\n");
fprintf(ins,"total energy value=%g\n", sum);
for(i=1; i<=params->node_num; i++) fprintf(ins, " %d", results->bestx[i]);
fprintf(ins,"\n");
for(i=1; i<=params->node_num; i++){
    for(j=1; j<=params->node_num; j++) fprintf(ins, " %d",results->bestmat[i][j]);
    fprintf(ins, "\n");
   }
fprintf(ins,"\n");
fprintf(ins, "mutation rate=%g \n\n",results->accept_loc/results->total_loc);
fclose(ins);
                                                                                                    
ins=fopen("ee.est", "a");
fprintf(ins, "delta=%g \n", delta);
if(ins==NULL){ printf("Can't write to file\n"); return 1; }
for(i=0; i<=results->sze; i++){
   fprintf(ins, "%5d  %10.6f  %10.6f  %10.6f  %g\n",i,hist[i][1],exp(hist[i][2]),
           hist[i][3],hist[i][2]);
   hist[i][3]=0.0;
 }
fclose(ins);

ins=fopen("ee.sol","a");

//print best network
for(sum=0.0, i=1; i<=params->node_num; i++) sum+=results->bestfvalue[i];
fprintf(ins,"total energy value=%g\n", sum);
for(i=1; i<=params->node_num; i++) fprintf(ins, " %d", results->bestx[i]);
fprintf(ins,"\n");
for(i=1; i<=params->node_num; i++){
    for(j=1; j<=params->node_num; j++) fprintf(ins, " %d",results->bestmat[i][j]);
    fprintf(ins, "\n");
   }
fprintf(ins,"\n");

//print the net values divided by total weight, not sure what this means
for(i=1; i<=params->node_num; i++) 
  for(j=1; j<=params->node_num; j++) net[i][j]/=totalweight;
for(i=1; i<=params->node_num; i++){
    for(j=1; j<=params->node_num; j++) fprintf(ins, " %g",net[i][j]);
    fprintf(ins, "\n");
   }
fprintf(ins,"\n");

fclose(ins);

free_dmatrix(hist,0,params->grid,1,3);
free_ivector(changelist,1,params->node_num);
free_dvector(fvalue,1,params->node_num);
free_ivector(x,1,params->node_num);
free_imatrix(mat,1,params->node_num,1,params->node_num);
free_dmatrix(net,1,params->node_num,1,params->node_num);

return 0;
}

void copyParamData(simParams* params, double *refden, int *states, char *data){
  //everything is already setup except for states, data and refdensity

  int rows = params->data_num;
  int cols = params->node_num;

  int *temp = (int*) data;
  int **result = malloc(sizeof(int*) * rows);

  for(int i=0; i<rows; i++){
    result[i] = malloc(sizeof(int)*cols);
    memcpy(result[i], temp, sizeof(int)*cols);
    temp += cols;
  }

  params->state=ivector(1,params->node_num);
  params->refden=dvector(0,params->grid);
  params->datax=imatrix(1,params->data_num,1,params->node_num);

  for(int i=0; i<rows; i++){
    for(int j=0; j<cols; j++){
      /*printf(" %d ", result[i][j]);*/
      params->datax[i+1][j+1] = result[i][j];
    }
  }
  
  /*printf("\n\nStates: \n");*/
  for(int i=0; i<cols; i++) {
    params->state[i+1]=states[i];
    /*printf(" %d ", states[i]);*/
    // Number of results->states in each column
  }

  /*printf("\n\nRefden: \n");*/
  for(int i=0; i<params->grid; i++){
    //copy the target reference density
    /*printf(" %f ", refden[i]);*/
    params->refden[i] = refden[i];
  }

  for(int i=0; i<rows; i++){
    free(result[i]);
  }
  free(result);

}

simResults* pyInitSimResults(simParams* params){
  simResults* ret = malloc(sizeof(simResults));

  ret->bestfvalue=dvector(1,params->node_num);
  ret->bestx=ivector(1,params->node_num);
  ret->bestmat=imatrix(1,params->node_num,1,params->node_num);
  ret->indicator=ivector(0,params->grid);

  return ret;
}

void freeSimResults(simParams* params, simResults* results){

  free_dvector(results->bestfvalue,1,params->node_num);
  free_ivector(results->bestx,1,params->node_num);
  free_imatrix(results->bestmat,1,params->node_num,1,params->node_num);
  free_ivector(results->indicator,0,params->grid);

  free(results);
}

int freeSimParams(simParams *params){

  free_dvector(params->refden,0,params->grid);
  free_ivector(params->state,1,params->node_num);
  free_imatrix(params->datax,1,params->data_num,1,params->node_num);
  free(params);
  return 0;
}

//Have to do some dumb copying from Python
/*simParams* pyInitSimParams(int rows, int cols, double *refden, int *states, char *data){*/
  /*int *temp = (int*) data;*/
  /*int **result = malloc(sizeof(int*) * rows);*/

  /*for(int i=0; i<rows; i++){*/
    /*result[i] = malloc(sizeof(int)*cols);*/
    /*memcpy(result[i], temp, sizeof(int)*cols);*/
    /*temp += cols;*/
  /*}*/

  /*simParams* ret = initSimParams(rows, cols, states, result);*/

  /*for(int i=0; i<rows; i++){*/
    /*free(result[i]);*/
  /*}*/
  /*free(result);*/

  /*for(int i=0; i<ret->grid; i++){*/
    /*//copy the target reference density*/
    /*ret->refden[i] = refden[i];*/
  /*}*/

  /*return ret;*/
/*}*/

/*//Default some of the common parameters for a simulation run*/
/*simParams* initSimParams(int rows, int cols, int *states, int **data){*/

  /*simParams* params = malloc(sizeof(simParams));*/

  /*params->prior_alpha=1.0;*/
  /*params->prior_gamma=0.1111;*/
  /*params->limparent=5;*/

  /*params->maxEE = 25000.0;*/
  /*params->lowE=8000.0;*/
  /*params->scale=1.0;*/
  /*params->range=0.0;*/
  /*params->rho=1.0;*/
  /*params->tau=1.0; */
  /*params->temperature=1.0;*/

  /*params->grid = ceil((params->maxEE + params->range - params->lowE) * params->scale);*/

  /*params->data_num=rows;*/
  /*params->node_num=cols;*/

  /*params->total_iteration=60000;*/
  /*params->burnin=300000;*/
  /*params->stepscale=100000;*/

  /*params->state=ivector(1,params->node_num);*/
  /*params->refden=dvector(0,params->grid);*/
  /*params->datax=imatrix(1,params->data_num,1,params->node_num);*/

  /*for(int i=0; i<rows; i++){*/
    /*for(int j=0; j<cols; j++){*/
      /*[>printf(" %d ", data[i][j]);<]*/
      /*params->datax[i+1][j+1] = data[i][j];*/
    /*}*/
  /*}*/

  /*for(int i=0; i<cols; i++) */
    /*params->state[i+1]=states[i];*/
    /*// Number of results->states in each column*/

  /*return params;*/
/*}*/

/*int main(int argc, char **argv)*/
/*{*/
    /*FILE *ins;*/
    /*int i,j,l;*/
    /*simParams* params = defaultSimParams();*/
    /*simResults* results = malloc(sizeof(simResults));*/

    /************** Data input  *********/
    /*ins=fopen("data/WBCD2.dat","r");*/
    /*if(ins==NULL){ printf("can't open datafile\n"); return 1; }*/
    /*for(i=1; i<=params->data_num; i++){*/
        /*for(j=1; j<=params->node_num-1; j++){ */
          /*l = fscanf(ins," %d",&results->datax[i][j]); */
          /*if(l!=1){ printf("Error reading data"); return 1; }*/
          /*results->datax[i][j]-=1;*/
          /*// subtract one to make the 9 features range from 0-9*/
        /*}*/
        /*l = fscanf(ins, " %d", &results->datax[i][params->node_num]);*/
        /*if(l!=1){ printf("Error reading data"); return 1; }*/
        /*results->datax[i][params->node_num]+=1; // I Assume this last entry is the label {1,2}*/
       /*}*/
    /*fclose(ins);*/
    /*// print the data*/
    /*[>for(i=1; i<=params->data_num; i++){ <]*/
        /*[>for(j=1; j<=params->node_num; j++) printf(" %d",results->datax[i][j]);<]*/
        /*[>printf("\n");<]*/
       /*[>}<]*/
    /***************************/

    /*int ret = run(params, results);*/

    /*freeSimParams(params);*/
    /*free(results);*/

    /*return ret;*/
/*}*/

