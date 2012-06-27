#include <math.h>
#include <stdlib.h>

#include "snet.h"
#include "lib.h"
#include "nrutil.h"
#include "cost.h"

int metmove(simParams *params, 
            simResults *results, 
            int *x,
            int **mat,
            double *fvalue,
            double **hist,
            double delta,
            int *region)
{
int *newx, **newmat;
int changelength, *changelist;
int i,j,k,k1,k2,i1,i2,j1,j2,accept,scheme; 
double *locfreq,*newfvalue, pxy, pyx;
double fold, fnew, un, r; 

   locfreq=dvector(0,results->sze-1);
   newx=ivector(0,params->node_num-1);
   newmat=imatrix(0,params->node_num-1,0,params->node_num-1);
   newfvalue=dvector(0,params->node_num-1);
   changelist=ivector(0,params->node_num-1);
   

   for(i=0; i<results->sze; i++) locfreq[i]=0.0;
   for(fold=0.0,i=0; i<params->node_num; i++) fold+=fvalue[i]; 
   if(fold>params->maxEE+params->range) k1=results->sze;
      else if(fold<params->lowE) k1=0;
         else k1=floor((fold-params->lowE)*params->scale);


    /* copy the old solution to the new one*/ 
    for(i=0; i<params->node_num; i++){
        newx[i]=x[i];
        newfvalue[i]=fvalue[i];
        for(j=0; j<params->node_num; j++) newmat[i][j]=mat[i][j];
       }

     
    un=rand()*1.0/RAND_MAX;
    if(un<1.0/3) scheme=1;
       else if(un<2.0/3) scheme=2;
          else scheme=3;

   if(scheme==1){ /* temporal order change */
      
      k=0;
      while(k<0 || k>params->node_num-2) 
          k=floor(rand()*1.0/RAND_MAX*(params->node_num-2));
      
      newx[k]=x[k+1]; newx[k+1]=x[k];

      changelength=2;
      changelist[0]=k; changelist[1]=k+1;

      for(j=k+2; j<params->node_num; j++){
          if(newmat[k][j]==1 || newmat[k+1][j]==1){
             changelength++;
             changelist[changelength-1]=j;
            }
          }
      
      fnew=cost(params,results,newx,newmat,newfvalue,changelist,changelength);
      pxy=pyx=1.0;
     }

    if(scheme==2){ /* skeletal change */ 

       i=0;
       while(i<0 || i>params->node_num-1)
          i=floor(rand()*1.0/RAND_MAX*(params->node_num-1));
       j=i;
       while(j<0 || j>params->node_num-1 || j==i)
          j=floor(rand()*1.0/RAND_MAX*(params->node_num-1));
       if(i<j){
          newmat[i][j]=1-newmat[i][j];
          changelength=1;
          changelist[0]=j;
         }
          else{
            newmat[j][i]=1-newmat[j][i];
            changelength=1;
            changelist[0]=i;
           }
       
       fnew=cost(params,results,newx,newmat,newfvalue,changelist,changelength);
       pxy=pyx=1.0;
     }

   
    if(scheme==3){ /* Double skeletal change */ 
        
       i1=i2=0; j1=j2=0;
       while(i1==i2 && j1==j2){  
         i1=0;
         while(i1<0 || i1>params->node_num-1)
             i1=floor(rand()*1.0/RAND_MAX*(params->node_num-1));
         j1=i1;
         while(j1<0 || j1>params->node_num-1 || j1==i1)
             j1=floor(rand()*1.0/RAND_MAX*(params->node_num-1));
         if(i1>j1){ k=i1; i1=j1; j1=k; }

         i2=0;
         while(i2<0 || i2>params->node_num-1)
             i2=floor(rand()*1.0/RAND_MAX*(params->node_num));
         j2=i2;
         while(j2<0 || j2>params->node_num-1 || j2==i2)
             j2=floor(rand()*1.0/RAND_MAX*(params->node_num)); 
         if(i2>j2){ k=i2; i2=j2; j2=k; }
        }
       

        newmat[i1][j1]=1-newmat[i1][j1];
        newmat[i2][j2]=1-newmat[i2][j2];

        if(j1==j2){ changelength=1; changelist[0]=j1; }
           else{ changelength=2; changelist[0]=j1; changelist[1]=j2; }
      
        fnew=cost(params,results,newx,newmat,newfvalue,changelist,changelength);
        pxy=pyx=1.0;
      }
 
    
  /* acceptance of new moves */

     if(fnew>params->maxEE+params->range) k2=results->sze-1;
       else if(fnew<params->lowE) k2=0;
         else k2=floor((fnew-params->lowE)*params->scale);
     results->indicator[k2]=1;

     r=hist[k1][1]-hist[k2][1]+(fold-fnew)/params->temperature+log(pyx/pxy);

        if(r>0.0) accept=1;
          else{
            un=-1.0;
            while(un<=0.0) un=rand()*1.0/RAND_MAX;
            if(un<exp(r)) accept=1;
               else accept=0;
           }
        
	

     if(accept==1){ 
        k1=k2; fold=fnew;
        for(i=0; i<params->node_num; i++){
            x[i]=newx[i];
            fvalue[i]=newfvalue[i];
            for(j=0; j<params->node_num; j++) mat[i][j]=newmat[i][j];
           }
        locfreq[k2]+=1.0;  results->accept_loc+=1.0; results->total_loc+=1.0;
       }
      else{ locfreq[k1]+=1.0; results->total_loc+=1.0; }


      if(accept == 1 && fnew<results->bestenergy){
         results->bestenergy=fnew;
         for(i=0; i<params->node_num; i++){
             results->bestx[i]=x[i];
             results->bestfvalue[i]=fvalue[i];
             for(j=0; j<params->node_num; j++) results->bestmat[i][j]=mat[i][j];
            }
          }
       
      *region=k1;


    for(i=0; i<results->sze; i++)
       if(results->indicator[i]==1){
          hist[i][1]+=delta*(locfreq[i]-params->refden[i]);
          hist[i][2]+=locfreq[i];
         }
     

 free_dvector(locfreq,0,results->sze-1);
 free_ivector(newx,0,params->node_num-1);
 free_imatrix(newmat,0,params->node_num-1,0,params->node_num-1);
 free_dvector(newfvalue,0,params->node_num-1);
 free_ivector(changelist,0,params->node_num-1);

return 0;
}
