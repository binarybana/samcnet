#include "comm.h"
#include "nrutil.h"
#include "nrutil.c"
#include "lib.c"
#include "cost.c"
#include "metmove2.c"

int main(int argc, char **argv)
{
int stime;
long ltime;
int i, j, l, k0, iter, grid, Repeat; //s,k1,k2
int region,changelength, *changelist, *x, **mat;
double delta,**hist,*fvalue,sum,ave,max,un;
double **net, sinw, totalweight;
FILE *ins, *ins2;
/*char buf[255];*/

ltime=time(NULL);
stime=(unsigned int)ltime/2;
/*srand(stime);*/
srand(912345);

grid=ceil((maxEE+range-lowE)*scale);

hist=dmatrix(0,grid,1,3);
refden=dvector(0,grid);
datax=imatrix(1,data_num,1,node_num);
changelist=ivector(1,node_num);
state=ivector(1,node_num);
fvalue=dvector(1,node_num); 
x=ivector(1,node_num);
mat=imatrix(1,node_num,1,node_num);
bestfvalue=dvector(1,node_num);
bestx=ivector(1,node_num);
bestmat=imatrix(1,node_num,1,node_num);
net=dmatrix(1,node_num,1,node_num);
indicator=ivector(0,grid);

/************** Data input  *********/
ins=fopen("WBCD2.dat","r");
if(ins==NULL){ printf("can't open datafile\n"); return 1; }
for(i=1; i<=data_num; i++){
    for(j=1; j<=node_num-1; j++){ 
      l = fscanf(ins," %d",&x[j]); 
      x[j]-=1; }
    // subtract one to make the 9 features range from 0-9
    l = fscanf(ins, " %d", &x[node_num]);
    x[node_num]+=1; // I Assume this last entry is the label {1,2}
    for(j=1; j<=node_num; j++) datax[i][j]=x[j]; 
    //^^ save a copy of the data (untampered)
   }
fclose(ins);

// print the data
/*for(i=1; i<=data_num; i++){ */
    /*for(j=1; j<=node_num; j++) printf(" %d",datax[i][j]);*/
    /*printf("\n");*/
   /*}*/

for(i=1; i<=node_num-1; i++) state[i]=10;
state[node_num]=2;
// Number of states in each column, static var

/***************************/

ins=fopen("ee.log","a");
fprintf(ins, "random seed=%d  t0=%d tau=%g rho=%g scale=%g iteration=%d\n",
    stime,stepscale,tau,rho,scale,total_iteration);
fclose(ins);

ins=fopen("ee.sum","a");
fprintf(ins, "random seed=%d  t0=%d tau=%g rho=%g  scale=%g iteration=%d\n",
    stime,stepscale,tau,rho,scale,total_iteration);
fclose(ins);


for(Repeat=1; Repeat<=1; Repeat++){

    maxE=maxEE; 
    grid=ceil((maxEE+range-lowE)*scale);

    /* Initialization */
    for(sum=0.0, i=0; i<=grid; i++){ refden[i]=(grid-i+1)*(grid-i+1); sum+=refden[i]; }
    for(i=0; i<=grid; i++) refden[i]/=sum;
    for(i=0; i<=grid; i++){
        hist[i][1]=lowE+i*1.0/scale;
        hist[i][2]=0.0;
        hist[i][3]=0.0;
       }
    

    /* initialization of the network */
    permut_sample(x,node_num); //get a 1-10 permutation
    for(i=1; i<=node_num; i++)
       for(j=1; j<=node_num; j++) mat[i][j]=0;
    for(i=1; i<=node_num; i++) mat[i][i]=1; 
    //^^ initialize mat to identity
     
    changelength=node_num;
    for(i=1; i<=changelength; i++) changelist[i]=i; //changelist== 1,2,3...
    bestenergy=cost(x,mat,fvalue,changelist,changelength);
    printf("initial energy=%g\n",bestenergy);
    
    for(i=1; i<=node_num; i++){
        maxE=bestenergy;
        bestx[i]=x[i];
        bestfvalue[i]=fvalue[i]; 
        for(j=1; j<=node_num; j++) bestmat[i][j]=mat[i][j];
       }

    if(maxE>maxEE || maxE<lowE){
       ins=fopen("ee.log","a");
       fprintf(ins, "initial maxE=%g\n", maxE);
       fprintf(ins,"The choice of maxEE is too large or the choice of lowE is too small.\n");
       fclose(ins);
      }

   for(i=0; i<=grid; i++) indicator[i]=0;
   for(sum=0.0,i=1; i<=node_num; i++) sum+=fvalue[i];
   if(sum>maxEE+range) nonempty=sze;
      else if(sum<lowE) nonempty=0;
         else nonempty=floor((sum-lowE)*scale); 
   indicator[nonempty]=1;

   for(i=1; i<=node_num; i++)
      for(j=1; j<=node_num; j++) net[i][j]=0.0;

   ins=fopen("ee.log","a"); 
   iter=1; accept_loc=total_loc=0.1; totalweight=0.0;
   while(iter<=total_iteration){

       if(iter<=WARM*stepscale) delta=rho;
          else delta=rho*exp(-tau*log(1.0*(iter-(WARM-1)*stepscale)/stepscale));

       metmove(x,mat,fvalue,hist,delta, &region);
       for(sum=0.0, i=1; i<=node_num; i++) sum+=fvalue[i]; 

       if(iter==burnin){
          i=0;
          while(i<=grid && indicator[i]==0) i++;
          nonempty=i;
         }

       sinw=1;
       if(iter>burnin){ 
        
          un=hist[nonempty][2];
          for(i=0; i<=grid; i++){
              if(indicator[i]==1) hist[i][2]-=un;
             }
          
          sinw=exp(hist[region][2]);
          for(i=1; i<=node_num; i++)
             for(j=i+1; j<=node_num; j++)
               if(mat[i][j]==1) net[x[i]][x[j]]+=sinw;
          totalweight+=sinw;
         }
 
          
       if(iter%10000==0){
          fprintf(ins,"%g %d %g %g\n",delta,iter,bestenergy,sum);
          printf("nonempty=%d hist=%g %g %g %g\n", nonempty,hist[nonempty][2],hist[nonempty+1][2],
                 hist[nonempty+2][2], hist[nonempty+3][2]);
          printf("delta=%g iter=%d bestenergy=%g current=%g weight=%g\n",delta,iter,
                 bestenergy,sum,sinw);
         }  

       iter++;

       if(iter%100000==0){
          ins2=fopen("ee.sol","a");
          for(sum=0.0, i=1; i<=node_num; i++) sum+=bestfvalue[i];
          fprintf(ins2,"total energy value=%g\n", sum);
          for(i=1; i<=node_num; i++) fprintf(ins2, " %d", bestx[i]);
          fprintf(ins2,"\n");
          for(i=1; i<=node_num; i++){
             for(j=1; j<=node_num; j++) fprintf(ins2, " %d",bestmat[i][j]);
             fprintf(ins2, "\n");
            }
          fprintf(ins2,"\n\n");
          fclose(ins2);
         }
     }
   fclose(ins);


 
    for(sum=0.0,k0=0,i=0; i<=sze; i++)
       if(hist[i][3]<=0.0){ sum+=refden[i]; k0++; }
    if(k0>0) ave=sum/k0;
       else ave=0.0;
    for(i=0; i<=sze; i++) hist[i][2]=hist[i][2]+log(refden[i]+ave);
    max=hist[0][2];
    for(i=1; i<=sze; i++)
       if(hist[i][2]>max) max=hist[i][2];
    for(sum=0.0, i=0; i<=sze; i++){ hist[i][2]-=max; sum+=exp(hist[i][2]); }
    for(i=0; i<=sze; i++) hist[i][2]=hist[i][2]-log(sum)+log(100.0);

    
    ins=fopen("ee.log","a"); 
    fprintf(ins, "node energy values:\n");
    for(sum=0.0, i=1; i<=node_num; i++){
        fprintf(ins, " %g",bestfvalue[i]);
        sum+=bestfvalue[i];
       }
    fprintf(ins,"\n");
    fprintf(ins,"total energy value=%g\n", sum);
    for(i=1; i<=node_num; i++) fprintf(ins, " %d", bestx[i]);
    fprintf(ins,"\n");
    for(i=1; i<=node_num; i++){
        for(j=1; j<=node_num; j++) fprintf(ins, " %d",bestmat[i][j]);
        fprintf(ins, "\n");
       }
    fprintf(ins,"\n");
    fprintf(ins, "mutation rate=%g \n\n",accept_loc/total_loc);
    fclose(ins);
                                                                                                        
    ins=fopen("ee.est", "a");
    fprintf(ins, "delta=%g \n", delta);
    if(ins==NULL){ printf("Can't write to file\n"); return 1; }
    for(i=0; i<=sze; i++){
       fprintf(ins, "%5d  %10.6f  %10.6f  %10.6f  %g\n",i,hist[i][1],exp(hist[i][2]),
               hist[i][3],hist[i][2]);
       hist[i][3]=0.0;
     }
   fclose(ins);


   ins=fopen("ee.sol","a");
   for(sum=0.0, i=1; i<=node_num; i++) sum+=bestfvalue[i];
   fprintf(ins,"total energy value=%g\n", sum);
   for(i=1; i<=node_num; i++) fprintf(ins, " %d", bestx[i]);
   fprintf(ins,"\n");
   for(i=1; i<=node_num; i++){
        for(j=1; j<=node_num; j++) fprintf(ins, " %d",bestmat[i][j]);
        fprintf(ins, "\n");
       }
   fprintf(ins,"\n");
   fclose(ins);


   for(i=1; i<=node_num; i++) 
      for(j=1; j<=node_num; j++) net[i][j]/=totalweight;
   ins=fopen("ee.sol","a");
   for(i=1; i<=node_num; i++){
        for(j=1; j<=node_num; j++) fprintf(ins, " %g",net[i][j]);
        fprintf(ins, "\n");
       }
   fprintf(ins,"\n");
   fclose(ins);

 }


free_dmatrix(hist,0,grid,1,3);
free_dvector(refden,0,grid);
free_imatrix(datax,1,data_num,1,node_num);
free_ivector(state,1,node_num);
free_ivector(changelist,1,node_num);
free_dvector(fvalue,1,node_num);
free_ivector(x,1,node_num);
free_imatrix(mat,1,node_num,1,node_num);
free_dvector(bestfvalue,1,node_num);
free_ivector(bestx,1,node_num);
free_imatrix(bestmat,1,node_num,1,node_num);
free_dmatrix(net,1,node_num,1,node_num);
free_ivector(indicator,0,grid);

return 0;
}
