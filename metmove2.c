int metmove(x,mat,fvalue,hist,delta,region)
int *x, **mat, *region;
double *fvalue,**hist,delta;
{
int *newx, **newmat, locrepeat=1;
int changelength, *changelist;
int i,j,k,k1,k2,i1,i2,j1,j2,G,accept,scheme; 
double *locfreq,*newfvalue, pxy, pyx;
double fold, fnew, un, r; 

   sze=floor((maxEE+range-lowE)*scale);

   locfreq=dvector(0,sze);
   newx=ivector(1,node_num);
   newmat=imatrix(1,node_num,1,node_num);
   newfvalue=dvector(1,node_num);
   changelist=ivector(1,node_num);
   

   for(i=0; i<=sze; i++) locfreq[i]=0.0;
   for(fold=0.0,i=1; i<=node_num; i++) fold+=fvalue[i]; 
   if(fold>maxEE+range) k1=sze;
      else if(fold<lowE) k1=0;
         else k1=floor((fold-lowE)*scale);


   for(G=1; G<=locrepeat; G++){

    /* copy the old solution to the new one*/ 
    for(i=1; i<=node_num; i++){
        newx[i]=x[i];
        newfvalue[i]=fvalue[i];
        for(j=1; j<=node_num; j++) newmat[i][j]=mat[i][j];
       }

     
    un=rand()*1.0/RAND_MAX;
    if(un<1.0/3) scheme=1;
       else if(un<2.0/3) scheme=2;
          else scheme=3;

   if(scheme==1){ /* temporal order change */
      
      k=0;
      while(k<1 || k>node_num-1) 
          k=floor(rand()*1.0/RAND_MAX*(node_num-1))+1;
      
      newx[k]=x[k+1]; newx[k+1]=x[k];

      changelength=2;
      changelist[1]=k; changelist[2]=k+1;

      for(j=k+2; j<=node_num; j++){
          if(newmat[k][j]==1 || newmat[k+1][j]==1){
             changelength++;
             changelist[changelength]=j;
            }
          }
      
      fnew=cost(newx,newmat,newfvalue,changelist,changelength);
      pxy=pyx=1.0;
     }

    if(scheme==2){ /* skeletal change */ 

       i=0;
       while(i<1 || i>node_num)
          i=floor(rand()*1.0/RAND_MAX*node_num)+1; 
       j=i;
       while(j<1 || j>node_num || j==i)
          j=floor(rand()*1.0/RAND_MAX*node_num)+1;
       if(i<j){
          newmat[i][j]=1-newmat[i][j];
          changelength=1;
          changelist[1]=j;
         }
          else{
            newmat[j][i]=1-newmat[j][i];
            changelength=1;
            changelist[1]=i;
           }
       
       fnew=cost(newx,newmat,newfvalue,changelist,changelength);
       pxy=pyx=1.0;
     }

   
    if(scheme==3){ /* Double skeletal change */ 
        
       i1=i2=0; j1=j2=0;
       while(i1==i2 && j1==j2){  
         i1=0;
         while(i1<1 || i1>node_num)
             i1=floor(rand()*1.0/RAND_MAX*node_num)+1;
         j1=i1;
         while(j1<1 || j1>node_num || j1==i1)
             j1=floor(rand()*1.0/RAND_MAX*node_num)+1;
         if(i1>j1){ k=i1; i1=j1; j1=k; }

         i2=0;
         while(i2<1 || i2>node_num)
             i2=floor(rand()*1.0/RAND_MAX*node_num)+1;
         j2=i2;
         while(j2<1 || j2>node_num || j2==i2)
             j2=floor(rand()*1.0/RAND_MAX*node_num)+1; 
         if(i2>j2){ k=i2; i2=j2; j2=k; }
        }
       

        newmat[i1][j1]=1-newmat[i1][j1];
        newmat[i2][j2]=1-newmat[i2][j2];

        if(j1==j2){ changelength=1; changelist[1]=j1; }
           else{ changelength=2; changelist[1]=j1; changelist[2]=j2; }
      
        fnew=cost(newx,newmat,newfvalue,changelist,changelength);
        pxy=pyx=1.0;
      }
 
    
  /* acceptance of new moves */

     if(fnew>maxEE+range) k2=sze;
       else if(fnew<lowE) k2=0;
         else k2=floor((fnew-lowE)*scale);
     indicator[k2]=1;

     r=hist[k1][2]-hist[k2][2]+(fold-fnew)/temperature+log(pyx/pxy);

        if(r>0.0) accept=1;
          else{
            un=-1.0;
            while(un<=0.0) un=rand()*1.0/RAND_MAX;
            if(un<exp(r)) accept=1;
               else accept=0;
           }
        
	

     if(accept==1){ 
        k1=k2; fold=fnew;
        for(i=1; i<=node_num; i++){
            x[i]=newx[i];
            fvalue[i]=newfvalue[i];
            for(j=1; j<=node_num; j++) mat[i][j]=newmat[i][j];
           }
        locfreq[k2]+=1.0;  accept_loc+=1.0; total_loc+=1.0;
       }
      else{ locfreq[k1]+=1.0; total_loc+=1.0; }


      if(accept==1 && fnew<bestenergy){
         bestenergy=fnew;
         for(i=1; i<=node_num; i++){
             bestx[i]=x[i];
             bestfvalue[i]=fvalue[i];
             for(j=1; j<=node_num; j++) bestmat[i][j]=mat[i][j];
            }
          }
       
      *region=k1;
   } /* end of local repeat */


    for(i=0; i<=sze; i++)
       if(indicator[i]==1){
          locfreq[i]/=locrepeat;
          hist[i][2]+=delta*(locfreq[i]-refden[i]);
          hist[i][3]+=locfreq[i];
         }
     

 free_dvector(locfreq,0,sze);
 free_ivector(newx,1,node_num);
 free_imatrix(newmat,1,node_num,1,node_num);
 free_dvector(newfvalue,1,node_num);
 free_ivector(changelist,1,node_num);

return 0;
}
