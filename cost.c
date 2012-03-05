double cost(x,mat,fvalue,changelist,changelength)
int *x,**mat,*changelist,changelength;
double *fvalue;
{
int **tab, **tab00;
int count, count00, num, num00, i, j, k, l, m, s, tep;
int numparent, parstate, parlist[node_num];
double sum, alphaijk, alphaik;


for(m=1; m<=changelength; m++){
    
    i=changelist[m]; 

    parstate=1; s=0;
    for(j=1; j<i; j++){
        if(mat[j][i]==1){ 
           parstate*=state[x[j]]; //accumulating the total number of parent sattes
           s++;
           parlist[s]=x[j]; //parent list
         }
       }
    numparent=s; // tot num of parents
    // Structure Prior, in log form:
    fvalue[i]=numparent*log(prior_gamma);

    // Number of parents limited to limparent
    if(numparent>limparent){ 
       for(j=1; j<=node_num; j++) fvalue[j]=1.0e+100;
       goto ABC;
      }
    
    /* 
    printf("cost1: %g %d\n", -1.0*fvalue[i], parstate);
    */

    /* parent state table */
    tab=imatrix(1,(parstate+1)*state[x[i]],1,2);
    tab00=imatrix(1,parstate+1,1,2);

    for(j=1; j<=parstate*state[x[i]]; j++)
       for(l=1; l<=2; l++) tab[j][l]=0;
    for(j=1; j<=parstate; j++) 
       for(l=1; l<=2; l++) tab00[j][l]=0;


    /* data summary: count N_{ijk} */
    count=count00=0;
    for(k=1; k<=data_num; k++){ 
       
        for(num00=0,s=numparent; s>=1; s--){
            tep=1;
            for(j=1; j<s; j++) tep*=10; //FIXME Shouldn't this be state[x[i]] instead of 10?
            num00+=datax[k][parlist[s]]*tep; 
           }
        num=num00*10+datax[k][x[i]]; 
        // ^^ Encode the current data row's state and parent values as a
        // numparents+1 digit decimal number. For data sets with too many
        // nodes >32 or >64, we should worry about overflow.
        //
        // Also, encode just the parents values into decimal number num00
     
        if(count00==0){
            count00++;
            tab00[count00][1]=num00;
            tab00[count00][2]+=1;
           }
        //tab00 and count00 only care about parent states.
        //first row of tab00 is the encoded parent configuration
        //second row of tab00 is the emprical count of the above configuration
        //count00 is the number of unique parent configs witnessed
          else{
            j=1;
            while(tab00[j][1]<num00 && j<=count00) j++;

            if(tab00[j][1]==num00) tab00[j][2]+=1;
               else{
                 //below: Insert the config,count tuple into the tab00 matrix
                 //so it remains sorted on the configs in the first row
                for(l=count00; l>=j; l--){
                    tab00[l+1][1]=tab00[l][1];
                    tab00[l+1][2]=tab00[l][2];
                   }
                tab00[j][1]=num00; tab00[j][2]+=1;
                count00++;
               }
           }


          // Now, do the same thing, but with state configurations included in
          // the encoded representation
        if(count==0){
            count++;
            tab[count][1]=num; 
            tab[count][2]+=1;
           }
          else{ 
            j=1;
            while(tab[j][1]<num && j<=count) j++; 
            
            if(tab[j][1]==num) tab[j][2]+=1;
               else{ 
                for(l=count; l>=j; l--){ 
                    tab[l+1][1]=tab[l][1]; 
                    tab[l+1][2]=tab[l][2]; 
                   } 
                tab[j][1]=num; tab[j][2]=1;
                count++;
               }
           }
       } /* end data summary */
          
     /* 
     printf("cost2: count=%d count00=%d\n", count, count00);
     for(k=1; k<=count; k++) printf("%d  %d %d\n",k,tab[k][1],tab[k][2]);
     for(k=1; k<=count00; k++) printf("%d  %d %d\n",k,tab00[k][1],tab00[k][2]);
     */
      
     alphaijk=prior_alpha/parstate/state[x[i]];
     alphaik=prior_alpha/parstate;
     
     fvalue[i]-=count*gammln(alphaijk);
     for(k=1; k<=count; k++) fvalue[i]+=gammln(tab[k][2]+alphaijk);
         
     fvalue[i]+=count00*gammln(alphaik);
     for(k=1; k<=count00; k++) fvalue[i]-=gammln(tab00[k][2]+alphaik);
     fvalue[i]*=-1.0;

     free_imatrix(tab,1,(parstate+1)*state[x[i]],1,2);
     free_imatrix(tab00,1,parstate+1,1,2);
   }

ABC:
  for(sum=0.0,m=1; m<=node_num; m++){
      sum+=fvalue[m];
     }

  return sum;
}
