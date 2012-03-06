#include <search.h>
#include <assert.h>

typedef struct {
  int config;
  int num;
} my_tentry;

my_tentry * make_my_tentry(int config, int num) {
  my_tentry *mt = (my_tentry *)calloc(sizeof(my_tentry),1);
  if(!mt){
    printf("calloc failure\n");
    exit(1);
  }
  mt->config = config;
  mt->num = num;
  return mt;
}

void mt_free_func(void *mt_data) {
  my_tentry *mt = mt_data;
  if(!mt) {
    return;
  }
  free(mt_data);
  return;
}

int mt_compare_func(const void *l, const void *r)
{
  const my_tentry *ml = l;
  const my_tentry *mr = r;
  if(ml->config < mr->config) {
    return -1;
  }
  if(ml->config > mr->config) {
    return 1;
  }
  return 0;
}

double accum = 0.0;
double alphaijk = 0.0;
double alphaik = 0.0;

void mt_add_func(const void *node, VISIT order, int level) {

  const my_tentry *p = *(const my_tentry **) node;

  if (order == postorder || order == leaf) {
    accum += gammln(p->num + alphaijk);
  }
}

void mt_sub_func(const void *node, VISIT order, int level) {

  const my_tentry *p = *(const my_tentry **) node;

  if (order == postorder || order == leaf) {
    accum -= gammln(p->num + alphaik);
  }
}

void mt_print_func(const void *node, VISIT order, int level) {

  const my_tentry *p = *(const my_tentry **) node;

  if (order == postorder || order == leaf) {
    printf("%d %d\n", p->config, p->num);
  }
}



double cost(x,mat,fvalue,changelist,changelength)
int *x,**mat,*changelist,changelength;
double *fvalue;
{
int count, count00, num, num00, i, j, k, m, s, tep;
int numparent, parstate, parlist[node_num];
double sum;

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
#ifdef TREE

    void *tree = NULL, *tree00 = NULL;
    my_tentry *re = 0, *retval = 0, *mt = 0;

#else
    int l;

    int **tab, **tab00;
    tab=imatrix(1,(parstate+1)*state[x[i]],1,2);
    tab00=imatrix(1,parstate+1,1,2);

    for(j=1; j<=(parstate+1)*state[x[i]]; j++)
       for(l=1; l<=2; l++) tab[j][l]=0;
    for(j=1; j<=(1+parstate); j++) 
       for(l=1; l<=2; l++) tab00[j][l]=0;
    for(j=1; j<=(parstate+1)*state[x[i]]; j++) {
       tab[j][1]=-1;
       tab[j][2]=0;
     }
    for(j=1; j<=parstate+1; j++) {
       tab00[j][1]=-1;
       tab00[j][2]=0;
     }
#endif


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
        
#ifdef NEW

        j=1;
        while(tab00[j][1]!=num00 && j<=count00) j++;
        if(j==count00+1) {
          count00++;
          tab00[j][1]=num00;
        }
        tab00[j][2]++;
     
        j=1;
        while(tab[j][1]!=num && j<=count) j++;
        if(j==count+1) {
          count++;
          tab[j][1]=num;
        }
        tab[j][2]++;
     
#elif defined TREE

        mt = make_my_tentry(num00,1);

        retval = tsearch(mt, &tree00, mt_compare_func);

        re = *(my_tentry **)retval;

        if(re != mt) {
          //already in the tree, we should add one its num
          re->num++;
          mt_free_func(mt);
        } else {
          //inserted, so we should increase count00
          count00++;
        }

        //********************
        //Now for num
        mt = make_my_tentry(num,1);

        retval = tsearch(mt, &tree, mt_compare_func);

        re = *(my_tentry **)retval;

        if(re != mt) {
          //already in the tree, we should add one its num
          re->num++;
          mt_free_func(mt);
        } else {
          //inserted, so we should increase count
          count++;
        }
          
#else
     
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
                tab00[j][1]=num00; tab00[j][2]=1;
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
#endif
       } /* end data summary */
          
     /*if(numparent==1) { */
       /*printf("count: %d, count00: %d, parstate*state: %d\n",count,count00,(parstate+1)*state[x[i]]);*/
       /*printf("numparent=%d parstate=%d\n", numparent, parstate);*/
       /*twalk(tree,mt_print_func);*/
       /*twalk(tree00,mt_print_func);*/
       /*for(k=1; k<=count; k++) printf("%d  %d %d\n",k,tab[k][1],tab[k][2]);*/
       /*for(k=1; k<=count00+1; k++) printf("%d  %d %d\n",k,tab00[k][1],tab00[k][2]);*/
       /*for(sum1=0,sum2=0,k=1; k<=count00+1;k++) {*/
         /*sum1+=tab00[k][2];*/
         /*sum2+=tab00[k][2];*/
       /*}*/
       /*printf("tab00 sum: %f  tab00 sum: %f \n", sum1,sum2);*/
     /*}*/
     
     alphaijk=prior_alpha/parstate/state[x[i]];
     alphaik=prior_alpha/parstate;

     accum = 0.0;
     accum-=count*gammln(alphaijk);
     accum+=count00*gammln(alphaik);

#ifdef TREE

     twalk(tree, mt_add_func);
     twalk(tree00, mt_sub_func);

     tdestroy(tree,mt_free_func);
     tdestroy(tree00,mt_free_func);

#else
      
     /*for(k=1; k<=count; k++) fvalue[i]+=gammln(tab[k][2]+alphaijk);*/
     /*for(k=1; k<=count00; k++) fvalue[i]-=gammln(tab00[k][2]+alphaik);*/
     for(k=1; k<=count; k++) accum+=gammln(tab[k][2]+alphaijk);
     for(k=1; k<=count00; k++) accum-=gammln(tab00[k][2]+alphaik);

     free_imatrix(tab,1,(parstate+1)*state[x[i]],1,2);
     free_imatrix(tab00,1,parstate+1,1,2);
#endif

   fvalue[i] += accum;
   fvalue[i]*=-1.0;


    /*if(numparent==1){*/
      /*printf("Numparent: %d ", numparent);*/
      /*printf("Accum: %g\n", accum);*/
      /*exit(1);*/
     /*}*/
  }
ABC:
  for(sum=0.0,m=1; m<=node_num; m++){
      sum+=fvalue[m];
     }

  return sum;
}
