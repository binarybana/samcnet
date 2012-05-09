#ifndef _LIB_H_
#define _LIB_H_

/* lib.h */
void ludcmp(double **a, int n, int *indx, double *d);
void lubksb(double **a, int n, int *indx, double *b);
double matrix_logdet(double **X, int n);
double matrix_inverse(double **X, double **Y, int n);
int matrix_inverse_diag(double **X, double **Y, double *diag, int n);
double matrix_trace(double **A, int p);
int matrix_sum(double **A, double **B, double **C, int n, int p);
int matrix_multiply(double **A, double **B, double **C, int n, int p, int m);
int matrix_vector_prod(double **A, double *b, double *d, int n, int p);
void copy_vector(double *a, double *b, int p);
void copy_matrix(double **a, double **b, int n, int p);
int choldc(double **a, int n, double **D);
double loggamma(double xx);
double logpum(int k);
double Rgamma(double a, double b);
double Rbeta(double b);
int RDirichlet(double *w, double *a, int k);
double gasdev(void);
double Rgasdev(double mean, double variance);
int RNORM(double *x, double *mu, double **Sigma, int p);
int Rwishart(double **B, double df, double **Sigma, int p);
double dloggamma(double x, double a, double b);
double dloggauss(double z, double mean, double variance);
double dlogstudent(double z, int k);
double DLOGGAUSS(double *z, double *mean, double **variance, int p);
double Dlogwishart(double **D, double df, double **Sigma, int p);
int uniform_direction(double *d, int n);
int dmaxclass(double *z, int n);
int imaxclass(int *z, int n);
int binary_trans(int k, int l, int *d);
double logsum(double a, double b);
double maxvector(double *x, int n);
double minvector(double *x, int n);
double sample_variance(double *x, int n);
double gammln(double xx);
void gser(double *gamser, double a, double x, double *gln);
void gcf(double *gammcf, double a, double x, double *gln);
double gammp(double a, double x);
double Gaussp(double x);
double diGamma(double z);
double correlation(double *z1, double *z2, int p);
int permut_sample(int *sam, int n);

#endif //_LIB_H_
