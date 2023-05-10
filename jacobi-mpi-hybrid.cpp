/* MPI-parallel FMM
 * 
 * 
 * 
 */
#include <stdio.h>
#include <math.h>
#include <cmath>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <mpi.h>
#include <string.h>

struct box{
  double Q;
  double* multipole; 
  double* local;
}

void box_initial(int qq, box b){
  Q = 0;
  b->multipole = (double *) calloc(sizeof(double), qq);
  b->local = (double *) calloc(sizeof(double), qq);

  for (int i=0;i<qq;i++){
    b->multipole[i] = 0;
    b->local[i] = 0;
  }
}

int binomialCoeff(int n, int k)
{
    // Base Cases
    if (k > n)
        return 0;
    if (k == 0 || k == n)
        return 1;
 
    // Recur
    return binomialCoeff(n - 1, k - 1)
           + binomialCoeff(n - 1, k);
}

int log_a_to_base_b(int a, int b)
{
    return log2(a) / log2(b);
}

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *lu, int lN, double invhsq) {
  int i;
  double tmp, gres = 0.0, lres = 0.0;

#pragma omp parallel for default(none) shared(lu,lN,invhsq) private(i,tmp) reduction(+:lres)
  for (i = 1; i <= lN; i++){
    tmp = ((2.0*lu[i] - lu[i-1] - lu[i+1]) * invhsq - 1);
    lres += tmp * tmp;
  }
  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}


int main(int argc, char * argv[]) {
  int mpirank, i, p, N, s, iter, max_iters;
  MPI_Status status, status1;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* get name of host running MPI process */
  /*
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);
  */

  N = 20000;
  int Nl = log_a_to_base_b ( N / s , 4);
  int Nb = (pow(4,Nl)-1)/3;
  int dim = pow(2,Nl-1);
  int dimsq;
  int q = 6;
  double par = 1.0/dim;
  struct box * grid = (box*) malloc(Nb * sizeof(box)); 

  #pragma omp parallel for{
    for(int i = 0;i<Nb;i++){
      box_initial(q,grid[i]);
    }
  }



  double* x = (double*) malloc(N * sizeof(double));
  double* y = (double*) malloc(N * sizeof(double));
  double* f = (double*) malloc(N * sizeof(double));
  double* u = (double*) malloc(N * sizeof(double));

  for (long i = 0; i < N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    f[i] = drand48();
    u[i] = 0;
  }

  int x_c;
  int y_c;
  double dx;
  double dy;
  double r;
  int b_c;
  int start = (pow(4,Nl-1)-1)/3;

  #pragma omp parallel for{
    for(int i=0;i<n;i++){
      x_c = x[i]/par;
      y_c = y[i]/par;
      dx = x[i] - (par*x_c+par/2.0);
      dy = y[i] - (par*y_c+par/2.0);
      r = sqrt(dx*dx+dy*dy);
      b_c = dim*y_c + x_c;
      grid[start+b_c]->Q += f[i];

      for (int j = 0;j<q;j++){
        grid[start+b_c]->multipole[j] += (-f[i]*pow(r,j+1))*1.0/(j+1);
      }
    }
  }

  double z0;
  double startchid;

  #pragma omp parallel for{
    for(int i = Nl-2;i>=0;i--){
      dim = pow(2,i); 
      dimsq = dim*dim;
      startchid =start;
      start -= dimsq;
      
      par = 1.0/dim/2;
      z0 = sqrt(2*par*par);

      for (int j=0;j<dimsq;j++){
        for (int k=0;k<4;k++){
          grid[start+j]->Q += grid[startchid+j*4+k]->Q;
          for (int l =0;l<q;l++){
            grid[start+j]->local += (-1.0*grid[startchid+j*4+k]->Q*pow(z0,l+1)/(l+1));
            for(int o=0;o<=l;o++){
              grid[start+j]->local[l] += binomialCoeff(l,o)*grid[startchid+j*4+k]->local[o]*pow(z0,l-k);
            }
          }
        }
      }


    }
  }

  



  /* compute number of unknowns handled by each process */
  
  
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();



  /* Allocation of vectors, including left/upper and right/lower ghost points */
  double * lu    = (double *) calloc(sizeof(double), lN + 2);
  double * lunew = (double *) calloc(sizeof(double), lN + 2);
  double * lutemp;

  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-5;

  /* initial residual */
  gres0 = compute_residual(lu, lN, invhsq);
  gres = gres0;

  for (iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {

#pragma omp parallel for default(none) shared(lN,lunew,lu,hsq)
    /* Jacobi step for local points */
    for (i = 1; i <= lN; i++){
      lunew[i]  = 0.5 * (hsq + lu[i - 1] + lu[i + 1]);
    }

    /* communicate ghost values */
    if (mpirank < p - 1) {
      /* If not the last process, send/recv bdry values to the right */
      MPI_Send(&(lunew[lN]), 1, MPI_DOUBLE, mpirank+1, 124, MPI_COMM_WORLD);
      MPI_Recv(&(lunew[lN+1]), 1, MPI_DOUBLE, mpirank+1, 123, MPI_COMM_WORLD, &status);
    }
    if (mpirank > 0) {
      /* If not the first process, send/recv bdry values to the left */
      MPI_Send(&(lunew[1]), 1, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
      MPI_Recv(&(lunew[0]), 1, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status1);
    }


    /* copy newu to u using pointer flipping */
    lutemp = lu; lu = lunew; lunew = lutemp;
    if (0 == (iter % 10)) {
      gres = compute_residual(lu, lN, invhsq);
      if (0 == mpirank) {
	printf("Iter %d: Residual: %g\n", iter, gres);
      }
    }
  }

  /* Clean up */
  free(lu);
  free(lunew);

  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == mpirank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();
  return 0;
}
