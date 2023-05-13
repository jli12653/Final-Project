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
};

void box_initial(int qq, box b){
  b.Q = 0;
  b.multipole = (double*) malloc(qq * sizeof(double));
  b.local = (double*) malloc(qq * sizeof(double));

  for (int i=0;i<qq;i++){
    b.multipole[i] = 0;
    b.local[i] = 0;
  }

  return;
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



int main(int argc, char * argv[]) {
  int rank, p, N, s;
  MPI_Status status;
  MPI_Request request_outh, request_inh;
  MPI_Request request_outv, request_inv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* get name of host running MPI process */
  /*
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);
  */

  N = 10000;
  s = 10;
  int Nl = log_a_to_base_b ( N / s , 4);
  int Nb = (pow(4,Nl)-1)/3;
  int dim = pow(2,Nl-1);
  int dimsq = dim*dim;
  int q = 6;
  double par = 1.0/dim;
  struct box * grid = (box*) malloc(Nb * sizeof(box)); 

  #pragma omp parallel for
    for(int i = 0;i<Nb;i++){
      box_initial(q,grid[i]);
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

  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  

  // Compute Multipole for finest level

  #pragma omp parallel for
    for(int i=0;i<N;i++){
      x_c = x[i]/par;
      y_c = y[i]/par;

      dx = x[i] - (par*x_c+par/2.0);
      dy = y[i] - (par*y_c+par/2.0);
      r = sqrt(dx*dx+dy*dy);
      b_c = dim*y_c + x_c;
      if (b_c >= dimsq) b_c = dimsq - 1;
      grid[start + b_c].Q += f[i];

      for (int j = 0;j<q;j++){
        grid[start+b_c].multipole[j] += (-f[i]*pow(r,j+1))*1.0/(j+1);
      }
    }
  printf("Compute Multipole for finest level done.\n");

  

  double z0;
  int startchid;

  // Multipole to Multipole

  
  /*
  #pragma omp parallel for
    for(int i = Nl-2;i>=0;i--){
      dim = pow(2,i); 
      dimsq = dim*dim;
      startchid =start;
      start -= dimsq;
      
      par = 1.0/dim/2;
      z0 = sqrt(2*par*par);

      for (int j=0;j<dimsq;j++){
        for (int k=0;k<4;k++){
          grid[start+j].Q += grid[startchid+j*4+k].Q;
          for (int l =0;l<q;l++){
            grid[start+j].local[l] += (-1.0*grid[startchid+j*4+k].Q*pow(z0,l+1)/(l+1));
            for(int o=0;o<=l;o++){
              grid[start+j].local[l] += binomialCoeff(l,o)*grid[startchid+j*4+k].local[o]*pow(z0,l-k);
            }
          }
        }
      }


    }

  printf("Multipole to Multipole done.\n");

  */

  // Multipole to Local
  /*
  int dimpar=0;
  int childx, childy,stss, x_i, y_i, boxid;
  int x1,x2,x3,y1,y2,y3;

  #pragma omp parallel for
    for(int i = 2; i<=Nl;i++){
      start = (pow(4,i-1)-1)/3;
      dim = pow(2,i);
      dimpar = dim/2;
      dimsq = dim*dim;
      double* sendh = (double*) malloc(dim * 2 * (q+1) * sizeof(double)); 
      double* sendv = (double*) malloc(dim * 2 * (q+1) * sizeof(double));
      double* rech = (double*) malloc(dim * 2 * (q+1) * sizeof(double)); 
      double* recv = (double*) malloc(dim * 2 * (q+1) * sizeof(double));


      
      MPI_Barrier(MPI_COMM_WORLD);

      for (int j=0;i<dim;j++){
        for (int k=0; k<dim;k++){
          x_i = j;
          y_i = k;
          boxid = start+x_i+y_i*dim;
          childx = x_i % 2;
          childy = y_i % 2;
          

          if(rank = 0){
            MPI_Irecv(rech, dim * 2 * (q+1), MPI_DOUBLE, 1, 123, MPI_COMM_WORLD, &request_inh);
            MPI_Irecv(recv, dim * 2 * (q+1), MPI_DOUBLE, 2, 124, MPI_COMM_WORLD, &request_inv);
          }
          if(rank = 1 ){
            MPI_Irecv(rech, dim * 2 * (q+1), MPI_DOUBLE, 0, 123, MPI_COMM_WORLD, &request_inh);
            MPI_Irecv(recv, dim * 2 * (q+1), MPI_DOUBLE, 3, 124, MPI_COMM_WORLD, &request_inv);
          }
          if(rank =2){
            MPI_Irecv(rech, dim * 2 * (q+1), MPI_DOUBLE, 3, 123, MPI_COMM_WORLD, &request_inh);
            MPI_Irecv(recv, dim * 2 * (q+1), MPI_DOUBLE, 0, 124, MPI_COMM_WORLD, &request_inv);
          }
          if(rank =3){
            MPI_Irecv(rech, dim * 2 * (q+1), MPI_DOUBLE, 2, 123, MPI_COMM_WORLD, &request_inh);
            MPI_Irecv(recv, dim * 2 * (q+1), MPI_DOUBLE, 1, 124, MPI_COMM_WORLD, &request_inv);
          }

            if (x_i/2 == dimpar - 1){
              stss = childx + y_i*2;
              sendh[stss] = grid[boxid].Q; 
              for(int ii = 1;ii<=q;ii++){
                sendh[stss+ii] = grid[boxid].multipole[ii-1];
              }
            }
            if (y_i/2 == dimpar - 1){
              stss = x_i + childy*2;
              sendh[stss] = grid[boxid].Q; 
              for(int ii = 1;ii<=q;ii++){
                sendv[stss+ii] = grid[boxid].multipole[ii-1];
              }
            }

          if(rank = 0){
            MPI_Isend(sendh,dim * 2 * (q+1), MPI_DOUBLE, 1, 123, MPI_COMM_WORLD, &request_outh);
            MPI_Isend(sendv, dim * 2 * (q+1), MPI_DOUBLE, 2, 124, MPI_COMM_WORLD, &request_outv);
          }
          if(rank = 1 ){
            MPI_Isend(sendh,dim * 2 * (q+1), MPI_DOUBLE, 0, 123, MPI_COMM_WORLD, &request_outh);
            MPI_Isend(sendv, dim * 2 * (q+1), MPI_DOUBLE, 3, 124, MPI_COMM_WORLD, &request_outv);
          }
          if(rank =2){
            MPI_Isend(sendh,dim * 2 * (q+1), MPI_DOUBLE, 3, 123, MPI_COMM_WORLD, &request_outh);
            MPI_Isend(sendv, dim * 2 * (q+1), MPI_DOUBLE, 0, 124, MPI_COMM_WORLD, &request_outv);
          }
          if(rank =3){
           MPI_Isend(sendh,dim * 2 * (q+1), MPI_DOUBLE, 2, 123, MPI_COMM_WORLD, &request_outh);
            MPI_Isend(sendv, dim * 2 * (q+1), MPI_DOUBLE, 1, 124, MPI_COMM_WORLD, &request_outv);
          }
          

          if (childx == 0){
            if (childy == 0){
              y1 = y_i-2;
              y2 = y_i+2;
              y3 = y_i+3;
              x1 = x_i-2;
              x2 = x_i+2;
              x3 = x_i+3;
            }
            else{
              y1 = y_i-3;
              y2 = y_i-2;
              y3 = y_i+3;
              x1 = x_i-2;
              x2 = x_i+2;
              x3 = x_i+3;
            }
          }
          else{
            if (childy == 0){
              y1 = y_i-2;
              y2 = y_i+2;
              y3 = y_i+3;
              x1 = x_i-3;
              x2 = x_i-2;
              x3 = x_i+3;
            }
            else{
              y1 = y_i-3;
              y2 = y_i-2;
              y3 = y_i+3;
              x1 = x_i-3;
              x2 = x_i-2;
              x3 = x_i+3;
            }
          }



          if (y1 >=0){
                for (int iii = i-2;i<=i+3;i++){
                  if (iii>=0 && iii<dim){

                  }
                }
              }
              if (y2 >=0){
                for (int iii = i-2;i<=i+3;i++){
                  if (iii>=0 && iii<dim){

                  }
                }
              }
              if (y3 >=0){
                for (int iii = i-2;i<=i+3;i++){
                  if (iii>=0 && iii<dim){

                  }
                }
              }
              if (x1 >=0){
                for (int jjj=j-1;jjj<=j+1;jjj+1){

                }
              }
              if (x2 >=0){
                for (int jjj=j-1;jjj<=j+1;jjj+1){
                  
                }
              }
              if (x3 >=0){
                for (int jjj=j-1;jjj<=j+1;jjj+1){
                  
                }
              }

          MPI_Wait(&request_outh, &status);
          MPI_Wait(&request_inh, &status);
          MPI_Wait(&request_outv, &status);
          MPI_Wait(&request_inv, &status);
          int x_h,y_h,x_v,y_v;

          if(rank = 0){
            x1 = dim-2;
            x2 = dim-1;
            y1 = dim-2;
            y2 = dim-1;
          }
          if(rank = 1 ){
            x1 = 0;
            x2 = 1;
            y1 = dim-2;
            y2 = dim-1;
          }
          if(rank =2){
            x1 = dim-2;
            x2 = dim-1;
            y1 = 0;
            y2 = 1;
          }
          if(rank =3){
            x1 = 0;
            x2 = 1;
            y1 = 0;
            y2 = 1;
          }

          int parid, fir;

          for ( x_h =0;x_h<2;x_h++){
            for (y_h = 0;y_h<dim;y_h++){
              if (x_h == 0){

                // only update x1
                parid = y_h/2;

                if(parid-1>=0){
                  fir = 2*(parid-1);


                }
                fir = 2*parid;

                if(parid+1<dim){
                  fir = 2*(parid+1);
                }
              }
              else{

                // only update x1, x2
                parid = y_h/2;

                if(parid-1>=0){
                  fir = 2*(parid-1);


                }
                fir = 2*parid;


                if(parid+1<dim){
                  fir = 2*(parid+1);
                }
              }
            }
          }

          for ( x_v =0;x_v<dim;x_v++){
            for (y_v = 0;y_v<2;y_v++){
              if (y_v == 0){

                // only update y1
                parid = x_v/2;

                if(parid-1>=0){
                  fir = 2*(parid-1);


                }
                fir = 2*parid;

                if(parid+1<dim){
                  fir = 2*(parid+1);
                }
              }
              else{
                // only update y1, y2
                parid = x_v/2;

                if(parid-1>=0){
                  fir = 2*(parid-1);


                }
                fir = 2*parid;


                if(parid+1<dim){
                  fir = 2*(parid+1);

                }
              }
            }
          }

        
          



        } 
      }

      free(sendh);
      free(sendv);
      free(rech);
      free(recv);

      
    }

    printf("Multipole to Local done.\n");
  */



  /*
  // Local to Local
  start = 0;
  startchid = 0;

  #pragma omp parallel for
    for(int i = 0;i<Nl;i++){
      start = startchid;
      dim = pow(2,i); 
      dimsq = dim*dim;
      startchid = start + dimsq;
      
      par = 1.0/dim/2;
      z0 = sqrt(2*par*par);

      for (int j=0;j<dimsq;j++){
        for (int k=0;k<4;k++){
          for (int l =0;l<q;l++){
            grid[startchid+j*4+k].local[l] += (-1.0*grid[start+j].Q*pow(z0,l+1)/(l+1));
            for(int o=0;o<=l;o++){
              grid[startchid+j*4+k].local[l] += binomialCoeff(l,o)*grid[start+j].local[o]*pow(z0,l-k);
            }
          }
        }
      }


    }
  printf("Local to Local done.\n");
  */
  
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == rank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }


  free(x);
  free(y);
  free(f);
  free(u);
  free(grid);
  MPI_Finalize();
  return 0;
}
