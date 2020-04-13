/**
   Written by: Vivek Kale
   Last Edited: December 18, 2019

   Description: Code for doing a dot product for two input vectors. 
   The code first performs an element-wise vector product and then obtains the dot product.
   TODO: find hardware profiling library for GPU. - Vivek
   TODO: check metric for calcuations.

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef _OPENACC
#include <accel.h>
#endif

#ifdef _OPENMP 
#include <omp.h>
#endif

#include <cuda.h>

//#ifdef PAPI
//#include <papi.h>
//#endif 

#include <mpi.h>

#define MAX_ITER 100000
#define NUM_ITERS 20
#define PROB_SIZE 10000

__global__ void mult( int *a, int *b, int *c ) {
    int tid = blockIdx.x; // handle the data at this index
    if (tid < N)
      c[tid] = a[tid]*b[tid];
  }
}


int main (int argc, char* argv[] )
{
  // Size of vectors 
  int n = 10000;
  // Number of outer iterations 
  int numIters = NUM_ITERS; 
  // current outer iteration (or timestep)
  // Input vectors
  double *restrict a;
  double *restrict b;
  // Output vector
  double *restrict c; 
  double t_start, t_end = 0.0;
  int p; // Number of processes.
  int myrank; // My process's rank. 
  int iter; // Current outer iteration (or timestep).

  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  if(argc < 1)
    {
      printf("usage: vecSum [probSize]\n");
      MPI_Finalize();
    }
  else if(argc > 1)
    {
      if(atoi(argv[i]) == -1) // If the user provides -1, set the problem size to a default problem size (could be the maximum size). 
	n = PROB_SIZE;
      else
	n = atoi(argv[1]); 
    }
  // Calculate the size, in bytes, of each vector.
  size_t bytes = n*sizeof(float);

  // Allocate memory for each vector 
  a = (float*)malloc(bytes);
  b = (float*)malloc(bytes);
  c = (float*)malloc(bytes);
  MPI_Comm_size(MPI_COMM_WORLD, &p);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank); // my MPI rank 
  // Assign a GPU to a particular MPI process
#ifdef _OPENACC
  acc_init(acc_device_nvidia);  // OpenACC call   // TODO: acc_device_nvidia sh
  const int num_dev = acc_get_num_devices(acc_device_nvidia);  // #GPUs
  const int dev_id = myrank % num_dev;         
  acc_set_device_num(dev_id,acc_device_nvidia); // assign GPU to one MPI process
  cout << "MPI process " << myrank << "  is assigned to GPU " << dev_id << "\n";
#elif CUDA

#endif
  // Initialize the content of input vectors, where vector a[i] = sin(i)^2 and vector b[i] = cos(i)^2
  unsigned int i;
  for(i=0; i<n; i++) {
    a[i] = sin(i)*sin(i);
    b[i] = cos(i)*cos(i);
  }

  // add simd 
  // change simd 

  t_start = MPI_Wtime();
#ifdef _OPENACC
#pragma acc data copy_in (a[0:n],b[0:n]) copy_out(c[0:n])
   {
    for (iter = 0; iter < NUM_ITERS; iter++)
      {  
	const float sum = 0.0;
#pragma acc kernels loop present_or_copyin(a[0:n], b[0:n]) independent
	{
	for(i=0; i<n; i++) {
	  c[i] = a[i]*b[i];
	  sum += c[i];
	}
	float g_sum;
	MPI_Allreduce(&sum,&g_sum,1,MPI_DOUBLE,MPI_SUM, MPI_COMM_WORLD);    
	}
      }
   }
#else

#endif
   
  t_end = MPI_Wtime();
  if(myrank == 1)
    {
      // Check that the result of the calculation is correct 
      sum = g_sum;
      printf("Result: %f\n", g_sum); 
      // Obtain timings and performance
      printf("\n Time in seconds : %g\n", t_end - t_start);
      printf("GFLOP/s         : %g\n",2.0e-9*N/(t_end - t_start));
      printf("GiByte/s       : %g\n", (2.0*N*sizeof(float)*(1024*1024*1024)/(t_end - t_start))); // divide by 1024^3 to get GiBytes/s 
    }
  // Release memory
  free(a);
  free(b);
  free(c);
  MPI_Finalize();
}
