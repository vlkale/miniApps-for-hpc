/*****************************************************************************
 * FILE: omp_dotprod_hybrid.c
 * DESCRIPTION:
 *   This simple program is the hybrid version of a dot product and the fourth
 *   of four codes used to show the progression from a serial program to a 
 *   hybrid MPI/OpenMP program.  The relevant codes are:
 *      - omp_dotprod_serial.c  - Serial version
 *      - omp_dotprod_openmp.c  - OpenMP only version
 *      - omp_dotprod_mpi.c     - MPI only version
 *      - omp_dotprod_hybrid.c  - Hybrid MPI and OpenMP version
 * Based on code by Blaise Barney (blaiseb@llnl.gov)
 ******************************************************************************/

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/* Define length of dot product vectors and number of OpenMP threads */
#define VECLEN 100
#define NUMTHREADS 8
#define NUM_TIMESTEPS 100

int main (int argc, char* argv[])
{
  int i, myid, tid, numprocs, len=VECLEN, threads=NUMTHREADS, timesteps = NUM_TIMESTEPS;
  double *a, *b;
  double mysum, allsum, sum, psum;
  double execTime = 0.0;

  /* MPI Initialization */
  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &myid); 

  /* 
   Each MPI task uses OpenMP to perform the dot product, obtains its partial sum, 
   and then calls MPI_Reduce to obtain the global sum.
  */
  if (myid == 0)
    printf("Starting omp_dotprod_hybrid. Using %d tasks...\n",numprocs);
  
  if(argc >= 2)
    {
      threads =  atoi(argv[1]);
    }
  if(argc >= 3)
    {
      threads =  atoi(argv[1]);
      len  =  atoi(argv[2]);
    }
  if(argc >= 4)
    {
      timesteps  =  atoi(argv[3]);
    }

  /* Assign storage for dot product vectors */
  a = (double*) malloc (len*threads*sizeof(double));
  b = (double*) malloc (len*threads*sizeof(double));
 
  /* Initialize dot product vectors */
  for (i=0; i<len*threads; i++) 
  {
    a[i]=1.0;
    b[i]=a[i];
  }

  /*
   Perform the dot product in an OpenMP parallel region for loop with a sum reduction
   For illustration purposes:
     - Explicitly sets number of threads
     - Gets and prints number of threads used
     - Each thread keeps track of its partial sum
  */

  /* Initialize OpenMP reduction sum */
  sum = 0.0;


#pragma omp parallel private(i,tid,psum) num_threads(threads)
  {
    psum = 0.0;
    tid = omp_get_thread_num();
    if (tid ==0)
      {
	threads = omp_get_num_threads();
	printf("Task %d using %d threads\n",myid, threads);
      }

    if(myid ==0)
    execTime = - omp_get_wtime();

#ifdef DOT_PROD
#pragma omp for reduction(+:sum)
    for (i=0; i<len*threads; i++)
      {
	sum += (a[i] * b[i]);
	psum = sum;
      }
#ifdef VERBOSE
    printf("Task %d thread %d partial sum = %f\n",myid, tid, psum);
#endif
  }


  /* Print this task's partial sum */
  mysum = sum;
#ifdef VERBOSE
  printf("Task %d partial sum = %f\n",myid, mysum);
#endif 
  /* After the dot product, perform a summation of results on each node */
  MPI_Reduce (&mysum, &allsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif 


#ifdef DAXPY
#pragma omp for reduction(+:sum)
    for (i=0; i<len*threads; i++)
      {
	sum += (a[i] * b[i]);
	psum = sum;
      }
#ifdef VERBOSE
    printf("Task %d thread %d partial sum = %f\n",myid, tid, psum);
#endif
  }
  /* Print this task's partial sum */
  mysum = sum;
#ifdef VERBOSE
  printf("Task %d partial sum = %f\n",myid, mysum);
#endif 
  /* After the dot product, perform a summation of results on each node */
  MPI_Reduce (&mysum, &allsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif 

#ifdef OMP_STENCIL
#pragma omp for reduction(+:sum)
    for (i=0; i<len*threads; i++)
      {
	sum += (a[i] * b[i]);
	psum = sum;
      }
#ifdef VERBOSE
    printf("Task %d thread %d partial sum = %f\n",myid, tid, psum);
#endif
  }
  /* Print this task's partial sum */
  mysum = sum;
#ifdef VERBOSE
  printf("Task %d partial sum = %f\n",myid, mysum);
#endif 
  /* After the dot product, perform a summation of results on each node */
  MPI_Reduce(&mysum, &allsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif

#ifdef STENCIL
#pragma omp for reduction(+:sum)
    for (i=0; i<len*threads; i++)
      {
	sum += (a[i] * b[i]);
	psum = sum;
      }
#ifdef VERBOSE
    printf("Task %d thread %d partial sum = %f\n",myid, tid, psum);
#endif
  }
  /* Print this task's partial sum */
  mysum = sum;
#ifdef VERBOSE
  printf("Task %d partial sum = %f\n",myid, mysum);
#endif 
  /* After the dot product, perform a summation of results on each node */
  MPI_Reduce (&mysum, &allsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif

// regular comm, load balanced 
#ifdef PICALC
#pragma omp for reduction(+:sum)
    for (i=0; i<len*threads; i++)
      {
	sum += (a[i] * b[i]);
	psum = sum;
      }
#ifdef VERBOSE
    printf("Task %d thread %d partial sum = %f\n",myid, tid, psum);
#endif
  }
  /* Print this task's partial sum */
  mysum = sum;
#ifdef VERBOSE
  printf("Task %d partial sum = %f\n",myid, mysum);
#endif 
  /* After the dot product, perform a summation of results on each node */
  MPI_Reduce (&mysum, &allsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif 


#ifdef SORTING
#pragma omp for reduction(+:sum)
    for (i=0; i<len*threads; i++)
      {
	sum += (a[i] * b[i]);
	psum = sum;
      }
#ifdef VERBOSE
    printf("Task %d thread %d partial sum = %f\n",myid, tid, psum);
#endif
  }
  /* Print this task's partial sum */
  mysum = sum;
#ifdef VERBOSE
  printf("Task %d partial sum = %f\n",myid, mysum);
#endif 
  /* After the dot product, perform a summation of results on each node */
  MPI_Reduce (&mysum, &allsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif 

#ifdef SORTING
#pragma omp for reduction(+:sum)
    for (i=0; i<len*threads; i++)
      {
	sum += (a[i] * b[i]);
	psum = sum;
      }
#ifdef VERBOSE
    printf("Task %d thread %d partial sum = %f\n",myid, tid, psum);
#endif
  }
  /* Print this task's partial sum */
  mysum = sum;
#ifdef VERBOSE
  printf("Task %d partial sum = %f\n",myid, mysum);
#endif 
  /* After the dot product, perform a summation of results on each node */
  MPI_Reduce (&mysum, &allsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif 

#ifdef NBODY
#pragma omp for reduction(+:sum)
    for (i=0; i<len*threads; i++)
      {
	sum += (a[i] * b[i]);
	psum = sum;
      }
#ifdef VERBOSE
    printf("Task %d thread %d partial sum = %f\n",myid, tid, psum);
#endif
  }
  /* Print this task's partial sum */
  mysum = sum;
#ifdef VERBOSE
  printf("Task %d partial sum = %f\n",myid, mysum);
#endif 
  /* After the dot product, perform a summation of results on each node */
  MPI_Reduce (&mysum, &allsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif 

// irregular comm., load balanced 
#ifdef INDACC
#pragma omp for reduction(+:sum)
    for (i=0; i<len*threads; i++)
      {
	sum += (a[i] * b[i]);
	psum = sum;
      }
#ifdef VERBOSE
    printf("Task %d thread %d partial sum = %f\n",myid, tid, psum);
#endif
  }
  /* Print this task's partial sum */
  mysum = sum;
#ifdef VERBOSE
  printf("Task %d partial sum = %f\n",myid, mysum);
#endif 
  /* After the dot product, perform a summation of results on each node */
  MPI_Reduce (&mysum, &allsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif 

// irregular comm., load imbalanced 
#ifdef SPMV
#pragma omp for reduction(+:sum)
    for (i=0; i<len*threads; i++)
      {
	sum += (a[i] * b[i]);
	psum = sum;
      }
#ifdef VERBOSE
    printf("Task %d thread %d partial sum = %f\n",myid, tid, psum);
#endif
  }
  /* Print this task's partial sum */
  mysum = sum;
#ifdef VERBOSE
  printf("Task %d partial sum = %f\n",myid, mysum);
#endif 
  /* After the dot product, perform a summation of results on each node */
  MPI_Reduce (&mysum, &allsum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
#endif 



  if(myid ==0)
    execTime += omp_get_wtime();

  if (myid == 0) 
    printf ("Done. Hybrid version: global sum  =  %f \n", allsum);

  if (myid == 0) 
    printf("Total time for dot prod = %f \n", execTime);

  free (a);
  free (b);
  MPI_Finalize();
}
