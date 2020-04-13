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

// regular comm, load imbalanced

      program dProd
      ! use mpi

      use iso_c_binding, only: c_int, c_double, c_ptr, c_null_ptr

    !DEC$ if defined(_OpenACC)
      !use acc_lib
!DEC$ ELSE
      use omp_lib
!DEC$ endif

      !#ifdef _OpenACC
      !use openacc
      !#else
      ! use omp_lib
      !#endif

      implicit none

      include 'mpif.h'
      ! include 'openaccf.h'

    integer ( kind = 4 ) rank
    integer ( kind = 4 ) size
    integer ( kind = 4 ) ierror

      !     runtime and openMP variables
      INTEGER NTHREADS
      INTEGER MAX_THREADS
      INTEGER TID
      integer tid_ind
    PARAMETER(MAX_THREADS= 16)
      real endLoopReturnVal

      !     loop iteration variables
      INTEGER i
      !     application specific variables
      double precision checkSum
      !     performance timing variables
    double precision :: startTime, endTime, totalTime
    integer, PARAMETER :: probSize = 10000000
    integer :: N
    integer, PARAMETER :: NUM_TIMESTEPS = 100

      integer num_dev
      integer dev_id

      character*80 arg
    integer (kind = 4) num_args
      ! !     app variables
      ! INTEGER N
      !INTEGER NUM_TIMESTEPS

    !TODO : get input arg for problem size to do tests

						   ! read(*,N)
      ! select case -  error handling of input

      INTEGER TIMESTEP
						   !real a(N)
						   !real b(N)
						   !real c(N)

						   ! TODO: add allocate for a, b and c  -done
    ! TODO: understand how/why the below pointer and allocate statements work
    real, pointer :: a(:,:)
    real, pointer :: b(:,:)
    real, pointer :: c(:)

    real, pointer :: u(:,:)
    real, pointer :: unew(:,:) 
      

    !      num_args = iargc()
    num_args = command_argument_count()
      num_args = 2
    if (num_args .gt. 1) then
			   call getarg(1, arg)
			   read(arg,*) N
      end if
    if (rank == 0) then
		     print *, 'The problem size  input is', N
		     call flush(6)
		     call flush(0)
      end if

    call MPI_INIT(ierror)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierror)
    call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierror)


    allocate(a(N, N))
    allocate(b(N))
    allocate(c(N))

    !DEC$ IF DEFINED(_OPENACC)
    ! call acc_init (acc_get_device_type())
      !#ifdef  _OPENACC
    if(rank == 0) then
		    ! print *, 'initializing openACC'
		    ! call flush(6)
		    !  call flush(0)
      end if
    !call acc_init(acc_get_device_type())
!DEC$ ENDIF
    ! num_dev = acc_get_num_devices(acc_get_device_type())
    ! dev_id = mod(rank, num_dev)
    ! call acc_set_device_num(dev_id, acc_get_device_type())
      ! assign GPU to one MPI process
    ! print *, 'MPI rank ', rank, ' assigned to GPU ' , dev_id
    !! call flush(6)
    ! call flush(0)

       !#endif
    !call acc_init(acc_device_nvidia)
    !$DEC IF DEFINED(_OPENACC)
    print *, 'OpenACC active'
    call flush(6)
    call flush(0)
!$DEC ELSE

!$OMP PARALLEL
    nthreads = omp_get_num_threads()
    tid = omp_get_thread_num()
    print *, 'Thread ' , tid , ' of ', nthreads , ' active.'
    call flush(6)
    call flush(0)
!$OMP END PARALLEL
!$DEC ENDIF

    DO i = 1, N
    a(i) = 1.0
    b(i) = 1.0
    c(i) = 1.0
      END DO

    call MPI_BARRIER(MPI_COMM_WORLD, ierror)
    !     TODO: consider array of per-thread and per-timestep timings
    startTime = MPI_Wtime()

    do timestep = 1, NUM_TIMESTEPS

         if (mod(id, 2) == 0) then
				call MPI_Irecv(indata, msgSize, MPI_FLOAT, &
					       mod((rank + 1), p) , 0 , & 
					       MPI_COMM_WORLD, requests, ierror)
         end if 
    if (mod(id, 2) == 1) then
			   call MPI_Irecv(indata, msgSize, MPI_FLOAT, &
					  mod((rank - 1), p), 0 , & 
					  MPI_COMM_WORLD, requests, ierror);
         end if

         if (mod(id,2)  == 0) then
				call MPI_Isend(outdata, msgSize, MPI_FLOAT, rank + 1, 0, &
					       MPI_COMM_WORLD, requests, ierror)
         end if

         if (mod(id, 2) == 1) then
				call MPI_Isend(outdata, msgSize, MPI_FLOAT, rank - 1, 0, &
					       MPI_COMM_WORLD, requests)
         end if
         
         call MPI_Waitall(numRequests, requests, MPI_STATUSES_IGNORE, &
			  ierror)

	 !DEC$ IF DEFINED(_OPENACC)
	 !$ACC parallel loop data present_or_copyin(a(0:n), b(0:n)), copy(c(0:n))
!DEC$ ELSE
!$OMP PARALLEL DO
!DEC$ ENDIF 
	   DO i = 1, N
	   c(i) = c(i) + a(i)*b(i)
         END DO
	   !DEC$ IF DEFINED(_OPENACC)
!$ACC END parallel loop
!DEC$ ELSE
!$OMP END PARALLEL DO
!DEC$ ENDIF
	   call MPI_BARRIER (MPI_COMM_WORLD, ierror)
      end do

		endTime = MPI_Wtime()

		  if (rank == 0) then
				   ! write(*,*), 'That took ', endTime - startTime, ' seconds.'
				   print *, 'print:That took ', endTime - startTime , ' seconds.'
				   call flush(6)
				   call flush(0)
      end if


	   call MPI_Finalize(ierror)
      stop

      end program dProd


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
