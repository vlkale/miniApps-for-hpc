/********************************************************
*
*  Description: Jacobi computation intended to be done in hybridized MPI+pthreads model. 
*  The kernel is based on Quinn et al.
*
*  
*  Last Revised: 2/25/2020  Vivek Kale 
*  Brookhaven National Laboratory / Stony Brook University
*************************************************************/

#include "../framework/hpcTuner.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include <omp.h>
#include <pthread.h>

#define MAX_THREADS 16
#define VERBOSE 0
#define EPSILON .0000001

#define BLOCKING_MPI 0

#define USE_HYBRID_MPI 1
#define DATA_COLLECTION 1
#define HISTOGRAM_GENERATION 1


#define JACOBI_ITERATIONS 1000

// Domain Decompostion 
/* #define A(i,j,k) ((i)*Z*Y + (j)*Z + (k)) */
#define A(i,j,k) ((j)*Z*X + (i)*Z + (k)) 
#define AA(j,k) ((j)*Z + (k))
 
FILE* loadBalanceData;
extern FILE* perfTuningData;

int numThreads;
int mySize;
int Z = 64, Y = 512, X = 64;
int Zdim, Ydim, Xdim;


int p;
int id ; 
pthread_mutex_t mutexdiff;
extern pthread_barrier_t myBarrier; 

int histogram[100];

int mallocMode =1; /* 1: first touch is  handled , 0:  first touch is not handled */


/* measurement and statistics info */
int workCounts[MAX_THREADS][JACOBI_ITERATIONS]; 
double workTimes[MAX_THREADS][JACOBI_ITERATIONS];

double totalOverhead[MAX_THREADS];
double totalOverheads = 0.0;
double workTimeTotal = -1.0;
double totalThreadIdleTime = -1.0;
double threadIdleTime[MAX_THREADS];

double totalExecutionTime = 0.0;
unsigned long long flops;

int curr; 

/* utility functions */
extern double getMax(double*, int, int);  
extern double getMin(double*, int, int);  
extern double getAvg(double*, int, int);  
extern double getRange(double*, int, int);  

/* application-specific to jacobi */ 
double* u;
double* w; // global - gets updated on each jacobi iteration. 
double* result;

double* myLeftBoundary;
double* myRightBoundary;
double* myLeftGhostCells;
double* myRightGhostCells;

double* jacobi3D(int threadID); 
void printMatrix(double* matrix, int id, int matDimX, int matDimY, int matDimZ);

int runExperiment(int tid)
{
  jacobi3D(tid);
}


/*  
CORE algorithm:   this contains the core 3D stencil algorithm.  
This decomposition can be specified through #define AA and AAA
*/



double* jacobi3D(int threadID)
{
  double coeff = 1.0/7.0;
  int startj = 0;
  int endj = 0;
  double diff;
  double global_diff =0.0;
  double communicationTime;
  double communicationTimeBegin ;
  double communicationTimeEnd;
  int i, j, k;
  int its =0;
  double tdiff, t_start, t_end;
  double tickTime;  
  int boundarySize;
  int ghostSize;  
  int Y_size;
  double threadIdleBegin = 0.0;
    
  boundarySize = (Z-2)*(Y-2);
  ghostSize = (Z-2)*(Y-2);
  
  /* we don't need to communicate boundary border values, so we allocate M-2 by N-2 space */
  if(threadID ==0)
    {
      myLeftBoundary = (double*) malloc(boundarySize*sizeof(double));
      myRightBoundary = (double*) malloc(boundarySize*sizeof(double)); 
      myLeftGhostCells = (double*) malloc(ghostSize*sizeof(double));
      myRightGhostCells = (double*) malloc(ghostSize*sizeof(double));
    }  
  if(threadID ==0 )
    {
      for(i = 0; i < numThreads ; i++)
	{
	  totalOverhead[i] = 0.0; 
	  threadIdleTime[i] = 0.0;
	  for(j = 0; j< JACOBI_ITERATIONS; j++)
	    {
	      workCounts[i][j] = 0;
	      workTimes[i][j]= 0.0;
	    }
	} 
      totalThreadIdleTime = 0.0;
      totalOverheads = 0.0;
      workTimeTotal = 0.0;
      totalThreadIdleTime = 0.0;
      flops = 0;
    }  
  MPI_Status status;

  MPI_Status statii[4];
  MPI_Request sendLeft;
  MPI_Request sendRight;
  MPI_Request recvLeft;
  MPI_Request recvRight;
  MPI_Request requests[4];

  its = 0;
  for( i = 0; i < 61; i++)
    histogram[i] = 0;
  diff = 0.0; 
  pthread_barrier_wait(&myBarrier);
  /*****   BEGIN  AN ITERATION of JACOBI STENCIL COMPUTATION ******/
  double previousIterationTime; 
  communicationTime = 0.0;
  if(threadID == 0 && id ==0)
    {
      t_start = MPI_Wtime();  
      previousIterationTime = t_start;
    }
  while (1)
    {      
      if(USE_HYBRID_MPI)
	{ 
	  if(threadID ==0 ) /*  num Processes should be greater than zero */
	    {
	      /* goto BARRIER; */
	      communicationTimeBegin =  MPI_Wtime();         
	      for (i = 1; i<  X -1 ; i++)
		for (j = 1; j < Y -1; j++)
		  myRightBoundary[AA(i,j)] = u[A(i,j,Z-1)]; 
	      for (i = 1; i< X -1 ; i++)
		for (j = 1; j < Y -1; j++)
		  myLeftBoundary[AA(i,j)] = u[A(i,j ,1)]; 
	      
	      if(!BLOCKING_MPI)
		{
		  int numRequests = 0;
		  if( id > 0 ) 
		    MPI_Irecv(myLeftGhostCells, ghostSize, MPI_DOUBLE, id - 1, 0 , MPI_COMM_WORLD, &requests[numRequests++]);
		  
		  if(id < p-1)
		    MPI_Irecv(myRightGhostCells, ghostSize, MPI_DOUBLE, id + 1, 0 , MPI_COMM_WORLD, &requests[numRequests++]);
		  
		  if (id > 0)
		    MPI_Isend(myLeftBoundary, boundarySize, MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &requests[numRequests++]);
		  
		  if (id < p - 1 ) 
		    MPI_Isend(myRightBoundary, boundarySize, MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD, &requests[numRequests++]); 		  
		  MPI_Waitall(numRequests, requests, MPI_STATUSES_IGNORE);
		}

	      else
		{
		  if (id > 0 )
		    { 
		      /*  printf("rank %d sending left boundary with size = %d \n" , id, boundarySize); */
		      MPI_Send(myLeftBoundary, boundarySize, MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD);  
		      /*  printf("rank %d done sending left boundary\n" , id); */
		    }
		  /* printf("rank %d just before sending right boundary\n" , id); */
		  if (id < p-1) 
		    { 
		      MPI_Recv(myRightGhostCells, ghostSize, MPI_DOUBLE, id +1, 0, MPI_COMM_WORLD, &status ); 
		    }
		  /* printf("rank %d  is here \n" , id); */
		  if (id < p-1)
		    {
		      /*   printf("rank %d sending right boundary with boundary size %d \n" , id, boundarySize); */
		      MPI_Send(myRightBoundary, boundarySize, MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
		      /* printf("rank %d Done sending right boundary with boundary size %d \n" , id, boundarySize); */ 
		    }
		  if (id > 0) /* if id  is 0 we  don't  receive because the ghost cells  are actually the boundary cells */
		    {
		      MPI_Recv(myLeftGhostCells, ghostSize, MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status); 
		      /*  printf("rank %d receiving a ghost cell\n" , id); */
		    }		  
		}
	    }
	}
  
      threadIdleBegin = MPI_Wtime();
      pthread_barrier_wait(&myBarrier);
      if( threadID==0 )
	    {
	      communicationTimeEnd = MPI_Wtime();
	      communicationTime += (communicationTimeEnd - communicationTimeBegin);
	    }   
      tdiff = 0.0; 
      // partitioning of slabs to threads. 
      startj = 1 + (Y/numThreads)*threadID;
      endj = startj + Y/numThreads;
      int rangeNext = 0;
      int workCount = 0;
      // TODO: Add PAPI here for collecting cache misses 
      
      threadIdleTime[threadID] +=  (MPI_Wtime() - threadIdleBegin);
      double accumulatedWorkTime = 0.0;
      
  	  double tWorkBegin = MPI_Wtime(); 
      
 //     #pragma omp for schedule (guided, 4) // can use user-defined schedules here. 
	    for(j = startj ; j < endj ; j++)
	     {
	      for(i = 1; i < X-1; i++)
		     {
		     for (k = 1; k < Z - 1; k++) 
		    /* printf("%d \n", &(w[A(i, j, k)]));  */
		    w[A(i,j,k)] = ( 
				   u[A(i-1,j,k)]+ u[A(i+1,j,k)] 
				   + u[A(i,j-1,k)] + u[A(i, j+1, k)]+
				   + u[A(i, j, k-1)] + u[A(i,j, k+1)]
				   + u[A(i, j, k)]    
				   )*coeff;
		     }
	     }
	   workCount += (endj - startj)*(X-2)*(Z-2); 
     // gather dequeue overheads from using dynamic scheduling. 
     
      threadIdleBegin = MPI_Wtime();
      if(DATA_COLLECTION)
	    {
	     workCounts[threadID][its] = workCount; 
	     /*  workTimes[threadID][its] = MPI_Wtime() - tWorkBegin; */
	     workTimes[threadID][its] = accumulatedWorkTime; 
   	}
      
      pthread_barrier_wait(&myBarrier);
      if(threadID ==0)
	{
	  if(1)
	    {
	      double* temp = w ;
	      w = u ; 
	      u = temp;
	    }
	  else
	    {
	      for (i = 0; i<X-1; i++)
	     	 for(j = 0; j< Y-1; j++)
		      for(k = 0; k < Z-1; k++)
		        w[A(i,j,k)] = u[A(i, j, k)];     
	    }
	}
  
 
      // TODO : need to add this back in .    
      if(threadID==0)
	{
	  /* MPI_Allreduce(&diff,  &global_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); */
	}
      pthread_barrier_wait(&myBarrier);
      /* END ITERATION  OF JACOBI STENCIL COMPUTATION */
      threadIdleTime[threadID] +=  (MPI_Wtime() - threadIdleBegin);
 
      /*   HISTOGRAM DATA COLLECTION */ // can do this in separate data collection library

      if( threadID==0  ) {
	    double endIterationTime = MPI_Wtime();
	    double timeForIteration = endIterationTime - previousIterationTime; 
	    previousIterationTime = endIterationTime; 
	    int bin = (int) (timeForIteration/.00025);  /* each bin is 250 microseconds */
    	if(bin > 60 ) bin = 60 ;
	    histogram[bin]++;
      }

    SKIP:
      its++;
      if ( /* (global_diff <= EPSILON) */ (its >= JACOBI_ITERATIONS) )
	{ 	
	  if (threadID ==0)
	    printf("converged or completed max jacobi iterations.\n");
	  break;
	} 
    }

  if(threadID == 0 && id == 0)
    {
      t_end = MPI_Wtime(); 
    }
  if(id == 0)
    totalExecutionTime = t_end - t_start;
  
  if( threadID ==0 && (rank == 0))
    printf("totalExection time on rank %d was  %f \n" ,rank, totalExecutionTime);
  double dataCollectionTime, dataCollectionTime_start;  
  if(threadID ==0)
    dataCollectionTime_start = MPI_Wtime();
  
  /****** COLLECTION OF PERFORMANCE STATS AFTER  ALL ITERATIONS COMPLETED *********/  
  double sumOverheadWorkTimes = 0.0; 
  double sumAvgWorkTimes =0.0;
  int numIterationsIncluded = 0;
  int y = 3;
  double x = 0.0;
  double lock_time =0.0;
  double barrier_time  = 0.0;
  if((threadID == 0) &&  (rank ==0) && (DATA_COLLECTION) )
    {
      workTimeTotal = 0.0; 
      for(i = 0 ;  i < JACOBI_ITERATIONS; i++)
	{
	  if(VERBOSE >= 3)
	    {
	      for(j = 0 ; j < numThreads; j++)
		printf("%d(%f), ", workCounts[j][i], workTimes[j][i]*1000.0 );
	      printf("\n");
	    }
	  double maxTime = 0.0;
	  double minTime = 99999.0;
	  double sumTime = 0.0;
	  int maxWork = 0;
	  int minWork = 999999;
	  int sumWork = 0;
	  for(j = 0; j < numThreads; j++)
	    {
	      if(VERBOSE >= 2)
		printf("accumulating work times\n");
	      maxTime = (maxTime < workTimes[j][i] ) ? (workTimes[j][i]): maxTime;
	      minTime = (minTime > workTimes[j][i] ) ? (workTimes[j][i]): minTime;
	      sumTime  += workTimes[j][i];
	      maxWork = (maxWork < workCounts[j][i] ) ? (workCounts[j][i]): maxWork;
	      minWork = (minWork > workCounts[j][i] ) ? (workCounts[j][i]): minWork;
	      sumWork  += workCounts[j][i]; 	   
	    }	
	  double timeBalance = (maxTime - minTime)/(sumTime/(1.0*numThreads));
	  double workBalance = ((double) (maxWork - minWork))/((1.0*sumWork)/(1.0*numThreads));
	  workTimeTotal += sumTime;
	  flops += 8*sumWork; 
	  if( (timeBalance < 0.05 )  && (workBalance <  0.01)) 
	    {    
	      sumAvgWorkTimes += sumTime/(1.0*numThreads);
	      numIterationsIncluded++;
	    }
	}
      if(rank ==0){
	double totalOverheads = 0.0;

	for(i = 0; i< numThreads; i++)
	  { 
	    totalOverheads += totalOverhead[i];
	    totalThreadIdleTime += threadIdleTime[i];
	  }
    
  loadBalanceData  = fopen("jacobiLoadBalance.dat", "a+");   
	if(!loadBalanceData)
	  printf("error opening optional performanceStats loadBalanceData. Is it in the same folder as jacobiHybridSlabs3D.c? \n");
	else
	  fprintf(loadBalanceData, "\t%d\t%d\t%d\t%f\n", p, numThreads, numIterationsIncluded, sumAvgWorkTimes/(1.0*numIterationsIncluded));
	fclose(loadBalanceData);
      }
    }
  if(threadID == 0 && id == 0)
    {
      printf("STENCIL\t3D-VH-SLAB\t%d\t%d\t%d\t%d\t%d\t%d\t%f\t%f\t%f\t%f\t%d\n" ,
	     X, Y,Z, p, numThreads,its, t_end - t_start, workTimeTotal, totalThreadIdleTime, communicationTime, numWorkSteal);
      dataCollectionTime = MPI_Wtime() - dataCollectionTime_start;
      printf("data Collection time = %f \t flops = %llu \t flops per second =  %llu \n",
	     dataCollectionTime,
	     flops,
	     (unsigned long long) ( flops/ (numThreads*(t_end - t_start))  ));
    }
  
  if(HISTOGRAM_GENERATION == 1)
    {
      if( (threadID == 0)  &&  (rank ==0) )
	{
	  printf("histogram for %d iterations for rank %d \n", its, rank);
	  int i ; 
	  for(i  = 0;  i <  61 ;  i++) // TODO: unhard code 61 - needs to be adjusted for how much granularity to see .
	    printf( "\t%f\t%d\n" ,   i*0.25, histogram[i]);
	}
    }
  pthread_barrier_wait(&myBarrier);
  return u; // only need this if we want display result. - also need a norm for correctness and a number for answer. 
}
