/********************************************************
*
*  Description: Jacobi computation intended to be done in hybridized MPI+pthreads model. 
*  The kernel is based on Quinn et al.
*
*  
*  Last Revised: 3/16/2020  Vivek Kale 
*  Brookhaven National Laboratory / Stony Brook University
*************************************************************/

//#include "../framework/hpcTuner.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "mpi.h"
#include <omp.h>
#include <pthread.h>

#include <iostream>

using namespace std;

#define MAX_THREADS 16
#define EPSILON .0000001

#define USE_HYBRID_MPI 1
#define BLOCKING_MPI 0
int POINTER_SWAP=1; // swap pointers of arrays instead of cell copy of arrays (set to 0 for the that)


#define JACOBI_ITERATIONS 5

// Domain Decompostion 

/* #define A(i,j,k) ((i)*Z*Y + (j)*Z + (k)) */ 
// the following is a slab decomposition 
#define A(i,j,k) ((j)*Z*X + (i)*Z + (k)) 
#define AA(j,k) ((j)*Z + (k))
 
#define NUM_CORES 16
#define NUM_NODES 1024


// Experimental methodology and data collection. 
#define NUM_TRIALS 1
#define HISTOGRAM_NUM_BINS 100
#define DATA_COLLECTION 1
#define HISTOGRAM_GENERATION 1
FILE* loadBalanceData;
FILE* perfTuningData;

FILE* perfTestOutput;
FILE* perfTestOutput_Temp;
//char* perfTestsFileName[127];
char perfTestsFileName[127];

int procID; 
int numprocs; 
int processesPerNode;

int skipCoreCount;
int numThreads;
int numthrds;
int mySize;


int numBins = HISTOGRAM_NUM_BINS;


// TODO: add RAJA 


// Need to create library for fortran 
/* --Library for scheduling strategy and variables and macros associated with the library -- */
//#include "vSched.h"
double constraint;
double fs;
// in the below macros, strat is how we specify the library
#define FORALL_BEGIN(strat, s,e, start, end, tid, numThds )  loop_start_ ## strat (s,e ,&start, &end, tid, numThds);  do {
#define FORALL_END(strat, start, end, tid)  } while( loop_next_ ## strat (&start, &end, tid));


/* -- Debugging -- */
 #define VERBOSE 0

/* --  Performance Measurement -- */
double totalTime = 0.0;
FILE* myfile;// output file for experimental data

/* --  Hardware Profiling -- */
// #define USE_PAPI 
 #ifdef USE_PAPI
 #include <papi.h>  // Comment these lines on PAPI out to make fully portable. Some platforms don't have the L2 and L3 cache misses available. TODO: need a way to check the counters in config or programmatically. 
#endif

int p;
int id ; 
pthread_mutex_t mutexdiff;
// extern pthread_barrier_t myBarrier; 
/* profiling */

double trialTimes[NUM_TRIALS];
double minThreadTrialTimes[NUM_TRIALS];
double maxThreadTrialTimes[NUM_TRIALS];
int numTrials = 1;

int histogram[HISTOGRAM_NUM_BINS]; // for performance visualization of distribution of iteration timings

// #define PAPI_PROFILING
// Library for hardware profiling.
#ifdef PAPI_PROFILING
#include <papi.h>
#endif

int mallocMode =1; /* 1: first touch is  handled , 0:  first touch is not handled */

/* measurement and statistics info */
int workCounts[MAX_THREADS][JACOBI_ITERATIONS]; 
double workTimes[MAX_THREADS][JACOBI_ITERATIONS];

double totalOverhead[MAX_THREADS];
double totalOverheads = 0.0;
double workTimeTotal = -1.0;
double totalThreadIdleTime = -1.0;
double threadTrialTime[MAX_THREADS][JACOBI_ITERATIONS];
double threadIdleTime[MAX_THREADS];

double totalExecutionTime = 0.0;
double startTime_Init, endTime_Init, startTime_Experiment, endTime_Experiment;

unsigned long long flops;

int curr; 

/* utility functions */
extern double getMax(double*, int, int);
extern double getMin(double*, int, int);
extern double getAvg(double*, int, int);
extern double getRange(double*, int, int);


/* experimental framework */ 
void preProcessInput(int input_argc, char** input_argv);
void initializeExperiment(int threadID, int p);

void printMatrix(double* matrix, int id, int matDimX, int matDimY, int matDimZ);
void collectPerfStats();

double* runExperiment(int tid, double* result);


/* application-specific to jacobi */ 
#define X_dim 16
#define Y_dim 64
#define Z_dim 16

int Z = 1, Y = 1, X = 1; // default dimensions 
int numTimesteps = 0; 
int Zdim, Ydim, Xdim;

double* u;
double* w; // global - gets updated on each jacobi iteration. 
double* result;

 int boundarySize = (Z-2)*(Y-2);
int ghostSize = (Z-2)*(Y-2);

//double myLeftBoundary[(X_dim-2)*(Y_dim-2)];
// double myRightBoundary[(X_dim -2)*(Y_dim -2)];
// double myLeftGhostCells[(Z_dim -2)*(Y_dim - 2)]; 
// double myRightGhostCells[(Z_dim -2)*(Y_dim -2)];

double* myLeftBoundary;
double* myRightBoundary;
double* myLeftGhostCells;
double* myRightGhostCells;

double* jacobi3D(int, double*);



double* runExperiment(int tid, double* result) // todo: really ought to change this to a void for generic type
{
  return jacobi3D(tid, result);
}

/*  
CORE algorithm:   this contains the core 3D stencil algorithm.  
This decomposition can be specified through #define AA and AAA
*/


double* jacobi3D(int threadID, double* resultMatrix)
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
  int Y_size;
  double threadIdleBegin = 0.0;

  MPI_Status status;
  MPI_Status statii[4];
  MPI_Request sendLeft;
  MPI_Request sendRight;
  MPI_Request recvLeft;
  MPI_Request recvRight;
  MPI_Request requests[4];
  
  its = 0;
  for( i = 0; i < numBins; i++)
    histogram[i] = 0;
  diff = 0.0; 
  
  ///  BEGIN  AN ITERATION of JACOBI STENCIL COMPUTATION
  double previousIterationTime; 
  communicationTime = 0.0;
  if(id ==0)
    {
#pragma omp master 
      {
	t_start = MPI_Wtime();  
	previousIterationTime = t_start;
      }
    }

  while (1)
    {
      #pragma omp master
       MPI_Barrier(MPI_COMM_WORLD);
      #pragma omp barrier 
    
  if(USE_HYBRID_MPI || (numprocs > 1))
	{ 
 /*  num Processes should be greater than zero */
    #pragma omp master 
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

		  //	  MPI_Waitall(numRequests, requests, MPI_STATUSES_IGNORE);
		  MPI_Waitall(numRequests, requests, statii);
		}
	      else
		{
		  if (id > 0 ) 
		      MPI_Send(myLeftBoundary, boundarySize, MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD);  
		  if (id < p-1) 
		      MPI_Recv(myRightGhostCells, ghostSize, MPI_DOUBLE, id +1, 0, MPI_COMM_WORLD, &status ); 		 
		  if (id < p-1)
		      MPI_Send(myRightBoundary, boundarySize, MPI_DOUBLE, id+1, 0, MPI_COMM_WORLD);
		  if (id > 0) /* if id  is 0 we  don't  receive because the ghost cells  are actually the boundary cells */
		      MPI_Recv(myLeftGhostCells, ghostSize, MPI_DOUBLE, id-1, 0, MPI_COMM_WORLD, &status);
		}
	    }
	}

      tdiff = 0.0; 
      // partitioning of slabs to threads. 
      // startj = 1 + (Y/numThreads)*threadID;
      // endj = startj + Y/numThreads;

      startj = 0;
      endj = Y;
      // TODO: Add PAPI here for collecting cache misses 
    
      // TODO: figure out how to separate gpu diff from node diff .
      // TODO: figure out how to make some number of threads (one per core of multi-core each control a GPU). 
      // TODO: tune num teams , thread limit , distschedule , chunk size 	  
      // patition loop between CPU and GPUs 
      // map(tofrom: gpudiff) 
	  /* Note:  The variable sum is now mapped with tofrom, for correctexecution with 4.5 (and pre-4.5) compliant compilers. See Devices Intro.S-17*/ 



      #pragma omp target map(to: u[0:(X*Y*Z)], v[0:(X*Y*Z)]) 
      #pragma omp teams num_teams(8) thread_limit(16) 
      #pragma omp distribute parallel for dist_schedule(static, 1024) schedule(static, 64) 

      // can use user-defined schedules here. 
      // #pragma omp for schedule (guided)
      for(j = startj ; j < endj ; j++)
	{
	  for(i = 1; i < X-1; i++)
	    {
	      for (k = 1; k < Z - 1; k++) 
		w[A(i,j,k)] = ( 
			       u[A(i-1,j,k)]+ u[A(i+1,j,k)] 
			       + u[A(i,j-1,k)] + u[A(i, j+1, k)]+
			       + u[A(i, j, k-1)] + u[A(i,j, k+1)]
			       + u[A(i, j, k)]    
				)*coeff;
	    }
	} 

#pragma omp master 
	{
	  if(POINTER_SWAP) 
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
      // TODO : need to add this back in .  
	  /* MPI_Allreduce(&diff,  &global_diff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); */
	}

#pragma omp single
	its++;
	
#pragma omp barrier 

	if ( /* (global_diff <= EPSILON) */ (its >= numTimesteps) )
	  {
	    // #pragma omp master
	    #ifdef VERBOSE
	    printf("converged or completed max jacobi iterations.\n");
	    #endif 
	    break;
	  } 
	else 
	  {
#pragma omp master 
	    {
#ifdef DEBUG 
	    cout << "Process " << procID <<  " finished Jacobi iteration " << its << endl;
#endif
	    }
	  }
    } // END WHILE loop of Jacobi Iterative Computation 

	/*   HISTOGRAM DATA COLLECTION */ // can do this in separate data collection library

#pragma omp master
	{
	    double endIterationTime = omp_get_wtime();
	    double timeForIteration = endIterationTime - previousIterationTime; 
	    previousIterationTime = endIterationTime; 
	    int bin = (int) (timeForIteration/.00025);  /* each bin is 250 microseconds */
	    if(bin > 60 ) bin = 60; // TODO: don't hardcode this. 
	    histogram[bin]++;
	}

	if(id == 0)
	  {
#pragma omp master 
	    t_end = MPI_Wtime(); 
	    totalExecutionTime = t_end - t_start;
	  }
	
	if( procID == 0)
	  {
#pragma omp master
	    printf("jacobi3D(): Total execution time on MPI process %d was %f \n" ,procID, totalExecutionTime);
	  }

#pragma omp barrier 
	resultMatrix = u; // only need this if we want display result. - also need a norm for correctness and a number for answer. 
	return u;

} // end jacobi3d

void initializeExperiment(int threadID, int p)
{
  int i;
  int j;   
  int k;
  int chunkSize;
  int iterations = 0;
  chunkSize = X*Y*Z;
  #pragma omp master
    {
       for(j=1; j < Y-1; j++)
	 for(i=1; i < X-1; i++)
 	  for(k=1; k < Z-1; k++)   
 	    {
 	      u[A(i, j, k)] = i*j*k*1.0;
 	      w[A(i, j, k)] = i*j*k*1.0;
 	    }
       for(i=0; i<X; i++)
	for(k=0; k<Z; k++)
 	  {
 	    u[A(i, 0, k)] = 101.0;
 	    u[A(i, Y-1, k)] = -101.0;
 	    w[A(i, 0, k)] = 101.0;
 	    w[A(i, Y-1, k)] = - 101.0;
 	  } 
       for(i=0; i<X; i++)
	 for(j=0; j<Y; j++)
 	  {
 	    u[A(i,j,0)] = 102.0;
 	    u[A(i,j,Y-1)] = -102.0;
 	    w[A(i,j,0)] = 102.0;
 	    w[A(i,j,Y-1)] = -102.0;
 	  }
       if(id == 0 )
 	{
 	  for(j=0; j <Y; j++)
 	    for(k=0; k<Z; k++) 
 	      {
 		u[A(0,j,k)] = 100.0;
 		w[A(0,j,k)] = 100.0;
 	      }
 	}
       if( id == p-1)
 	{
 	  for(j=0; j <Y; j++)
 	    for(k = 0; k<Z; k++) 
 	      {
 		u[A(X-1,j,k)] = -100.0;
 		w[A(X-1,j,k)] = -100.0;
 	      }
 	}

//       /* we don't need to communicate boundary border values, so we allocate M-2 by N-2 space */

// 	flops = 0;
    }
} // end Initialize 

 void preProcessInput(int input_argc, char** input_argv)
 {
   /* this is to take care of inconsistent naming I have done(TODO: change variable names) */
   p = numprocs;
   id = procID;
   numThreads = 4;
   MPI_Barrier(MPI_COMM_WORLD);
   if(input_argc <= 1)
     {
       printf("Usage: mpirun -n [numprocesses] jacobi-hybrid [numThreads] [Xdim][Ydim][Zdim] [<Num_Iters>][num_trials>]\n");
       exit(-1);
       MPI_Finalize();
     }
//   // TODO: need to pass back an object of variables to make this an application specific function
//   //TODO : need flags to parse input 
   if(input_argc == 2)
     {
       numThreads = atoi(input_argv[1]);        
       Xdim = 4;
       Ydim = 16;
       Zdim = 4;
       numTimesteps = JACOBI_ITERATIONS;
       numTrials = NUM_TRIALS;
     }
   if(input_argc == 5)
     {
       numThreads = atoi(input_argv[1]);  
       Xdim = atoi(input_argv[2]); 
       Ydim = atoi(input_argv[3]); 
       Zdim = atoi(input_argv[4]); 
       numTimesteps = JACOBI_ITERATIONS;
       numTrials = NUM_TRIALS;
     }
   if(input_argc == 6)
     {
       numThreads = atoi(input_argv[1]);  
       Xdim = atoi(input_argv[2]); 
       Ydim = atoi(input_argv[3]); 
       Zdim = atoi(input_argv[4]); 
       numTimesteps = atoi(input_argv[5]);
       numTrials = NUM_TRIALS;
     }
   if(input_argc == 7)
     {
       numThreads = atoi(input_argv[1]);
       Xdim = atoi(input_argv[2]); 
       Ydim = atoi(input_argv[3]); 
       Zdim = atoi(input_argv[4]); 
       numTimesteps = atoi(input_argv[5]); 
       numTrials = atoi(input_argv[6]); 
     }

   if(input_argc == 8)
     {
       numThreads = atoi(input_argv[1]);
       Xdim = atoi(input_argv[2]); 
       Ydim = atoi(input_argv[3]); 
       Zdim = atoi(input_argv[4]); 
       numTimesteps = atoi(input_argv[5]); 
       numTrials = atoi(input_argv[6]); 
       numBins = atoi(input_argv[7]); // user can set num bins (depends on numTimesteps, can also be dependent on numTrials) need an automated way to do this. 
     }

//   if(procID ==0)
//     printf("numthreads = %d ,   Xdim = %d , Ydim = %d , Zdim = %d  \n", numThreads,  Xdim ,  Ydim, Zdim);
  
   if (Xdim%numprocs !=0) /* application-specific */
     {
       if(procID == 0)
 	printf("WARNING: Length of X dimension is not divisible by number of processes. \n" );
     }
   if(Ydim%numThreads != 0)
     {
       if(procID == 0)
 	printf("WARNING: Ydim specified is not divisible by the number of threads. \n" );
     }
   /*  application-specific partitioning */
   Y =  (Ydim + 2);  
   //  Y_dynamic = atoi(input_argv[2]); 
   // Y_static = (int) (Ydim - Y_dynamic); 
//   // dynamic_ratio = (1.0*Y_dynamic)/(1.0*Ydim);  
   Z = Zdim + 2;
   X = (Xdim/numprocs) + 2;

//   /* initialize thread mutices for diffs */ 
   pthread_mutex_init (&mutexdiff, NULL);
 }

 void experimentCleanUp()
 {
   // TODO: This can cause problems on some machines. Uncomment below if it does
   // return;
   #ifdef VERBOSE 
  if(procID ==0)
    cout << "cleaning up experiment" << endl;
#endif 
   free(u);
   free(w);
   // free(myLeftBoundary);
   //free(myRightBoundary);
   //free(myLeftGhostCells );
   //free(myRightGhostCells);
   #ifdef VERBOSE
   if (procID == 0)
     cout << " ended experiment cleanup." << endl;
   #endif 

 }

 void nodeCleanUp()
 {
   pthread_mutex_destroy(&mutexdiff);
 }

void printResults(double* result, int id)
 {
   /// application-specific
   if (id == 0)
     cout << "MPI+OpenMP jacobi results shown below." << endl;
  
   if( (id == 0) || (id == 1) || (id == p-2) || (id == p -1)  ) 
     printMatrix(result, id, Z , Y, X);//prints horizontal slas  , with each slab separated by dashed lines 
 }



void printIterTimingHistograms(int _procID)
{  
  cout << "Histogram of timestep times across " << numTimesteps <<  " timesteps, for MPI rank " << _procID << endl;
  cout << "TimestepTimeRange(ms)\tNumTimestepsInRange " << endl;
  for(int i = 0; i < numBins; i++)// the upper bound ought to have a cutoff for insignificant bins(where the number of items is zero)
    {
	printf( "\t%f\t\t%d\n" ,   i*0.25, histogram[i]);
    }
}

void printPerfViz(int id)
{
  printIterTimingHistograms(id);
}

void setupOutFile(char** argv)
{

  int numCharsFromBuffer = 127;
  int charCount = snprintf(perfTestsFileName, numCharsFromBuffer, "outFile%ld_%d_%d_%d_%d_%d_%d.dat",atol(argv[3]), atoi(argv[4]), atoi(argv[1]), atoi(argv[2]), atoi(argv[9]), atoi(argv[5]) , atoi(argv[8]));                                                                                  
  perfTestOutput = fopen("outFilePerfTests.dat", "a+"); 
  perfTestOutput_Temp = fopen(perfTestsFileName, "w");
  if(perfTestOutput != NULL)  {
    fprintf(perfTestOutput, "#\t%ld_%d_%d_%d_%d_%d_%d\n", 
	    atol(argv[3]), atoi(argv[4]), atoi(argv[1]), atoi(argv[2]),                                                                            atoi(argv[9]), atoi(argv[5]), atoi(argv[8]));
    fprintf(perfTestOutput_Temp, "#\t%ld_%d_%d_%d_%d_%d_%d\n", 
	    atol(argv[3]), atoi(argv[4]),  atoi(argv[1]),atoi(argv[2]),                                                          
	    atoi(argv[9]), atoi(argv[5]),  atoi(argv[8]) );                                                                        
  }
  fclose(perfTestOutput);
  fclose(perfTestOutput_Temp);
} // end setupOutfile

/*
void collectPerfStats(int threadID, int its)	
{
 double dataCollectionTime, dataCollectionTime_start;  
  if(threadID ==0)
    dataCollectionTime_start = MPI_Wtime();
  
  // COLLECTION OF PERFORMANCE STATS AFTER  ALL ITERATIONS COMPLETED 
  double sumOverheadWorkTimes = 0.0; 
  double sumAvgWorkTimes =0.0;
  int numIterationsIncluded = 0;
  int y = 3;
  double x = 0.0;
  double lock_time =0.0;
  double barrier_time  = 0.0;
  if((threadID == 0) &&  (procID ==0) && (DATA_COLLECTION) )
    {
      workTimeTotal = 0.0; 
      for(int i = 0 ;  i < JACOBI_ITERATIONS; i++)
	{
	  if(VERBOSE >= 3)
	    {
	      for(int j = 0 ; j < numThreads; j++)
		printf("%d(%f), ", workCounts[j][i], workTimes[j][i]*1000.0 );
	      printf("\n");
	    }
	  double maxTime = 0.0;
	  double minTime = 99999.0;
	  double sumTime = 0.0;
	  int maxWork = 0;
	  int minWork = 999999;
	  int sumWork = 0;
	  for(int j = 0; j < numThreads; j++)
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

      if(procID ==0){
	double totalOverheads = 0.0;
	for(int i = 0; i< numThreads; i++)
	  { 
	    totalOverheads += totalOverhead[i];
	    totalThreadIdleTime += threadIdleTime[i];
	  } 
       loadBalanceData = fopen("jacobiLoadBalance.dat", "a+");   
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
	     X, Y, Z, p, numThreads,its, t_end - t_start, workTimeTotal, totalThreadIdleTime, communicationTime, numWorkSteal);
      dataCollectionTime = MPI_Wtime() - dataCollectionTime_start;
      printf("data Collection time = %f \t flops = %llu \t flops per second =  %llu \n",
	     dataCollectionTime,
	     flops,
	     (unsigned long long) ( flops/ (numThreads*(t_end - t_start))  ));
    }
  
  if(HISTOGRAM_GENERATION == 1)
    {
      if( (threadID == 0)  &&  (procID ==0) )
	{
	  printf("histogram for %d iterations for procID %d \n", its, procID);
	  int i ; 
	  for(i  = 0;  i <  61 ;  i++) // TODO: unhard code 61 - needs to be adjusted for how much granularity to see .
	    printf( "\t%f\t%d\n" ,   i*0.25, histogram[i]);
	}
    }
}
*/

void printMatrix(double* matrix, int id, int matDimX, int matDimY, int matDimZ)
{   
  int i;
  int j;
  int k;
  cout << "Begin matrix print for MPI process " << id << endl;
  //  return;
  for(j = 0; j < matDimY; j++)
    {
      for(i = 0; i < matDimZ; i++)
	{
	  for(k = 0; k <matDimX; k++)
	    {
	      printf("%3f ", matrix[A(i,j,k)]);
	    }
	  cout << endl;
	}
      cout << endl << "---------------- end slice " << j <<  "--------------------------------------" << endl;
    }
  cout << endl << "end Matrix print for process " << id << endl; 
}  

double getAvg(double* myArr, int size)
{
  int iter;
  double currSum;
  double average;
  for (iter = 0; iter < size; iter++)
    currSum += myArr[iter];
  average = currSum/(size*1.0); 
  return average;
}

double  getMax( double* myArr, int size     )
{
  int iter;
  double currMax= 0.0;
  for(iter = 0; iter < size; iter++)
    if(currMax < myArr[iter])
      currMax = myArr[iter];
  return currMax;
} 

double getMin (double* myArr, int size)
{
  int iter;
  double currMin = 99999.0;
  for(iter = 0; iter < size; iter++)
    if(currMin > myArr[iter])
      currMin = myArr[iter];
  return currMin;
}

double getRange(double* myArr, int size)
{
  double min = getMin(myArr, size);
  double max = getMax(myArr, size);
  return (max - min);
}


// function to use if compiling code standalone < --- hpctuner will take care of this in reality.
int main(int argc, char** argv)	
{
  int rcProc;
  long i;
  void *status;
  MPI_Init (&argc, &argv);
  MPI_Comm_size (MPI_COMM_WORLD, &numprocs);
  MPI_Comm_rank (MPI_COMM_WORLD, &procID);
 
  int numCores = NUM_CORES;
  int numNodes = NUM_NODES;
  /*  cpu_set_t *cpuset; */
  size_t size;
  double* resultMatrix; 
  if(argc >= 2)
    {
      numThreads = atoi(argv[1]);
      preProcessInput(argc, argv);      
      skipCoreCount = 0;
    }
  else
    if(procID == 0)
      {
	printf("Usage: mpirun -n [numthreadsPerProcess][problemSizeDimensions] [<numCoresPerNode>] [<numNodes>] \n");
	exit(1);
	MPI_Finalize();
      }
  if(procID == 0)
    {
      printf("Beginning computation. Using %d processes and %d threads per process \n", numprocs ,numThreads );
      //setupOutFile(perfTestsFileName, argv);
    }

  //  processesPerNode = numprocs/numNodes;
  // int procIDWithinNode = procID%processesPerNode;

  MPI_Barrier(MPI_COMM_WORLD);

  int dataSize = X*Y*Z;
  resultMatrix = (double*) malloc(sizeof(double)*dataSize); 
  u = (double*) malloc(sizeof(double)*dataSize); 
  w = (double*) malloc(sizeof(double)*dataSize);
  boundarySize = (Z-2)*(Y-2);
  ghostSize = (Z-2)*(Y-2); 
  myLeftBoundary = (double*) malloc(boundarySize*sizeof(double));
  myRightBoundary = (double*) malloc(boundarySize*sizeof(double)); 
  myLeftGhostCells = (double*) malloc(ghostSize*sizeof(double));
   myRightGhostCells = (double*) malloc(ghostSize*sizeof(double));

   int tid;
#pragma omp parallel private(tid) num_threads(numThreads)
      {
        tid = omp_get_thread_num();

#ifdef VERBOSE
	if(VERBOSE >=0)
	  cout << "Rank " << procID << " thread ID " << omp_get_thread_num() << " tid "  << tid << " initializing" << endl;
#endif


#pragma omp master 
	startTime_Init = omp_get_wtime();
	initializeExperiment(tid, numprocs);
#pragma omp master
	endTime_Init = omp_get_wtime();
      }

      MPI_Barrier(MPI_COMM_WORLD);

#pragma omp parallel private(tid)
      {
        int tid = omp_get_thread_num();
      /* BEGIN MAIN EXPERIMENTATION  */
      startTime_Experiment = 0.0;
      endTime_Experiment = 0.0;

#ifdef VERBOSE
	if(VERBOSE >=0)
	  cout << "Rank " << procID << " thread ID " << omp_get_thread_num() << " tid "  << tid << " initializing" << endl;
#endif

  #pragma omp barrier
      startTime_Experiment = omp_get_wtime(); // TODO: should really make this timer an inlined function 
      double* resMat;
      // resMat =
      runExperiment(tid, resultMatrix);
    
#pragma omp master
      {
	  endTime_Experiment = MPI_Wtime();
      }
      //MPI_Barrier(MPI_COMM_WORLD);    

#ifdef SHOWOUTPUT
      // printResults(resultMatrix, procID);
      printResults(resMat, procID);
#endif
#ifdef SHOWPERFVIZ
 printPerfViz(procID); 
#endif 

 #pragma omp master
 {
   //    endTime_Experiment = omp_get_wtime(); 
   if (procID == 0) 
     {
       endTime_Experiment = MPI_Wtime(); 
       cout << "Time for experiment's trial = " << endTime_Experiment - startTime_Experiment << endl;
     }
 }

 } // end omp parallel
    
      experimentCleanUp();

//  END MAIN EXPERIMENTATION	
// MPI_Barrier(MPI_COMM_WORLD);

// nodeCleanUp(); 

//exit(-1);
 rcProc = MPI_Finalize();
 cout << "Exit code is: " << rcProc << endl;

 return 0;
} // end MAIN
