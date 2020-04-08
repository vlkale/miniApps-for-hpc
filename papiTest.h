
//include files

#include <pthread.h>


#include <stdio.h>
#include <stdlib.h>
#include <errno.h>


// Libraries for Performance Measurement
// for timers
#include <cstdlib>
#include <sys/time.h>
#include <sys/resource.h>

// for lower-level Perf. Profiling

//#define PAPI_PROFILING

#ifdef PAPI_PROFILING
#include <papi.h>
#endif

double get_wtime(void);

#define MAXTHRDS 16
//#define _GNU_SOURCE

pthread_barrier_t myBarrier;
pthread_mutex_t myMutex;
pthread_attr_t attr;
pthread_t callThd[MAXTHRDS];

int numThreads;

int arrSize;

// app data structures for dot product
double* a;
double* b;
double* c;

// for scalar product
double* d;

// app data structures for spMV
double* x;
double* x_next;
double** solverMatrix;

// app data structures for stencil
double** grid1;
double** grid2;

double globalProd = 0.0;
double scalarProdCheckSum = 0.0;

//hardware-specific
//TODO: need to find a way to obtain this from system.  Make sure this works for any number of specified threads.
int numCores = 16;

//strategy specification
// app specification
// TO DO: avoid repeated code in functions


// version with daxpy low-level profile

void* daxpy1(void* tid)
{
// pthread binding management - system-specific
  int s, cNum;
  cpu_set_t cpuset;
  pthread_t thread;
  thread = pthread_self();

  double startTime, endTime, totalTime;
  long t = *((int*) tid);
  int myTid = (int) t;

  CPU_ZERO(&cpuset);
  CPU_SET(myTid, &cpuset);
  s = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
  if (s != 0)
    printf("error: with setting pthread affinity. Error code = %d\n", s);
//Initializing PAPI stuff

#ifdef PAPI_PROFILING
  if(myTid == 0)
  {
    PAPI_library_init(PAPI_VER_CURRENT); // should be in the main program
    PAPI_thread_init(pthread_self);  // the library needs to know which call to make.
  }



// PAPI perf counters
  unsigned long int papi_tid;
  int Events[3]  = {PAPI_L2_DCM, PAPI_SR_INS, PAPI_LD_INS};
  int retval;
  int EventSet = PAPI_NULL;
  long_long values[3];
  if ((papi_tid = PAPI_thread_id()) == (unsigned long int)-1)
  {
    printf("error with assigning the papi thread ID\n");
    exit(1);
  }
//   printf("Initial PAPI thread id is:\t%lu\n", papi_tid);
  if(PAPI_create_eventset(&EventSet) != PAPI_OK)
    printf("Error with creating event set \n");
  if(PAPI_add_event(EventSet, Events[0]) != PAPI_OK)
    printf("Error with creating event set 0 \n");
  if(PAPI_add_event(EventSet, Events[1]) != PAPI_OK)
    printf("Error with creating event set 1 \n");
  if(PAPI_add_event(EventSet, Events[2]) != PAPI_OK)
    printf("Error with creating event set 2 \n");
#endif

// do computation
for(int i = myTid ; i < arrSize; i+=numThreads)
    c[i] += a[i]*b[i];

#ifdef PAPI_PROFILING
// end PAPI perf counters

if (PAPI_read(EventSet, values) != PAPI_OK)
  printf("Error with reading values from EventSet\n");

if (PAPI_stop(EventSet, values) != PAPI_OK)
  printf("Error with stopping counters!\n");

printf("Total Prof: L1 data cache misses, reported from papi tid %lld:\t%lld\n", papi_tid, values[0]);
printf("Total Prof: data TLB misses, reported from papi tid %lld:\t%lld\n", papi_tid, values[1]);
printf("Total Prof: L1 data cache hit rate, reported from papi tid %lld:\t%f\n", papi_tid, 1.0 - 1.0*(values[0])/(1.0*(values[1] + values[2])));
#endif

if(myTid == 0)
    printf("Time(secs) to run daxpy1, reported from threadID %d:\t%f\n", myTid, totalTime);
}

// using high-level PAPI interface
void* daxpy2(void* tid)
{
// pthread binding management - system-specific
  int s, cNum;
  cpu_set_t cpuset;
  pthread_t thread;
  thread = pthread_self();

  double startTime, endTime, totalTime;
  long t = *((int*) tid);
  int myTid = (int) t;

  CPU_ZERO(&cpuset);
  CPU_SET(myTid, &cpuset);
  s = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
  if (s != 0)
    printf("error: with setting pthread affinity. Error code = %d\n", s);
//Initializing PAPI stuff

#ifdef PAPI_PROFILING
  if(myTid == 0)
  {
    PAPI_library_init(PAPI_VER_CURRENT); // should be in the main program
    PAPI_thread_init(pthread_self);  // the library needs to know which call to make.
  }

// PAPI perf counters  (using PAPI high-level interface
  unsigned long int papi_tid;
  int num_hwcntrs = 3;                                                          int Events[3]  = {PAPI_L2_DCM, PAPI_SR_INS, PAPI_LD_INS};
  long_long values[3];

  if ((papi_tid = PAPI_thread_id()) == (unsigned long int)-1)
  {
    printf("error with assigning the papi thread ID\n");
    exit(1);
  }
  printf("Initial PAPI thread id is:\t%lu\n", papi_tid);
  if (PAPI_start_counters(Events, num_hwcntrs) != PAPI_OK)                            printf("Error with creating event set \n");
#endif
//  do simple computation using block strategy for static scheduling
for(int i = myTid ; i < arrSize; i+=numThreads)
    c[i] += a[i]*b[i];

#ifdef PAPI_PROFILING
// end PAPI perf counters
if (PAPI_read_counters(values, num_hwcntrs) != PAPI_OK)                       printf("Error with reading counters!\n");
if(PAPI_stop_counters(values, num_hwcntrs) != PAPI_OK)                           printf("Error with stopping counters!\n");


//if(papi_tid == 0)
printf("Total Prof: L1 data cache misses, reported from papi tid %lld:\t%lld\n", papi_tid, values[0]);
printf("Total Prof: data TLB misses, reported from papi tid %lld:\t%lld\n", papi_tid, values[1]);
printf("Total Prof: L1 data cache hit rate, reported from papi tid %lld:\t%f\n", papi_tid, 1.0 - 1.0*(values[0])/(1.0*(values[1] + values[2])));
#endif

if(myTid == 0)
    printf("Time(secs) to run daxpy, reported from threadID %d:\t%f\n", myTid, totalTime);
}


// version with low-level interface

//dot Product application
int main(int argc, char* argv[] )
{
  if(argc >= 2)
  {
    arrSize = atoi(argv[1]);
    numThreads = atoi(argv[2]);
  }
  else
  {
    printf("Usage: a.out [problem size] [<number of threads>]\n");
    exit(1); // can be system specific
  }
//initialize app data structure
  a = (double*) malloc(sizeof(double)*arrSize);
  b = (double*) malloc(sizeof(double)*arrSize);
  c = (double*) malloc(sizeof(double)*arrSize);

  // needed for stencil
  for(int i = 0; i < arrSize; i++)
  {
    a[i] = 2.0;
    b[i] = 3.0;
    c[i] = 3.0 ;
    srand(time(NULL));
  }
  globalProd = 0.0;

//initialize pthread
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

//initialize barrier and mutex
  pthread_barrier_init(&myBarrier, NULL, numThreads);
  pthread_mutex_init(&myMutex, NULL);

  void* thread_status;
  int* thdsArr = (int*) malloc(sizeof(int)*numThreads);

//print parallelism characteristics
  printf("Number of Threads per Process:\t%d\n", numThreads);

//print application characteristics or strategy
  printf("Problem size:\t%d\n", arrSize);


  printf("Alg. Strategy: Block\n");


//run the daxpy  (begin threaded computation region for the dot Product application)
  for(int threadID = 0; threadID < numThreads; threadID++)
  {
    thdsArr[threadID] = threadID; // add this to ensure we don't get duplicated threadID
    pthread_create(&callThd[threadID], &attr, daxpy1, &thdsArr[threadID]);
  }
  pthread_attr_destroy(&attr);

// TODO: check if this is correct
  for(int i=0;i<numThreads;i++)
    pthread_join(callThd[i], &thread_status);

// correctness checks for application
  printf("correctness check for Dot Product application: global prod = %f \n", globalProd);

  for (int i = 0 ; i < arrSize ; i++)
    scalarProdCheckSum += x[i];

  printf("correctness check for Scalar Product application: checkSum = %f \n", scalarProdCheckSum);
//TODO: check to see why we can't get the right result for cache miss rate below

}


// functions for obtaining results
double get_wtime(void)
{
  struct rusage ruse;
//  getrusage(RUSAGE_SELF, &ruse);
//see documentation , need to use rusage thread
  getrusage(RUSAGE_THREAD, &ruse);
  return( (double)(ruse.ru_utime.tv_sec+ruse.ru_utime.tv_usec / 1000000.0) );
}

double get_profile(void)
{


}
