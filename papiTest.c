//This is a pthreads code with daxpy computation that is used to illustrate PAPI usage in an application program and to test PAPI on a given platform.

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
// Libraries for timers.
#include <cstdlib>
#include <sys/time.h>
#include <sys/resource.h>

//#define PAPI_PROFILING
// Library for hardware profiling.
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

//hardware-specific
int numCores = 16;

// using high-level PAPI interface
void* daxpy(void* tid)
{
// pthread binding management - system-specific - Not used. 
  int s, cNum;
  //cpu_set_t cpuset;
  pthread_t thread;
  thread = pthread_self();

  double wtime;
  long t = *((int*) tid);
  int myTid = (int) t;
 
  // CPU_ZERO(&cpuset);
  //CPU_SET(myTid, &cpuset);
  //s = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
  //if (s != 0)
  //  printf("error: with setting pthread affinity. Error code = %d\n", s);

//Initializing variables and counters related to PAPI. 

#ifdef PAPI_PROFILING
  unsigned long int papi_tid;

  int num_hwcntrs =2;
  int Events[2] = {PAPI_L1_DCM, PAPI_L2_DCM}; // Obtain L1 and L2 data cache misses.
  long_long values[2];

  if ((papi_tid = PAPI_thread_id()) == (unsigned long int)-1)
  {
    printf("error with assigning the papi thread ID\n");
    exit(1);
  }
  printf("Initial PAPI thread id is:\t%lu\n", papi_tid);
  if (PAPI_start_counters(Events, num_hwcntrs) != PAPI_OK)
    printf("Error with creating event set \n");
#endif
//  Do computation using block strategy with static scheduling
 wtime = - get_wtime();
for(int i = myTid ; i < arrSize; i+=numThreads)
    c[i] += a[i]*b[i];

 wtime += get_wtime();

#ifdef PAPI_PROFILING
// end PAPI perf counters
if (PAPI_read_counters(values, num_hwcntrs) != PAPI_OK)  printf("Error with reading counters!\n");
if (PAPI_stop_counters(values, num_hwcntrs) != PAPI_OK)   printf("Error with stopping counters!\n");
printf("Total Prof: L1 data cache misses, reported from papi tid %lld:\t%lld\n", papi_tid, values[0]);
printf("Total Prof: data TLB misses, reported from papi tid %lld:\t%lld\n", papi_tid, values[1]);
printf("Total Prof: L1 data cache hit rate, reported from papi tid %lld:\t%f\n", papi_tid, 1.0 - 1.0*(values[0])/(1.0*(values[1] + values[2])));
#endif

if(myTid == 0)
    printf("Time(secs) to run daxpy, reported from threadID %d:\t%f\n", myTid, wtime);
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
  double daxpyCheckSum = 0.0; // used for correctness check. 

#ifdef PAPI_PROFILING
  PAPI_library_init(PAPI_VER_CURRENT); // This function call should be in the main function.
  PAPI_thread_init(pthread_self);  // The library needs to know which call to make (to identify a thread).
#endif

//initialize pthread
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

//initialize barrier and mutex
  pthread_barrier_init(&myBarrier, NULL, numThreads);
  pthread_mutex_init(&myMutex, NULL);

  void* thread_status;
  int* thdsArr = (int*) malloc(sizeof(int)*numThreads);

  printf("Number of Threads:\t%d\n", numThreads);
  printf("Problem size:\t%d\n", arrSize);

  //Run the threaded computation region performing daxpy computation.

  for(int threadID = 0; threadID < numThreads; threadID++)
  {
    thdsArr[threadID] = threadID; // Used to avoid duplicate threadID.
    pthread_create(&callThd[threadID], &attr, daxpy, &thdsArr[threadID]);
  }
  pthread_attr_destroy(&attr);
  for(int i=0;i<numThreads;i++)
    pthread_join(callThd[i], &thread_status);

// correctness checks for application

  for (int i = 0 ; i < arrSize; i++)
    daxpyCheckSum += c[i];

  printf("Correctness check for daxpy application code: checkSum = %f \n", daxpyCheckSum);
}


// thread-safe timer functions for obtaining timings. 
double get_wtime(void)
{
  struct rusage ruse;
  // getrusage(RUSAGE_SELF, &ruse);
//see documentation , need to use rusage thread
  getrusage(RUSAGE_THREAD, &ruse);
  return( (double)(ruse.ru_utime.tv_sec+ruse.ru_utime.tv_usec / 1000000.0) );
}

