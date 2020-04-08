#include make-bw.def

# macos
include make.def
# cab and surface

# surface / int xeon

# To use the compiler that compiles code below, you must first type 'use mvapich2-pgi-2.1' on the commandline.
#include make-surface.def

all: papiTest

papiTestf: papiTest.f
	$(FC) papiTest.f $(FCFLAGS) -o papiTestMPIOpenMPf

papiTest: papiTest.C
	$(CC) papiTest.C $(CCFLAGS) $(OPENMPFLAGS) -o papiTest -DPAPI_PROFILING

test: papiTest
	mpirun -n 2 ./papiTest 128

clean:
	rm -rf papiTestMPIOpenMPf papiTestMPIOpenMPc *.core
