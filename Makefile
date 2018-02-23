include make-bw.def

# macos
#include make-macos.def
# cab and surface

# surface / int xeon

# To use the compiler that compiles code below, you must first type 'use mvapich2-pgi-2.1' on the commandline.
#include make-surface.def

all: stencil_MPIopenaccf stencil_MPIopenmpf jacobi_MPIopenaccf jacobi_MPIopenmpf stencil_MPIopenaccf-ua jacobi_MPIopenaccf-ua stencil_MPIopenmpf-ua jacobi_MPIopenmpf-ua

stencil_MPIf: stencil.f
	$(FC) stencil.f $(FCFLAGS) -o stencil_MPIf

stencil_MPIopenaccf: stencil.f
	$(FC) stencil.f $(FCFLAGS) $(OPENACCFLAGS) -o stencil_MPIopenaccf

stencil_MPIopenaccf-ua: stencil.f
	$(FC) stencil.f $(FCFLAGS) $(OPENACCFLAGS) -DUSE_ALLOCATABLE -o stencil_MPIopenaccf-ua

stencil_MPIopenmpf: stencil.f
	$(FC) stencil.f $(FCFLAGS) $(OPENMPFLAGS) -o stencil_MPIopenmpf

stencil_MPIopenmpf-ua: stencil.f
	$(FC) stencil.f $(FCFLAGS) $(OPENMPFLAGS) -DUSE_ALLOCATABLE -o stencil_MPIopenmpf-ua

stencil_MPI: stencil.c
	 $(CC) stencil.c $(CCFLAGS) -o stencil_MPI

stencil_MPIopenacc: stencil.c
	$(CC) stencil.c $(CCFLAGS) $(OPENACCFLAGS) -o stencil_MPIopenacc

stencil_MPIopenmp: stencil.c
	$(CC) stencil.c $(CCFLAGS) $(OPENMPFLAGS) -o stencil_MPIopenmp

jacobi_MPIf: jacobi.f
	$(FC) jacobi.f $(FCFLAGS) -o jacobi_MPIf

jacobi_MPIopenaccf: jacobi.f
	$(FC) jacobi.f $(FCFLAGS) $(OPENACCFLAGS) -o jacobi_MPIopenaccf

jacobi_MPIopenaccf-ua: jacobi.f
	$(FC) jacobi.f $(FCFLAGS) $(OPENACCFLAGS) -DUSE_ALLOCATABLE -o jacobi_MPIopenaccf-ua

jacobi_MPIopenmpf: jacobi.f
	$(FC) jacobi.f $(FCFLAGS) $(OPENMPFLAGS) -o jacobi_MPIopenmpf

jacobi_MPIopenmpf-ua: jacobi.f
	$(FC) jacobi.f $(FCFLAGS) $(OPENMPFLAGS) -DUSE_ALLOCATABLE -o jacobi_MPIopenmpf-ua

jacobi_MPIopenmpf-ua_withod: jacobi.f
	$(FC) jacobi.f $(FCFLAGS) $(OPENMPFLAGS) -DUSE_ALLOCATABLE -o jacobi_MPIopenmpf-ua_withod

clean:
	rm -rf stencil_MPIopenaccf jacobi_MPIopenaccf stencil_MPIopenaccf-ua jacobi_MPIopenaccf-ua stencil_MPIopenmpf jacobi_MPIopenmpf stencil_MPIopenmpf-ua jacobi_MPIopenmpf-ua  stencil_MPIf jacobi_MPIf stencil_MPIf-ua jacobi_MPIf-ua *.core