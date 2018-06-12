# Description

This is a set of small codes representative of larger application programs run on supercomputers. The reason that these codes were developed was to experiment with performance optimizations done on larger application programs.

The code named jacobi.f is a code of Jacobi relaxation written in Fortran implemented in MPI+OpenACC that I wrote during my postdoctoral research position at University of Illinois at Urbana-Champaign to identify performance and compiler optimizations for an MPI+CUDA code for a plasma combustion simulation program that I was assigned to work on for the XPACC project. 

The code named ptytomo.C is a mini-app MPI+CUDA code compiled run and tested on NVIDIA Tesla P100 representative of a larger X-ray imaging program implemented with MPI+CUDA and run on NVIDIA Tesla P100. The mini-app code is used to guide improving performance, in particular compiler flag optimizations, loop transformations, and new half-precision CUDA functions available for the P100, of the larger application code involving a 2-D image reconstruction using ptychography followed by 3-D image reconstruction using tomography.

# Notes on Installing and Running


# TODO
