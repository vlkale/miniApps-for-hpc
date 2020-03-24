! Description: 2D Jacobi relaxation example code used for experimenting with over-decomposition strategies along with GPU-ization.

! This code file does not use functions for each part of the jacobi computation, i.e., it is straightline, or non-functionized.
!
! Author: Vivek Kale
!
! Last Edited: 3/20/2020


! TODO: check why loop labeled seq gets warning message of not being parallielizable

      MODULE util
      contains

      subroutine printSample(arr, dimRow, dimCol, sampleSizeRow, &
      sampleSizeCol)

      real, pointer :: arr(:,:) ! assigning to m, m
      integer :: dimRow
      integer :: dimCol
      integer :: sampleSizeRow
      integer :: sampleSizeCol
      integer :: krow
      integer :: kcol

      integer :: rank  = 0

      integer i
      integer j

      krow = dimRow/sampleSizeRow
      kcol = dimCol/sampleSizeCol

      write(*,*) 'rank', rank, ' krow', krow , 'dimRow', dimRow
      do i = 1, dimRow, krow
         write(*,*) ''
         do j = 1, dimCol, kcol
            write(*,"(f8.3)", advance="no") arr(i,j)
         end do
      end do

      write(*,*) ''
      end subroutine printSample
      end MODULE util

      program stencil
      use util

      use iso_c_binding, only: c_int, c_double, c_ptr, c_null_ptr

      use omp_lib

      implicit none

      ! Include the mpif.h file corresponding to the compiler that compiles this code.
      include 'mpif.h'

      ! Variables for MPI

      integer ( kind = 4 ) :: rank
      integer ( kind = 4 ) :: size
      integer ( kind = 4 ) :: ierror
      integer :: numRequests
      INTEGER, allocatable :: ARRAY_OF_REQUESTS(:)
      integer :: requestCounter = 1

      ! Variables for openMP
      INTEGER NTHDS
      INTEGER MAX_THREADS
      INTEGER TID
      PARAMETER(MAX_THREADS= 16)

      !  application specific variables

      !    Variables for parameters of application
      INTEGER TIMESTEP
      integer, PARAMETER :: probSize = 512
      integer :: N ! x dimension  of mesh  -TODO: change variable name to Nx
      integer :: Ny ! y dimension of mesh - TODO:
      integer, PARAMETER :: NUM_TIMESTEPS = 100
      integer, parameter :: NUM_FLOP = 10

      integer :: nSteps
      integer :: FLOP
      integer ht ! height for the halo and data array

      !   data arrays for 2D mesh
      real, allocatable :: a(:,:)
      real, allocatable :: b(:,:)
      real, allocatable :: temp(:,:) ! used for pointer swapping (not used right now).

      ! Buffers and variables for border exchange
      real, allocatable :: topBoundary(:)
      real, allocatable :: bottomBoundary(:)
      real, allocatable :: topHalo(:)
      real, allocatable :: bottomHalo(:)
      integer :: msgSize

      ! used to check correctness
      integer :: xsum
      double precision checkSum
      integer r

      !     loop iteration variables
      INTEGER :: i, j

      INTEGER :: k ! Used for loop that allows varying computation per step right now.

      !  Variables for timing
      double precision :: startTime, endTime, totalTime

      integer num_dev
      integer dev_id
      integer :: numRanksPerNode = 2

      ! Variables  for program Input
      character*80 arg
      integer (kind = 4) num_args

      num_args = command_argument_count() ! TODO: Fix this to ensure the function is returning expected arguments.
      if (num_args .gt. 3) then
          call getarg(1, arg)
          read(arg,*) N
          call getarg(2, arg)
          read(arg,*) Ny
          call getarg(3, arg)
          read(arg,*) FLOP
          call getarg(4, arg)
          read(arg,*) nSteps
       else
          N = probSize
          Ny = probSize
          FLOP = NUM_FLOP
          nSteps = NUM_TIMESTEPS
      end if

      if (rank == 0) then
         print *, 'The problem size input is', N, 'by', Ny
         call flush(6)
         call flush(0)
      end if

      call MPI_INIT(ierror)
      call MPI_COMM_SIZE(MPI_COMM_WORLD, size, ierror)
      call MPI_COMM_RANK(MPI_COMM_WORLD, rank, ierror)

      allocate(ARRAY_OF_REQUESTS(size))

      do i = 1, size
         ARRAY_OF_REQUESTS(i) = MPI_REQUEST_NULL
      end do

      ARRAY_OF_REQUESTS = MPI_REQUEST_NULL
      ht = 2 + N/size

      ! TODO: change the below from N to Ny

      print *,  "stencil.f: Allocating arrays."
      call flush(6)
      call flush(0)
      allocate(a(ht, N+2))
      allocate(b(ht, N+2))
      allocate(topBoundary(N+2))
      allocate(bottomBoundary(N+2))
      allocate(topHalo(N+2))  !  need this for GPU to copy out
      allocate(bottomHalo(N+2))  ! need this for GPU to copy out

! #ifdef HAVE_OPENACC

  !    call acc_init(acc_get_device_type())

   !   num_dev = acc_get_num_devices(acc_get_device_type())
   !   dev_id = mod(rank, num_dev)
      !  assign GPU to one MPI process
  !    print *, 'MPI rank ', rank, ' assigned to GPU ' , dev_id
  !    call flush(6)
  !    call flush(0)
  !    call acc_set_device_num(dev_id, acc_get_device_type())
  !    nthds = 2880 !TODO: find this number programmatically using a function of openacc runtime

   !    if (rank == 0) then
   !       print *, 'stencil.f: Initialized OpenACC.'
   !       call flush(6)
   !       call flush(0)
   !    end if

! elif HAVE_OPENMP

!$OMP PARALLEL
      nthds = omp_get_num_threads()
      tid = omp_get_thread_num()
      print *, 'Thread ' , omp_get_thread_num() + 1 , ' of ', nthds , ' active.'
      call flush(6)
      call flush(0)
!$OMP END PARALLEL
! endif
      ! set size of MPI isend/irecv/waitall messages
      msgSize = N+2

      ! Initial values assigned to data array / matrix , where all values of the cells of data array / matrix are 50.0.
      ! Realistically, we should set the array cells of the below arrays to initial values stored in an input file, but we leave this out now.

      a = 50.0

      ! Assign initial values to buffers
      topBoundary = 50.0
      bottomBoundary = 50.0
      topHalo = 50.0
      bottomHalo = 50.0

      call MPI_BARRIER(MPI_COMM_WORLD, ierror)

      startTime = MPI_Wtime()


      do timestep = 1, nSteps
         requestCounter = 1
         if (rank .ne. 0) then
            call MPI_Irecv(topHalo, msgSize, MPI_REAL, rank-1, 0 , &
            MPI_COMM_WORLD, array_of_requests(requestCounter), ierror)
            requestCounter = requestCounter + 1
            call MPI_Isend(topBoundary, msgSize, MPI_REAL, rank-1, 0 , &
            MPI_COMM_WORLD, array_of_requests(requestCounter), ierror)
            requestCounter= requestCounter + 1
         end if
!     sending Boundary , so need to copy from data array to topBoundary
!     and bottom Boundary
         if (rank .ne. size-1) then
            call MPI_Irecv(bottomHalo, msgSize, MPI_REAL, rank + 1, 0 , &
            MPI_COMM_WORLD, array_of_requests(requestCounter), ierror)
            requestCounter= requestCounter + 1
            call MPI_Isend(bottomBoundary, msgSize, MPI_REAL, rank + 1, 0, MPI_COMM_WORLD, &
            array_of_requests(requestCounter), ierror)
            requestCounter= requestCounter + 1
         end if

         call MPI_Waitall(requestCounter - 1, array_of_requests, &
         MPI_STATUSES_IGNORE, ierror)

! TODO: organize this to work for both OpenACC and OpenMP


! Note: when using MPI, each MPI process invokes the below.
! Consider different schemes here, including GPU direct and copying the entire data array into the matrix

! Note : we  dont need create(k) since we have acc loop seq 

           if (rank == 0) then
           !$OMP parallel
           !$OMP do
              do i = 2, N+1
                 topHalo(i) = a(2, i)
              end do
           !$OMP end do
           !$OMP end parallel
           end if

           if (rank == (size-1)) then
           !$OMP parallel
           !$OMP do schedule(static, 4)
              do i = 2, N+1
                 bottomHalo(i) = a(ht-1, i)
              end do
           !$OMP end do
           !$OMP end parallel

           end if

           !$OMP parallel
           !$OMP do schedule(static, 4)
           do i = 1, N+2
              a(1,i) = topHalo(i)
           end do
           !$OMP end do
           !$OMP end parallel

           !$OMP parallel
           !$OMP do
           do i = 1, N+2
              a(ht,i) = bottomHalo(i)
           end do
           !$OMP end do
           !$OMP end parallel

!$OMP PARALLEL
!$OMP DO SCHEDULE(STATIC, 4)
           do i = 2, ht-1
              a(i, 1) = a(i, 2)
              a(i, N+2) = a(i, N+1)
           end do
!$OMP END DO
!$OMP END PARALLEL

           ! reset boundaries
           if(rank == 0) then
              a(1,2) = 0.0
              a(2,1) = 0.0
           end if
           if (rank == (size - 1)) then
              a(ht , n+1) = 100.0
              a(ht -1, n+2) = 100.0
           end if


! We dont do loop interchange for now.

do k = 1, FLOP
!$OMP PARALLEL private(j)
!$OMP DO SCHEDULE(STATIC, 4) collapse(2)
   do i = 2, ht-1
         do j = 2, N+1
                 b(i,j)= 0.2*(a(i,j) + a(i-1,j) + &
                 a(i+1,j) + a(i,j-1) + a(i,j+1))
              end do
           end do
!$OMP END DO
!$OMP END PARALLEL

!$OMP PARALLEL private(j)
!$OMP DO SCHEDULE(STATIC, 4) collapse(2)
           do i = 2, ht-1
              do j = 2, n+1
                 a(i,j) = b(i,j)
              end do
           end do
!$OMP END DO
!$OMP END PARALLEL
        end do

!$OMP PARALLEL
!$OMP DO SCHEDULE(STATIC, 4)
           do i = 1, N+2
              topBoundary(i) = a(2, i)
           end do
!$OMP END DO
!$OMP END PARALLEL

!$OMP PARALLEL
!$OMP DO SCHEDULE(STATIC, 4)
           do i = 1, N+2
              bottomBoundary(i) = a(ht-1, i)
           end do
!$OMP END DO
!$OMP END PARALLEL

      call MPI_BARRIER (MPI_COMM_WORLD, ierror)
      end do ! end timestep


      endTime = MPI_Wtime()

      if (rank == (N/3)/(ht-2)) then
         r = mod((N/3), ht-2 )
         write(*, *) N/3,':', a(r, N/3), a(r,2*N/3)
      end if
      if (rank == (2*N/3)/(ht-2)) then
         r = mod((2*N/3), ht-2)
         write(*, *) 2*N/3,':', a(r, N/3), a(r,2*N/3)
      end if

      if (rank == 0) then
         print *, 'That took ', endTime - startTime , ' seconds.'
         call flush(6)
         call flush(0)
         Open( unit=10, file="outfile-meshcomp-surface.dat", access="sequential", form="formatted", &
         status="unknown", position="append")
         ! TODO: may need to find a unsigned long for problem size

         write(10, '(A, I7, I7, I7, A, A, I7, I7, A, f8.3, f8.3, f8.3)') '\t stn', N*Ny, FLOP, nSteps, '\t ifp', '\t omp', size, &
                nthds, '\t ptr', endTime - startTime, 0.0, 0.0

         close(10)

      end if
      call MPI_Finalize(ierror)
      stop

      end program stencil
