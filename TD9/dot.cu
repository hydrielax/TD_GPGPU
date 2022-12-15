/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

 #include <stdio.h>
 #define imin(a,b) (a<b?a:b)
 
 const int N = 1024 * 1024;
 const int threadsPerBlock = 256;
 const int blocksPerGrid = (N+threadsPerBlock-1) / threadsPerBlock ;
 
 __global__
 void dot( double *a, double *b, double *c ) {
     // TODO
 }
 
 
 int main( void ) {
     double   *h_a, *h_b, *partial_c;
     double   *d_a, *d_b, *d_partial_c;
     double   res;
 
     // allocate memory on the cpu side
     h_a = (double*)malloc( N*sizeof(double) );
     h_b = (double*)malloc( N*sizeof(double) );
     partial_c = (double*)malloc( blocksPerGrid*sizeof(double) );
 
     // allocate the memory on the GPU
     cudaMalloc( (void**)&d_a, N*sizeof(double) );
     cudaMalloc( (void**)&d_b, N*sizeof(double) );
     cudaMalloc( (void**)&d_partial_c, blocksPerGrid*sizeof(double) );
 
     // fill in the host memory with data
     for (int i = 0; i < N; i++) {
         h_a[i] = i;
         h_b[i] = i*2;
     }
 
     // copy the arrays 'a' and 'b' to the GPU
     cudaMemcpy( d_a, h_a, N*sizeof(double), cudaMemcpyHostToDevice );
     cudaMemcpy( d_b, h_b, N*sizeof(double), cudaMemcpyHostToDevice ); 
 
     cudaEvent_t start, stop;
     cudaEventCreate( &start );
     cudaEventCreate( &stop );
     cudaEventRecord( start, 0 );
 
    // call dot
 
     cudaEventRecord( stop, 0 );
     cudaEventSynchronize( stop );
     float elapsedTime;
     cudaEventElapsedTime( &elapsedTime, start, stop );
     printf("Total time for %d elements was %f ms\n", N, elapsedTime );
 
     // copy the array 'c' back from the GPU to the CPU
     cudaMemcpy( partial_c, d_partial_c, blocksPerGrid*sizeof(double), cudaMemcpyDeviceToHost );
 
     // finish up on the CPU side
     res = 0;
     for (int i=0; i<blocksPerGrid; i++) {
         res += partial_c[i];
     }
 
     #define sum_squares(x)  (x*(x+1)*(2*x+1)/6)
     printf( "Does GPU value %.6g = %.6g?\n", res,
              2 * sum_squares( (double)(N - 1) ) );
 
     // free memory on the gpu side
     cudaFree( d_a );
     cudaFree( d_b );
     cudaFree( d_partial_c );
 
     // free memory on the cpu side
     free( h_a );
     free( h_b );
     free( partial_c );
 }
 