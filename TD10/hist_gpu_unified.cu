
#include <stdio.h>
#include <string.h>
#include "text.h"
#include <iostream>

#define NB_ASCII_CHAR 128

const int threadsPerBlock = 256;

__global__ void kernel(char *buffer, int size, u_int *histo) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    int stride = blockDim.x * gridDim.x;

    while (i < size) {
        atomicAdd(&(histo[buffer[i]]), 1);
        i += stride;
    }
}

int main(void)
{
    int len = strlen(h_str);
    printf("len:%d\n", len);
    int size = len * sizeof(char);

    // GPU COMPUTATION

    // create variables
    char *d_str;
    u_int *histo;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // allocate & copy
    cudaMalloc( (void**) &d_str, size );
    cudaMemcpy(d_str, h_str, size, cudaMemcpyHostToDevice);

    // share memory
    cudaMallocManaged(&histo, NB_ASCII_CHAR * sizeof(u_int));

    // nb blocks
    const int nb_blocks = (len + threadsPerBlock - 1) / threadsPerBlock;

    // call kernel
    kernel<<<nb_blocks, threadsPerBlock>>>(d_str, len, histo);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Total time for %d elements was %f ms\n", len, elapsedTime);

    // PRINT
    for (int bean = 0; bean < NB_ASCII_CHAR; bean++)
    {
        std::cout << (char)bean << " : " << histo[bean] << std::endl;
    }

    cudaFree(histo);
    cudaFree(d_str);

    return 0;
}
