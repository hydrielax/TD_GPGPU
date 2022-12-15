#include <stdio.h>

/* experiment with N */
/* how large can it be? */
#define N (2047 * 2047)
#define THREADS_PER_BLOCK 512

__global__ void vector_add(int *a, int *b, int *c)
{
	/* insert code to calculate the index properly using blockIdx.x, blockDim.x, threadIdx.x */
	int index = threadIdx.x + blockDim.x * blockIdx.x;
	if (index < N)
	{
		c[index] = a[index] + b[index];
	}
}

int main()
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	/* record start time */
	cudaEventRecord(start, 0);

	int *a, *b, *c;
	// int *a, *b, *c;

	int size = N * sizeof(int);

	/* allocate space for device copies of a, b, c */
	cudaMallocManaged(&a, size);
	cudaMallocManaged(&b, size);
	cudaMallocManaged(&c, size);

	/* allocate space for host copies of a, b, c and setup input values */

	// a = (int *)malloc( size );
	// b = (int *)malloc( size );
	// c = (int *)malloc( size );

	for (int i = 0; i < N; i++)
	{
		a[i] = b[i] = i;
		// c[i] = 0;
	}

	/* copy inputs to device */
	/* fix the parameters needed to copy data to the device */
	// cudaMemcpy(a, a, size, cudaMemcpyHostToDevice);
	// cudaMemcpy(b, b, size, cudaMemcpyHostToDevice);

	/* take the number of blocks */
	int nb_blocks = ceil((float)N / THREADS_PER_BLOCK);

	/* launch the kernel on the GPU */
	/* insert the launch parameters to launch the kernel properly using blocks and threads */
	vector_add<<<nb_blocks, THREADS_PER_BLOCK>>>(a, b, c);

	cudaDeviceSynchronize();

	/* copy result back to host */
	/* fix the parameters needed to copy data back to the host */
	// cudaMemcpy(c, c, size, cudaMemcpyDeviceToHost);

	printf("c[%d] = %d\n", 0, c[0]);
	printf("c[%d] = %d\n", 5, c[5]);
	printf("c[%d] = %d\n", N - 1, c[N - 1]);

	/* clean up */

	// free(a);
	// free(b);
	// free(c);
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);

	/* record finish time */
	cudaEventRecord(stop, 0);
	/* wait GPU event */
	cudaEventSynchronize(stop);

	/* compute and print ellapsed time between start and stop */
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time to execute %3.1f ms\n", elapsedTime);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
} /* end main */
