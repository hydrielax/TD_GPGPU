#include <stdio.h>
#include <math.h>

#define N (2048 * 2048)
#define THREADS_PER_BLOCK 512
#define NB_STREAMS 4
#define SEGMENT_SIZE (1024 * 128)

#define CREATE_CUDAEVENT     \
	cudaEvent_t start, stop; \
	cudaEventCreate(&start); \
	cudaEventCreate(&stop);

#define START_CUDAEVENT cudaEventRecord(start, 0);
#define STOP_AND_PRINT_CUDAEVENT(txt)                       \
	cudaEventRecord(stop, 0);                               \
	cudaEventSynchronize(stop);                             \
	{                                                       \
		float elapsedTime;                                  \
		cudaEventElapsedTime(&elapsedTime, start, stop);    \
		printf("Time to %s %3.1f ms\n", #txt, elapsedTime); \
	}

__global__ void vector_add(int *a, int *b, int *c)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	c[index] = a[index] + b[index];
}

void stream_addition(int *a, int *b, int *c)
{
	CREATE_CUDAEVENT
	int *d_a[NB_STREAMS];
	int *d_b[NB_STREAMS];
	int *d_c[NB_STREAMS];
	cudaStream_t streams[NB_STREAMS];
	// int size = N * sizeof(int) / NB_STREAMS;

	for (int i = 0 ; i < NB_STREAMS ; i++) {
		cudaStreamCreate(&(streams[i]));
		cudaMalloc((void **)&(d_a[i]), SEGMENT_SIZE * sizeof(int));
		cudaMalloc((void **)&(d_b[i]), SEGMENT_SIZE * sizeof(int));
		cudaMalloc((void **)&(d_c[i]), SEGMENT_SIZE * sizeof(int));
	}

	START_CUDAEVENT
	
	for (int i = 0 ; i < N ; i += SEGMENT_SIZE * NB_STREAMS) {

		for (int s = 0 ; s < NB_STREAMS ; s++) {
			cudaMemcpyAsync(
				d_a[s],
				a + i + SEGMENT_SIZE * s,
				SEGMENT_SIZE * sizeof(int),
				cudaMemcpyHostToDevice,
				streams[s]
			);
			cudaMemcpyAsync(
				d_b[s],
				b + i + SEGMENT_SIZE * s,
				SEGMENT_SIZE * sizeof(int),
				cudaMemcpyHostToDevice,
				streams[s]
			);
		}

		for (int s = 0 ; s < NB_STREAMS ; s++) {
			vector_add<<<SEGMENT_SIZE / THREADS_PER_BLOCK, THREADS_PER_BLOCK, 0, streams[s]>>>(d_a[s], d_b[s], d_c[s]);
		}

		for (int s = 0 ; s < NB_STREAMS ; s++) {
			cudaMemcpyAsync(
				c + i + SEGMENT_SIZE * s,
				d_c[s],
				SEGMENT_SIZE * sizeof(int),
				cudaMemcpyDeviceToHost,
				streams[s]
			);
		}
		
	}
	
	for (int s = 0 ; s < NB_STREAMS ; s++) {
		cudaStreamDestroy(streams[s]);
	}
	STOP_AND_PRINT_CUDAEVENT(computation)
	
	/* clean up */
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

void addition(int *a, int *b, int *c)
{
	CREATE_CUDAEVENT
	int size = N * sizeof(int);
	int *d_a, *d_b, *d_c;

	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	START_CUDAEVENT
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
	STOP_AND_PRINT_CUDAEVENT(memcpy h2d)

	START_CUDAEVENT
	vector_add<<<(N + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
	STOP_AND_PRINT_CUDAEVENT(computation)

	START_CUDAEVENT
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
	STOP_AND_PRINT_CUDAEVENT(memcpy d2h)

	/* clean up */
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

int main()
{
	int *a, *b, *c;
	int size = N * sizeof(int);

	/* Pinned memory */
	cudaHostAlloc((void **)&a, size, cudaHostAllocDefault);
	cudaHostAlloc((void **)&b, size, cudaHostAllocDefault);
	cudaHostAlloc((void **)&c, size, cudaHostAllocDefault);

	for (int i = 0; i < N; i++)
	{
		a[i] = b[i] = i;
		c[i] = 0;
	}

	printf("Addition with default stream\n");
	addition(a, b, c);

	printf("c[0] = %d\n", c[0]);
	printf("c[%d] = %d\n", N - 1, c[N - 1]);

	// Reinitialisation
	for (int i = 0; i < N; i++)
	{
		c[i] = 0;
	}

	/*< Add a call to your function with streams >*/
	printf("Addition with streams\n");
	stream_addition(a, b, c);

	printf("c[0] = %d\n", c[0]);
	printf("c[%d] = %d\n", N - 1, c[N - 1]);

	/* clean up */
	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(c);

	return 0;
}
