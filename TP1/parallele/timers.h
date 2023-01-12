#ifndef TIMERS_H
#define TIMERS_H

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
		printf("\tTime to %s %3.1f ms\n", #txt, elapsedTime); \
	}

#endif