#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  int device;
  
  for (device = 0; device < deviceCount; ++device) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Device n°%d\n", device);
    printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Name: %s\n", deviceProp.name);
    printf("Multiprocessor Count: %d\n", deviceProp.multiProcessorCount);
    printf("Clock Rate (kHz): %d\n", deviceProp.clockRate);
  }

  return 0;
}