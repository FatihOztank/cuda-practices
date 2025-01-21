#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

void printGPUProp(const cudaDeviceProp &deviceProp)
{
    printf("Device name: %s\n", deviceProp.name);
    printf(" Device UUID: %lu\n", deviceProp.uuid);
    printf(" Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf(" Total Global Memory: %lu MB\n", deviceProp.totalGlobalMem / (1024 * 1024));
    printf(" Shared Memory per Block: %lu KB\n", deviceProp.sharedMemPerBlock / 1024);
    printf(" Registers per Block: %d\n", deviceProp.regsPerBlock);
    printf(" Warp Size: %d\n", deviceProp.warpSize);
    printf(" Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
    printf(" Max Threads per Multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
    printf(" Number of Multiprocessors: %d\n", deviceProp.multiProcessorCount);
    printf(" Clock Rate: %d MHz\n", deviceProp.clockRate / 1000);
    printf(" Memory Clock Rate: %d MHz\n", deviceProp.memoryClockRate / 1000);
    printf(" Memory Bus Width: %d bits\n", deviceProp.memoryBusWidth);
    printf(" L2 Cache Size: %d KB\n", deviceProp.l2CacheSize / 1024);
    // this should be 0 for compute capability < 8.0. When available, it allows storing a portion of the L2 cache as persistent global storage
    printf(" Persisting L2 Max Cache Size: %d KB\n", deviceProp.persistingL2CacheMaxSize / 1024);
    ////////
    printf(" Max Thread Dimensions: [%d,%d,%d]\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf(" Max Grid Dimensions: [%d,%d,%d]\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf(" Is Multi GPU Environment: %s\n", deviceProp.isMultiGpuBoard ? "true" : "false");
    
    printf("\n");
}

int main(int argc, char const *argv[])
{
    int currDevice;
    int deviceCount;
    cudaGetDevice(&currDevice);
    cudaGetDeviceCount(&deviceCount);

    for (int device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printGPUProp(deviceProp);
    }

    return 0;
}
