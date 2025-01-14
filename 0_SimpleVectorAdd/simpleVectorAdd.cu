#include <cuda_runtime.h>
// #include <iostream>
// #include <algorithm>
#include <string>
#include <chrono>
#include <execution>
#include "simpleVectorAdd.cuh"
// alias in c++
using Microseconds = std::chrono::microseconds;

using HighResClock = std::chrono::high_resolution_clock;

__global__ void vectorAdd(float *a, float *b, float *c, int numElems)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElems)
    {
        c[i] = a[i] + b[i];
    }
}

void vectorAddCPU(float *a, float *b, float *c, int arrSize)
{
    for (int i = 0; i < arrSize; ++i)
    {
        c[i] = a[i] + b[i];
    }
}

void vectorAddCPUParallel(float *a, float *b, float *c, int arrSize)
{
    std::for_each(std::execution::par, a, a + arrSize, [=](float &elem)
                  {
        int i = &elem - a;
        c[i] = a[i] + b[i]; });
}

int main(int argc, char *argv[])
{
    int arraySize = std::stoi(argv[1]);
    int n_threads = 4;
    int n_blocks = (arraySize + n_threads - 1) / n_threads;

    size_t size = arraySize * sizeof(float);

    float *host_a = (float *)malloc(size);
    float *host_b = (float *)malloc(size);
    float *host_c = (float *)malloc(size);
    float *host_d = (float *)malloc(size);
    float *host_e = (float *)malloc(size);

    float *device_a = NULL;
    float *device_b = NULL;
    float *device_c = NULL;

    cudaError_t err = cudaSuccess;

    std::generate(host_a, host_a + arraySize, []()
                  { return 1; });

    std::generate(host_b, host_b + arraySize, []()
                  { return 2; });

    safeCudaMalloc((void *&)device_a, size);
    safeCudaMalloc((void *&)device_b, size);
    safeCudaMalloc((void *&)device_c, size);

    // printArray(host_b, arraySize);

    safeCudaMemcpy((void *&)device_a, (void *&)host_a, size, cudaMemcpyHostToDevice);
    safeCudaMemcpy((void *&)device_b, (void *&)host_b, size, cudaMemcpyHostToDevice);

    printf("CUDA kernel launch with %d blocks of %d threads\n", n_blocks, n_threads);
    auto gpuCalcStart = std::chrono::high_resolution_clock::now();
    vectorAdd<<<n_blocks, n_threads>>>(device_a, device_b, device_c, arraySize);
    auto gpuCalcEnd = std::chrono::high_resolution_clock::now();
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    auto gpuDuration = std::chrono::duration_cast<std::chrono::microseconds>(gpuCalcEnd - gpuCalcStart);

    safeCudaMemcpy((void *&)host_c, (void *&)device_c, size, cudaMemcpyDeviceToHost);
    // printArray(host_c, arraySize);

    auto cpuCalcStart = HighResClock::now();
    vectorAddCPU(host_a, host_b, host_d, arraySize);
    auto cpuCalcEnd = HighResClock::now();

    auto cpuDuration = std::chrono::duration_cast<Microseconds>(cpuCalcEnd - cpuCalcStart);
    // printArray(host_d, arraySize);

    auto cpuParallelCalcStart = HighResClock::now();
    vectorAddCPUParallel(host_a, host_b, host_e, arraySize);
    auto cpuParallelCalcEnd = HighResClock::now();

    auto cpuParallelDuration = std::chrono::duration_cast<Microseconds>(cpuParallelCalcEnd - cpuParallelCalcStart);

    // printArray(host_e, arraySize);

    safeCudaFree((void *&)device_a);
    safeCudaFree((void *&)device_b);
    safeCudaFree((void *&)device_c);

    free(host_a);
    free(host_b);
    free(host_c);
    free(host_d);
    free(host_e);

    printf("CPU took %ld microseconds to complete array sum for %d elements\n", cpuDuration.count(), arraySize);
    printf("CPU took %ld microseconds to complete array sum for %d elements with parallelism\n", cpuParallelDuration.count(), arraySize);
    printf("CUDA took %ld microseconds to complete array sum for %d elements\n", gpuDuration.count(), arraySize);

    return 0;
}