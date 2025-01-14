#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
void safeCudaMalloc(void *&devPtr, size_t size)
{
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&devPtr, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void safeCudaMemcpy(void *&dst, void *&src, size_t size, cudaMemcpyKind direction)
{
    cudaError_t err = cudaSuccess;
    err = cudaMemcpy(dst, src, size, direction);
    if (err != cudaSuccess)
    {
        fprintf(stderr,
                "Failed to copy vector from host to device (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void safeCudaFree(void *&devPtr)
{
    cudaError_t err = cudaSuccess;
    err = cudaFree(devPtr);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void printArray(float *arr, int arrSize)
{
    std::string strArr = "[";
    std::for_each(arr, arr + arrSize, [&strArr, arr, arrSize](float &elem)
                  {
        int i = &elem - arr;
        strArr += std::to_string(elem);
        if (i + 1 != arrSize){
            strArr += ",";
        } });
    strArr += "]\n";
    std::cout << strArr;
}