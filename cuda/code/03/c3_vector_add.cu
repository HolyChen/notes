#include <cstdint>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_helper.h"

// kernel definition
__global__ void vec_add(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}


int main()
{
    const int N = 5;
    float h_A[N] = { 0., 1., 2., 3., 4. };
    float h_B[N] = { 9., 8., 7., 6., 5. };
    float h_C[N];

    float *d_A, *d_B, *d_C;

    // malloc for arrays on device
    cc(cudaMalloc(&d_A, sizeof(float) * N));
    cc(cudaMalloc(&d_B, sizeof(float) * N));
    cc(cudaMalloc(&d_C, sizeof(float) * N));

    // copy memory from host to device
    cc(cudaMemcpy(d_A, h_A, sizeof(float) * N, cudaMemcpyHostToDevice));
    cc(cudaMemcpy(d_B, h_B, sizeof(float) * N, cudaMemcpyHostToDevice));

    // kernel invocation with N threads
    vec_add<<<1, N >>>(d_A, d_B, d_C);

    // waiting until device completes
    cc(cudaDeviceSynchronize());

    // copy result from device to host
    cc(cudaMemcpy(h_C, d_C, sizeof(float) * N, cudaMemcpyDeviceToHost));

    cc(cudaFree(d_A));
    cc(cudaFree(d_B));
    cc(cudaFree(d_C));

    std::cout << "A: ";
    for (int i = 0; i < N; i++)
    {
        std::cout << h_A[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "B: ";
    for (int i = 0; i < N; i++)
    {
        std::cout << h_B[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "C: ";
    for (int i = 0; i < N; i++)
    {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    // output should be:
    // A: 0 1 2 3 4
    // B: 9 8 7 6 5
    // C: 9 9 9 9 9

    return 0;
}