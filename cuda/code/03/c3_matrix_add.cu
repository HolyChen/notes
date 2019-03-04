#include <cstdint>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_helper.h"

// matrix addition
template <int M, int N>
__global__ void mat_add(float* A, float* B, float* C)
{
    // note, the layout of threads is column major
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    C[row * N + col] = A[row * N + col] + B[row * N + col];
}


int main()
{
    const int M = 24;
    const int N = 16;

    float h_A[M][N];
    float h_B[M][N];
    float h_C[M][N];

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            h_A[i][j] = -i * N - j;
            if (i == j)
            {
                h_B[i][j] = i * N + j + 1;
            }
            else
            {
                h_B[i][j] = i * N + j;
            }
        }
    }

    float *d_A, *d_B, *d_C;

    // malloc for arrays on device
    cc(cudaMalloc(&d_A, sizeof(float) * M * N));
    cc(cudaMalloc(&d_B, sizeof(float) * M * N));
    cc(cudaMalloc(&d_C, sizeof(float) * M * N));

    // copy memory from host to device
    cc(cudaMemcpy(d_A, h_A, sizeof(float) * M * N, cudaMemcpyHostToDevice));
    cc(cudaMemcpy(d_B, h_B, sizeof(float) * M * N, cudaMemcpyHostToDevice));

    dim3 num_of_block = { 4, 3 };
    dim3 thread_per_block = { 4, 8 };

    // kernel invocation with N threads
    mat_add<M, N> << <num_of_block, thread_per_block >> > (d_A, d_B, d_C);

    // waiting until device completed
    cc(cudaDeviceSynchronize());

    // copy result from device to host
    cc(cudaMemcpy(h_C, d_C, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    cc(cudaFree(d_A));
    cc(cudaFree(d_B));
    cc(cudaFree(d_C));

    std::cout << "A:\n";
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << h_A[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "B:\n";
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << h_B[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "C:\n";
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << h_C[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Output of matrix C:
    // 1 if on main diagonal
    // 0 otherwise

    return 0;
}