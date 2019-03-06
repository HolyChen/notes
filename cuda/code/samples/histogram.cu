#include <iostream>
#include <cstdint>
#include <memory>
#include <random>
#include <functional>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_helper.h"

const size_t BLOCK_SIZE = 128u;

// 按块划分，每个线程处理一个连续的块
// | _ _ (0) _ _ | _ _ (1) _ _ | ... | _ _ (nthread - 1) _ _ |
__global__
void kernel_histogram_naive_block(const char* input, int* result, const size_t len, const size_t n_bin)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    // 一块的大小
    int section = (len - 1) / (blockDim.x * gridDim.x) + 1;
    int start = i * section;

    for (int k = 0; k < section; k++)
    {
        if (start + k < len)
        {
            int c = input[start + k];
            if (c >= 0 && c < n_bin)
            {
                atomicAdd(&result[c], 1);
            }
        }
    }
}

// 交错划分，每个线程处理一个连续的块
// |(0)(1) ... (nthread - 1) |(0)(1) ... (nthread - 1) | ... |
__global__
void kernel_histogram_naive_interleaved(const char* input, int* result, const size_t len, const size_t n_bin)
{
    int section = blockDim.x * gridDim.x;

    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < len; i += section)
    {
        int c = input[i];
        if (c >= 0 && c < n_bin)
        {
            atomicAdd(&result[c], 1);
        }
    }
}

// 使用共享内存处理
__global__
void kernel_histogram_privatized(const char* input, int* result, const size_t len, const size_t n_bin)
{
    extern __shared__ int local_hist[];
    
    for (int bid = threadIdx.x; bid < n_bin; bid += blockDim.x)
    {
        local_hist[bid] = 0;
    }

    __syncthreads();

    int section = blockDim.x * gridDim.x;

    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < len; i += section)
    {
        int c = input[i];
        if (c >= 0 && c < n_bin)
        {
            atomicAdd(&local_hist[c], 1);
        }
    }

    __syncthreads();

    // merge

    for (int bid = threadIdx.x; bid < n_bin; bid += blockDim.x)
    {
        atomicAdd(&result[bid], local_hist[bid]);
    }
}


std::unique_ptr<int[]> gpu_histogram(const std::unique_ptr<char[]>& input, const size_t len, const size_t n_bin, 
    void(*kernel)(const char*, int *, size_t, size_t))
{
    std::unique_ptr<int[]> h_result(new int[n_bin]);

    char *d_input = nullptr;
    int *d_result = nullptr;

    cc(cudaMalloc(&d_input, sizeof(char) * len));
    cc(cudaMemcpy(d_input, input.get(), sizeof(char) * len, cudaMemcpyHostToDevice));

    cc(cudaMalloc(&d_result, sizeof(int) * n_bin));
    cc(cudaMemset(d_result, 0, sizeof(int) * n_bin));
    
    if (kernel != kernel_histogram_privatized)
    {
        kernel<<<(len - 1) / (128 * BLOCK_SIZE) + 1, BLOCK_SIZE>>>(d_input, d_result, len, n_bin);
    }
    else
    {
        kernel<<<(len - 1) / (128 * BLOCK_SIZE) + 1, BLOCK_SIZE, sizeof(int) * n_bin>>>(d_input, d_result, len, n_bin);
    }

    cc(cudaDeviceSynchronize());

    cc(cudaMemcpy(h_result.get(), d_result, sizeof(int) * n_bin, cudaMemcpyDeviceToHost));

    cudaFree(d_input);
    cudaFree(d_result);

    return h_result;
}

std::unique_ptr<int[]> cpu_histogram(const std::unique_ptr<char[]>& input, const size_t len, const size_t n_bin)
{
    std::unique_ptr<int[]> result(new int[n_bin]);

    std::fill(result.get(), result.get() + n_bin, 0);

    for (int i = 0; i < len; i++)
    {
        int c = input[i];
        if (c >= 0 && c < n_bin)
        {
            result[c]++;
        }
    }
    
    return result;
}

bool valid(const std::unique_ptr<int[]>& h_result, const std::unique_ptr<int[]>& d_result, const size_t len)
{
    bool is_valid = true;

    for (auto i = 0; i < len; i++)
    {
        auto delta = h_result[i] - d_result[i];
        if (delta != 0)
        {
            is_valid = false;
            printf("At [%d]: %d vs %d\n", i, h_result[i], d_result[i]);
        }
    }
    if (is_valid)
    {
        printf("All OK\n");
    }
    else
    {
        printf("Somewhere error\n");
    }

    return is_valid;
}

enum class DataDistribution
{
    UNIFORM,
    BERNOULLI
};

void test(const size_t len, const size_t n_bin, DataDistribution type = DataDistribution::UNIFORM)
{
    std::unique_ptr<char[]> input(new char[len]);

    if (type == DataDistribution::UNIFORM)
    {
        std::default_random_engine rd;
        std::uniform_int_distribution<uint32_t> dis(0, 127);

#pragma omp parallel for
        for (auto i = 0; i < len; i++)
        {
            input[i] = dis(rd);
        }

        printf("Uniform: \n");
    }
    else
    {
        std::default_random_engine rd;
        std::bernoulli_distribution dis(0.5);

#pragma omp parallel for
        for (auto i = 0; i < len; i++)
        {
            input[i] = dis(rd) ? 1 : 0;
        }

        printf("Bernoulli: \n");
    }

    TimeCounter<> tcpu;
    auto h_result = cpu_histogram(input, len, n_bin);
    tcpu.output("CPU: ");

    {
        TimeCounter<> tgpu;
        auto d_result = gpu_histogram(input, len, n_bin, kernel_histogram_naive_block);
        tgpu.output("GPU Naive Block: ");
        valid(h_result, d_result, n_bin);
    }

    {
        TimeCounter<> tgpu;
        auto d_result = gpu_histogram(input, len, n_bin, kernel_histogram_naive_interleaved);
        tgpu.output("GPU Naive Interleaved: ");
        valid(h_result, d_result, n_bin);
    }

    {
        TimeCounter<> tgpu;
        auto d_result = gpu_histogram(input, len, n_bin, kernel_histogram_privatized);
        tgpu.output("GPU Privatized: ");
        valid(h_result, d_result, n_bin);
    }
    printf("\n\n");
}

int main()
{
    const size_t len = 600'000'000;
    const size_t n_bin = 128u;

    printf("Length: %zu    Bins: %zu\n", len, n_bin);

    test(len, n_bin, DataDistribution::UNIFORM);
    test(len, n_bin, DataDistribution::BERNOULLI);

    return 0;
}