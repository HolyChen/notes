#include <iostream>
#include <cstdint>
#include <memory>
#include <random>
#include <functional>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_helper.h"

const size_t MINI_KERNEL_SIZE = 1024ull;
const size_t STREAM_KERNEL_SIZE = 512ull;
const size_t STREAM_MAX_KERNEL = 65536ull;

// 并行归约：单block Kogge-Stone算法
__global__
void kernel_prefix_sum_Kogge_Stone(const float* vec, float* result, const size_t len)
{
    __shared__ float data_section[MINI_KERNEL_SIZE];

    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    data_section[threadIdx.x] = tid < len ? vec[tid] : 0.0f;

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        __syncthreads();
        if (threadIdx.x >= stride)
        {
            // 不同的warp执行顺序的不同可能会导致数据覆盖
            auto tmp = data_section[threadIdx.x - stride];
            __syncthreads();
            data_section[threadIdx.x] += tmp;
        }
    }

    result[tid] = data_section[threadIdx.x];
}

// 并行归约：单block Brent-Kung算法
__global__
void kernel_prefix_sum_Brent_Kung(const float* vec, float* result, const size_t len)
{
    __shared__ float data_section[MINI_KERNEL_SIZE * 2];

    const int tid = 2 * blockDim.x * blockIdx.x + threadIdx.x;

    // 每个线程加载两个数据
    data_section[threadIdx.x] = tid < len ? vec[tid] : 0.0f;
    data_section[blockDim.x + threadIdx.x] = tid + blockDim.x < len ? vec[tid + blockDim.x] : 0.0f;

    // Forwarding
    for (int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        __syncthreads();

        int id_in_sec = (threadIdx.x + 1) * stride * 2 - 1;
        if (id_in_sec < blockDim.x * 2)
        {
            data_section[id_in_sec] += data_section[id_in_sec - stride];
        }
    }

    // Backwarding
    // 这里从blockDim.x / 2开始，是因为用一个线程处理两个线程
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        __syncthreads();

        int id_in_sec = (threadIdx.x + 1) * stride * 2 - 1;
        if (id_in_sec + stride < blockDim.x * 2)
        {
            data_section[id_in_sec + stride] += data_section[id_in_sec];
        }
    }

    __syncthreads();

    if (tid < len)
    {
        result[tid] = data_section[threadIdx.x];
    }
    if (blockDim.x + tid)
    {
        result[blockDim.x + tid] = data_section[blockDim.x + threadIdx.x];
    }
}

// 并行归约：流式算法

__device__ int flags[STREAM_MAX_KERNEL];
__device__ volatile float previous_sum[STREAM_MAX_KERNEL];
__device__ int block_counter[1];

__global__
void kernel_prefix_sum_Stream(const float* vec, float* result, const size_t len)
{
    __shared__ int bid;

    // 令线程按数据顺序调度
    if (threadIdx.x == 0)
    {
         bid = atomicAdd(block_counter, 1);

        // clear flag
        atomicExch(&flags[bid], 0);
    }

    __syncthreads();

    // 使用Brent-Kung算法做第一步逐段加和
    __shared__ float data_section[STREAM_KERNEL_SIZE * 2];

    int id = threadIdx.x + 2 * bid * blockDim.x;

    data_section[threadIdx.x] = id < len ? vec[id] : 0.0f;
    data_section[blockDim.x + threadIdx.x] = blockDim.x + id < len ? vec[blockDim.x + id] : 0.0f;

    for (int stride = 1; stride <= blockDim.x; stride *= 2)
    {
        __syncthreads();

        int id_in_sec = (threadIdx.x + 1) * 2 * stride - 1;
        if (id_in_sec < blockDim.x * 2)
        {
            // 由于误差传播，改用max
            //data_section[id_in_sec] += data_section[id_in_sec - stride];
            data_section[id_in_sec] = max(data_section[id_in_sec], data_section[id_in_sec - stride]);
        }
    }

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        __syncthreads();

        int id_in_sec = (threadIdx.x + 1) * 2 * stride - 1;
        if (id_in_sec + stride < blockDim.x * 2)
        {
            // 由于误差传播，改用max
            //data_section[id_in_sec + stride] += data_section[id_in_sec];
            data_section[id_in_sec + stride] = max(data_section[id_in_sec + stride], data_section[id_in_sec]);
        }
    }

    __syncthreads();

    // 数据传递

    __shared__ float presum;

    if (threadIdx.x == 0)
    {
        if (bid != 0)
        {
            while (atomicAdd(&flags[bid - 1], 0) == 0) { /* wait */ };
            presum = previous_sum[bid - 1];

            //previous_sum[bid] = presum + data_section[blockDim.x * 2 - 1];
            previous_sum[bid] = max(presum, data_section[blockDim.x * 2 - 1]);

            __threadfence();

            atomicExch(&flags[bid], 1);
        }
        else
        {
            presum = 0.0f;

            //previous_sum[bid] = presum + data_section[blockDim.x * 2 - 1];
            previous_sum[bid] = max(presum, data_section[blockDim.x * 2 - 1]);

            __threadfence();

            atomicAdd(&flags[bid], 1);
        }

    }

    __syncthreads();

    //data_section[threadIdx.x] += presum;
    //data_section[blockDim.x + threadIdx.x] += presum;

    data_section[threadIdx.x] = max(data_section[threadIdx.x], presum);
    data_section[blockDim.x + threadIdx.x] = max(data_section[blockDim.x + threadIdx.x], presum);

    if (id < len)
    {
        result[id] = data_section[threadIdx.x];
    }
    if (blockDim.x + id < len)
    {
        result[blockDim.x + id] = data_section[blockDim.x + threadIdx.x];
    }
}

// 由于误差传播，改用max替换sum
std::unique_ptr<float[]> cpu_prefix_sum(const std::unique_ptr<float[]>& vec, const size_t len)
{
    std::unique_ptr<float[]> result(new float[len]);

    float sum = 0.0;

    for (int i = 0; i < len; i++)
    {
        //sum += vec[i];
        sum = std::max(sum, vec[i]);
        result[i] = sum;
    }

    return result;
}

std::unique_ptr<float[]> gpu_prefix_sum(const std::unique_ptr<float[]>& h_vec, const size_t len)
{
    std::unique_ptr<float[]> h_result(new float[len]);

    float *d_vec = nullptr, *d_result = nullptr;

    cc(cudaMalloc(&d_vec, sizeof(float) * len));
    cc(cudaMemcpy(d_vec, h_vec.get(), sizeof(float) * len, cudaMemcpyDefault));

    cc(cudaMalloc(&d_result, sizeof(float) * len));

    void *addr_block_counter = nullptr;
    cc(cudaGetSymbolAddress(&addr_block_counter, block_counter));
    cc(cudaMemset(addr_block_counter, 0, sizeof(int)));

    //kernel_prefix_sum_Kogge_Stone<<<1, MINI_KERNEL_SIZE>>>(d_vec, d_result, len);
    //kernel_prefix_sum_Brent_Kung<<<1, MINI_KERNEL_SIZE>>>(d_vec, d_result, len);
    kernel_prefix_sum_Stream<<<(len + 2 * STREAM_KERNEL_SIZE - 1) / (2 * STREAM_KERNEL_SIZE), STREAM_KERNEL_SIZE>>>(d_vec, d_result, len);
    cc(cudaDeviceSynchronize());

    cc(cudaMemcpy(h_result.get(), d_result, sizeof(float) * len, cudaMemcpyDefault));

    cc(cudaFree(d_vec));
    cc(cudaFree(d_result));

    return h_result;
}

bool valid(const std::unique_ptr<float[]>& h_result, const std::unique_ptr<float[]>& d_result, const size_t len)
{
    bool is_valid = true;

    for (auto i = 0; i < len; i++)
    {
        auto delta = h_result[i] - d_result[i];
        delta = delta >= 0.0f ? delta : -delta;
        if (delta > 0.00001)
        {
            is_valid = false;
            printf("At [%d]: %f vs %f\n", i, h_result[i], d_result[i]);
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

int main()
{
    std::default_random_engine rd;
    std::uniform_real_distribution<float> dis(0, 0.001);

    size_t len =  STREAM_MAX_KERNEL * STREAM_KERNEL_SIZE * 2; // 2 * MINI_KERNEL_SIZE;

    printf("%zu\n", len);

    std::unique_ptr<float[]> vec(new float[len]);

    for (int i = 0; i < len; i++)
    {
        vec[i] = dis(rd);
    }

    printf("---- CPU ----\n");
    TimeCounter<> tcpu;
    auto h_result = cpu_prefix_sum(vec, len);
    std::cout << tcpu.tell<std::chrono::milliseconds>() << std::endl;


    printf("---- GPU ----\n");
    TimeCounter<> tgpu;
    auto g_result = gpu_prefix_sum(vec, len);
    std::cout << tgpu.tell<std::chrono::milliseconds>() << std::endl;

    valid(h_result, g_result, len);

    return 0;
}