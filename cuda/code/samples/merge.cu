#include <iostream>
#include <cstdint>
#include <memory>
#include <random>
#include <functional>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_helper.h"

const uint32_t BLOCK_SIZE = 128u;
const uint32_t TILE_SIZE = 2048u;

using ElemType = int;

__device__ __host__
void merge_sequential(const ElemType* a, const int m, const ElemType* b, const int n, ElemType* c)
{
    auto i = 0, j = 0, k = 0;
    while (i < m && j < n)
    {
        if (a[i] <= b[j])
        {
            c[k++] = a[i++];
        }
        else
        {
            c[k++] = b[j++];
        }
    }

    while (i < m)
    {
        c[k++] = a[i++];
    }

    while (j < n)
    {
        c[k++] = b[j++];
    }
}

__device__ __host__
int co_rank(const int k, const ElemType* a, const int m, const ElemType* b, const int n)
{
    int i = k > m ? m : k;
    int j = k - i;
    int i_low = k - n > 0 ? k - n : 0;
    int j_low = k - m > 0 ? k - m : 0;

    int delta = 0;
    bool found = false;

    while (!found)
    {
        if (i > 0 && j < n && a[i - 1] > b[j])
        {
            delta = (i - i_low + 1) / 2;
            i -= delta;
            j_low = j;
            j += delta;
        }
        else if (j > 0 && i < m && b[j - 1] >= a[i])
        {
            delta = (j - j_low + 1) / 2;
            j -= delta;
            i_low = i;
            i += delta;
        }
        else
        {
            found = true;
        }
    }

    return i;
}

__global__
void kernel_merge_naive(const ElemType* a, const int m, const ElemType* b, const int n, ElemType* c)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    auto section = (m + n - 1) / (blockDim.x * gridDim.x) + 1;
    auto c_cur = tid * section;
    auto c_next = min((tid + 1) * section, m + n);
    auto i_cur = co_rank(c_cur, a, m, b, n);
    auto j_cur = c_cur - i_cur;
    auto i_next = co_rank(c_next, a, m, b, n);
    auto j_next = c_next - i_next;

    merge_sequential(a + i_cur, i_next - i_cur, b + j_cur, j_next - j_cur, c + c_cur);
}

__global__
void kernel_merge_tiled(const ElemType* a, const int m, const ElemType* b, const int n, ElemType* c)
{
    extern __shared__ ElemType shareABs[];

    ElemType* as = &shareABs[0];
    ElemType* bs = &shareABs[TILE_SIZE];
    ElemType* cs = &shareABs[TILE_SIZE * 2];

    auto section = (m + n - 1) / gridDim.x + 1;
    auto c_begin = min(blockIdx.x * section, m + n);
    auto c_end = min((blockIdx.x + 1) * section, m + n);

    if (threadIdx.x == 0)
    {
        as[0] = co_rank(c_begin, a, m, b, n);
        as[1] = co_rank(c_end, a, m, b, n);
    }

    __syncthreads();
    
    auto a_begin = as[0], a_end = as[1];
    auto b_begin = c_begin - a_begin, b_end = c_end - a_end;
    auto a_length = a_end - a_begin;
    auto b_length = b_end - b_begin;
    auto c_length = c_end - c_begin;
    auto a_consumed = 0;
    auto b_consumed = 0;

    // assume(TILE_SIZE % blockDim.x == 0);

    __syncthreads();

    auto sub_sec = TILE_SIZE / blockDim.x;

    int count = 0;
    while (count < section)
    {
        for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x)
        {
            if (i + a_consumed + a_begin < a_end)
            {
                as[i] = a[i + a_consumed + a_begin];
            }
        }

        for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x)
        {
            if (i + b_consumed + b_begin < b_end)
            {
                bs[i] = b[i + b_consumed + b_begin];
            }
        }

        __syncthreads();

        auto as_length = min(TILE_SIZE, a_length - a_consumed);
        auto bs_length = min(TILE_SIZE, b_length - b_consumed);

        auto k_cur = min(threadIdx.x * sub_sec, c_length - (a_consumed + b_consumed));
        auto k_next = min((threadIdx.x + 1) * sub_sec, c_length - (a_consumed + b_consumed));
        auto i_cur = co_rank(k_cur, as, as_length, bs, bs_length);
        auto i_next = co_rank(k_next, as, as_length, bs, bs_length);
        auto j_cur = k_cur - i_cur;
        auto j_next = k_next - i_next;

        merge_sequential(as + i_cur, i_next - i_cur, bs + j_cur, j_next - j_cur, cs + k_cur);

        __syncthreads();

        for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x)
        {
            if (i + count + c_begin < c_end)
            {
                c[i + count + c_begin] = cs[i];
            }
        }

        count += TILE_SIZE;

        a_consumed += co_rank(TILE_SIZE, as, as_length, bs, bs_length);
        b_consumed = count - a_consumed;

        __syncthreads();
    }
}

std::unique_ptr<ElemType[]> gpu_merge_naive(const ElemType* h_a, const int m, const ElemType* h_b, const int n)
{
    ElemType *d_a, *d_b, *d_c;

    cc(cudaMalloc(&d_a, sizeof(ElemType) * m));
    cc(cudaMalloc(&d_b, sizeof(ElemType) * n));
    cc(cudaMalloc(&d_c, sizeof(ElemType) * (m + n)));

    cc(cudaMemcpy(d_a, h_a, sizeof(ElemType) * m, cudaMemcpyHostToDevice));
    cc(cudaMemcpy(d_b, h_b, sizeof(ElemType) * n, cudaMemcpyHostToDevice));

    kernel_merge_naive<<<(m + n - 1) / (16 * BLOCK_SIZE) + 1 , BLOCK_SIZE>>>(d_a, m, d_b, n, d_c);

    cc(cudaDeviceSynchronize());

    std::unique_ptr<ElemType[]> h_c(new ElemType[m + n]);
    cc(cudaMemcpy(h_c.get(), d_c, sizeof(ElemType) * (m + n), cudaMemcpyDeviceToHost));

    cc(cudaFree(d_a));
    cc(cudaFree(d_b));
    cc(cudaFree(d_c));

    return h_c;
}

std::unique_ptr<ElemType[]> gpu_merge_tiled(const ElemType* h_a, const int m, const ElemType* h_b, const int n)
{
    ElemType *d_a, *d_b, *d_c;

    cc(cudaMalloc(&d_a, sizeof(ElemType) * m));
    cc(cudaMalloc(&d_b, sizeof(ElemType) * n));
    cc(cudaMalloc(&d_c, sizeof(ElemType) * (m + n)));

    cc(cudaMemcpy(d_a, h_a, sizeof(ElemType) * m, cudaMemcpyHostToDevice));
    cc(cudaMemcpy(d_b, h_b, sizeof(ElemType) * n, cudaMemcpyHostToDevice));

    kernel_merge_tiled<<<(m + n - 1) / (8 * TILE_SIZE) + 1, 128, sizeof(ElemType) * TILE_SIZE * 3>>>(d_a, m, d_b, n, d_c);

    cc(cudaDeviceSynchronize());

    std::unique_ptr<ElemType[]> h_c(new ElemType[m + n]);
    cc(cudaMemcpy(h_c.get(), d_c, sizeof(ElemType) * (m + n), cudaMemcpyDeviceToHost));

    cc(cudaFree(d_a));
    cc(cudaFree(d_b));
    cc(cudaFree(d_c));

    return h_c;
}

std::unique_ptr<ElemType[]> cpu_merge(const ElemType* a, const int m, const ElemType* b, const int n)
{
    std::unique_ptr<ElemType[]> c(new ElemType[m + n]);

    merge_sequential(a, m, b, n, c.get());

    return c;
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

int main()
{
    const size_t m = 100'000'000;
    const size_t n = 100'000'000 - 1;

    std::default_random_engine rd;
    std::uniform_int_distribution<int> dis(0xFFFFFFFF, 0x7FFFFFFF);

    std::unique_ptr<ElemType[]> a(new ElemType[m]);
    std::unique_ptr<ElemType[]> b(new ElemType[n]);

#pragma omp parallel for
    for (auto i = 0; i < m; i++)
    {
        a[i] = dis(rd);
    }

#pragma omp parallel for
    for (auto i = 0; i < n; i++)
    {
        b[i] = dis(rd);
    }

    std::stable_sort(a.get(), a.get() + m);
    std::stable_sort(b.get(), b.get() + n);

    TimeCounter<> tcpu;
    auto h_result = cpu_merge(a.get(), m, a.get(), n);
    tcpu.output("CPU: ");

    {
        TimeCounter<> tgpu;
        auto d_result = gpu_merge_naive(a.get(), m, a.get(), n);
        tgpu.output("GPU Naive: ");

        valid(h_result, d_result, m + n);
    }

    {
        TimeCounter<> tgpu;
        auto d_result = gpu_merge_tiled(a.get(), m, a.get(), n);
        tgpu.output("GPU Tiled: ");

        valid(h_result, d_result, m + n);
    }

    return 0;
}