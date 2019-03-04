#include <iostream>
#include <cstdint>
#include <memory>
#include <random>
#include <functional>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_helper.h"

#pragma region CONV_1D

// 这里假定1D卷积核大小不超过125
const int MAX_CONV_KERNEL_SIZE_1D = 125;
// 每个block大小
const int TILE_SIZE_1D = 128;

__constant__ float d_m_1d[MAX_CONV_KERNEL_SIZE_1D];

// 共享内存中保存接缝元素和中间元素 YYYXX...XXXYYY
__global__ void kernel_conv_1d_1(
    const float* n,
    float* result,
    const size_t len_n,
    const size_t len_m
)
{
    __shared__ float Ns[TILE_SIZE_1D + MAX_CONV_KERNEL_SIZE_1D - 1];

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int last_block_end = blockIdx.x * blockDim.x;
    int next_block_begin = (blockIdx.x + 1) * blockDim.x;
    int half_len_m = len_m / 2;

    if (threadIdx.x < half_len_m)
    {
        int left = last_block_end - (half_len_m - threadIdx.x);
        Ns[threadIdx.x] = left >= 0 ? n[left] : 0.0f;

        int right = next_block_begin + threadIdx.x;

        Ns[threadIdx.x + blockDim.x + half_len_m] = right < len_n ? n[right] : 0.0f;
    }

    Ns[threadIdx.x + half_len_m] = i < len_n ? n[i] : 0.0f;

    __syncthreads();

    float sum = 0.0f;

    for (auto k = 0; k < len_m; k++)
    {
        sum += Ns[threadIdx.x + k] * d_m_1d[k];
    }

    __syncthreads();

    if (i < len_n)
    {
        result[i] = sum;
    }
}

// 共享内存中只保存中间元素，接缝元素从全局内存或者L2缓存中获取，如果恰好存在的话
__global__ void kernel_conv_1d_2(
    const float* n,
    float* result,
    size_t len_n,
    size_t len_m
)
{
    __shared__ float Ns[TILE_SIZE_1D];

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int last_block_end = blockIdx.x * blockDim.x;
    int next_block_begin = (blockIdx.x + 1) * blockDim.x;
    int half_len_m = len_m / 2;

    Ns[threadIdx.x] = i < len_n ? n[i] : 0.0f;

    __syncthreads();

    float sum = 0.0f;

    for (auto k = 0; k < len_m; k++)
    {
        int p = i + k - half_len_m;
        if (p >= 0 && p < len_n)
        {
            if (p >= last_block_end && p < next_block_begin)
            {
                sum += Ns[threadIdx.x + k - half_len_m] * d_m_1d[k];
            }
            else
            {
                sum += n[p] * d_m_1d[k];
            }
        }
    }

    __syncthreads();

    if (i < len_n)
    {
        result[i] = sum;
    }
}

std::unique_ptr<float[]> gpu_conv_1d(
    const std::unique_ptr<float[]>& h_n,
    const std::unique_ptr<float[]>& h_m,
    const size_t len_n,
    const size_t len_m,
    void(*kernel) (const float*,
        float*,
        const size_t,
        const size_t) // 这里是kernel的类型
)
{
    float *d_n, *d_result;
    void *addr_d_m;

    cudaGetSymbolAddress(&addr_d_m, d_m_1d);

    cc(cudaMalloc(&d_n, sizeof(float) * len_n));
    cc(cudaMalloc(&d_result, sizeof(float) * len_n));
    cc(cudaMemcpy(d_n, h_n.get(), sizeof(float) * len_n, cudaMemcpyHostToDevice));
    cc(cudaMemcpy(addr_d_m, h_m.get(), sizeof(float) * len_m, cudaMemcpyHostToDevice));

    kernel<<<(len_n + TILE_SIZE_1D - 1) / TILE_SIZE_1D, TILE_SIZE_1D>>>(d_n, d_result, len_n, len_m);
    cc(cudaDeviceSynchronize());

    float *h_result = new float[len_n];
    cc(cudaMemcpy(h_result, d_result, sizeof(float) * len_n, cudaMemcpyDeviceToHost));

    cc(cudaFree(d_n));
    cc(cudaFree(d_result));

    return std::unique_ptr<float[]>(h_result);
}


std::unique_ptr<float[]> cpu_conv_1d(
    const std::unique_ptr<float[]>& n,
    const std::unique_ptr<float[]>& m,
    const size_t len_n,
    const size_t len_m)
{
    float* result = new float[len_n];

    int half_m = len_m / 2;

#pragma omp parallel for
    for (auto i = 0; i < len_n; i++)
    {
        float sum = 0.0f;
        for (auto j = 0; j < len_m; j++)
        {
            int k = i + j - half_m;
            if (k >= 0 && k < len_n)
            {
                sum += n[k] * m[j];
            }
        }
        result[i] = sum;
    }

    return std::unique_ptr<float[]>(result);
}

void valid_1d(const std::unique_ptr<float[]>& h_result, const std::unique_ptr<float[]>& d_result, const size_t len)
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
}

#pragma endregion

#pragma region CONV_2D

__inline__ __host__ __device__ float get_2d(
    const float* mat,
    const int row,
    const int col,
    const int width)
{
    return mat[row * width + col];
}

__inline__ __host__ __device__ float& get_2d(
    float* mat,
    const int row,
    const int col,
    const int width)
{
    return mat[row * width + col];
}

inline float get_2d(
    const std::unique_ptr<float[]>& mat,
    const int row,
    const int col,
    const int width)
{
    return mat[row * width + col];
}

inline float& get_2d(
    std::unique_ptr<float[]>& mat,
    const int row,
    const int col,
    const int width)
{
    return mat[row * width + col];
}

std::unique_ptr<float[]> cpu_conv_2d(
    const std::unique_ptr<float[]>& n,
    const std::unique_ptr<float[]>& m,
    const int width_n,
    const int height_n,
    const int width_m,
    const int height_m)
{
    std::unique_ptr<float[]> result(new float[width_n * height_n]);
    
    const int half_width_m = width_m / 2, half_height_m = height_m / 2;

#pragma omp parallel for
    for (int i = 0; i < height_n; i++)
    {
        for (int j = 0; j < width_n; j++)
        {
            float sum = 0.0f;
            for (int p = -half_height_m; p < height_m - half_height_m; p++)
            {
                for (int q = -half_width_m; q < width_m - half_width_m; q++)
                {
                    if (i + p >= 0 && i + p < height_n && j + q >= 0 && j + q < width_n)
                    {
                        sum += get_2d(n, i + p, j + q, width_n) * get_2d(m, half_height_m + p, half_width_m + q, width_m);
                    }
                }
            }
            get_2d(result, i, j, width_n) = sum;
        }
    }

    return result;
}

const int MAX_CONV_KERNEL_SIZE_2D = 16; // 假设2d核最大是16x16
const int TILE_SIZE_2D = 32; // 2d tile的大小为32x32

__constant__ float d_m_2d[MAX_CONV_KERNEL_SIZE_2D * MAX_CONV_KERNEL_SIZE_2D];

__global__ void kernel_conv_2d_1(
    const float* n,
    float *result,
    const size_t pitch_n,
    const size_t pitch_result,
    const size_t width_n,
    const size_t height_n,
    const size_t width_m,
    const size_t height_m
)
{
    __shared__ float Ns[TILE_SIZE_2D][TILE_SIZE_2D];

    int half_width_m = width_m / 2, half_height_m = height_m / 2;

    int out_x = blockIdx.x * (blockDim.x - width_m + 1) + threadIdx.x;
    int out_y = blockIdx.y * (blockDim.y - height_m + 1) + threadIdx.y;

    if (out_y - half_height_m >= 0 && out_y - half_height_m < height_n &&
        out_x - half_width_m >= 0 && out_x - half_width_m < width_n)
    {
        Ns[threadIdx.y][threadIdx.x] = get_2d(n, out_y - half_height_m, out_x - half_width_m, pitch_n / sizeof(float));
    }
    else
    {
        Ns[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    float sum = 0.0;

    if (threadIdx.x < (blockDim.x - width_m + 1) && threadIdx.y < (blockDim.y - height_m + 1) &&
        out_x < width_n && out_y < height_n)
    {
        for (int i = 0; i < height_m; i++)
        {
            for (int j = 0; j < width_m; j++)
            {
                sum += Ns[threadIdx.y + i][threadIdx.x + j] * get_2d(d_m_2d, i, j, width_m);
            }
        }

        get_2d(result, out_y, out_x, pitch_result / sizeof(float)) = sum;
    }
}

std::unique_ptr<float[]> gpu_conv_2d(const std::unique_ptr<float[]>& h_n,
    const std::unique_ptr<float[]>& h_m,
    const int width_n,
    const int height_n,
    const int width_m,
    const int height_m)
{
    std::unique_ptr<float[]> h_result(new float[width_n * height_n]);

    const int half_width_m = width_m / 2, half_height_m = height_m / 2;

    cudaPitchedPtr d_n = make_cudaPitchedPtr(nullptr, 0, sizeof(float) * width_n, height_n);
    cc(cudaMallocPitch(&d_n.ptr, &d_n.pitch, d_n.xsize, d_n.ysize));
    cc(cudaMemcpy2D(d_n.ptr, d_n.pitch, h_n.get(), sizeof(float) * width_n, d_n.xsize, d_n.ysize, cudaMemcpyHostToDevice));

    void* addr_d_m_2d = nullptr;
    cc(cudaGetSymbolAddress(&addr_d_m_2d, d_m_2d));
    cc(cudaMemcpy(addr_d_m_2d, h_m.get(), sizeof(float) * width_m * height_m, cudaMemcpyHostToDevice));

    cudaPitchedPtr d_result = make_cudaPitchedPtr(nullptr, 0, sizeof(float) * width_n, height_n);
    cc(cudaMallocPitch(&d_result.ptr, &d_result.pitch, d_result.xsize, d_result.ysize));

    const int valid_size_x = TILE_SIZE_2D - width_m + 1;
    const int valid_size_y = TILE_SIZE_2D - height_m + 1;
    dim3 n_blocks((width_n + valid_size_x - 1) / valid_size_x, (height_n + valid_size_y - 1) / valid_size_y);
    dim3 n_threads(TILE_SIZE_2D, TILE_SIZE_2D);

    kernel_conv_2d_1<<<n_blocks, n_threads>>>(
        static_cast<float*>(d_n.ptr), static_cast<float*>(d_result.ptr), d_n.pitch, d_result.pitch, width_n, height_n, width_m, height_m);
    
    cc(cudaDeviceSynchronize());

    cc(cudaMemcpy2D(h_result.get(), sizeof(float) * width_n, d_result.ptr, d_result.pitch, width_n * sizeof(float), height_n, cudaMemcpyDeviceToHost));

    cc(cudaFree(d_n.ptr));
    cc(cudaFree(d_result.ptr));

    return h_result;
}

void valid_2d(const std::unique_ptr<float[]>& h_result, const std::unique_ptr<float[]>& d_result, const size_t width, const size_t height)
{
    bool is_valid = true;

    for (auto i = 0; i < height; i++)
    {
        for (auto j = 0; j < width; j++)
        {
            auto delta = h_result[i * width + j] - d_result[i * width + j];
            delta = delta >= 0.0f ? delta : -delta;
            if (delta > 0.00001)
            {
                is_valid = false;
                printf("At [%d, %d]: %f vs %f\n", i, j, h_result[i * width + j], d_result[i * width + j]);
            }
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
}
#pragma endregion


int main()
{

    std::default_random_engine rd;
    std::uniform_real_distribution<float> dis(-1.0, 1.0);

#pragma region CONV_1D_TEST
    const int len_n = 100'000'000, len_m = 13; // the length of array and kernel

    printf("---- 1D ----\n");
    printf("Length of data: %d    Length of kernel: %d\n", len_n, len_m);

    std::unique_ptr<float[]> n_1d(new float[len_n]);
    std::unique_ptr<float[]> m_1d(new float[len_m]);

#pragma omp parallel for
    for (auto i = 0; i < len_n; i++)
    {
        n_1d[i] = dis(rd);
    }

    for (auto i = 0; i < len_m; i++)
    {
        m_1d[i] = dis(rd);
    }

    std::cout << "1D CPU Method: " << std::endl;
    TimeCounter<> tcpu;
    auto h_result_1d = cpu_conv_1d(n_1d, m_1d, len_n, len_m);
    std::cout << tcpu.tell<std::chrono::milliseconds>() << std::endl;

    std::cout << "1D GPU Method 1: " << std::endl;
    TimeCounter<> tgpu1;
    auto d_result1_1d = gpu_conv_1d(n_1d, m_1d, len_n, len_m, kernel_conv_1d_1);
    std::cout << tgpu1.tell<std::chrono::milliseconds>() << std::endl;
    valid_1d(d_result1_1d, h_result_1d, len_n);


    std::cout << "1D GPU Method 2: " << std::endl;
    TimeCounter<> tgpu2;
    auto d_result2_1d = gpu_conv_1d(n_1d, m_1d, len_n, len_m, kernel_conv_1d_2);
    std::cout << tgpu2.tell<std::chrono::milliseconds>() << std::endl;
    valid_1d(d_result2_1d, h_result_1d, len_n);
#pragma endregion

    const int width_n = 30000, height_n = 2000;
    const int width_m = 7, height_m = 8;

    printf("---- 2D ----\n");
    printf("Size of data: (%d, %d)    Size of kernel: (%d, %d)\n", width_n, height_n, width_m, height_m);

    std::unique_ptr<float[]> n_2d(new float[width_n * height_n]);
    std::unique_ptr<float[]> m_2d(new float[width_m * height_m]);

#pragma omp parallel for
    for (auto i = 0; i < height_n; i++)
    {
        for (auto j = 0; j < width_n; j++)
        {
            n_2d[i * width_n + j] = dis(rd);
        }
    }

    for (auto i = 0; i < height_m; i++)
    {
        for (auto j = 0; j < width_m; j++)
        {
            m_2d[i * width_m + j] = dis(rd);
        }
    }

    std::cout << "2D CPU Method: " << std::endl;
    TimeCounter<> tcup;
    auto h_result_2d = cpu_conv_2d(n_2d, m_2d, width_n, height_n, width_m, height_m);
    std::cout << tcup.tell<std::chrono::milliseconds>() << std::endl;

    std::cout << "2D GPU Method: " << std::endl;
    TimeCounter<> tgup;
    auto d_result_2d = gpu_conv_2d(n_2d, m_2d, width_n, height_n, width_m, height_m);
    std::cout << tgup.tell<std::chrono::milliseconds>() << std::endl;

    valid_2d(h_result_2d, d_result_2d, width_n, height_n);

    return 0;
}