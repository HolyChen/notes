#include <iostream>
#include <cstdint>
#include <memory>
#include <random>
#include <functional>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cuda_helper.h"

const size_t BLOCK_SIZE = 128u;

struct SparseMat_CSR
{
    uint32_t n_row = 0u;
    uint32_t n_col = 0u;
    uint32_t n_elem = 0u;
    float* elems = nullptr;
    uint32_t* col_index = nullptr;
    uint32_t* row_ptr = nullptr;

    SparseMat_CSR() = default;

    SparseMat_CSR(uint32_t n_row, uint32_t n_col, uint32_t n_elem)
        : n_row(n_row), n_col(n_col), n_elem(n_elem)
    {
        if (n_row * n_col * n_elem > 0)
        {
            elems = new float[n_elem];
            col_index = new uint32_t[n_elem];
            row_ptr = new uint32_t[n_row + 1];
        }
    }

    ~SparseMat_CSR()
    {
        if (elems) delete[] elems;
        if (col_index) delete[] col_index;
        if (row_ptr) delete[] row_ptr;
    }
};

struct Vector
{
    uint32_t n = 0;
    float* elems = nullptr;

    Vector() = default;

    Vector(uint32_t n)
        : n(n)
    {
        if (n > 0)
        {
            elems = new float[n];
        }
    }

    Vector(const Vector& ano)
        : n(ano.n)
    {
        if (n > 0)
        {
            elems = new float[n];
            std::copy(ano.elems, ano.elems + n, elems);
        }
    }

    Vector(Vector&& ano)
        : n(ano.n)
    {
        if (n > 0)
        {
            ano.n = 0;
            elems = ano.elems;
            ano.elems = nullptr;
        }
    }

    ~Vector()
    {
        if (elems) delete[] elems;
    }
};

// SpMV/CSR 稀疏矩阵-向量乘法，压缩稀疏行(Compress Sparse Row, CSR)结构算法
__global__
void kernel_spMV_CSR(const uint32_t mat_n_row, const uint32_t mat_n_col,
    const float* mat_elems, const uint32_t* mat_col_index, const uint32_t* mat_row_ptr,
    const float* x_elems, float* result)
{
    auto tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < mat_n_row)
    {
        float sum = 0.0f;
        for (auto i = mat_row_ptr[tid], end = mat_row_ptr[tid + 1]; i < end; i++)
        {
            sum += mat_elems[i] * x_elems[mat_col_index[i]];
        }

        result[tid] = sum;
    }
}

Vector gpu_spMV_CSR(const SparseMat_CSR& h_mat, const Vector& h_x)
{
    Vector h_result(h_mat.n_row);
    std::fill(h_result.elems, h_result.elems + h_mat.n_row, 0.0f);

    float *d_mat_elems = nullptr;
    cc(cudaMalloc(&d_mat_elems, sizeof(float) * h_mat.n_elem));
    cc(cudaMemcpy(d_mat_elems, h_mat.elems, sizeof(float) * h_mat.n_elem, cudaMemcpyHostToDevice));

    uint32_t *d_mat_col_index = nullptr;
    cc(cudaMalloc(&d_mat_col_index, sizeof(uint32_t) * h_mat.n_elem));
    cc(cudaMemcpy(d_mat_col_index, h_mat.col_index, sizeof(uint32_t) * h_mat.n_elem, cudaMemcpyHostToDevice));

    uint32_t *d_mat_row_ptr = nullptr;
    cc(cudaMalloc(&d_mat_row_ptr, sizeof(uint32_t) * (h_mat.n_row + 1)));
    cc(cudaMemcpy(d_mat_row_ptr, h_mat.row_ptr, sizeof(uint32_t) * (h_mat.n_row + 1), cudaMemcpyHostToDevice));

    uint32_t d_x_n = h_x.n;
    float *d_x_elems = nullptr;
    cc(cudaMalloc(&d_x_elems, sizeof(float) * h_x.n));
    cc(cudaMemcpy(d_x_elems, h_x.elems, sizeof(float) * h_x.n, cudaMemcpyHostToDevice));

    uint32_t d_result_n = h_result.n;
    float *d_result_elems = nullptr;
    cc(cudaMalloc(&d_result_elems, sizeof(float) * d_result_n));

    kernel_spMV_CSR<<<(h_mat.n_row - 1) / BLOCK_SIZE + 1, BLOCK_SIZE>>>(h_mat.n_row, h_mat.n_col, d_mat_elems,
        d_mat_col_index, d_mat_row_ptr, d_x_elems, d_result_elems);
    
    cc(cudaDeviceSynchronize());

    cc(cudaMemcpy(h_result.elems, d_result_elems, sizeof(float) * d_result_n, cudaMemcpyDeviceToHost));

    cc(cudaFree(d_mat_elems));
    cc(cudaFree(d_mat_col_index));
    cc(cudaFree(d_mat_row_ptr));
    cc(cudaFree(d_x_elems));
    cc(cudaFree(d_result_elems));

    return h_result;
}

__global__
void kernel_spMV_ELL(const uint32_t mat_n_row, const uint32_t mat_n_col,
    const float* mat_elems, const uint32_t* mat_col_index, const float* x_elems, float* result)
{
    auto tid = threadIdx.x + blockDim.x * blockIdx.x;
    
    if (tid < mat_n_row)
    {
        float sum = 0.0f;
        for (auto i = 0; i < mat_n_col; i++)
        {
            sum += mat_elems[i * mat_n_row + tid] * x_elems[mat_col_index[i * mat_n_row + tid]];
        }

        result[tid] = sum;
    }
}

Vector gpu_spMV_ELL(const SparseMat_CSR& h_mat, const Vector& h_x)
{
    Vector h_result(h_mat.n_row);
    std::fill(h_result.elems, h_result.elems + h_mat.n_row, 0.0f);

    uint32_t mat_n_row = h_mat.n_row;
    // 非零元素最多的行的非零元素个数
    uint32_t d_row_length = 0;
    for (auto i = 0; i < h_mat.n_row; i++)
    {
        d_row_length = std::max(d_row_length, h_mat.row_ptr[i + 1] - h_mat.row_ptr[i]);
    }
    
    float *padded_mat_elems = new float[mat_n_row * d_row_length];
    uint32_t *padded_mat_col_index = new uint32_t[mat_n_row * d_row_length];
    
    std::fill(padded_mat_elems, padded_mat_elems + mat_n_row * d_row_length, 0.0f);
    std::fill(padded_mat_col_index, padded_mat_col_index + mat_n_row * d_row_length, 0u);

    // 以转置的方式储存
    for (auto i = 0; i < h_mat.n_row; i++)
    {
        int count = 0;
        for (auto j = h_mat.row_ptr[i], end = h_mat.row_ptr[i + 1]; j < end; j++)
        {
            padded_mat_elems[count * mat_n_row + i] = h_mat.elems[j];
            padded_mat_col_index[count * mat_n_row + i] = h_mat.col_index[j];
            count++;
        }
    }

    uint32_t *d_mat_col_index = nullptr;
    cc(cudaMalloc(&d_mat_col_index, sizeof(uint32_t) * (mat_n_row * d_row_length)));
    cc(cudaMemcpy(d_mat_col_index, padded_mat_col_index, sizeof(uint32_t) * (mat_n_row * d_row_length), cudaMemcpyHostToDevice));

    float *d_mat_elems = nullptr;
    cc(cudaMalloc(&d_mat_elems, sizeof(float) * (mat_n_row * d_row_length)));
    cc(cudaMemcpy(d_mat_elems, padded_mat_elems, sizeof(float) * (mat_n_row * d_row_length), cudaMemcpyHostToDevice));

    delete[] padded_mat_col_index;
    delete[] padded_mat_elems;

    uint32_t d_x_n = h_x.n;
    float *d_x_elems = nullptr;
    cc(cudaMalloc(&d_x_elems, sizeof(float) * h_x.n));
    cc(cudaMemcpy(d_x_elems, h_x.elems, sizeof(float) * h_x.n, cudaMemcpyHostToDevice));

    uint32_t d_result_n = h_result.n;
    float *d_result_elems = nullptr;
    cc(cudaMalloc(&d_result_elems, sizeof(float) * d_result_n));

    kernel_spMV_ELL<<<(h_mat.n_row - 1) / BLOCK_SIZE + 1, BLOCK_SIZE >>>(h_mat.n_row, d_row_length, d_mat_elems, d_mat_col_index, d_x_elems, d_result_elems);

    cc(cudaDeviceSynchronize());

    cc(cudaMemcpy(h_result.elems, d_result_elems, sizeof(float) * d_result_n, cudaMemcpyDeviceToHost));

    cc(cudaFree(d_mat_elems));
    cc(cudaFree(d_mat_col_index));
    cc(cudaFree(d_x_elems));
    cc(cudaFree(d_result_elems));

    return h_result;
}

Vector cpu_spMV(const SparseMat_CSR& mat, const Vector& x)
{
    Vector result(mat.n_row);
    std::fill(result.elems, result.elems + mat.n_row, 0.0f);

    for (auto i = 0; i < mat.n_row; i++)
    {
        float sum = 0.0f;
        
        for (auto j = mat.row_ptr[i], end = mat.row_ptr[i + 1]; j < end; j++)
        {
            sum += mat.elems[j] * x.elems[mat.col_index[j]];
        }

        result.elems[i] = sum;
    }

    return std::move(result);
}

bool valid(const float* h_result, const float* d_result, const size_t len)
{
    bool is_valid = true;

    for (auto i = 0; i < len; i++)
    {
        auto delta = h_result[i] - d_result[i];
        delta = delta > 0 ? delta : -delta;
        if (delta > 1e-5)
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
    //Vector v(4);

    //for (auto i = 0; i < v.n; i++)
    //{
    //    v.elems[i] = 1.0;
    //}

    //SparseMat_CSR mat(4, 4, 7);
    //mat.elems[0] = 3;
    //mat.elems[1] = 1;
    //mat.elems[2] = 2;
    //mat.elems[3] = 4;
    //mat.elems[4] = 1;
    //mat.elems[5] = 1;
    //mat.elems[6] = 1;

    //mat.col_index[0] = 0;
    //mat.col_index[1] = 2;
    //mat.col_index[2] = 1;
    //mat.col_index[3] = 2;
    //mat.col_index[4] = 3;
    //mat.col_index[5] = 0;
    //mat.col_index[6] = 3;
    //
    //mat.row_ptr[0] = 0;
    //mat.row_ptr[1] = 2;
    //mat.row_ptr[2] = 2;
    //mat.row_ptr[3] = 5;
    //mat.row_ptr[4] = 7;

    const uint32_t n_row = 3000;
    const uint32_t n_col = 2000;

    std::default_random_engine rd;
    std::normal_distribution<double> normal_dis(0, 3);
    std::uniform_real_distribution<double> uniform_dis(0.0, 1.0);

    std::vector<float> elems;
    std::vector<uint32_t> col_index;
    std::vector<uint32_t> row_ptr;

    uint32_t count = 0;

    row_ptr.emplace_back(count);

    for (auto i = 0; i < n_row; i++)
    {
        for (auto j = 0; j < n_col; j++)
        {
            // 分别用于均匀分布和不均匀分布
            //if (((i % 15 == 4 ? 0.1 : 0) + 1.0) * normal_dis(rd) > 7.0)
            if (normal_dis(rd) > 7.0)
            {
                elems.emplace_back(uniform_dis(rd));
                col_index.emplace_back(j);
                count++;
            }
        }

        row_ptr.emplace_back(count);
    }

    Vector v(n_row);
    SparseMat_CSR mat(n_row, n_col, count);

    std::copy(elems.begin(), elems.end(), mat.elems);
    std::copy(col_index.begin(), col_index.end(), mat.col_index);
    std::copy(row_ptr.begin(), row_ptr.end(), mat.row_ptr);

    for (auto i = 0; i < n_row; i++)
    {
        v.elems[i] = uniform_dis(rd);
    }

    printf("Row: %d    Col: %d    N: %d\n", n_row, n_col, count);

    auto h_result = cpu_spMV(mat, v);
    auto d_result_CSR = gpu_spMV_CSR(mat, v);
    auto d_result_ELL = gpu_spMV_ELL(mat, v);

    std::cout << "CSR: ";
    valid(h_result.elems, d_result_CSR.elems, n_row);

    std::cout << "ELL: ";
    valid(h_result.elems, d_result_ELL.elems, n_row);

    //for (int i = 0; i < h_result.n; i++)
    //{
    //    printf("%f ", h_result.elems[i]);
    //}

    //printf("\n");

    //for (int i = 0; i < h_result.n; i++)
    //{
    //    printf("%f ", d_result_CSR.elems[i]);
    //}

    //printf("\n");

    //for (int i = 0; i < h_result.n; i++)
    //{
    //    printf("%f ", d_result_ELL.elems[i]);
    //}

    //printf("\n");

    return 0;
}