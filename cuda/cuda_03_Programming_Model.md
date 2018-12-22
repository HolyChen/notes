# Chapter 3 - 编程模型

References: [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-selection)

## 异构编程

CUDA编程模型假设CUDA程序运行在一个物理分离的**设备**(*device*)上，它是运行C程序的**主机**(*host*)的**协处理器**(*coprocessor*)。最常见的情况是，主机是CPU，设备是GPU。

图heterogeneous programming给出了一个CUDA异构编程的例子。图中，带有波浪线的箭头表示线程，*kernel*是运行在设备上的程序代码。可以看到，在主机中的代码以顺序的方式执行，在调用在设备上运行的函数`kernel0`和`kernel1`后，代码在设备中以并行方式执行。

> 图 heterogeneous programming
>
> ![heterogeneous-programming.png](./resources/heterogeneous-programming.png)

CUDA编程模型假设主机和设备独立维护各自的内存空间，称为**主机内存**(*host memory*)和**设备内存**(*device memory*)。程序通过调用CUDA运行时函数对设备内存进行管理，包括内存分配、释放以及数据传输等。

*Unified Memory*提供了桥接主机、设备内存空间的**托管内存**(*managed memory*)，这使得用户可以使用一个具有共同地址空间的、单一、一致的内存镜像访问所有CPU和GPU上的内存空间。

## Kernel
用户可以通过编写*kernel*来实现设备上并行运行的函数，其语法与C语言中的函数类似，可以通过kernel call的语法`<<<...>>>`进行调用，其中`<<<...>>>`中的内容称为**执行配置**(*[Execution Configuration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration)*)。

以下为一个代码示例，函数`vec_add`对两个单精度浮点数数组`A`和`B`进行相加，并储存到数组`C`中，每个线程对其中一位上的数字进行相加。

```c++
// ------ code 3.1 ------

// kernel definition
__global__ void vec_add(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    ...
    // kernel invocation with N threads
    vec_add<<<1, N>>>(A, B, C);
    ...
}
```
上述代码中，主函数`main`中调用了名为`vec_add`的kernel用于执行向量加法，```<<<...>>>```中的含义为使用`1`个**线程块**(*thread block*)，每个线程块有`N`个线程。

在kernel中，`threadIdx`是CUDA C语言扩展中的一个内置变量，称作**线程索引**(*threadIdx*)表示在一个线程块内的线程在`x,y,z`三个维度的`id`，即一个3维无符号整形向量`uint3`。为了方便在IDE中使用，可以在编写程序时包含头文件`<device_launch_parameters.h>`。

kernel前的`__global__`是一个**函数执行空间限定符**(*[Function Execution Space Specifiers](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-declaration-specifiers)*)，用于将函数声明为一个kernel，它表示这个函数在设备中运行，并且，它既可被主机调用，也可以被设备调用。被声明为`__global__`的函数的返回值必须为`void`，并且不可以作为类的成员函数。

以下为CUDA 10.0中所有的函数执行空间限定符：

| FESS | 说明  |
|---|---|
| `__device__`  |  1. 在设备上运行；<br/> 2. 仅可以由设备调用；<br/> 3. `__global__`和`__device__`不可以被用于同一个函数|
| `__global__` | 1. 在设备上运行；<br/> 2. 可以由主机调用； <br/> 3. 在计算能力3.2及以上的设备中，可以由设备调用； <br/> 4. 必须返回`void`且不可以作为类的成员函数； <br/> 5. 任何对`__global__`函数的调用都必须指定它的执行配置；<br/> 6. 对于`__global__`函数的调用是**异步**的，这意味着函数调用会在设备执行完成前返回。
| `__host__` | 1. 在主机上运行；<br/> 2. 仅可以由主机调用；<br/> 3. `__global__`和`__host__`不可以被用于同一个函数；<br/> 4. `__device__`和`__host__`可以用于同一个函数，常见于通过宏`__CUDA_ARCH__`对主机代码或设备代码进行条件编译。
| `__noinline__` <br/> `__forceinline__` | 1. 通常，如果设备编译器在编译时会将适合内联的代码自动内联；<br/> 2. `__noinline__`用于提示编译器如果可能的话，不要内联该函数； <br/> 3. `__forceinline__`强制编译器内联该函数；<br/>4. 这两个限定符不能够被同时使用，并且函数**限定符**(*qualifier*)都不可用于内联函数。|

向量加法的完整代码如下。其中，`cudaDeviceSynchronize`用于等待设备计算完成，`cudaMalloc`，`cudaMemcpy`以及`cudaFree`为内存管理相关的函数，分别是为设备中的变量分配空间、在设备和主机间进行数据复制，释放设备中分配的内存空间，这些函数将在[Chapter 4](./cuda_04_Memory.md)详细说明。
```c++
// ------ code 3.2 ------

#include <cstdint>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// kernel definition
__global__ void vec_add(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}


int main()
{
    const int N = 5;
    float host_A[N] = { 0., 1., 2., 3., 4. };
    float host_B[N] = { 9., 8., 7., 6., 5. };
    float host_C[N];

    float *A, *B, *C;

    // malloc for arrays on device
    cudaMalloc(&A, sizeof(float) * N);
    cudaMalloc(&B, sizeof(float) * N);
    cudaMalloc(&C, sizeof(float) * N);

    // copy memory from host to device
    cudaMemcpy(A, host_A, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(B, host_B, sizeof(float) * N, cudaMemcpyHostToDevice);

    // kernel invocation with N threads
    vec_add<<<1, N >>>(A, B, C);

    // waiting until device completes
    cudaDeviceSynchronize();

    // copy result from device to host
    cudaMemcpy(host_C, C, sizeof(float) * N, cudaMemcpyDeviceToHost);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    std::cout << "A: ";
    for (int i = 0; i < N; i++)
    {
        std::cout << host_A[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "B: ";
    for (int i = 0; i < N; i++)
    {
        std::cout << host_B[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "C: ";
    for (int i = 0; i < N; i++)
    {
        std::cout << host_C[i] << " ";
    }
    std::cout << std::endl;

    // output should be:
    // A: 0 1 2 3 4
    // B: 9 8 7 6 5
    // C: 9 9 9 9 9

    return 0;
}
```

## 线程层次

前面提到，线程`threadIdx`是一个3维向量，通过设置其维度，可以构成一维、二维或三维的线程块，这提供了一种自然的方式用于计算向量、矩阵或是张量。

线程块的维度用内置变量`blockDim`表示，这是一个三维的无符号整形向量，其类型为`dim3`，也具有`x`，`y`和`z`三个分量，表示线程块对应维度有几个线程。因此，在一个线程块中，`threadIdx={x, y, z}`的线程的是线程块中第`threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x`个线程。

进一步的，线程块可以组成**网格**(*grid*)，在网格中，线程块的索引用表示为内置的三维向量`blockIdx`。对应的，网格的维度用三维向量`gridDim`表示。

线程与线程块的排布都是**列主序**(*column Major*)的，即`x`表示在一行中的位置，`y`表示在一列中的位置。如图grid of thread blocks所示

> 图 grid of thread blocks
>
> ![grid-of-thread-blocks.png](./resources/grid-of-thread-blocks.png)

在计算线程在整个网格中的位置`tid`时，方法如下（以二维为例）：

```c++
x = blockIdx.x * blockDim.x + threadIdx.x;
y = blockIdx.y * blockDim.y + threadIdx.y;
tid = y * blockDim.x * gridDim.x + x;
```

线程块的执行必须要有独立性，即彼此间没有执行顺序的依赖。它们即应能顺序执行，也能并行执行。这允许线程块可以被以任何顺序、任意核心数量进行调度。

同属一个线程块内的线程可以通过**共享内存**(*shared Memory*)进行数据共享，也可以通过`__syncthreads()`指令函数进行同步。当块内的线程调用`__syncthreads()`时，它会被阻塞直到块内所有的线程均运行至此。为了线程间能够高效的协作，共享内存有望是靠近处理器核心的低延迟内存，例如L1缓存，`__synthreads__`也有望是轻量级的。共享内存及线程同步的使用将在[Chapter 4](./cuda_04_Memory.md)看到。

这里一个给出使用二维线程计算矩阵加法的例子。

```cpp
// ------ code 3.3 ------

#include <cstdint>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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

    float host_A[M][N];
    float host_B[M][N];
    float host_C[M][N];

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            host_A[i][j] = -i * N - j;
            if (i == j)
            {
                host_B[i][j] = i * N + j + 1;
            }
            else
            {
                host_B[i][j] = i * N + j;
            }
        }
    }

    float *A, *B, *C;

    // malloc for arrays on device
    cudaMalloc(&A, sizeof(float) * M * N);
    cudaMalloc(&B, sizeof(float) * M * N);
    cudaMalloc(&C, sizeof(float) * M * N);

    // copy memory from host to device
    cudaMemcpy(A, host_A, sizeof(float) * M * N, cudaMemcpyHostToDevice);
    cudaMemcpy(B, host_B, sizeof(float) * M * N, cudaMemcpyHostToDevice);

    dim3 num_of_block = { 4, 3 };
    dim3 thread_per_block = { 4, 8 };

    // kernel invocation with N threads
    mat_add<M,N><<<num_of_block, thread_per_block >>>(A, B, C);

    // waiting until device completes
    cudaDeviceSynchronize();

    // copy result from device to host
    cudaMemcpy(host_C, C, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    std::cout << "A:\n";
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << host_A[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "B:\n";
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << host_B[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "C:\n";
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << host_C[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Output of matrix C:
    // 1 if on main diagonal
    // 0 otherwise

    return 0;
}
```

在kernel中，可以看到，矩阵的行由`y`坐标计算得到，列则有`x`坐标计算得到。这里为了简化程序，使用一维数组替代了二维数组。在[Chapter 4](./cuda_04_Memory.md)中，将会看到直接创建二维、三维数组的例子。

## 内存层次

CUDA的每个线程都具有自己私有的局部内存；每个线程块具有共享内存，它为块内所有线程可见。所有线程都可以访问全局内存。除此之外，还有全局的只读内存空间，即常量内存空间和纹理内存空间。如图memory hierarchy所示。

> 图 memory hierarchy
>
> ![memory-hierarchy.png](./resources/memory-hierarchy.png)


详细内容可见于[设备内存访问](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)。

