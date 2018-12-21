# Chapter 3 - 编程模型

References: [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-selection)


## Kernel
用户可以编写称作*kernel*的函数，来定义在设备上运行的并行代码。kernel与C语言中的函数类似，可以通过*kernel call*的语法`<<<...>>>`进行调用，其中`<<<...>>>`中的内容称为**执行配置**(*[Execution Configuration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration)*)。

以下为一个代码示例，函数`vecAdd`对两个单精度浮点数数组`A`和`B`进行相加，并储存到数组`C`中，每个线程对其中一位上的数字进行相加。

```c++
// ------ code 3.1 ------

// kernel definition
__global__ void vecAdd(float* A, float* B, float* C)
{
    uint32_t i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    ...
    // kernel invocation with N threads
    vecAdd<<<1, N>>>(A, B, C);
    ...
}
```
上述代码中，主函数`main`中调用了名为`vecAdd`的kernel用于执行向量加法，```<<<...>>>```中的含义为使用`1`个**线程块**(*thread block*)，每个线程块有`N`个线程。

在kernel中，`threadIdx`是CUDA C语言扩展中的一个内置变量，称作**线程索引**(*threadIdx*)表示在一个线程块内的线程在`x,y,z`三个维度的`id`，即一个3维无符号整形向量`uint3`。为了方便在IDE中使用，可以在编写程序时包含头文件`<device_launch_parameters.h>`。

kernel前的`__global__`是一个**函数执行空间限定符**(*[Function Execution Space Specifiers](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-declaration-specifiers)*)，用于将函数声明为一个kernel，它表示这个函数在设备中运行，并且，它既可被主机调用，也可以被设备调用。被声明为`__global__`的函数的返回值必须为`void`，并且不可以作为类的成员函数。

以下为CUDA 10.0中所有的函数执行空间限定符：

| FESS | 说明  |
|---|---|
| `__device__`  |  1. 在设备上运行；<br/> 2. 仅可以由设备调用；<br/> 3. `__global__`和`__device__`不可以被用于同一个函数|
| `__global__` | 1. 在设备上运行；<br/> 2. 可以由主机调用； <br/> 3. 在计算能力3.2及以上的设备中，可以由设备调用； <br/> 4. 必须返回`void`且不可以作为类的成员函数； <br/> 5. 任何对`__global__`函数的调用都必须指定它的执行配置；<br/> 6. 对于`__global__`函数的调用是**异步**的，这意味着函数调用会在设备执行完成前返回。
| `__host__` | 1. 在主机上运行；<br/> 2. 仅可以由主机调用；<br/> 3. `__global__`和`__host__`不可以被用于同一个函数；<br/> 4. `__device__`和`__host__`可以用于同一个函数，常见于通过宏`__CUDA_ARCH__`对主机代码或设备代码进行条件编译。
| `__noinline__` <br/> `__forceinline__` | 1. 通常，如果设备编译器在编译时会将适合内联的代码自动内联；<br/> 2. `__noinline__`用于提示编译器如果可能的话，不要内联该函数； <br/> 3. `__forceinline__`强制编译器内联该函数；<br/>4. 这两个限定符不能够被同时使用，并且函数**限定符**(*qualifier*)都不可用于内联函数。|

向量加法的完整代码如下。其中，`cudaDeviceSynchronize`用于等待设备计算完成，`cudaMalloc`，`cudaMemcpy`以及`cudaFree`为内存管理相关的函数，将在[第4章](./04-Memory.md)介绍。
```c++
// ------ code 3.2 ------

#include <cstdint>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// kernel definition
__global__ void vecAdd(float* A, float* B, float* C)
{
    uint32_t i = threadIdx.x;
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
    vecAdd<<<1, N >>>(A, B, C);

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

进一步的，线程块可以组成**网格**(*Grid*)，在网格中，线程块的索引用表示为内置的三维向量`blockIdx`。对应的，网格的维度用三维向量`gridDim`表示。

线程与线程块的排布都是**列主序**(*Column Major*)的，即`x`表示在一行中的位置，`y`表示在一列中的位置。

例如：
```
Blcok (0, 0) Block(1, 0) Block(2, 0)...
Blcok (0, 1) Block(1, 1) Block(2, 1)...
```

在计算线程在整个网格中的位置`tid`时，方法如下（以二维为例）：

```c++
x = blockIdx.x * blockDim.x + threadIdx.x;
y = blockIdx.y * blockDim.y + threadIdx.y;
tid = y * blockDim.x * gridDim.x + x;
```

