# Chapter 03 - 编程模型

References: [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-selection)


## Kernel
用户可以编写称作*kernel*的函数，来定义在设备上运行的并行代码。kernel与C语言中的函数类似，可以通过*kernel call*的语法`<<<...>>>`进行调用，其中`<<<...>>>`中的内容称为**执行配置**(*[Execution Configuration](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration)*)。

例如：
```cpp
// ------ code 3.1 ------

// Kernel definition
__global__ void VecAdd(float* A, float* B, float* C)
{
    int i = threadIdx.x;
    C[i] = A[i] + B[i];
}

int main()
{
    ...
    // Kernel invocation with N threads
    VecAdd<<<1, N>>>(A, B, C);
    ...
}
```
上述代码中，主函数`main`中调用了名为`VecAdd`的kernel用于执行向量加法，```<<<...>>>```中的含义为使用`1`个**线程块**(*thread block*)，每个线程块有`N`个线程。

在kernel中，`threadIdx`是CUDA C语言扩展中的一个内置变量，表示在一个线程块内的线程在`x,y,z`三个维度的`id`，即一个三维`unsigned int`型向量。为了方便在IDE中使用，可以在编写程序时包含头文件`<device_launch_parameters.h>`。

kernel前的`__global__`是一个**函数执行空间限定符**(*[Function Execution Space Specifiers](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#function-declaration-specifiers)*)，用于将函数声明为一个kernel，它表示这个函数在设备中运行，并且，它既可被主机调用，也可以被设备调用。被声明为`__global__`的函数的返回值必须为`void`，并且不可以作为类的成员函数。

以下为CUDA 10.0中所有的函数执行空间限定符：

| FESS | 说明  |
|---|---|
| `__device__`  |  1. 在设备上运行；<br/> 2. 仅可以由设备调用；<br/> 3. `__global__`和`__device__`不可以被用于同一个函数|
| `__global__` | 1. 在设备上运行；<br/> 2. 可以由主机调用； <br/> 3. 在计算能力3.2及以上的设备中，可以由设备调用； <br/> 4. 必须返回`void`且不可以作为类的成员函数； <br/> 5. 任何对`__global__`函数的调用都必须指定它的执行配置；<br/> 6. 对于`__global__`函数的调用是**异步**的，这意味着函数调用会在设备执行完成前返回。
| `__host__` | 1. 在主机上运行；<br/> 2. 仅可以由主机调用；<br/> 3. `__global__`和`__host__`不可以被用于同一个函数；<br/> 4. `__device__`和`__host__`可以用于同一个函数，常见于通过宏`__CUDA_ARCH__`对主机代码或设备代码进行条件编译。
| `__noinline__` <br/> `__forceinline__` | 1. 通常，如果设备编译器在编译时会将适合内联的代码自动内联；<br/> 2. `__noinline__`用于提示编译器如果可能的话，不要内联该函数； <br/> 3. `__forceinline__`强制编译器内联该函数；<br/>4. 这两个限定符不能够被同时使用，并且函数**限定符**(*qualifier*)都不可用于内联函数。|