# Chapter 4 - 内存

## 设备内存

CUDA运行时为主机和设备的内存管理提供了一组接口，包括内存分配、释放、数据传输等。设备内存既可以被分配为**线性内存**(*linear memory*)，也可以被分配为**CUDA 数组**(*CUDA arrays*)。其中CUDA数组是不透明内存布局，对纹理获取(texture fetching)有着特殊的优化。

线性内存位于设备中的40位地址空间中，可以通过指针进行访问，因此可以实现二叉树等链式结构。它通常通过`cudaMalloc`进行分配，`cudaFree`进行释放，而数据的传输可以通过`cudaMemcpy`进行。

在此前计算向量加法的程序中，这些函数被多次用到。

```cpp
// ------ code 4.1 ------

// kernel defination...

int main()
{
    // ...

    // allocate for arrays on device
    cudaMalloc(&A, sizeof(float) * N);
    cudaMalloc(&B, sizeof(float) * N);
    cudaMalloc(&C, sizeof(float) * N);

    // copy memory from host to device
    cudaMemcpy(A, host_A, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(B, host_B, sizeof(float) * N, cudaMemcpyHostToDevice);

    // ...

    // copy result from device to host
    cudaMemcpy(host_C, C, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // deallocate
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    // ...

    return 0;
}
```

线性内存也可以通过`cudaMallocPitch`和`cudaMalloc3d`进行分配。建议将这两个函数用于分配2D或3D数组，这样可以保证分配的数组的满足[Device Memory Accesses](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses)中的对齐需求，以及让行取值或是2D内存与其他设备内存间的复制(`cudaMemcpy2D`，`cudaMemcpy3D`)获得最好性能。

