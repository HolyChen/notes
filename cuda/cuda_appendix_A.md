# 附录 A

## API

用于检测CUDA API调用是否正确的宏。在Debug模式下，这个宏将返回值接受为函数参数，通过lambda函数对其进行检测，如果发生错误则输出错误信息、文件及行号。

```cpp
// ------ code A.1 ------

#ifdef NDEBUG

#define cc(return_value) return_value

#else

#define cc(return_value) \
[&]() { \
    cudaError_t cudaStatus = return_value; \
    if (cudaStatus != cudaSuccess) \
    { \
        fprintf(stderr, "cuda error %d - %s\nat: (File %s : Line %d)\n\n", cudaStatus, cudaGetErrorString(cudaStatus), __FILE__, __LINE__); \
        exit(-1);  \
    }   \
        \
    return cudaStatus; \
}()

#endif // NDEBUG | for cc
```