# 附录 A

目录

- [错误检测-cc](#错误检测-cc)

## 错误检测-cc

用于检测CUDA API调用是否正确的宏。当发生错误时，将输出错误信息并退出程序。

```cpp
// ------ code A.1 ------

inline cudaError_t _cuda_call(cudaError_t return_value, const char* file, size_t line)
{
    cudaError_t cudaStatus = return_value; 
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "cuda error %d - %s\nat: (File %s : Line %d)\n\n", cudaStatus, cudaGetErrorString(cudaStatus), file, line);
        exit(-1); // Unexcepted Terminal
    }   
    return cudaStatus;
}

#define cc(return_value) _cuda_call(return_value, __FILE__, __LINE__)

```