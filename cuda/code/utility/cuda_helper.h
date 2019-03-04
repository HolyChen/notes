#ifndef _CUDA_HELPER_H_
#define _CUDA_HELPER_H_

#include <cassert>
#include <chrono>
#include <sstream>
#include <string>

#if defined(__device__)

inline cudaError_t _cuda_call(cudaError_t return_value, const char* file, size_t line)
{
    cudaError_t cudaStatus = return_value;
    if (cudaStatus != cudaSuccess)
    {
        fprintf(stderr, "\n\n    CUDA ERROR %d - %s\nin %s : (%d)\n\n", cudaStatus, cudaGetErrorString(cudaStatus), file, line);
        exit(-1);
    }
    return cudaStatus;
}

#define cc(return_value) _cuda_call(return_value, __FILE__, __LINE__)

#define cec() cc(cudaPeekAtLastError())

#endif // __CUDA_ARCH__



template <typename Duration>
constexpr const char* duration_name();

// Time counter
// Args:
//   Clock: clock type, default = std::chrono::steady_clock
template <typename Clock = std::chrono::steady_clock>
struct TimeCounter final
{
    using TimePoint = typename Clock::time_point;

    TimePoint start_tp = Clock::now();

    TimeCounter() = default;

    template <typename Duration = typename std::chrono::microseconds, typename Rep = float>
    inline std::string tell() const
    {
        std::chrono::duration<Rep, typename Duration::period> diff = Clock::now() - start_tp;

        std::stringstream ss;
        ss << diff.count() << duration_name<Duration>() << std::endl;

        return ss.str();
    }

    inline void reset()
    {
        start_tp = std::chrono::steady_clock::now();
    }
};

template <>
constexpr const char* duration_name<std::chrono::hours>()
{
    return "h";
}

template <>
constexpr const char* duration_name<std::chrono::minutes>()
{
    return "min";
}

template <>
constexpr const char* duration_name<std::chrono::seconds>()
{
    return "s";
}

template <>
constexpr const char* duration_name<std::chrono::milliseconds>()
{
    return "ms";
}

template <>
constexpr const char* duration_name<std::chrono::microseconds>()
{
    return "us";
}

template <>
constexpr const char* duration_name<std::chrono::nanoseconds>()
{
    return "ns";
}

#endif // !_CUDA_HELPER_H_
