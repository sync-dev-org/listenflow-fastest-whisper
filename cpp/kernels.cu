// TensorRT calls each projection with 128 threads and ROWS output rows per block.
#include <cuda_fp16.h>

__device__ __forceinline__ float half_round(float value)
{
    return __half2float(__float2half_rn(value));
}

#ifdef BLOCK
extern "C" __global__ void bias_fp16(
    half const* x_ptr, half const* bias_ptr, half* out_ptr, int count)
{
    // Unsigned indexing also covers INT32_MAX without signed overflow.
    unsigned int offset = blockIdx.x * BLOCK + threadIdx.x;
    if (offset < static_cast<unsigned int>(count))
        out_ptr[offset] = __float2half_rn(__half2float(x_ptr[offset]) + __half2float(*bias_ptr));
}
#endif

#if defined(WIDTH) && defined(ROWS) && defined(DEPTH)
static_assert(ROWS == 4 && DEPTH % 128 == 0 && (DEPTH & (DEPTH - 1)) == 0 && WIDTH <= DEPTH,
    "The plugin launches four warps with a power-of-two padded reduction");

__device__ __forceinline__ float dot_row(half const* x, half const* weight, int row, int output)
{
    constexpr int values_per_lane = DEPTH / 32;
    float products[values_per_lane];
    int lane = threadIdx.x % 32;
    // Each warp owns one row. Four adjacent half values per lane allow vector
    // loads while keeping all products exact in fp32 before the balanced sum.
    #pragma unroll
    for (int i = 0; i < values_per_lane; ++i) {
        int column = (i / 4) * 128 + lane * 4 + i % 4;
        products[i] = column < WIDTH && row < output
            ? __half2float(x[column]) * __half2float(weight[row * WIDTH + column]) : 0.0f;
    }
    #pragma unroll
    for (int stride = values_per_lane / 2; stride > 0; stride /= 2) {
        #pragma unroll
        for (int i = 0; i < stride; ++i)
            products[i] += products[i + stride];
    }
    float result = products[0];
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        result += __shfl_down_sync(0xffffffffU, result, offset);
    return result;
}

#ifndef OUTPUT
extern "C" __global__ void qkv_fp16(half const* x_ptr, half const* weight_ptr,
    half const* q_bias_ptr, half const* v_bias_ptr, half* out_ptr)
{
    int row = blockIdx.x * ROWS + threadIdx.x / 32;
    float result = dot_row(x_ptr, weight_ptr, row, 3 * WIDTH);
    if (threadIdx.x % 32 == 0 && row < 3 * WIDTH) {
        float q = row < WIDTH ? __half2float(q_bias_ptr[row]) : 0.0f;
        float v = row >= 2 * WIDTH ? __half2float(v_bias_ptr[row - 2 * WIDTH]) : 0.0f;
        out_ptr[row] = __float2half_rn((result + q) + v);
    }
}
#else
extern "C" __global__ void fc1_fp16(
    half const* x_ptr, half const* weight_ptr, half const* bias_ptr, half* out_ptr)
{
    int row = blockIdx.x * ROWS + threadIdx.x / 32;
    float dot = dot_row(x_ptr, weight_ptr, row, OUTPUT);
    if (threadIdx.x % 32 == 0 && row < OUTPUT) {
        // Every edge of the adopted fp16 graph rounds independently, including
        // the dot before bias and the fp16 constant used to scale erf's input.
        float value = half_round(half_round(dot) + __half2float(bias_ptr[row]));
        float scaled = half_round(value * 0.70703125f);
        float erf_value = half_round(erff(scaled));
        float gate = half_round(1.0f + erf_value);
        float half_value = half_round(value * 0.5f);
        out_ptr[row] = __float2half_rn(half_value * gate);
    }
}
#endif
#endif
