#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <torch/extension.h>
#include <ATen/ATen.h>
//#include <ATen/cuda/CUDAContext.h>

#include <ATen/hip/HIPContext.h>

/* Includes, hip */
#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>

#include "type_shim_hip.h"
#include "type_shim.h"
#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                    \
    if(error != hipSuccess)                       \
    {                                             \
        fprintf(stderr,                           \
                "Hip error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),         \
                error,                            \
                __FILE__,                         \
                __LINE__);                        \
        exit(EXIT_FAILURE);                       \
    }
#endif
#ifndef CHECK_HIPBLAS_ERROR
#define CHECK_HIPBLAS_ERROR(error)                              \
if(error != HIPBLAS_STATUS_SUCCESS)                         \
{                                                           \
    fprintf(stderr, "hipBLAS error: ");                     \
    if(error == HIPBLAS_STATUS_NOT_INITIALIZED)             \
        fprintf(stderr, "HIPBLAS_STATUS_NOT_INITIALIZED");  \
    if(error == HIPBLAS_STATUS_ALLOC_FAILED)                \
        fprintf(stderr, "HIPBLAS_STATUS_ALLOC_FAILED");     \
    if(error == HIPBLAS_STATUS_INVALID_VALUE)               \
        fprintf(stderr, "HIPBLAS_STATUS_INVALID_VALUE");    \
    if(error == HIPBLAS_STATUS_MAPPING_ERROR)               \
        fprintf(stderr, "HIPBLAS_STATUS_MAPPING_ERROR");    \
    if(error == HIPBLAS_STATUS_EXECUTION_FAILED)            \
        fprintf(stderr, "HIPBLAS_STATUS_EXECUTION_FAILED"); \
    if(error == HIPBLAS_STATUS_INTERNAL_ERROR)              \
        fprintf(stderr, "HIPBLAS_STATUS_INTERNAL_ERROR");   \
    if(error == HIPBLAS_STATUS_NOT_SUPPORTED)               \
        fprintf(stderr, "HIPBLAS_STATUS_NOT_SUPPORTED");    \
    if(error == HIPBLAS_STATUS_INVALID_ENUM)                \
        fprintf(stderr, "HIPBLAS_STATUS_INVALID_ENUM");     \
    if(error == HIPBLAS_STATUS_UNKNOWN)                     \
        fprintf(stderr, "HIPBLAS_STATUS_UNKNOWN");          \
    fprintf(stderr, "\n");                                  \
    exit(EXIT_FAILURE);                                     \
}
#endif


// BF16 inputs and BF16 accumulation
void gemmex_wrapper_fp16(
    hipblasHandle_t handle,
    hipblasOperation_t transa,
    hipblasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha,
    at::BFloat16* A,
    int lda,
    at::BFloat16* B,
    int ldb,
    const float* beta,
    at::BFloat16* C,
    int ldc) {
    //std::cout << "bf16 bf16" << std::endl;
    CHECK_HIPBLAS_ERROR(hipblasGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      HIP_R_16BF,
      lda,
      B,
      HIP_R_16BF,
      ldb,
      beta,
      C,
      HIP_R_16BF,
      ldc,
      HIPBLAS_COMPUTE_32F,
      HIPBLAS_GEMM_DEFAULT));
}

// FP16 inputs and FP16 accumulation
void gemmex_wrapper_fp16(
    //cublasHandle_t handle,
    //cublasOperation_t transa,
    //cublasOperation_t transb,
    hipblasHandle_t handle,
    hipblasOperation_t transa,
    hipblasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha,
    at::Half* A,
    int lda,
    at::Half* B,
    int ldb,
    const float* beta,
    at::Half* C,
    int ldc) {
    //std::cout << "fp16 fp16" << std::endl;
    CHECK_HIPBLAS_ERROR(hipblasGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      //CUDA_R_16F,
      HIP_R_16F,
      lda,
      B,
      //CUDA_R_16F,
      HIP_R_16F,
      ldb,
      beta,
      C,
      HIP_R_16F,
      ldc,
      HIPBLAS_COMPUTE_32F,
      HIPBLAS_GEMM_DEFAULT));
}

hipblasHandle_t g_hipblas_handle = nullptr;

template <typename T>
void wgrad_gemm_accum_fp16_cuda(T *input, T *d_output, T *d_weight, int in_dim, int hidden_dim, int out_dim) {
    //cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    if(g_hipblas_handle == nullptr)
        CHECK_HIPBLAS_ERROR(hipblasCreate(&g_hipblas_handle));
    hipStream_t stream;
    hipblasGetStream(g_hipblas_handle, &stream);
    const float alpha = 1.0;
    const float beta  = 1.0;
    gemmex_wrapper_fp16(
        g_hipblas_handle,
        //CUBLAS_OP_N,
        //CUBLAS_OP_T,
        HIPBLAS_OP_N,
        HIPBLAS_OP_T,
        in_dim,
        out_dim,
        hidden_dim,
        &alpha,
        input,
        in_dim,
        d_output,
        out_dim,
        &beta,
        d_weight,
        in_dim);
} 

template void wgrad_gemm_accum_fp16_cuda<at::Half>(at::Half *input, at::Half *d_output, at::Half *d_weight, int in_dim, int hidden_dim, int out_dim);
template void wgrad_gemm_accum_fp16_cuda<at::BFloat16>(at::BFloat16 *input, at::BFloat16 *d_output, at::BFloat16 *d_weight, int in_dim, int hidden_dim, int out_dim);

void wgrad_gemm_accum_fp16_cuda_stub(
  at::Tensor &input,
  at::Tensor &d_output,
  at::Tensor &d_weight
) {
    at::Tensor input_2d, d_output_2d;
    // input tensor: collapse to the first dim
    auto in_sizes = input.sizes();
    if (input.dim() > 2) {
        input_2d = input.view({-1, in_sizes[in_sizes.size() - 1]});
    } else {
        input_2d = input;
    }
    // d_output tensor: collapse to the first dim
    auto d_out_sizes = d_output.sizes();
    if (d_output.dim() > 2) {
        d_output_2d = d_output.view({-1, d_out_sizes[d_out_sizes.size() - 1]});
    } else {
        d_output_2d = d_output;
    }

    const int hidden_dim = input_2d.size(0);
    const int in_dim = input_2d.size(1);
    const int out_dim = d_weight.size(0);

    DISPATCH_HALF_AND_BFLOAT(input_2d.scalar_type(), "wgrad_gemm_accum_fp16",
        wgrad_gemm_accum_fp16_cuda<scalar_t>(
            input_2d.data_ptr<scalar_t>(),
            d_output_2d.data_ptr<scalar_t>(),
            d_weight.data_ptr<scalar_t>(),
            in_dim,
            hidden_dim,
            out_dim);
    );
}
/*
// BF16 inputs and BF16 accumulation
void gemmex_wrapper_fp16(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha,
    at::BFloat16* A,
    int lda,
    at::BFloat16* B,
    int ldb,
    const float* beta,
    at::BFloat16* C,
    int ldc) {
  TORCH_CUDABLAS_CHECK(cublasGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      CUDA_R_16BF,
      lda,
      B,
      CUDA_R_16BF,
      ldb,
      beta,
      C,
      CUDA_R_16BF,
      ldc,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// FP16 inputs and FP16 accumulation
void gemmex_wrapper_fp16(
    cublasHandle_t handle,
    cublasOperation_t transa,
    cublasOperation_t transb,
    int m,
    int n,
    int k,
    const float* alpha,
    at::Half* A,
    int lda,
    at::Half* B,
    int ldb,
    const float* beta,
    at::Half* C,
    int ldc) {
  TORCH_CUDABLAS_CHECK(cublasGemmEx(
      handle,
      transa,
      transb,
      m,
      n,
      k,
      alpha,
      A,
      CUDA_R_16F,
      lda,
      B,
      CUDA_R_16F,
      ldb,
      beta,
      C,
      CUDA_R_16F,
      ldc,
      CUDA_R_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

template <typename T>
void wgrad_gemm_accum_fp16_cuda(T *input, T *d_output, T *d_weight, int in_dim, int hidden_dim, int out_dim) {
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cudaStream_t stream;
    cublasGetStream(handle, &stream);
    const float alpha = 1.0;
    const float beta  = 1.0;

    gemmex_wrapper_fp16(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_T,
        in_dim,
        out_dim,
        hidden_dim,
        &alpha,
        input,
        in_dim,
        d_output,
        out_dim,
        &beta,
        d_weight,
        in_dim);
} 

template void wgrad_gemm_accum_fp16_cuda<at::Half>(at::Half *input, at::Half *d_output, at::Half *d_weight, int in_dim, int hidden_dim, int out_dim);
template void wgrad_gemm_accum_fp16_cuda<at::BFloat16>(at::BFloat16 *input, at::BFloat16 *d_output, at::BFloat16 *d_weight, int in_dim, int hidden_dim, int out_dim);

void wgrad_gemm_accum_fp16_cuda_stub(
  at::Tensor &input,
  at::Tensor &d_output,
  at::Tensor &d_weight
) {
    at::Tensor input_2d, d_output_2d;
    // input tensor: collapse to the first dim
    auto in_sizes = input.sizes();
    if (input.dim() > 2) {
        input_2d = input.view({-1, in_sizes[in_sizes.size() - 1]});
    } else {
        input_2d = input;
    }
    // d_output tensor: collapse to the first dim
    auto d_out_sizes = d_output.sizes();
    if (d_output.dim() > 2) {
        d_output_2d = d_output.view({-1, d_out_sizes[d_out_sizes.size() - 1]});
    } else {
        d_output_2d = d_output;
    }

    const int hidden_dim = input_2d.size(0);
    const int in_dim = input_2d.size(1);
    const int out_dim = d_weight.size(0);

    DISPATCH_HALF_AND_BFLOAT(input_2d.scalar_type(), "wgrad_gemm_accum_fp16",
        wgrad_gemm_accum_fp16_cuda<scalar_t>(
            input_2d.data_ptr<scalar_t>(),
            d_output_2d.data_ptr<scalar_t>(),
            d_weight.data_ptr<scalar_t>(),
            in_dim,
            hidden_dim,
            out_dim);
    );
}
*/