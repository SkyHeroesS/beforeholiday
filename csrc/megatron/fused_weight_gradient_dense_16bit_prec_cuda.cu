#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "type_shim.h"

/* Includes, HIP */
#include <hipblaslt/hipblaslt-ext.hpp>

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
#ifndef CHECK_HIPBLASLT_ERROR
#define CHECK_HIPBLASLT_ERROR(error)                                                      \
    if(error != HIPBLAS_STATUS_SUCCESS)                                                   \
    {                                                                                     \
        fprintf(stderr, "hipBLASLt error(Err=%d) at %s:%d\n", error, __FILE__, __LINE__); \
        fprintf(stderr, "\n");                                                            \
        exit(EXIT_FAILURE);                                                               \
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
    int batch_count,
    float& alpha,
    float& beta,
    at::BFloat16* A,
    at::BFloat16* B,
    at::BFloat16* C,
    at::BFloat16* D,
    void*   d_workspace,
    int64_t  max_workspace_size,
    hipStream_t   stream) 
{
    hipblaslt_ext::GemmPreference gemmPref;
    gemmPref.setMaxWorkspaceBytes(max_workspace_size);
    hipblaslt_ext::Gemm gemm(
        handle, transa, transb, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIPBLAS_COMPUTE_32F);

    hipblaslt_ext::GemmEpilogue
        epilogue; // No action needed, default is HIPBLASLT_EPILOGUE_DEFAULT. (Gemm only)
    hipblaslt_ext::GemmInputs inputs;
    inputs.a     = A;
    inputs.b     = B;
    inputs.c     = C;
    inputs.d     = D;
    inputs.alpha = &alpha;
    inputs.beta  = &beta;
    gemm.setProblem(m, n, k, batch_count, epilogue, inputs);

    const int                                     request_solutions = 1;
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    CHECK_HIPBLASLT_ERROR(gemm.algoGetHeuristic(request_solutions, gemmPref, heuristicResult));

    if(heuristicResult.empty())
    {
        std::cerr << "No valid solution found!" << std::endl;
        return;
    }

    // In this sample, the workspace is already allocated with max_workspace_size
    // If not, calculate the needed workspace_size and allocate d_workspace here
    // uint64_t workspace_size = 0;
    // for(int i = 0; i < returnedAlgoCount; i++)
    //     workspace_size = max(workspace_size, heuristicResult[i].workspaceSize);
    // CHECK_HIP_ERRORhipMalloc(&d_workspace, workspace_size));

    // Make sure to initialize every time when algo changes
    CHECK_HIPBLASLT_ERROR(gemm.initialize(heuristicResult[0].algo, d_workspace));
    CHECK_HIPBLASLT_ERROR(gemm.run(stream));
    return;
}

// FP16 inputs and FP16 accumulation
void gemmex_wrapper_fp16(
    hipblasLtHandle_t handle,
    hipblasOperation_t transa,
    hipblasOperation_t transb,
    int m,
    int n,
    int k,
    int batch_count,
    float& alpha,
    float& beta,
    at::Half* A,
    at::Half* B,
    at::Half* C,
    at::Half* D,
    void*   d_workspace,
    int64_t  max_workspace_size,
    hipStream_t   stream) 
{
       hipblaslt_ext::GemmPreference gemmPref;
    gemmPref.setMaxWorkspaceBytes(max_workspace_size);
    hipblaslt_ext::Gemm gemm(
        handle, transa, transb, HIP_R_16F, HIP_R_16F, HIP_R_16F, HIP_R_16F, HIPBLAS_COMPUTE_32F);

    hipblaslt_ext::GemmEpilogue
        epilogue; // No action needed, default is HIPBLASLT_EPILOGUE_DEFAULT. (Gemm only)
    hipblaslt_ext::GemmInputs inputs;
    inputs.a     = A;
    inputs.b     = B;
    inputs.c     = C;
    inputs.d     = D;
    inputs.alpha = &alpha;
    inputs.beta  = &beta;
    gemm.setProblem(m, n, k, batch_count, epilogue, inputs);

    const int                                     request_solutions = 1;
    std::vector<hipblasLtMatmulHeuristicResult_t> heuristicResult;
    CHECK_HIPBLASLT_ERROR(gemm.algoGetHeuristic(request_solutions, gemmPref, heuristicResult));

    if(heuristicResult.empty())
    {
        std::cerr << "No valid solution found!" << std::endl;
        return;
    }

    // In this sample, the workspace is already allocated with max_workspace_size
    // If not, calculate the needed workspace_size and allocate d_workspace here
    // uint64_t workspace_size = 0;
    // for(int i = 0; i < returnedAlgoCount; i++)
    //     workspace_size = max(workspace_size, heuristicResult[i].workspaceSize);
    // CHECK_HIP_ERRORhipMalloc(&d_workspace, workspace_size));

    // Make sure to initialize every time when algo changes
    CHECK_HIPBLASLT_ERROR(gemm.initialize(heuristicResult[0].algo, d_workspace));
    CHECK_HIPBLASLT_ERROR(gemm.run(stream));
    return;
}


//hipblasLtHandle_t g_hipblas_handle = nullptr;

template <typename T>
void wgrad_gemm_accum_fp16_cuda(T *input, T *d_output,  T *dc_tensor, T *d_weight,int in_dim, int hidden_dim, int out_dim) {
    //cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    //if(g_hipblas_handle == nullptr)
    //   CHECK_HIPBLAS_ERROR(hipblasCreate(&g_hipblas_handle));
    hipblasLtHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    hipStream_t stream;
    hipblasGetStream(handle, &stream);
    float alpha = 1.0;
    float beta  = 1.0;
    const int batch_count = 1;
    void*   d_workspace;
    int64_t max_workspace_size = 32*1024*1024;
    if(max_workspace_size > 0)
        CHECK_HIP_ERROR(hipMalloc(&d_workspace, max_workspace_size));
    gemmex_wrapper_fp16(
        handle,
        HIPBLAS_OP_N,
        HIPBLAS_OP_T,
        in_dim,      //m
        out_dim,     //n
        hidden_dim,  //k
        batch_count,
        alpha,
        beta,
        input,      //da
        d_output,   //db
        dc_tensor,  //dc
        d_weight,   //dd
        d_workspace,
        max_workspace_size,
        stream);
} 

template void wgrad_gemm_accum_fp16_cuda<at::Half>(at::Half *input, at::Half *d_output,  at::Half *dc_tensor, at::Half *d_weight, int in_dim, int hidden_dim, int out_dim);
template void wgrad_gemm_accum_fp16_cuda<at::BFloat16>(at::BFloat16 *input, at::BFloat16 *d_output, at::BFloat16 *dc_tensor, at::BFloat16 *d_weight,  int in_dim, int hidden_dim, int out_dim);

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

    at::Tensor dc_tensor = at::empty_like(d_weight);
    dc_tensor.copy_(d_weight);
    //at::Tensor dst_tensor = at::zeros_like(d_weight);

    const int hidden_dim = input_2d.size(0);  //k
    const int in_dim = input_2d.size(1);   //m
    const int out_dim = d_weight.size(0);  //n

    DISPATCH_HALF_AND_BFLOAT(input_2d.scalar_type(), "wgrad_gemm_accum_fp16",
        wgrad_gemm_accum_fp16_cuda<scalar_t>(
            input_2d.data_ptr<scalar_t>(),
            d_output_2d.data_ptr<scalar_t>(),
            dc_tensor.data_ptr<scalar_t>(),
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

    const int hidden_dim = input_2d.size(0); //k
    const int in_dim = input_2d.size(1); //m
    const int out_dim = d_weight.size(0);  //n

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