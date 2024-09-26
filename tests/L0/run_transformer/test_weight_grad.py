import torch
import fused_weight_gradient_mlp_cuda
import torch.nn as nn
import sys
import math, pdb
from torch.testing._internal import common_utils

torch.backends.cuda.matmul.allow_tf32 = False

class TestFusedWeightGradDenseCuda(common_utils.TestCase):
    SEED: int = 123456
    configs = [
        #BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_SIZE, OUTPUT_SIZE
        (1, 14, 1531, 500),
        (1, 32, 1536, 256),
        (1, 64, 3584, 512),
        (1, 128, 4096, 1024),
        (8, 32, 1536, 256),
        (8, 64, 3584, 2048),
        (8, 128, 4096, 4096),       
    ]
    def cosin_similarity(self, vector_a, vector_b):

        vector_a = vector_a.to(dtype=torch.float64, device="cuda")
        vector_b = vector_b.to(dtype=torch.float64, device="cuda")
        dot_product = torch.dot(vector_a, vector_b)

        if torch.isinf(dot_product):
            print("Warning: Dot product is inf.")
            return 0

        norm_a = torch.norm(vector_a)
        norm_b = torch.norm(vector_b)
        return dot_product / ((norm_a * norm_b) + 0.00001)


    def backward_matrix_operation(self, alpha, beta, A, B, C):

        A_sizes = A.size()
        if A.dim() > 2:
            new_shape = (-1, A_sizes[-1])
            reshaped_A = A.view(new_shape)
        else:
            reshaped_A = A   
        
        B_sizes = B.size()
        if B.dim() > 2:
            new_shape = (-1, B_sizes[-1])
            reshaped_B = B.view(new_shape)
        else:
            reshaped_B = B  
        
        AB = reshaped_B.float().t().matmul(reshaped_A.float())
        scaled_AB = alpha * AB
        
        output_type = C.dtype
        C_temp = C.float()
        
        C_temp.data.mul_(beta)
        C_temp.data.add_(scaled_AB)

        C.copy_(C_temp).to(dtype=output_type, device="cuda")
        return C


    def RunWeightGradDense(self, ab_type, cd_type):
        for BATCH_SIZE, SEQUENCE_LENGTH, HIDDEN_SIZE, OUTPUT_SIZE in self.configs:
            print(f"BATCH_SIZE = {BATCH_SIZE}, SEQUENCE_LENGTH = {SEQUENCE_LENGTH}, HIDDEN_SIZE = {HIDDEN_SIZE}, OUTPUT_SIZE = {OUTPUT_SIZE}")
            input_shape = (SEQUENCE_LENGTH, BATCH_SIZE, HIDDEN_SIZE) # k*m
            output_shape = (SEQUENCE_LENGTH, BATCH_SIZE, OUTPUT_SIZE) # k*n
            weight_shape = (OUTPUT_SIZE, HIDDEN_SIZE)
            
            torch.manual_seed(self.SEED)

            input = torch.randn(input_shape, requires_grad=False, device="cuda") # A
            output = torch.randn(output_shape, requires_grad=False, device="cuda") #B
            weight = torch.randn(weight_shape, requires_grad=False, device="cuda") # C

            ref_weight = weight.clone()
            
            if ab_type == 'fp16':
                input = input.half()
                output = output.half()
            elif ab_type == 'bf16':
                input = input.bfloat16()
                output = output.bfloat16()
        
            if cd_type == 'fp16':
                weight = weight.half()
                ref_weight = ref_weight.half()
            elif cd_type == 'bf16':
                weight = weight.bfloat16()
                ref_weight = ref_weight.bfloat16()

            if cd_type == 'fp32':
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                    input, output, weight
                )
            else: 
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                    input, output, weight
                )
            
            #ref linear
            alpha = 1.0
            beta = 1.0
            self.backward_matrix_operation(alpha, beta, input, output, ref_weight)

            #cosim similarity
            weight = weight.view(-1)
            ref_weight = ref_weight.view(-1)

            #compare
            similarity = self.cosin_similarity(weight, ref_weight)
            print("similarity :", similarity)
            if similarity > 0.999:
                print("========PASS======")
            else:
                print("========FAIL======")
               

    def test_fused_weight_grad_dense_cuda_fp32(self):
        self.RunWeightGradDense('fp32', 'fp32')
    def test_fused_weight_grad_dense_cuda_infp16(self):
        self.RunWeightGradDense('fp16', 'fp32')
    def test_fused_weight_grad_dense_cuda_fp16(self):
        self.RunWeightGradDense('fp16', 'fp16')
    def test_fused_weight_grad_dense_cuda_inbf16(self):
        self.RunWeightGradDense('bf16', 'fp32')
    def test_fused_weight_grad_dense_cuda_bf16(self):
        self.RunWeightGradDense('bf16', 'bf16')


if __name__ == "__main__":
    common_utils.run_tests()
