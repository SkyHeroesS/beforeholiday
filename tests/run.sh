#!/bin/bash

echo -e "\n***************input and output are fp32, weight is fp32******************\n"
python test_kernel.py fp32 fp32
echo -e "\n***************input and output are fp16, weight is fp32******************\n"
python test_kernel.py fp16 fp32
echo -e "\n***************input and output are bf16, weight is fp32******************\n"
python test_kernel.py bf16 fp32
echo -e "\n***************input and output are fp16, weight is fp16******************\n"
python test_kernel.py fp16 fp16
echo -e "\n***************input and output are bf16, weight is bf16******************\n"
python test_kernel.py bf16 bf16
