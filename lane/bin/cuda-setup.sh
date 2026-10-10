#!/usr/bin/env bash
# Install minimal CUDA 13 toolkit (nvcc + cudart/cublas dev headers) on ai-host
# from NVIDIA's ubuntu2604 apt repo. Log everything; no secrets.
set -euo pipefail
cd /tmp
echo "[1/4] keyring $(date)"
curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2604/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
echo "[2/4] apt update $(date)"
sudo apt-get update -qq
echo "[3/4] install $(date)"
sudo apt-get install -y -qq cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-libraries-dev-13-0 cuda-cccl-13-0
echo "[4/4] verify $(date)"
/usr/local/cuda-13.0/bin/nvcc --version | tail -1
ls /usr/local/cuda-13.0/lib64/ | grep -E "cublasLt|cudart" | head -4
echo "CUDA_SETUP_DONE"
