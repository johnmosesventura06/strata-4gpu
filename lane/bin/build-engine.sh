#!/bin/bash
export PATH=/usr/local/cuda-13.3/bin:$PATH
cd ~/strata-4gpu/strata || exit 1
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.3/bin/nvcc || exit 1
cmake --build build --target strata
