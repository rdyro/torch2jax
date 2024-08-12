#!/usr/bin/env bash

env \
  JAXFI_LOAD_SYSTEM_CUDA_LIBS="true" \
  XLA_PYTHON_CLIENT_ALLOCATOR="platform" \
  JAX_ENABLE_X64="true" \
  pytest tests

  #LD_PRELOAD="/usr/local/cuda/lib64/libcudart.so:$LD_PRELOAD" \
