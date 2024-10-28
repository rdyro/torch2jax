#ifndef _GPU_IMPL_H
#define _GPU_IMPL_H

#include <cuda.h>
#include <cuda_runtime.h>

#include "main.h"

using namespace std;
namespace py = pybind11;

ffi::Error gpu_apply_torch_call_impl(cudaStream_t stream, 
    ffi::RemainingArgs args, ffi::RemainingRets rets, ffi::Dictionary attrs);

py::dict GPURegistrations();

#endif