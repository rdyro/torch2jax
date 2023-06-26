#include "gpu_impl.h"

py::dict GPURegistrations() {
  py::dict dict;
  dict["gpu_torch_call_f32"] = encapsulateFunction(gpu_apply_torch_call<float>);
  dict["gpu_torch_call_f64"] = encapsulateFunction(gpu_apply_torch_call<double>);
  return dict;
}