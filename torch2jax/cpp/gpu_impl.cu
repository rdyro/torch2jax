#include "gpu_impl.h"

py::dict GPURegistrations() {
  py::dict dict;
  // dict["gpu_torch_call_f32"] =
  // encapsulateFunction(gpu_apply_torch_call<float>);
  // dict["gpu_torch_call_f64"] =
  // encapsulateFunction(gpu_apply_torch_call<double>);
  dict["gpu_torch_call"] = encapsulateFunction(gpu_apply_torch_call);
  return dict;
}

// template <typename T>
void gpu_apply_torch_call(cudaStream_t stream, void **buffers,
                          const char *opaque, size_t opaque_len) {
  /* ---------------------------------------------------------------------------
  The GPU version of this routine just deserializes the descriptor and calls the
  main `apply_torch_call` routine.
  --------------------------------------------------------------------------- */

  DescriptorDataAccessor da(reinterpret_cast<const int64_t*>(opaque), nullptr);
  DynamicTorchCallDescriptor d;
  deserialize_descriptor(d, da);

  // apply_torch_call<T>(buffers, d);
  apply_torch_call(buffers, d);
}