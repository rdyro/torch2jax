#include "gpu_impl.h"

ffi::Error gpu_apply_torch_call_impl(cudaStream_t stream, 
  ffi::RemainingArgs args, ffi::RemainingRets rets, ffi::Dictionary attrs) {
  /* ---------------------------------------------------------------------------
  The GPU version of this routine just deserializes the descriptor and calls the
  main `apply_torch_call` routine.
  --------------------------------------------------------------------------- */
  return apply_torch_call(args, rets, string(attrs.get<string_view>("fn_id").value()), torch::kCUDA);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    gpu_apply_torch_call, gpu_apply_torch_call_impl,
    ffi::Ffi::Bind()
    .Ctx<ffi::PlatformStream<cudaStream_t>>()
    .RemainingArgs()
    .RemainingRets()
    .Attrs()
);

py::dict GPURegistrations() {
  py::dict dict;
  dict["torch_call"] = EncapsulateFfiCall(gpu_apply_torch_call);
  return dict;
}

ffi::ErrorOr<TorchCallDevice> actual_device(torch::DeviceType device_type, void* buffer) {
  if (device_type == torch::kCPU) return TorchCallDevice{torch::kCPU, 0};
  CUdevice device_ordinal;
  CUresult err = cuPointerGetAttribute(&device_ordinal, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, (CUdeviceptr)buffer);
  if (err != CUDA_SUCCESS) {
    const char* msg = nullptr;
    cuGetErrorString(err, &msg);
    return ffi::Unexpected(ffi::Error::Internal(string("torch2jax: cannot query the CUDA device of an XLA buffer: ") +
                                                (msg ? msg : to_string(static_cast<int>(err)))));
  }
  return TorchCallDevice{torch::kCUDA, device_ordinal};
}
