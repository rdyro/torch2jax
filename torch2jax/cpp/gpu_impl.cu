#include "gpu_impl.h"

ffi::Error gpu_apply_torch_call_impl(cudaStream_t stream, 
  ffi::RemainingArgs args, ffi::RemainingRets rets, ffi::Dictionary attrs) {
  /* ---------------------------------------------------------------------------
  The GPU version of this routine just deserializes the descriptor and calls the
  main `apply_torch_call` routine.
  --------------------------------------------------------------------------- */
  apply_torch_call(args, rets, string(attrs.get<string_view>("fn_id").value()),
                  torch::kCUDA);
  return ffi::Error::Success();
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

TorchCallDevice actual_device(torch::DeviceType device_type, void* buffer) {
  if (device_type == torch::kCPU) return {torch::kCPU, 0};
  CUdevice device_ordinal;
  CUresult err = cuPointerGetAttribute((void*)&device_ordinal, 
                                       CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, 
                                       (CUdeviceptr)buffer);
  assert(err == CUDA_SUCCESS);
  return {torch::kCUDA, device_ordinal};
}