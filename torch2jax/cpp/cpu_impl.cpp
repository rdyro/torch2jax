#include "cpu_impl.h"

ffi::Error cpu_apply_torch_call_impl(ffi::RemainingArgs args, 
  ffi::RemainingRets rets, ffi::Dictionary attrs) {
  /* ---------------------------------------------------------------------------
  The general strategy for the CPU version of the torch call is as follows:
    1. create a descriptor from the input constants in the `in` buffer
    2. wrap the, here passed separately, input and output buffers as a single
       array of pointers
    3. call the main `apply_torch_call` routine
  --------------------------------------------------------------------------- */
  apply_torch_call(args, rets, string(attrs.get<string_view>("fn_id").value()), 
                   torch::kCPU);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    cpu_apply_torch_call, cpu_apply_torch_call_impl,
    ffi::Ffi::Bind()
    .RemainingArgs()
    .RemainingRets()
    .Attrs()
);

pybind11::dict CPURegistrations() {
  pybind11::dict dict;
  dict["torch_call"] = EncapsulateFfiCall(cpu_apply_torch_call);
  return dict;
}
