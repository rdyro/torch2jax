#include "cpu_impl.h"

pybind11::dict CPURegistrations() {
  pybind11::dict dict;
  // dict["cpu_torch_call_f32"] =
  // encapsulateFunction(cpu_apply_torch_call<float>);
  // dict["cpu_torch_call_f64"] =
  // encapsulateFunction(cpu_apply_torch_call<double>);
  dict["cpu_torch_call"] = encapsulateFunction(cpu_apply_torch_call);
  return dict;
}

void cpu_apply_torch_call(void *out_tuple, const void **in) {
  /* ---------------------------------------------------------------------------
  The general strategy for the CPU version of the torch call is as follows:
    1. create a descriptor from the input constants in the `in` buffer
    2. wrap the, here passed separately, input and output buffers as a single
       array of pointers
    3. call the main `apply_torch_call` routine
  --------------------------------------------------------------------------- */

  DescriptorDataAccessor da(nullptr, reinterpret_cast<const int32_t *>(in[0]));
  DynamicTorchCallDescriptor d;
  deserialize_descriptor(d, da);

  // 2. wrap the, here passed separately, input and output buffers as a single
  // array of pointers
  void **buffers = new void *[d.nargin + d.nargout];
  for (int64_t i = 0; i < d.nargin; i++)
    buffers[i] = const_cast<void *>(in[i + 1]);
  if (d.nargout == 0) return;
  void **out =
      d.nargout == 1 ? &out_tuple : reinterpret_cast<void **>(out_tuple);
  for (int64_t i = 0; i < d.nargout; i++) buffers[d.nargin + i] = out[i];

  // 3. call the main `apply_torch_call` routine
  // apply_torch_call<T>(buffers, d);
  apply_torch_call(buffers, d);

  delete[] buffers;
}