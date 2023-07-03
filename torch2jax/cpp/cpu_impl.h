#ifndef _CPU_IMPL_H
#define _CPU_IMPL_H

#include "main.h"

using namespace std;
namespace py = pybind11;

template <typename T>
void cpu_apply_torch_call(void *out_tuple, const void **in) {
  /* ---------------------------------------------------------------------------
  The general strategy for the CPU version of the torch call is as follows:
    1. create a descriptor from the input constants in the `in` buffer
    2. wrap the, here passed separately, input and output buffers as a single
       array of pointers
    3. call the main `apply_torch_call` routine
  --------------------------------------------------------------------------- */

  const void** in_ptr_cpy = in;
  DynamicTorchCallDescriptor d = deserialize_cpu_descriptor(&in_ptr_cpy);

  // 2. wrap the, here passed separately, input and output buffers as a single
  // array of pointers
  void **buffers = new void *[d.nargin + d.nargout];
  for (int64_t i = 0; i < d.nargin; i++)
    buffers[i] = const_cast<void *>(in_ptr_cpy[i]);
  if (d.nargout == 0) return;
  void **out =
      d.nargout == 1 ? &out_tuple : reinterpret_cast<void **>(out_tuple);
  for (int64_t i = 0; i < d.nargout; i++) buffers[d.nargin + i] = out[i];

  // 3. call the main `apply_torch_call` routine
  apply_torch_call<T>(buffers, d);

  delete[] buffers;
}

py::dict CPURegistrations();

#endif