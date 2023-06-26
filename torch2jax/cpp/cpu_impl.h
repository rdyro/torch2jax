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

  // 1. create a descriptor from the input constants in the `in` buffer
  // the input buffer contains the following:
  // i) number of input arguments
  // ii) for each input argument:
  //  a) number of dimensions
  //  b) shape (one integer per dimension)
  // iii) number of output arguments
  // iv) for each output argument:
  //  a) number of dimensions
  //  b) shape (one integer per dimension)

  // 1. (a) create the empty descriptor
  TorchCallDescriptor d;
  // 1. (b) assign the CPU device to the descriptor
  d.device = {torch::kCPU, 0};
  // 1. (c) obtain the shapes of the input arguments
  int64_t k = 0;
  d.nargin = *reinterpret_cast<const int64_t *>(in[k++]);
  assert(d.nargin < MAX_NARGIN);
  for (int64_t i = 0; i < d.nargin; i++) {
    d.shapes_in[i].ndim = *reinterpret_cast<const int64_t *>(in[k++]);
    for (int64_t j = 0; j < d.shapes_in[i].ndim; j++) {
      d.shapes_in[i].shape[j] = *reinterpret_cast<const int64_t *>(in[k++]);
    }
  }
  // 1. (d) obtain the shapes of the output arguments
  d.nargout = *reinterpret_cast<const int64_t *>(in[k++]);
  assert(d.nargin < MAX_NARGOUT);
  for (int64_t i = 0; i < d.nargout; i++) {
    d.shapes_out[i].ndim = *reinterpret_cast<const int64_t *>(in[k++]);
    for (int64_t j = 0; j < d.shapes_out[i].ndim; j++) {
      d.shapes_out[i].shape[j] = *reinterpret_cast<const int64_t *>(in[k++]);
    }
  }
  // 1. (e) obtain the call id
  snprintf(d.id, MAX_ID_LEN, "%ld", *reinterpret_cast<const int64_t *>(in[k++]));

  // 2. wrap the, here passed separately, input and output buffers as a single
  // array of pointers
  void **buffers = new void *[d.nargin + d.nargout];
  for (int64_t i = 0; i < d.nargin; i++)
    buffers[i] = const_cast<void *>(in[k++]);
  if (d.nargout == 0) return;
  void **out =
      d.nargout == 1 ? &out_tuple : reinterpret_cast<void **>(out_tuple);
  for (int64_t i = 0; i < d.nargout; i++) buffers[d.nargin + i] = out[i];

  // 3. call the main `apply_torch_call` routine
  apply_torch_call<T>(buffers, d);

  delete[] buffers;
}

//void cpu_torch_call_f32(void *out_tuple, const void **in) {
//  cpu_apply_torch_call<float>(out_tuple, in);
//}
//
//void cpu_torch_call_f64(void *out_tuple, const void **in) {
//  cpu_apply_torch_call<double>(out_tuple, in);
//}

py::dict CPURegistrations();

#endif