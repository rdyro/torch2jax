#ifndef _GPU_IMPL_H
#define _GPU_IMPL_H

#include "main.h"
#include <cuda.h>

using namespace std;
namespace py = pybind11;

template <typename T>
void gpu_apply_torch_call(cudaStream_t stream, void **buffers,
                          const char *opaque, size_t opaque_len) {
  /* ---------------------------------------------------------------------------
  The GPU version of this routine just deserializes the descriptor and calls the
  main `apply_torch_call` routine.
  --------------------------------------------------------------------------- */
  const TorchCallDescriptor &d =
      *unpackDescriptor<TorchCallDescriptor>(opaque, opaque_len);
  apply_torch_call<T>(buffers, d);
}

py::dict GPURegistrations();

#endif