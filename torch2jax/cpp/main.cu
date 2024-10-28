#ifdef TORCH2JAX_WITH_CUDA

#include "main.h"
#include "gpu_impl.h"
#include "cpu_impl.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cpu_registrations", &CPURegistrations);
  m.def("gpu_registrations", &GPURegistrations);
}

#endif