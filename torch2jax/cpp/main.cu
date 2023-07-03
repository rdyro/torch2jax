#ifdef TORCH2JAX_WITH_CUDA

#include "main.h"
#include "gpu_impl.h"
#include "cpu_impl.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cpu_registrations", &CPURegistrations);
  m.def("gpu_registrations", &GPURegistrations);
  m.def("serialize_gpu_descriptor", &serialize_gpu_descriptor);
  m.def("serialize_cpu_descriptor", &serialize_cpu_descriptor);
}

#endif