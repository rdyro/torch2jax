#ifndef TORCH2JAX_WITH_CUDA

#include "main.h"
#include "cpu_impl.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cpu_registrations", &CPURegistrations);
  m.def("serialize_cpu_descriptor", &serialize_cpu_descriptor);
}

#endif