#ifdef TORCH2JAX_WITH_CUDA

#include "main.h"
#include "gpu_impl.h"
#include "cpu_impl.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cpu_registrations", &CPURegistrations);
  m.def("gpu_registrations", &GPURegistrations);
  m.def("serialize_cpu_descriptor", &serialize_cpu_descriptor);
  m.def("serialize_gpu_descriptor", &serialize_gpu_descriptor);
  m.attr("DEVICE_TYPE_CPU") = DEVICE_TYPE_CPU;
  m.attr("DEVICE_TYPE_CUDA") = DEVICE_TYPE_CUDA;
  m.attr("DATA_TYPE_UINT8") = DATA_TYPE_UINT8;
  m.attr("DATA_TYPE_BOOL") = DATA_TYPE_BOOL;
  m.attr("DATA_TYPE_INT8") = DATA_TYPE_INT8;
  m.attr("DATA_TYPE_INT16") = DATA_TYPE_INT16;
  m.attr("DATA_TYPE_INT32") = DATA_TYPE_INT32;
  m.attr("DATA_TYPE_INT64") = DATA_TYPE_INT64;
  m.attr("DATA_TYPE_FLOAT16") = DATA_TYPE_FLOAT16;
  m.attr("DATA_TYPE_FLOAT32") = DATA_TYPE_FLOAT32;
  m.attr("DATA_TYPE_FLOAT64") = DATA_TYPE_FLOAT64;
}

#endif