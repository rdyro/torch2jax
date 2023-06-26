#include "cpu_impl.h"

pybind11::dict CPURegistrations() {
  pybind11::dict dict;
  dict["cpu_torch_call_f32"] = encapsulateFunction(cpu_apply_torch_call<float>);
  dict["cpu_torch_call_f64"] = encapsulateFunction(cpu_apply_torch_call<double>);
  return dict;
}
