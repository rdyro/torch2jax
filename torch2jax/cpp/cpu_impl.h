#ifndef _CPU_IMPL_H
#define _CPU_IMPL_H

#include "main.h"

using namespace std;
namespace py = pybind11;

//template <typename T>
void cpu_apply_torch_call(void *out_tuple, const void **in);

py::dict CPURegistrations();

#endif
