#ifndef _CPU_IMPL_H
#define _CPU_IMPL_H

#include "main.h"


using namespace std;
namespace py = pybind11;

ffi::Error cpu_apply_torch_call_impl(ffi::RemainingArgs args, 
    ffi::RemainingRets rets, ffi::Dictionary attrs);

py::dict CPURegistrations();

#endif