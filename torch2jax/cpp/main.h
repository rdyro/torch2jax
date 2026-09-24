#ifndef _MAIN_H_
#define _MAIN_H_

#include <Python.h>
#include <pybind11/pybind11.h>
#include <stdio.h>
#include <torch/extension.h>

#ifdef TORCH2JAX_WITH_CUDA
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAEvent.h>
#endif

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace std;
namespace py = pybind11;
namespace ffi = xla::ffi;

////////////////////////////////////////////////////////////////////////////////

struct TorchCallDevice {
  torch::DeviceType type;
  int64_t index;
};

////////////////////////////////////////////////////////////////////////////////

/// @brief Converts a C++ function to a PyCapsule
/// @return PyCapsule of the function
template <typename T>
py::capsule EncapsulateFfiCall(T *fn) {
  // This check is optional, but it can be helpful for avoiding invalid handlers.
  static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                "Encapsulated function must be and XLA FFI handler");
  return py::capsule(reinterpret_cast<void *>(fn));
}

////////////////////////////////////////////////////////////////////////////////

std::optional<torch::ScalarType> torch_dtype(ffi::DataType dtype);

ffi::ErrorOr<TorchCallDevice> actual_device(torch::DeviceType device_type, void* buffer);

////////////////////////////////////////////////////////////////////////////////

/// @brief The main torch call routine, wraps JAX arrays as Torch tensors and
/// calls the torch fn
ffi::Error apply_torch_call(ffi::RemainingArgs args, ffi::RemainingRets rets, const string& fn_id,
                            torch::DeviceType device_type, void* stream = nullptr);

#endif