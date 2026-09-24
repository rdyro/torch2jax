#ifndef _MAIN_H_
#define _MAIN_H_

#include <Python.h>
#include <pybind11/pybind11.h>
#include <stdio.h>
#include <torch/extension.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace std;
namespace py = pybind11;
namespace ffi = xla::ffi;

////////////////////////////////////////////////////////////////////////////////

#define DEVICE_TYPE_CPU 0
#define DEVICE_TYPE_CUDA 1

#define DATA_TYPE_BOOL 0
#define DATA_TYPE_UINT8 1
#define DATA_TYPE_INT8 2
#define DATA_TYPE_INT16 3
#define DATA_TYPE_INT32 4
#define DATA_TYPE_INT64 5
#define DATA_TYPE_FLOAT16 6
#define DATA_TYPE_FLOAT32 7
#define DATA_TYPE_FLOAT64 8
#define DATA_TYPE_BFLOAT16 9

#define MASK32BIT 0xFFFFFFFF

////////////////////////////////////////////////////////////////////////////////

struct TorchCallDevice {
  torch::DeviceType type;
  int64_t index;
};

struct DynamicShapeDtype {
  int64_t ndim;
  vector<int64_t> shape;
  int64_t dtype;
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
                            torch::DeviceType device_type);

#endif