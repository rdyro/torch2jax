#ifndef _MAIN_H_
#define _MAIN_H_

#include <Python.h>
#include <pybind11/pybind11.h>
#include <stdio.h>
#include <torch/extension.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

using namespace std;
namespace py = pybind11;

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

struct DynamicTorchCallDescriptor {
  string id;
  TorchCallDevice device;
  int64_t nargin;
  int64_t nargout;
  vector<DynamicShapeDtype> shapes_in;
  vector<DynamicShapeDtype> shapes_out;
};

////////////////////////////////////////////////////////////////////////////////

class DescriptorDataAccessor {
 public:
  DescriptorDataAccessor(const int64_t *data64, const int32_t *data32)
      : data64(data64), data32(data32) {}

  int64_t get(int64_t i) const {
    if (this->data64 != nullptr) {
      return (this->data64)[i];
    } else {
      int64_t upper = (this->data32)[2 * i] & MASK32BIT;
      int64_t lower = (this->data32)[2 * i + 1] & MASK32BIT;
      return (upper << 32) | lower;
    }
  }

 private:
  const int64_t *data64;
  const int32_t *data32;
};

////////////////////////////////////////////////////////////////////////////////

string tolower(string &s);

// https://en.cppreference.com/w/cpp/numeric/bit_cast
template <class To, class From>
typename std::enable_if<sizeof(To) == sizeof(From) &&
                            std::is_trivially_copyable<From>::value &&
                            std::is_trivially_copyable<To>::value,
                        To>::type
bit_cast(const From &src) noexcept {
  static_assert(std::is_trivially_constructible<To>::value,
                "This implementation additionally requires destination type to "
                "be trivially constructible");

  To dst;
  memcpy(&dst, &src, sizeof(To));
  return dst;
}

/// @brief Converts a C++ function to a PyCapsule
/// @return PyCapsule of the function
template <typename T>
py::capsule encapsulateFunction(T *fn) {
  return py::capsule(bit_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
}

////////////////////////////////////////////////////////////////////////////////

vector<int64_t> serialize_cpu_descriptor(int64_t id, int64_t device_type,
                                         int64_t device_index,
                                         vector<vector<int64_t>> &shape_in,
                                         vector<int64_t> &dtype_in,
                                         vector<vector<int64_t>> &shape_out,
                                         vector<int64_t> &dtype_out);
py::bytes serialize_gpu_descriptor(int64_t id, int64_t device_type,
                                   int64_t device_index,
                                   vector<vector<int64_t>> &shape_in,
                                   vector<int64_t> &dtype_in,
                                   vector<vector<int64_t>> &shape_out,
                                   vector<int64_t> &dtype_out);
int64_t deserialize_descriptor(DynamicTorchCallDescriptor &d,
                               const DescriptorDataAccessor &data);

////////////////////////////////////////////////////////////////////////////////

// template <typename T>
// torch::TensorOptions tensor_options(T *buffer, TorchCallDevice &device) {
//   throw runtime_error(string("Buffer type not supported") +
//                       string(typeid(T).name()) + string("\n"));
//   return torch::TensorOptions();
// }
// torch::TensorOptions tensor_options(float *buffer,
//                                     const TorchCallDevice device);
// torch::TensorOptions tensor_options(double *buffer,
//                                     const TorchCallDevice device);

torch::TensorOptions tensor_dtype(torch::TensorOptions opts,
                                  const int64_t dtype);
torch::TensorOptions tensor_device(torch::TensorOptions opts,
                                   const TorchCallDevice device);

torch::TensorOptions tensor_options(int64_t dtype,
                                    const TorchCallDevice device);

////////////////////////////////////////////////////////////////////////////////

/// @brief The main torch call routine, wraps JAX arrays as Torch tensors and
/// calls the torch fn
/// @tparam T
/// @param buffers Array of pointers to input and then output buffers
/// @param d The Torch call descriptor, contains input & output shapes and
/// device and call id
// template <typename T>
void apply_torch_call(void **buffers, const DynamicTorchCallDescriptor &d);

#endif