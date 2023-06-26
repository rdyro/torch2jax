#ifndef _TORCH_CALL_H_
#define _TORCH_CALL_H_

// #include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

using namespace std;
namespace py = pybind11;

////////////////////////////////////////////////////////////////////////////////

#define MAX_DIMS 7
#define MAX_NARGIN 20
#define MAX_NARGOUT 20
#define MAX_ID_LEN 256

struct Shape {
  size_t ndim;
  size_t shape[MAX_DIMS];
};

struct Device {
  torch::DeviceType type;
  int64_t index;
};

struct TorchCallDescriptor {
  char id[MAX_ID_LEN];
  Device device;
  int64_t nargin;
  int64_t nargout;
  Shape shapes_in[MAX_NARGIN];
  Shape shapes_out[MAX_NARGOUT];
};

////////////////////////////////////////////////////////////////////////////////

// https://en.cppreference.com/w/cpp/numeric/bit_cast
template <class To, class From>
typename std::enable_if<sizeof(To) == sizeof(From) &&
                            std::is_trivially_copyable<From>::value &&
                            std::is_trivially_copyable<To>::value,
                        To>::type
bit_cast(const From& src) noexcept {
  static_assert(std::is_trivially_constructible<To>::value,
                "This implementation additionally requires destination type to "
                "be trivially constructible");

  To dst;
  memcpy(&dst, &src, sizeof(To));
  return dst;
}

////////////////////////////////////////////////////////////////////////////////

/// @brief Create (Py)Torch tensor options: device and the data type
/// @tparam T
/// @param buffer The buffer to inform the datatype, it is not used
/// @param device The device struct of the tensors, device type (cpu/cuda) and
/// device ordinal index
/// @return TensorOptions object
template <typename T>
torch::TensorOptions tensor_options(T* buffer, Device& device) {
  throw runtime_error(string("Buffer type not supported") +
                      string(typeid(T).name()) + string("\n"));
  return torch::TensorOptions();
}

torch::TensorOptions tensor_options(float* buffer, const Device device) {
  if (device.type == torch::kCPU) {
    return torch::TensorOptions().dtype(torch::kFloat32).device(device.type);
  } else {
    return torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device.type, device.index);
  }
}

torch::TensorOptions tensor_options(double* buffer, const Device device) {
  if (device.type == torch::kCPU) {
    return torch::TensorOptions().dtype(torch::kFloat64).device(device.type);
  } else {
    return torch::TensorOptions()
        .dtype(torch::kFloat64)
        .device(device.type, device.index);
  }
}

/// @brief Convert a string to lowercase (like Python's .lower())
/// @param s
/// @return lowercase string
string tolower(string& s) {
  string str(s);
  transform(str.begin(), str.end(), str.begin(),
            [](unsigned char c) { return tolower(c); });
  return str;
}

////////////////////////////////////////////////////////////////////////////////

/// @brief Convert a custom descriptor struct to a string of bytes
/// @tparam T
/// @param descriptor The descriptor struct
/// @return string of bytes representation of the descriptor
template <typename T>
string packDescriptorAsString(const T& descriptor) {
  return string(bit_cast<const char*>(&descriptor), sizeof(T));
}

/// @brief Convert a bytes object to a custom descriptor type
/// @tparam T
/// @param opaque bytes representation of the descriptor
/// @param opaque_len length of the bytes object
/// @return
template <typename T>
const T* unpackDescriptor(const char* opaque, size_t opaque_len) {
  if (opaque_len != sizeof(T)) {
    throw runtime_error("Invalid opaque object size");
  }
  return bit_cast<const T*>(opaque);
}

/// @brief Packs a custom descriptor types as Python bytes
/// @tparam T
/// @param descriptor The descriptor struct
/// @return Python bytes object
template <typename T>
py::bytes packDescriptor(const T& descriptor) {
  return py::bytes(packDescriptorAsString(descriptor));
}

/// @brief Converts a C++ function to a PyCapsule
/// @return PyCapsule of the function
template <typename T>
py::capsule encapsulateFunction(T* fn) {
  return py::capsule(bit_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET");
}

////////////////////////////////////////////////////////////////////////////////

#endif