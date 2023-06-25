#ifndef _TORCH_CALL_H_
#define _TORCH_CALL_H_

//#include <cuda_runtime_api.h>
#include <pybind11/pybind11.h>

#include <cstddef>
#include <cstdint>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

using namespace std;
namespace py = pybind11;

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

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
string packDescriptorAsString(const T& descriptor) {
  return string(bit_cast<const char*>(&descriptor), sizeof(T));
}

template <typename T>
const T* unpackDescriptor(const char* opaque, size_t opaque_len) {
  if (opaque_len != sizeof(T)) {
    throw runtime_error("Invalid opaque object size");
  }
  return bit_cast<const T*>(opaque);
}

template <typename T>
py::bytes packDescriptor(const T& descriptor) {
  return py::bytes(packDescriptorAsString(descriptor));
}

template <typename T>
py::capsule encapsulateFunction(T* fn) {
  return py::capsule(bit_cast<void*>(fn), "xla._CUSTOM_CALL_TARGET");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

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

#endif