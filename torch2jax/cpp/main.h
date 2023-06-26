#ifndef _MAIN_H_
#define _MAIN_H_

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <stdio.h>

#include <pybind11/pybind11.h>
#include <Python.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

using namespace std;
namespace py = pybind11;

////////////////////////////////////////////////////////////////////////////////

#define MAX_DIMS 7
#define MAX_NARGIN 20
#define MAX_NARGOUT 20
#define MAX_ID_LEN 256

struct Shape {
  int64_t ndim;
  int64_t shape[MAX_DIMS];
};

struct TorchCallDevice {
  torch::DeviceType type;
  int64_t index;
};

struct TorchCallDescriptor {
  char id[MAX_ID_LEN];
  TorchCallDevice device;
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
torch::TensorOptions tensor_options(T* buffer, TorchCallDevice& device) {
  throw runtime_error(string("Buffer type not supported") +
                      string(typeid(T).name()) + string("\n"));
  return torch::TensorOptions();
}
torch::TensorOptions tensor_options(float* buffer, const TorchCallDevice device);
torch::TensorOptions tensor_options(double* buffer, const TorchCallDevice device);
string tolower(string& s);

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

/// @brief The main torch call routine, wraps JAX arrays as Torch tensors and
/// calls the torch fn
/// @tparam T
/// @param buffers Array of pointers to input and then output buffers
/// @param d The Torch call descriptor, contains input & output shapes and
/// device and call id
template <typename T>
void apply_torch_call(void **buffers, const TorchCallDescriptor &d) {
  /* ---------------------------------------------------------------------------
  The general strategy for the torch call is as follows:
    1. wrap the input buffers as Torch tensors
    2. bind the input tensors to the Python module in an indentifiable place
    3. call the identifiable Python torch function which can find those inputs
    4. unwrap the output tensors and copy them to the output buffers
  --------------------------------------------------------------------------- */

  const int64_t nargin = d.nargin;
  const int64_t nargout = d.nargout;

  py::gil_scoped_acquire release;
  py::list my_list;

  // 1. wrap the input buffers as Torch tensors
  for (int64_t i = 0; i < nargin; i++) {
    auto size = torch::IntArrayRef((int64_t *)d.shapes_in[i].shape,
                                   (size_t)d.shapes_in[i].ndim);
    T *buf = reinterpret_cast<T *>(buffers[i]);
    auto options = tensor_options(buf, d.device);
    torch::Tensor tharray = torch::from_blob(buf, size, options);
    my_list.append(THPVariable_Wrap(tharray));
  }

  // 2. bind the input tensors to the Python module in an indentifiable place
  auto mod = py::module_::import("torch");
  mod.attr((string("_torch2jax_args_") + string(d.id)).c_str()) = my_list;
  // 3. call the identifiable Python torch function which can find those inputs
  py::tuple results =
      mod.attr((string("_torch2jax_fn_") + string(d.id)).c_str())();

  // 4. unwrap the output tensors and copy them to the output buffers
  assert(results.size() == nargout);
  for (int64_t i = 0; i < nargout; i++) {
    auto size = torch::IntArrayRef((int64_t *)d.shapes_out[i].shape,
                                   (size_t)d.shapes_out[i].ndim);
    T *buf = reinterpret_cast<T *>(buffers[nargin + i]);
    auto options = tensor_options(buf, d.device);
    torch::Tensor tharray = torch::from_blob(buf, size, options);
    PyObject *out = results[i].ptr();
    THPVariable_Check(out);
    tharray.copy_(THPVariable_Unpack(out));
  }
}

////////////////////////////////////////////////////////////////////////////////

/// @brief Create a Torch call CUDA descriptor struct from Python
/// @param id The unique identifier for this call
/// @param device_str Device string, either "cpu" or "cuda"
/// @param device_index Device ordinal index (e.g. 0 means first GPU), for CPU
/// this is ignored
/// @param shape_in List of shapes of inputs
/// @param shape_out List of shapes of outputs
/// @return A Python bytes object containing the serialized descriptor
py::bytes build_torch_call_descriptor(int64_t id, string &device_str,
                                      int64_t device_index,
                                      vector<vector<int64_t>> &shape_in,
                                      vector<vector<int64_t>> &shape_out);


////////////////////////////////////////////////////////////////////////////////

#endif