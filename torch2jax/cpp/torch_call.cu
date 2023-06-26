#include <Python.h>
#include <cuda.h>
#include <pybind11/pybind11.h>
#include <stdio.h>
#include <torch/extension.h>

#include "torch_call.h"

using namespace std;
namespace py = pybind11;

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
  mod.attr((string("_torch_call_args_") + string(d.id)).c_str()) = my_list;
  // 3. call the identifiable Python torch function which can find those inputs
  py::tuple results =
      mod.attr((string("_torch_call_fn_") + string(d.id)).c_str())();

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

template <typename T>
void gpu_apply_torch_call(cudaStream_t stream, void **buffers,
                          const char *opaque, size_t opaque_len) {
  /* ---------------------------------------------------------------------------
  The GPU version of this routine just deserializes the descriptor and calls the
  main `apply_torch_call` routine.
  --------------------------------------------------------------------------- */
  const TorchCallDescriptor &d =
      *unpackDescriptor<TorchCallDescriptor>(opaque, opaque_len);
  apply_torch_call<T>(buffers, d);
}

template <typename T>
void cpu_apply_torch_call(void *out_tuple, const void **in) {
  /* ---------------------------------------------------------------------------
  The general strategy for the CPU version of the torch call is as follows:
    1. create a descriptor from the input constants in the `in` buffer
    2. wrap the, here passed separately, input and output buffers as a single
       array of pointers
    3. call the main `apply_torch_call` routine
  --------------------------------------------------------------------------- */

  // 1. create a descriptor from the input constants in the `in` buffer
  // the input buffer contains the following:
  // i) number of input arguments
  // ii) for each input argument:
  //  a) number of dimensions
  //  b) shape (one integer per dimension)
  // iii) number of output arguments
  // iv) for each output argument:
  //  a) number of dimensions
  //  b) shape (one integer per dimension)

  // 1. (a) create the empty descriptor
  TorchCallDescriptor d;
  // 1. (b) assign the CPU device to the descriptor
  d.device = {torch::kCPU, 0};
  // 1. (c) obtain the shapes of the input arguments
  int64_t k = 0;
  d.nargin = *reinterpret_cast<const int64_t *>(in[k++]);
  assert(d.nargin < MAX_NARGIN);
  for (int64_t i = 0; i < d.nargin; i++) {
    d.shapes_in[i].ndim = *reinterpret_cast<const int64_t *>(in[k++]);
    for (int64_t j = 0; j < d.shapes_in[i].ndim; j++) {
      d.shapes_in[i].shape[j] = *reinterpret_cast<const int64_t *>(in[k++]);
    }
  }
  // 1. (d) obtain the shapes of the output arguments
  d.nargout = *reinterpret_cast<const int64_t *>(in[k++]);
  assert(d.nargin < MAX_NARGOUT);
  for (int64_t i = 0; i < d.nargout; i++) {
    d.shapes_out[i].ndim = *reinterpret_cast<const int64_t *>(in[k++]);
    for (int64_t j = 0; j < d.shapes_out[i].ndim; j++) {
      d.shapes_out[i].shape[j] = *reinterpret_cast<const int64_t *>(in[k++]);
    }
  }
  // 1. (e) obtain the call id
  snprintf(d.id, MAX_ID_LEN, "%d", *reinterpret_cast<const int64_t *>(in[k++]));

  // 2. wrap the, here passed separately, input and output buffers as a single
  // array of pointers
  void **buffers = new void *[d.nargin + d.nargout];
  for (int64_t i = 0; i < d.nargin; i++)
    buffers[i] = const_cast<void *>(in[k++]);
  if (d.nargout == 0) return;
  void **out =
      d.nargout == 1 ? &out_tuple : reinterpret_cast<void **>(out_tuple);
  for (int64_t i = 0; i < d.nargout; i++) buffers[d.nargin + i] = out[i];

  // 3. call the main `apply_torch_call` routine
  apply_torch_call<T>(buffers, d);

  delete[] buffers;
}

////////////////////////////////////////////////////////////////////////////////

void gpu_torch_call_f32(cudaStream_t stream, void **buffers, const char *opaque,
                        size_t opaque_len) {
  gpu_apply_torch_call<float>(stream, buffers, opaque, opaque_len);
}

void gpu_torch_call_f64(cudaStream_t stream, void **buffers, const char *opaque,
                        size_t opaque_len) {
  gpu_apply_torch_call<double>(stream, buffers, opaque, opaque_len);
}

void cpu_torch_call_f32(void *out_tuple, const void **in) {
  cpu_apply_torch_call<float>(out_tuple, in);
}

void cpu_torch_call_f64(void *out_tuple, const void **in) {
  cpu_apply_torch_call<double>(out_tuple, in);
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
                                      vector<vector<int64_t>> &shape_out) {
  assert(shape_in.size() < MAX_NARGIN);
  assert(shape_out.size() < MAX_NARGOUT);
  device_str = tolower(device_str);
  assert(device_str == "cpu" || device_str == "cuda");

  TorchCallDescriptor d;
  snprintf(d.id, MAX_ID_LEN, "%d", id);

  d.device.type = device_str == "cpu" ? torch::kCPU : torch::kCUDA;
  d.device.index = device_index;

  d.nargin = shape_in.size();
  for (int64_t i = 0; i < d.nargin; i++) {
    d.shapes_in[i].ndim = shape_in[i].size();
    for (int64_t j = 0; j < d.shapes_in[i].ndim; j++)
      d.shapes_in[i].shape[j] = shape_in[i][j];
  }

  d.nargout = shape_out.size();
  for (int64_t i = 0; i < d.nargout; i++) {
    d.shapes_out[i].ndim = shape_out[i].size();
    for (int64_t j = 0; j < d.shapes_out[i].ndim; j++)
      d.shapes_out[i].shape[j] = shape_out[i][j];
  }
  return packDescriptor(d);
}

////////////////////////////////////////////////////////////////////////////////

pybind11::dict GPURegistrations() {
  pybind11::dict dict;
  dict["gpu_torch_call_f32"] = encapsulateFunction(gpu_torch_call_f32);
  dict["gpu_torch_call_f64"] = encapsulateFunction(gpu_torch_call_f64);
  return dict;
}

pybind11::dict CPURegistrations() {
  pybind11::dict dict;
  dict["cpu_torch_call_f32"] = encapsulateFunction(cpu_torch_call_f32);
  dict["cpu_torch_call_f64"] = encapsulateFunction(cpu_torch_call_f64);
  return dict;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cpu_registrations", &CPURegistrations);
  m.def("gpu_registrations", &GPURegistrations);
  m.def("build_torch_call_descriptor", &build_torch_call_descriptor);
}