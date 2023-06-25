// This file contains the GPU implementation of our op. It's a pretty typical
// CUDA kernel and I make no promises about the quality of the code or the
// choices made therein, but it should get the point accross.

#include <Python.h>
#include <cuda.h>
#include <pybind11/pybind11.h>
#include <stdio.h>
#include <torch/extension.h>

#include "torch_call.h"

using namespace std;
namespace py = pybind11;

////////////////////////////////////////////////////////////////////////////////

void ThrowIfError(cudaError_t error) {
  if (error != cudaSuccess) {
    throw runtime_error(cudaGetErrorString(error));
  }
}

////////////////////////////////////////////////////////////////////////////////

template <typename T>
torch::TensorOptions tensor_options(T *buffer, Device &device) {
  throw runtime_error(string("Buffer type not supported") + string("\n"));
  return torch::TensorOptions();
}

torch::TensorOptions tensor_options(float *buffer, const Device device) {
  if (device.type == torch::kCPU) {
    return torch::TensorOptions().dtype(torch::kFloat32).device(device.type);
  } else {
    return torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device.type, device.index);
  }
}

torch::TensorOptions tensor_options(double *buffer, const Device device) {
  if (device.type == torch::kCPU) {
    return torch::TensorOptions().dtype(torch::kFloat64).device(device.type);
  } else {
    return torch::TensorOptions()
        .dtype(torch::kFloat64)
        .device(device.type, device.index);
  }
}

string tolower(string &s) {
  string str(s);
  transform(str.begin(), str.end(), str.begin(),
            [](unsigned char c) { return tolower(c); });
  return str;
}

////////////////////////////////////////////////////////////////////////////////

template <typename T>
void apply_torch_call(void **buffers, const TorchCallDescriptor &d) {
  const int64_t nargin = d.nargin;
  const int64_t nargout = d.nargout;

  py::gil_scoped_acquire release;
  py::list my_list;

  for (int64_t i = 0; i < nargin; i++) {
    auto size = torch::IntArrayRef((int64_t *)d.shapes_in[i].shape,
                                   (size_t)d.shapes_in[i].ndim);
    T *buf = reinterpret_cast<T *>(buffers[i]);
    auto options = tensor_options(buf, d.device);
    torch::Tensor tharray = torch::from_blob(buf, size, options);
    my_list.append(THPVariable_Wrap(tharray));
  }

  auto mod = py::module_::import("torch");
  mod.attr((string("_torch_call_args_") + string(d.id)).c_str()) = my_list;
  py::tuple results =
      mod.attr((string("_torch_call_fn_") + string(d.id)).c_str())();

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
  const TorchCallDescriptor &d =
      *unpackDescriptor<TorchCallDescriptor>(opaque, opaque_len);
  apply_torch_call<T>(buffers, d);
}

template <typename T>
void cpu_apply_torch_call(void *out_tuple, const void **in) {
  TorchCallDescriptor d;
  d.device = {torch::kCPU, 0};
  int64_t k = 0;
  d.nargin = *reinterpret_cast<const int64_t *>(in[k++]);
  assert(d.nargin < MAX_NARGIN);
  for (int64_t i = 0; i < d.nargin; i++) {
    d.shapes_in[i].ndim = *reinterpret_cast<const int64_t *>(in[k++]);
    for (int64_t j = 0; j < d.shapes_in[i].ndim; j++) {
      d.shapes_in[i].shape[j] = *reinterpret_cast<const int64_t *>(in[k++]);
    }
  }
  d.nargout = *reinterpret_cast<const int64_t *>(in[k++]);
  assert(d.nargin < MAX_NARGOUT);
  for (int64_t i = 0; i < d.nargout; i++) {
    d.shapes_out[i].ndim = *reinterpret_cast<const int64_t *>(in[k++]);
    for (int64_t j = 0; j < d.shapes_out[i].ndim; j++) {
      d.shapes_out[i].shape[j] = *reinterpret_cast<const int64_t *>(in[k++]);
    }
  }
  snprintf(d.id, MAX_ID_LEN, "%d", *reinterpret_cast<const int64_t *>(in[k++]));

  void **buffers = new void *[d.nargin + d.nargout];
  for (int64_t i = 0; i < d.nargin; i++)
    buffers[i] = const_cast<void *>(in[k++]);
  if (d.nargout == 0) return;
  void **out =
      d.nargout == 1 ? &out_tuple : reinterpret_cast<void **>(out_tuple);
  for (int64_t i = 0; i < d.nargout; i++) buffers[d.nargin + i] = out[i];

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

py::bytes build_torch_call_descriptor(string &id, string &device_str,
                                      int64_t device_index,
                                      vector<vector<int64_t>> &shape_in,
                                      vector<vector<int64_t>> &shape_out) {
  assert(shape_in.size() < MAX_NARGIN);
  assert(shape_out.size() < MAX_NARGOUT);
  assert(id.size() < MAX_ID_LEN);
  device_str = tolower(device_str);
  assert(device_str == "cpu" || device_str == "cuda");

  TorchCallDescriptor d;
  snprintf(d.id, MAX_ID_LEN, "%s", id.c_str());

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