#include "main.h"

torch::TensorOptions tensor_options(float* buffer, const TorchCallDevice device) {
  if (device.type == torch::kCPU) {
    return torch::TensorOptions().dtype(torch::kFloat32).device(device.type);
  } else {
    return torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device.type, device.index);
  }
}

torch::TensorOptions tensor_options(double* buffer, const TorchCallDevice device) {
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

py::bytes build_torch_call_descriptor(int64_t id, string &device_str,
                                      int64_t device_index,
                                      vector<vector<int64_t>> &shape_in,
                                      vector<vector<int64_t>> &shape_out) {
  assert(shape_in.size() < MAX_NARGIN);
  assert(shape_out.size() < MAX_NARGOUT);
  device_str = tolower(device_str);
  assert(device_str == "cpu" || device_str == "cuda");

  TorchCallDescriptor d;
  snprintf(d.id, MAX_ID_LEN, "%ld", id);

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