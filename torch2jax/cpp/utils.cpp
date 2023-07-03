#include "main.h"

torch::TensorOptions tensor_options(float *buffer,
                                    const TorchCallDevice device) {
  if (device.type == torch::kCPU) {
    return torch::TensorOptions().dtype(torch::kFloat32).device(device.type);
  } else {
    return torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(device.type, device.index);
  }
}

torch::TensorOptions tensor_options(double *buffer,
                                    const TorchCallDevice device) {
  if (device.type == torch::kCPU) {
    return torch::TensorOptions().dtype(torch::kFloat64).device(device.type);
  } else {
    return torch::TensorOptions()
        .dtype(torch::kFloat64)
        .device(device.type, device.index);
  }
}

////////////////////////////////////////////////////////////////////////////////

DynamicTorchCallDescriptor deserialize_gpu_descriptor(const char *opaque,
                                                      size_t opaque_len) {
  DynamicTorchCallDescriptor d;
  const int64_t *op_int = reinterpret_cast<const int64_t *>(opaque);
  int64_t total_op_len = 4 * 8;  // id, device, nargin, nargout
  assert(opaque_len >= total_op_len);
  d.id = to_string(*op_int++);
  d.device = {torch::kCUDA, *op_int++};
  d.nargin = *op_int++;
  for (int64_t i = 0; i < d.nargin; i++) {
    total_op_len += 8;
    assert(opaque_len >= total_op_len);
    d.shapes_in.emplace_back();
    d.shapes_in[i].ndim = *op_int++;
    total_op_len += d.shapes_in[i].ndim * 8;
    assert(opaque_len >= total_op_len);
    for (int64_t j = 0; j < d.shapes_in[i].ndim; j++)
      d.shapes_in[i].shape.push_back(*op_int++);
  }
  d.nargout = *op_int++;
  for (int64_t i = 0; i < d.nargout; i++) {
    total_op_len += 8;
    assert(opaque_len >= total_op_len);
    d.shapes_out.emplace_back();
    d.shapes_out[i].ndim = *op_int++;
    total_op_len += d.shapes_out[i].ndim * 8;
    assert(opaque_len >= total_op_len);
    for (int64_t j = 0; j < d.shapes_out[i].ndim; j++)
      d.shapes_out[i].shape.push_back(*op_int++);
  }
  return d;
}

py::bytes serialize_gpu_descriptor(int64_t id, int64_t device_index,
                                   vector<vector<int64_t>> &shape_in,
                                   vector<vector<int64_t>> &shape_out) {
  vector<int64_t> descriptor;
  descriptor.push_back(id);
  descriptor.push_back(device_index);
  descriptor.push_back(shape_in.size());
  for (int64_t i = 0; i < shape_in.size(); i++) {
    descriptor.push_back(shape_in[i].size());
    for (int64_t j = 0; j < shape_in[i].size(); j++)
      descriptor.push_back(shape_in[i][j]);
  }
  descriptor.push_back(shape_out.size());
  for (int64_t i = 0; i < shape_out.size(); i++) {
    descriptor.push_back(shape_out[i].size());
    for (int64_t j = 0; j < shape_out[i].size(); j++)
      descriptor.push_back(shape_out[i][j]);
  }
  return py::bytes(reinterpret_cast<char *>(descriptor.data()),
                   descriptor.size() * sizeof(int64_t));
}

////////////////////////////////////////////////////////////////////////////////

DynamicTorchCallDescriptor deserialize_cpu_descriptor(const void ***in_ptr) {
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
  // 2. lastly, advance the pointer in to now only include tensor data

  const void **in = *in_ptr;
  DynamicTorchCallDescriptor d;
  int64_t k = 0;

  // 1. deserialize the call id
  char buffer[32];
  snprintf(buffer, 32, "%ld", *reinterpret_cast<const int64_t *>(in[k++]));
  d.id = string(buffer);

  d.device = {torch::kCPU, 0};

  // 1. deserialize the shapes of the input arguments
  d.nargin = *reinterpret_cast<const int64_t *>(in[k++]);
  for (int64_t i = 0; i < d.nargin; i++) {
    d.shapes_in.emplace_back();
    d.shapes_in[i].ndim = *reinterpret_cast<const int64_t *>(in[k++]);
    for (int64_t j = 0; j < d.shapes_in[i].ndim; j++) {
      d.shapes_in[i].shape.push_back(
          *reinterpret_cast<const int64_t *>(in[k++]));
    }
  }
  // 1. deserialize the shapes of the output arguments
  d.nargout = *reinterpret_cast<const int64_t *>(in[k++]);
  for (int64_t i = 0; i < d.nargout; i++) {
    d.shapes_out.emplace_back();
    d.shapes_out[i].ndim = *reinterpret_cast<const int64_t *>(in[k++]);
    for (int64_t j = 0; j < d.shapes_out[i].ndim; j++) {
      d.shapes_out[i].shape.push_back(
          *reinterpret_cast<const int64_t *>(in[k++]));
    }
  }

  // 2. lastly, advance the pointer in to now only include tensor data
  *in_ptr = in + k;
  return d;
}

vector<int64_t> serialize_cpu_descriptor(int64_t id,
                                         vector<vector<int64_t>> &shape_in,
                                         vector<vector<int64_t>> &shape_out) {
  vector<int64_t> descriptor;
  descriptor.push_back(id);
  descriptor.push_back(shape_in.size());
  for (int64_t i = 0; i < shape_in.size(); i++) {
    descriptor.push_back(shape_in[i].size());
    for (int64_t j = 0; j < shape_in[i].size(); j++)
      descriptor.push_back(shape_in[i][j]);
  }
  descriptor.push_back(shape_out.size());
  for (int64_t i = 0; i < shape_out.size(); i++) {
    descriptor.push_back(shape_out[i].size());
    for (int64_t j = 0; j < shape_out[i].size(); j++)
      descriptor.push_back(shape_out[i][j]);
  }
  return descriptor;
}

////////////////////////////////////////////////////////////////////////////////