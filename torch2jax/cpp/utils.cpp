#include "main.h"

torch::TensorOptions tensor_dtype(torch::TensorOptions opts,
                                  const int64_t dtype) {
  switch (dtype) {
    case DATA_TYPE_BOOL:
      return opts.dtype(torch::kBool);
    case DATA_TYPE_UINT8:
      return opts.dtype(torch::kUInt8);
    case DATA_TYPE_INT8:
      return opts.dtype(torch::kInt8);
    case DATA_TYPE_INT16:
      return opts.dtype(torch::kInt16);
    case DATA_TYPE_INT32:
      return opts.dtype(torch::kInt32);
    case DATA_TYPE_INT64:
      return opts.dtype(torch::kInt64);
    case DATA_TYPE_FLOAT16:
      return opts.dtype(torch::kFloat16);
    case DATA_TYPE_FLOAT32:
      return opts.dtype(torch::kFloat32);
    case DATA_TYPE_FLOAT64:
      return opts.dtype(torch::kFloat64);
    default:
      assert(false);
      return opts;
  }
}

torch::TensorOptions tensor_device(torch::TensorOptions opts,
                                   const TorchCallDevice device) {
  if (device.type == torch::kCPU)
    return opts.device(device.type);
  else
    return opts.device(device.type, device.index);
}

torch::TensorOptions tensor_options(int64_t dtype,
                                    const TorchCallDevice device) {
  return tensor_device(tensor_dtype(torch::TensorOptions(), dtype), device);
}

////////////////////////////////////////////////////////////////////////////////

void push_back_int64(vector<int64_t> &v, int64_t x) {
  v.push_back((x >> 32) & MASK32BIT);
  v.push_back(x & MASK32BIT);
}

vector<int64_t> serialize_cpu_descriptor(int64_t id, int64_t device_type,
                                         int64_t device_index,
                                         vector<vector<int64_t>> &shape_in,
                                         vector<int64_t> &dtype_in,
                                         vector<vector<int64_t>> &shape_out,
                                         vector<int64_t> &dtype_out) {
  vector<int64_t> descriptor;
  push_back_int64(descriptor, id);
  push_back_int64(descriptor, device_type);
  push_back_int64(descriptor, device_index);
  push_back_int64(descriptor, shape_in.size());
  for (int64_t i = 0; i < shape_in.size(); i++) {
    push_back_int64(descriptor, shape_in[i].size());
    for (int64_t j = 0; j < shape_in[i].size(); j++)
      push_back_int64(descriptor, shape_in[i][j]);
    push_back_int64(descriptor, dtype_in[i]);
  }
  push_back_int64(descriptor, shape_out.size());
  for (int64_t i = 0; i < shape_out.size(); i++) {
    push_back_int64(descriptor, shape_out[i].size());
    for (int64_t j = 0; j < shape_out[i].size(); j++)
      push_back_int64(descriptor, shape_out[i][j]);
    push_back_int64(descriptor, dtype_out[i]);
  }
  return descriptor;
}

py::bytes serialize_gpu_descriptor(int64_t id, int64_t device_type,
                                   int64_t device_index,
                                   vector<vector<int64_t>> &shape_in,
                                   vector<int64_t> &dtype_in,
                                   vector<vector<int64_t>> &shape_out,
                                   vector<int64_t> &dtype_out) {
  vector<int64_t> descriptor;
  descriptor.push_back(id);
  descriptor.push_back(device_type);
  descriptor.push_back(device_index);
  descriptor.push_back(shape_in.size());
  for (int64_t i = 0; i < shape_in.size(); i++) {
    descriptor.push_back(shape_in[i].size());
    for (int64_t j = 0; j < shape_in[i].size(); j++)
      descriptor.push_back(shape_in[i][j]);
    descriptor.push_back(dtype_in[i]);
  }
  descriptor.push_back(shape_out.size());
  for (int64_t i = 0; i < shape_out.size(); i++) {
    descriptor.push_back(shape_out[i].size());
    for (int64_t j = 0; j < shape_out[i].size(); j++)
      descriptor.push_back(shape_out[i][j]);
    descriptor.push_back(dtype_out[i]);
  }
  return py::bytes(reinterpret_cast<char *>(descriptor.data()),
                   descriptor.size() * sizeof(int64_t));
}

int64_t deserialize_descriptor(DynamicTorchCallDescriptor &d,
                               const DescriptorDataAccessor &data) {
  int64_t k = 0;
  d.id = to_string(data.get(k++));
  int64_t device_type = data.get(k++);
  if (device_type == DEVICE_TYPE_CPU) {
    d.device = {torch::kCPU, data.get(k++)};
  } else {
    d.device = {torch::kCUDA, data.get(k++)};
  }
  d.nargin = data.get(k++);
  for (int64_t i = 0; i < d.nargin; i++) {
    d.shapes_in.emplace_back();
    d.shapes_in[i].ndim = data.get(k++);
    for (int64_t j = 0; j < d.shapes_in[i].ndim; j++)
      d.shapes_in[i].shape.push_back(data.get(k++));
    d.shapes_in[i].dtype = data.get(k++);
  }
  d.nargout = data.get(k++);
  for (int64_t i = 0; i < d.nargout; i++) {
    d.shapes_out.emplace_back();
    d.shapes_out[i].ndim = data.get(k++);
    for (int64_t j = 0; j < d.shapes_out[i].ndim; j++)
      d.shapes_out[i].shape.push_back(data.get(k++));
    d.shapes_out[i].dtype = data.get(k++);
  }
  return k;
}

/// @brief The main torch call routine, wraps JAX arrays as Torch tensors and
/// calls the torch fn
/// @tparam T
/// @param buffers Array of pointers to input and then output buffers
/// @param d The Torch call descriptor, contains input & output shapes and
/// device and call id
// template <typename T>
void apply_torch_call(void **buffers, const DynamicTorchCallDescriptor &d) {
  /* ---------------------------------------------------------------------------
  The general strategy for the torch call is as follows:
    1. wrap the input buffers as Torch tensors
    2. bind the input tensors to the Python module in an identifiable place
    3. call the identifiable Python torch function which can find those inputs
    4. unwrap the output tensors and copy them to the output buffers
  ---------------------------------------------------------------------------
*/

  py::gil_scoped_acquire acquire;
  py::list my_list;

#ifdef TORCH2JAX_WITH_CUDA
  if (d.device.type == torch::kCUDA) {
    //printf("Detected CUDA, will synchronize...\n");
    //fflush(stdout);
    torch::cuda::synchronize();
  }
#endif

  // 1. wrap the input buffers as Torch tensors
  for (int64_t i = 0; i < d.nargin; i++) {
    auto size = torch::IntArrayRef((int64_t *)d.shapes_in[i].shape.data(),
                                   (size_t)d.shapes_in[i].ndim);
    // T *buf = reinterpret_cast<T *>(buffers[i]);
    auto options = tensor_options(d.shapes_in[i].dtype, d.device);
    // torch::Tensor tharray = torch::from_blob(buf, size, options);
    torch::Tensor tharray = torch::from_blob(buffers[i], size, options);
    my_list.append(THPVariable_Wrap(tharray));
  }

  // 2. bind the input tensors to the Python module in an identifiable place
  auto mod = py::module_::import("torch");
  mod.attr((string("_torch2jax_args_") + string(d.id)).c_str()) = my_list;
  // 3. call the identifiable Python torch function which can find those
  // inputs
  py::tuple results =
      mod.attr((string("_torch2jax_fn_") + string(d.id)).c_str())();

  // py::gil_scoped_release release;

  // 4. unwrap the output tensors and copy them to the output buffers
  assert(results.size() == d.nargout);
  for (int64_t i = 0; i < d.nargout; i++) {
    auto size = torch::IntArrayRef((int64_t *)d.shapes_out[i].shape.data(),
                                   (size_t)d.shapes_out[i].ndim);
    // T *buf = reinterpret_cast<T *>(buffers[nargin + i]);
    auto options = tensor_options(d.shapes_out[i].dtype, d.device);
    // torch::Tensor tharray = torch::from_blob(buf, size, options);
    torch::Tensor tharray =
        torch::from_blob(buffers[d.nargin + i], size, options);
    PyObject *out = results[i].ptr();
    THPVariable_Check(out);
    tharray.copy_(THPVariable_Unpack(out));
  }

#ifdef TORCH2JAX_WITH_CUDA
  if (d.device.type == torch::kCUDA) {
    //printf("Detected CUDA, will synchronize...\n");
    //fflush(stdout);
    torch::cuda::synchronize();
  }
#endif
}