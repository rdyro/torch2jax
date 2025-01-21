#include "main.h"

torch::TensorOptions tensor_dtype(torch::TensorOptions opts,
                                  ffi::DataType dtype) {
  // PRED: DataType
  // S2: DataType
  // S4: DataType
  // S8: DataType
  // S16: DataType
  // S32: DataType
  // S64: DataType
  // U2: DataType
  // U4: DataType
  // U8: DataType
  // U16: DataType
  // U32: DataType
  // U64: DataType
  // F8E3M4: DataType
  // F8E4M3: DataType
  // F8E4M3FN: DataType
  // F8E4M3B11FNUZ: DataType
  // F8E4M3FNUZ: DataType
  // F8E5M2: DataType
  // F8E5M2FNUZ: DataType
  // BF16: DataType
  // F16: DataType
  // F32: DataType
  // F64: DataType
  // C64: DataType
  // C128: DataType
  switch (dtype) {
    case ffi::DataType::PRED:
      return opts.dtype(torch::kBool);
    case ffi::DataType::U8:
      return opts.dtype(torch::kUInt8);
    case ffi::DataType::S8:
      return opts.dtype(torch::kInt8);
    case ffi::DataType::S16:
      return opts.dtype(torch::kInt16);
    case ffi::DataType::S32:
      return opts.dtype(torch::kInt32);
    case ffi::DataType::S64:
      return opts.dtype(torch::kInt64);
    case ffi::DataType::F16:
      return opts.dtype(torch::kFloat16);
    case ffi::DataType::BF16:
      return opts.dtype(torch::kBFloat16);
    case ffi::DataType::F32:
      return opts.dtype(torch::kFloat32);
    case ffi::DataType::F64:
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

torch::TensorOptions tensor_options(ffi::DataType dtype,
                                    const TorchCallDevice device) {
  return tensor_device(tensor_dtype(torch::TensorOptions(), dtype), device);
}

////////////////////////////////////////////////////////////////////////////////


/// @brief The main torch call routine, wraps JAX arrays as Torch tensors and
/// calls the torch fn
/// @param args input buffer
/// @param rets output buffers
/// @param fn_id call fn id
/// @param device_type the accelerator type, device id is detected from pointer
void apply_torch_call(ffi::RemainingArgs args, ffi::RemainingRets rets,
    const string& fn_id, torch::DeviceType device_type) {
  /* ---------------------------------------------------------------------------
  The general strategy for the torch call is as follows:
    1. wrap the input buffers as Torch tensors
    2. bind the input tensors to the Python module in an identifiable place
    3. call the identifiable Python torch function which can find those inputs
    4. unwrap the output tensors and copy them to the output buffers
  --------------------------------------------------------------------------- */

  py::gil_scoped_acquire acquire;
  py::list my_list;

  // 1. wrap the input buffers as Torch tensors
  set<int64_t> cuda_device_idxs_seen;
  for (int64_t i = 0; i < args.size(); i++) {
    auto arg = args.get<ffi::AnyBuffer>(i).value();
    auto dims = arg.dimensions();
    vector<int64_t> shape(dims.begin(), dims.end());
    auto size = torch::IntArrayRef(shape.data(), (size_t)dims.size());

    void* data_ptr = arg.untyped_data();
    TorchCallDevice device_desc = actual_device(device_type, (void*)data_ptr);
    if (device_desc.type == torch::kCUDA) {
      cuda_device_idxs_seen.insert(device_desc.index);
    }
    auto options = tensor_options(arg.element_type(), device_desc);

    torch::Tensor tharray = torch::from_blob(data_ptr, size, options);
    my_list.append(THPVariable_Wrap(tharray));
  }

  py::gil_scoped_acquire release;

#ifdef TORCH2JAX_WITH_CUDA
  if (device_type == torch::kCUDA) {
    for (auto cuda_device_idx : cuda_device_idxs_seen) {
      torch::cuda::synchronize(cuda_device_idx);
    }
  }
#endif

  // 2. bind the input tensors to the Python module in an identifiable place
  auto mod = py::module_::import("torch");
  // 3. call the identifiable Python torch function which can find those inputs
  py::tuple results =
      mod.attr((string("_torch2jax_fn_") + fn_id).c_str())(my_list);

  // 4. unwrap the output tensors and copy them to the output buffers
  cuda_device_idxs_seen.clear();
  for (int64_t i = 0; i < rets.size(); i++) {
    auto ret = *(rets.get<ffi::AnyBuffer>(i).value());
    auto dims = ret.dimensions();
    vector<int64_t> shape(dims.begin(), dims.end());
    auto size = torch::IntArrayRef(shape.data(), (size_t)dims.size());

    void* data_ptr = ret.untyped_data();
    TorchCallDevice device_desc = actual_device(device_type, (void*)data_ptr);
    if (device_desc.type == torch::kCUDA) {
      cuda_device_idxs_seen.insert(device_desc.index);
    }
    auto options = tensor_options(ret.element_type(), device_desc);

    torch::Tensor tharray = torch::from_blob(data_ptr, size, options);
    PyObject *out = results[i].ptr();
    THPVariable_Check(out);
    tharray.copy_(THPVariable_Unpack(out));
  }

#ifdef TORCH2JAX_WITH_CUDA
  if (device_type == torch::kCUDA) {
    for (auto cuda_device_idx : cuda_device_idxs_seen) {
      torch::cuda::synchronize(cuda_device_idx);
    }
  }
#endif
}

#ifndef TORCH2JAX_WITH_CUDA
TorchCallDevice actual_device(torch::DeviceType device_type, void* buffer) {
  return {torch::kCPU, 0};
}
#endif