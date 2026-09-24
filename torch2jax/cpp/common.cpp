#include "main.h"

std::optional<torch::ScalarType> torch_dtype(ffi::DataType dtype) {
  switch (dtype) {
    case ffi::DataType::PRED: return torch::kBool;
    case ffi::DataType::U8: return torch::kUInt8;
    case ffi::DataType::U16: return torch::kUInt16;
    case ffi::DataType::U32: return torch::kUInt32;
    case ffi::DataType::U64: return torch::kUInt64;
    case ffi::DataType::S8: return torch::kInt8;
    case ffi::DataType::S16: return torch::kInt16;
    case ffi::DataType::S32: return torch::kInt32;
    case ffi::DataType::S64: return torch::kInt64;
    case ffi::DataType::F16: return torch::kFloat16;
    case ffi::DataType::BF16: return torch::kBFloat16;
    case ffi::DataType::F32: return torch::kFloat32;
    case ffi::DataType::F64: return torch::kFloat64;
    case ffi::DataType::C64: return torch::kComplexFloat;
    case ffi::DataType::C128: return torch::kComplexDouble;
    case ffi::DataType::F8E4M3FN: return torch::kFloat8_e4m3fn;
    case ffi::DataType::F8E5M2: return torch::kFloat8_e5m2;
    case ffi::DataType::F8E4M3FNUZ: return torch::kFloat8_e4m3fnuz;
    case ffi::DataType::F8E5M2FNUZ: return torch::kFloat8_e5m2fnuz;
    default: return std::nullopt;
  }
}

/// @brief Wrap an XLA buffer as a (non-owning) Torch tensor on the device the buffer lives on
static ffi::Error wrap_buffer(ffi::AnyBuffer buf, torch::DeviceType device_type, set<int64_t>& cuda_devices,
                              torch::Tensor& out) {
  auto dtype = torch_dtype(buf.element_type());
  if (!dtype)
    return ffi::Error::InvalidArgument("torch2jax: unsupported XLA FFI dtype code " +
                                       to_string(static_cast<int>(buf.element_type())));
  auto dims = buf.dimensions();
  auto dev = actual_device(device_type, buf.untyped_data());
  if (dev.has_error()) return dev.error();
  if (dev->type == torch::kCUDA) cuda_devices.insert(dev->index);
  auto device = dev->type == torch::kCPU ? torch::Device(torch::kCPU) : torch::Device(dev->type, dev->index);
  auto options = torch::TensorOptions().dtype(*dtype).device(device);
  vector<int64_t> shape(dims.begin(), dims.end());
  out = torch::from_blob(buf.untyped_data(), shape, options);
  return ffi::Error::Success();
}

static void synchronize(const set<int64_t>& cuda_devices) {
#ifdef TORCH2JAX_WITH_CUDA
  for (auto idx : cuda_devices) torch::cuda::synchronize(idx);
#endif
}

/// @brief The main torch call routine, wraps JAX arrays as Torch tensors and
/// calls the torch fn
/// @param args input buffer
/// @param rets output buffers
/// @param fn_id call fn id
/// @param device_type the accelerator type, device id is detected from pointer
ffi::Error apply_torch_call(ffi::RemainingArgs args, ffi::RemainingRets rets, const string& fn_id,
                            torch::DeviceType device_type) {
  /* ---------------------------------------------------------------------------
  The general strategy for the torch call is as follows:
    1. wrap the input buffers as Torch tensors
    2. call the identifiable Python torch function registered on the torch module
    3. validate the output tensors and copy them to the output buffers
  --------------------------------------------------------------------------- */

  // Attach a Python thread state to this thread if it doesn't have one.
  // XLA FFI calls apply_torch_call from a C++ thread with no PyThreadState;
  // in free-threading builds this does not acquire any lock — it only
  // registers the thread with the interpreter (see Python docs:
  // "C API Extension Support for Free Threading"). Declared first so it is
  // destroyed last, after all pybind11 objects below.
  py::gil_scoped_acquire py_guard;

  // 1. wrap the input buffers as Torch tensors
  set<int64_t> cuda_devices;
  py::list inputs(args.size());
  for (size_t i = 0; i < args.size(); i++) {
    torch::Tensor t;
    if (auto err = wrap_buffer(args.get<ffi::AnyBuffer>(i).value(), device_type, cuda_devices, t); !err.success())
      return err;
    inputs[i] = py::reinterpret_steal<py::object>(THPVariable_Wrap(t));
  }
  if (device_type == torch::kCUDA) synchronize(cuda_devices);

  // 2. call the identifiable Python torch function registered on the torch module
  py::tuple results = py::module_::import("torch").attr(("_torch2jax_fn_" + fn_id).c_str())(inputs);
  if (results.size() != rets.size())
    return ffi::Error::InvalidArgument("torch2jax: the torch function returned " + to_string(results.size()) +
                                       " outputs, but " + to_string(rets.size()) + " were expected");

  // 3. validate the output tensors and copy them to the output buffers
  cuda_devices.clear();
  for (size_t i = 0; i < rets.size(); i++) {
    torch::Tensor dst;
    if (auto err = wrap_buffer(*rets.get<ffi::AnyBuffer>(i).value(), device_type, cuda_devices, dst); !err.success())
      return err;
    PyObject* out = results[i].ptr();
    if (!THPVariable_Check(out))
      return ffi::Error::InvalidArgument("torch2jax: output " + to_string(i) + " of the torch function is not a tensor");
    const torch::Tensor& src = THPVariable_Unpack(out);
    if (src.sizes() != dst.sizes() || src.scalar_type() != dst.scalar_type()) {
      std::ostringstream msg;
      msg << "torch2jax: output " << i << " of the torch function is " << src.scalar_type() << src.sizes()
          << ", but " << dst.scalar_type() << dst.sizes() << " was expected (check `output_shapes`)";
      return ffi::Error::InvalidArgument(msg.str());
    }
    dst.copy_(src);
  }
  if (device_type == torch::kCUDA) synchronize(cuda_devices);
  return ffi::Error::Success();
}

#ifndef TORCH2JAX_WITH_CUDA
ffi::ErrorOr<TorchCallDevice> actual_device(torch::DeviceType device_type, void* buffer) {
  return TorchCallDevice{torch::kCPU, 0};
}
#endif
