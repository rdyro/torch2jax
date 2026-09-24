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
static ffi::Error wrap_buffer(ffi::AnyBuffer buf, torch::DeviceType device_type, vector<torch::Tensor>& out) {
  auto dtype = torch_dtype(buf.element_type());
  if (!dtype)
    return ffi::Error::InvalidArgument("torch2jax: unsupported XLA FFI dtype code " +
                                       to_string(static_cast<int>(buf.element_type())));
  auto dims = buf.dimensions();
  vector<int64_t> shape(dims.begin(), dims.end());
  auto dev = actual_device(device_type, buf.untyped_data());
  if (dev.has_error()) return dev.error();
  auto device = dev->type == torch::kCPU ? torch::Device(torch::kCPU) : torch::Device(dev->type, dev->index);
  out.push_back(torch::from_blob(buf.untyped_data(), shape, torch::TensorOptions().dtype(*dtype).device(device)));
  return ffi::Error::Success();
}

/// @brief The main torch call routine, wraps JAX arrays as Torch tensors and
/// calls the torch fn
/// @param args input buffer
/// @param rets output buffers
/// @param fn_id call fn id
/// @param device_type the accelerator type, device id is detected from pointer
/// @param stream the XLA CUDA stream (cudaStream_t) the torch computation is enqueued on, nullptr on CPU
ffi::Error apply_torch_call(ffi::RemainingArgs args, ffi::RemainingRets rets, const string& fn_id,
                            torch::DeviceType device_type, void* stream) {
  /* ---------------------------------------------------------------------------
  The general strategy for the torch call is as follows:
    1. wrap the input and output buffers as Torch tensors
    2. make XLA's stream the current torch stream (after torch's own stream), so
       the torch computation is ordered with respect to the XLA computation and
       prior torch work without device synchronization
    3. call the identifiable Python torch function registered on the torch module
    4. validate the output tensors and copy them to the output buffers
    5. order torch's own stream after XLA's stream for later torch work
  --------------------------------------------------------------------------- */

  // Attach a Python thread state to this thread if it doesn't have one.
  // XLA FFI calls apply_torch_call from a C++ thread with no PyThreadState;
  // in free-threading builds this does not acquire any lock — it only
  // registers the thread with the interpreter (see Python docs:
  // "C API Extension Support for Free Threading"). Declared first so it is
  // destroyed last, after all pybind11 objects below.
  py::gil_scoped_acquire py_guard;

  // 1. wrap the input and output buffers as Torch tensors
  vector<torch::Tensor> ins, outs;
  for (size_t i = 0; i < args.size(); i++)
    if (auto err = wrap_buffer(args.get<ffi::AnyBuffer>(i).value(), device_type, ins); err.failure()) return err;
  for (size_t i = 0; i < rets.size(); i++)
    if (auto err = wrap_buffer(*rets.get<ffi::AnyBuffer>(i).value(), device_type, outs); err.failure()) return err;

  // 2. make XLA's stream the current torch stream (after torch's own stream)
#ifdef TORCH2JAX_WITH_CUDA
  std::optional<c10::cuda::CUDAStream> torch_stream, xla_stream;
  std::optional<c10::cuda::CUDAStreamGuard> stream_guard;
  if (device_type == torch::kCUDA && (!ins.empty() || !outs.empty())) {
    auto device_idx = (!ins.empty() ? ins[0] : outs[0]).device().index();
    torch_stream = c10::cuda::getCurrentCUDAStream(device_idx);
    xla_stream = c10::cuda::getStreamFromExternal(static_cast<cudaStream_t>(stream), device_idx);
    at::cuda::CUDAEvent event;  // torch work queued before the call (e.g., a weight update) happens first
    event.record(*torch_stream);
    event.block(*xla_stream);
    stream_guard.emplace(*xla_stream);
  }
#endif

  // 3. call the identifiable Python torch function registered on the torch module
  py::list inputs(ins.size());
  for (size_t i = 0; i < ins.size(); i++) inputs[i] = py::reinterpret_steal<py::object>(THPVariable_Wrap(ins[i]));
  py::tuple results = py::module_::import("torch").attr(("_torch2jax_fn_" + fn_id).c_str())(inputs);
  if (results.size() != rets.size())
    return ffi::Error::InvalidArgument("torch2jax: the torch function returned " + to_string(results.size()) +
                                       " outputs, but " + to_string(rets.size()) + " were expected");

  // 4. validate the output tensors and copy them to the output buffers
  for (size_t i = 0; i < outs.size(); i++) {
    PyObject* out = results[i].ptr();
    if (!THPVariable_Check(out))
      return ffi::Error::InvalidArgument("torch2jax: output " + to_string(i) + " of the torch function is not a tensor");
    const torch::Tensor& src = THPVariable_Unpack(out);
    if (src.sizes() != outs[i].sizes() || src.scalar_type() != outs[i].scalar_type()) {
      std::ostringstream msg;
      msg << "torch2jax: output " << i << " of the torch function is " << src.scalar_type() << src.sizes()
          << ", but " << outs[i].scalar_type() << outs[i].sizes() << " was expected (check `output_shapes`)";
      return ffi::Error::InvalidArgument(msg.str());
    }
    outs[i].copy_(src);
  }
#ifdef TORCH2JAX_WITH_CUDA
  // 5. order torch's own stream after XLA's stream
  if (torch_stream) {  // torch work queued after the call sees the state the torch fn modified (e.g., buffers)
    at::cuda::CUDAEvent event;
    event.record(*xla_stream);
    event.block(*torch_stream);
  }
#endif
  return ffi::Error::Success();
}

#ifndef TORCH2JAX_WITH_CUDA
ffi::ErrorOr<TorchCallDevice> actual_device(torch::DeviceType device_type, void* buffer) {
  return TorchCallDevice{torch::kCPU, 0};
}
#endif
