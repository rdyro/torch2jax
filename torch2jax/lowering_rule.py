from typing import Callable
from types import ModuleType

import torch
import numpy as np
from jaxlib.hlo_helpers import custom_call
from jax.interpreters import mlir


def dtype_translation(cpp_module: ModuleType, dtype: np.dtype) -> int:
    if dtype == bool:
        return cpp_module.DATA_TYPE_BOOL
    elif dtype == np.uint8:
        return cpp_module.DATA_TYPE_UINT8
    elif dtype == np.int8:
        return cpp_module.DATA_TYPE_INT8
    elif dtype == np.int16:
        return cpp_module.DATA_TYPE_INT16
    elif dtype == np.int32:
        return cpp_module.DATA_TYPE_INT32
    elif dtype == np.int64:
        return cpp_module.DATA_TYPE_INT64
    elif dtype == np.float16:
        return cpp_module.DATA_TYPE_FLOAT16
    elif dtype == np.float32:
        return cpp_module.DATA_TYPE_FLOAT32
    elif dtype == np.float64:
        return cpp_module.DATA_TYPE_FLOAT64
    else:
        raise ValueError("Unsupported dtype: {}".format(dtype))


def _torch_call_lowering(
    ctx, *args, cpp_module: ModuleType = None, platform: str = None, id: int = 17
) -> Callable:
    assert platform in ["cpu", "gpu"]
    device = torch.device(platform if platform == "cpu" else "cuda")
    assert device.type in ["cuda", "cpu"]
    op_name = platform + "_torch_call"
    out_types = [
        mlir.ir.RankedTensorType.get(aval.shape, mlir.dtype_to_ir_type(aval.dtype))
        for aval in ctx.avals_out
    ]
    in_shapes = [list(mlir.ir.RankedTensorType(arg.type).shape) for arg in args]
    out_shapes = [list(x.shape) for x in ctx.avals_out]
    torch_call_in_dtypes = [
        dtype_translation(cpp_module, np.dtype(x.dtype)) for x in ctx.avals_in
    ]
    torch_call_out_dtypes = [
        dtype_translation(cpp_module, np.dtype(x.dtype)) for x in ctx.avals_out
    ]
    in_layouts = [tuple(range(len(shape) - 1, -1, -1)) for shape in in_shapes]
    out_layouts = [tuple(range(len(shape) - 1, -1, -1)) for shape in out_shapes]
    if platform == "cpu":
        desc = cpp_module.serialize_cpu_descriptor(
            int(id),
            cpp_module.DEVICE_TYPE_CPU,
            0,
            in_shapes,
            torch_call_in_dtypes,
            out_shapes,
            torch_call_out_dtypes,
        )
        ret = custom_call(
            op_name,
            out_types=out_types,
            operands=[mlir.ir_constant(z) for z in desc] + list(args),
            operand_layouts=[() for _ in desc] + in_layouts,
            result_layouts=out_layouts,
        )
    elif platform == "gpu":
        device_idx = device.index if device.index is not None else 0
        opaque = cpp_module.serialize_gpu_descriptor(
            int(id),
            cpp_module.DEVICE_TYPE_CUDA,
            int(device_idx),
            in_shapes,
            torch_call_in_dtypes,
            out_shapes,
            torch_call_out_dtypes,
        )
        ret = custom_call(
            op_name,
            out_types=out_types,
            operands=list(args),
            operand_layouts=in_layouts,
            result_layouts=out_layouts,
            backend_config=opaque,
        )
    else:
        raise NotImplementedError(f"Unsupported platform: {platform}")
    return (ret,) if len(out_layouts) == 1 else ret
