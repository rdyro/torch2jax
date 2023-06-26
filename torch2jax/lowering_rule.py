from typing import Callable
from types import ModuleType

import torch
import numpy as np
from jaxlib.hlo_helpers import custom_call
from jax.interpreters import mlir, xla

from .utils import _shapes_to_ir_constants


def _torch_call_lowering(
    ctx, *args, cpp_module: ModuleType = None, platform: str = None, id: int = 17
) -> Callable:
    assert platform in ["cpu", "gpu"]
    device = torch.device(platform if platform == "cpu" else "cuda")
    np_dtype = np.dtype(ctx.avals_in[0].dtype)
    assert np_dtype in [np.float32, np.float64] and device.type in ["cuda", "cpu"]
    op_name = platform + "_torch_call_" + ("f32" if np_dtype == np.float32 else "f64")
    out_types = [
        mlir.ir.RankedTensorType.get(aval.shape, mlir.dtype_to_ir_type(aval.dtype))
        for aval in ctx.avals_out
    ]
    in_shapes = [list(mlir.ir.RankedTensorType(arg.type).shape) for arg in args]
    out_shapes = [list(x.shape) for x in ctx.avals_out]
    in_layouts = [tuple(range(len(shape) - 1, -1, -1)) for shape in in_shapes]
    out_layouts = [tuple(range(len(shape) - 1, -1, -1)) for shape in out_shapes]
    device_idx = device.index if device.index is not None else 0
    opaque = cpp_module.build_torch_call_descriptor(
        id, device.type, device_idx, in_shapes, out_shapes
    )
    if platform == "cpu":
        shape_desc = _shapes_to_ir_constants(in_shapes, out_shapes)
        id_desc = [np.int64(id)]
        desc = shape_desc + id_desc
        ret = custom_call(
            op_name,
            out_types=out_types,
            operands=[mlir.ir_constant(z) for z in desc] + list(args),
            operand_layouts=[() for _ in desc] + in_layouts,
            result_layouts=out_layouts,
        )
    elif platform == "gpu":
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
