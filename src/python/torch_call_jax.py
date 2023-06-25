from __future__ import annotations

from functools import partial
import time
from typing import Callable

import torch
from torch import Tensor
import numpy as np
from jax import core, dtypes, lax
from jax import numpy as jnp
from jax.abstract_arrays import ShapedArray
from jax.interpreters import ad, batching, mlir, xla
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call


def _register_all(mod):
    for _name, _value in mod.cpu_registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="cpu")
    for _name, _value in mod.gpu_registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")


def _shapes_to_ir_constants(in_shapes, out_shapes):
    vals = [len(in_shapes)]
    for shape in in_shapes:
        vals.append(len(shape))
        vals.extend(shape)
    vals.append(len(out_shapes))
    for shape in out_shapes:
        vals.append(len(shape))
        vals.extend(shape)
    return [np.int64(val) for val in vals]


####################################################################################################


def torch_call(*args):
    global _torch_prim
    return _torch_prim.bind(*args)


####################################################################################################


def _torch_call_abstract(*args):
    return (ShapedArray(args[0].shape, args[0].dtype),)


def _torch_call_lowering(ctx, *args, mod=None, platform=None, id: int = 17):
    assert platform in ["cpu", "gpu"]
    device = torch.device(platform if platform == "cpu" else "cuda")
    np_dtype = np.dtype(ctx.avals_in[0].dtype)
    assert np_dtype in [np.float32, np.float64] and device.type in ["cuda", "cpu"]
    op_name = platform + "_torch_call_" + ("f32" if np_dtype == np.float32 else "f64")
    ir_dtype = mlir.ir.RankedTensorType(args[0].type)
    in_shapes = [list(mlir.ir.RankedTensorType(arg.type).shape) for arg in args]
    out_shapes = [list(x.shape) for x in ctx.avals_out]
    in_layouts = [tuple(range(len(shape) - 1, -1, -1)) for shape in in_shapes]
    out_layouts = [tuple(range(len(shape) - 1, -1, -1)) for shape in out_shapes]
    device_idx = device.index if device.index is not None else 0
    opaque = mod.build_torch_call_descriptor(
        f"{id:d}", device.type, device_idx, in_shapes, out_shapes
    )
    if platform == "cpu":
        shape_desc = _shapes_to_ir_constants(in_shapes, out_shapes)
        id_desc = [np.int64(id)]
        desc = shape_desc + id_desc
        ret = custom_call(
            op_name,
            out_types=[ir_dtype for _ in out_shapes],
            operands=[mlir.ir_constant(z) for z in desc] + list(args),
            operand_layouts=[() for _ in desc] + in_layouts,
            result_layouts=out_layouts,
        )
    elif platform == "gpu":
        ret = custom_call(
            op_name,
            out_types=[ir_dtype for _ in out_shapes],
            operands=list(args),
            operand_layouts=in_layouts,
            result_layouts=out_layouts,
            backend_config=opaque,
        )
    else:
        raise NotImplementedError(f"Unsupported platform: {platform}")
    return (ret,) if len(out_layouts) == 1 else ret


# *********************************************
# *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# *********************************************
def build(mod):
    _register_all(mod)

    # global _torch_prim
    # _torch_prim = core.Primitive("torch_call")
    # _torch_prim.multiple_results = True
    # _torch_prim.def_impl(partial(xla.apply_primitive, _torch_prim))
    # _torch_prim.def_abstract_eval(_torch_call_abstract)

    # for platform in ["cpu", "gpu"]:
    #    mlir.register_lowering(
    #        _torch_prim,
    #        partial(_torch_call_lowering, mod=mod, platform=platform),
    #        platform=platform,
    #    )

    # def _torch_call_fn_17():
    #    args = torch._torch_call_args_17
    #    sums = []
    #    for i, arg in enumerate(args):
    #        sums.append(arg.sum())
    #    return (args[0] * sums[1] + args[1] * sums[0],)

    # torch._torch_call_fn_17 = _torch_call_fn_17


def wrap_torch_fn(mod: "Module", fn: Callable, args: list[Tensor], id: int = 17):
    torch_prim = core.Primitive(f"torch_call_{id}")
    torch_prim.multiple_results = True
    torch_prim.def_impl(partial(xla.apply_primitive, torch_prim))

    # call the pytorch function to infer shapes
    out = fn(*args)
    assert isinstance(out, (tuple, list, Tensor))
    out = (out,) if isinstance(out, Tensor) else tuple(out)
    dtype_map = {torch.float32: np.float32, torch.float64: np.float64}

    def _torch_call_abstract(*args):
        dtype = args[0].dtype
        return tuple(ShapedArray(z.shape, dtype) for z in out)

    torch_prim.def_abstract_eval(_torch_call_abstract)

    for platform in ["cpu", "gpu"]:
        mlir.register_lowering(
            torch_prim,
            partial(_torch_call_lowering, mod=mod, platform=platform, id=id),
            platform=platform,
        )

    def torch_call_fn_():
        args = getattr(torch, f"_torch_call_args_{id:d}")
        out = fn(*args)
        return (out,) if isinstance(out, Tensor) else tuple(out)

    setattr(torch, f"_torch_call_fn_{id:d}", torch_call_fn_)

    def wrapped_fn(*args):
        dtype, device = args[0].dtype, args[0].device
        assert all(arg.dtype == dtype for arg in args)
        return torch_prim.bind(*args)

    return wrapped_fn
