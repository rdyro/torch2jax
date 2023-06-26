from __future__ import annotations

from typing import Callable
from functools import partial

import torch
from torch import Tensor
from jax.interpreters import mlir, xla
from jax import core, dtypes, lax
from jax.abstract_arrays import ShapedArray

from .compile import compile_and_import_module
from .lowering_rule import _torch_call_lowering


def torch2jax(
    fn: Callable,
    id: int = 17,
    example_args: list[Tensor] | tuple[Tensor] = None,
    output_shapes: list[tuple[int]] | tuple[tuple[int]] = None,
    output_shapes_fn: Callable = None,
) -> Callable:
    assert example_args is not None or output_shapes is not None or output_shapes_fn is not None
    cpp_module = compile_and_import_module()

    torch_prim = core.Primitive(f"torch_call_{id}")
    torch_prim.multiple_results = True
    torch_prim.def_impl(partial(xla.apply_primitive, torch_prim))
    # call the pytorch function to infer shapes
    if output_shapes is not None:
        assert isinstance(output_shapes, (tuple, list)) and all(
            isinstance(shape, (tuple, list)) for shape in output_shapes
        )

        def _torch_call_abstract(*args):
            dtype = args[-1].dtype
            assert all(arg.dtype == dtype for arg in args)
            return tuple(ShapedArray(shape, dtype) for shape in output_shapes)

    elif output_shapes_fn is not None:

        def _torch_call_abstract(*args):
            dtype = args[-1].dtype
            assert all(arg.dtype == dtype for arg in args)
            return tuple(ShapedArray(shape, dtype) for shape in output_shapes_fn(*args))

    else:
        out = fn(*example_args)
        assert isinstance(out, (tuple, list, Tensor))
        out = (out,) if isinstance(out, Tensor) else tuple(out)

        def _torch_call_abstract(*args):
            dtype = args[-1].dtype
            assert all(arg.dtype == dtype for arg in args)
            return tuple(ShapedArray(z.shape, dtype) for z in out)

    torch_prim.def_abstract_eval(_torch_call_abstract)

    for platform in ["cpu", "gpu"]:
        mlir.register_lowering(
            torch_prim,
            partial(_torch_call_lowering, cpp_module=cpp_module, platform=platform, id=id),
            platform=platform,
        )

    def torch_call_fn_():
        args = getattr(torch, f"_torch_call_args_{id:d}")
        out = fn(*args)
        return (out,) if isinstance(out, Tensor) else tuple(out)

    setattr(torch, f"_torch_call_fn_{id:d}", torch_call_fn_)

    def wrapped_fn(*args):
        dtype = args[-1].dtype
        assert all(arg.dtype == dtype for arg in args)
        return torch_prim.bind(*args)

    return wrapped_fn
