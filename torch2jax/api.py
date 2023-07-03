from __future__ import annotations

from typing import Callable, Any
from functools import partial

import torch
from torch import Tensor
from jax.interpreters import mlir, xla
from jax import core, ShapeDtypeStruct
from jax.abstract_arrays import ShapedArray
from jax.tree_util import tree_flatten, tree_unflatten, tree_structure, tree_map

from .compile import compile_and_import_module
from .lowering_rule import _torch_call_lowering
from .utils import _find_unique_id


def torch2jax_v1(
    fn: Callable,
    example_args: list[Tensor] | tuple[Tensor] = None,
    output_shapes: list[tuple[int]] | tuple[tuple[int]] = None,
    output_shapes_fn: Callable = None,
) -> Callable:
    assert example_args is not None or output_shapes is not None or output_shapes_fn is not None
    cpp_module = compile_and_import_module()
    id = _find_unique_id()

    torch_prim = core.Primitive(f"torch_call_{id}")
    torch_prim.multiple_results = True
    torch_prim.def_impl(partial(xla.apply_primitive, torch_prim))

    # inferring shapes #############################################################################
    if output_shapes is not None:
        # call the pytorch function to infer shapes
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
    # inferring shapes #############################################################################

    # lowering ####################################################################################
    for platform in ["cpu", "gpu"]:
        mlir.register_lowering(
            torch_prim,
            partial(_torch_call_lowering, cpp_module=cpp_module, platform=platform, id=id),
            platform=platform,
        )
    # lowering ####################################################################################

    def torch_call_fn_():
        args = getattr(torch, f"_torch2jax_args_{id:d}")
        out = fn(*args)
        return (out,) if isinstance(out, Tensor) else tuple(out)

    setattr(torch, f"_torch2jax_fn_{id:d}", torch_call_fn_)

    def wrapped_fn(*args):
        dtype = args[-1].dtype
        assert all(arg.dtype == dtype for arg in args)
        return torch_prim.bind(*args)

    return wrapped_fn


def torch2jax(
    fn: Callable,
    *example_args: Any,
    example_kw: Any | None = None,
    example_kwargs: Any | None = None,
    output_shapes: "NestedShapeContainer" = None,
    input_struct: "NestedContainer" | None = None,
) -> Callable:
    # check for presence of example_args and example_kw
    msg = "Please provide either example_kw or example_kwargs, not both."
    assert example_kw is None or example_kwargs is None, msg
    if example_kwargs is not None:
        example_kw = example_kwargs
    has_kw = example_kw is not None

    if input_struct is None:
        if has_kw:
            input_struct = tree_structure((example_args, example_kw))
        else:
            input_struct = tree_structure(example_args)

    if output_shapes is None:
        output = fn(*example_args, **example_kw) if has_kw else fn(*example_args)
        output_flat, output_struct = tree_flatten(output)
        output_shapes = [torch.Size(x.shape) for x in output_flat]
    else:
        output_shapes, output_struct = tree_flatten(output_shapes)
        msg = "Please provide all shapes as torch.Size or jax.ShapeDtypeStruct (dtype is ignored)."
        assert all(
            isinstance(x, (torch.Size, ShapedArray, ShapeDtypeStruct)) for x in output_shapes
        ), msg
        output_shapes = [
            x if isinstance(x, torch.Size) else torch.Size(x.shape) for x in output_shapes
        ]

    def flat_fn(*args_flat):
        if has_kw:
            args, kw = tree_unflatten(input_struct, args_flat)
            ret = fn(*args, **kw)
        else:
            args = tree_unflatten(input_struct, args_flat)
            ret = fn(*args)
        return tree_flatten(ret)[0]

    wrapped_fn_v1 = torch2jax_v1(flat_fn, output_shapes=output_shapes)

    if has_kw:

        def wrapped_fn(*args, **kw):
            ret = wrapped_fn_v1(*tree_flatten((args, kw))[0])
            return tree_unflatten(output_struct, ret)

    else:

        def wrapped_fn(*args):
            ret = wrapped_fn_v1(*tree_flatten(args)[0])
            return tree_unflatten(output_struct, ret)

    return wrapped_fn
