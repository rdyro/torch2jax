from __future__ import annotations

from typing import Callable, Any
from functools import partial
from warnings import warn

import torch
from torch import Tensor, Size
from jax import numpy as jnp
from jax.interpreters import mlir, xla, batching
from jax import core, ShapeDtypeStruct

# from jax.abstract_arrays import ShapedArray
from jax.core import ShapedArray
from jax.tree_util import tree_flatten, tree_unflatten, tree_structure, PyTreeDef, tree_map

from .compile import compile_and_import_module
from .lowering_rule import _torch_call_lowering
from .utils import find_unique_id, dtype_t2j, normalize_shapes


def torch2jax_flat(
    fn: Callable,
    example_args: list[Tensor] | tuple[Tensor] = None,
    output_shapes: list[tuple[int]] | tuple[tuple[int]] = None,
    use_torch_vmap: bool = True,
) -> Callable:
    """Define a jit-compatible JAX function that calls a PyTorch function. Flat
    arguments and outputs.

    Args:
        fn (Callable): PyTorch function.
        example_args (list[Tensor] | tuple[Tensor], optional): Example arguments. Defaults to None.
        output_shapes (list[tuple[int]] | tuple[tuple[int]], optional): Output shapes (or shapes
                                                                        with dtype).  Defaults to
                                                                        None.
        use_torch_vmap (bool, optional): Whether to use torch.vmap for jax batching rule.
                                         Alternatively use a dumb for loop. Defaults to True.

    Returns:
        Callable: Wrapped jit-compatible jax function.
    """
    # assert example_args is not None or output_shapes is not None or output_shapes_fn is not None
    cpp_module = compile_and_import_module()
    id = find_unique_id()

    torch_prim = core.Primitive(f"torch_call_{id}")
    torch_prim.multiple_results = True
    torch_prim.def_impl(partial(xla.apply_primitive, torch_prim))

    # inferring shapes #############################################################################
    if output_shapes is not None:
        # call the pytorch function to infer shapes
        assert isinstance(output_shapes, (tuple, list)) and all(
            isinstance(shape, (tuple, list, ShapedArray, ShapeDtypeStruct, Size))
            or shape is None
            or hasattr(shape, "shape")
            for shape in output_shapes
        )

        def _torch_call_abstract(*args):
            output_shapes_ = [
                Size(shape) if isinstance(shape, (list, tuple)) else shape
                for shape in output_shapes
            ]
            return normalize_shapes(output_shapes_, args)

    else:
        with torch.no_grad():
            out = fn(*example_args)
        assert isinstance(out, (tuple, list, Tensor))
        out = (out,) if isinstance(out, Tensor) else tuple(out)

        def _torch_call_abstract(*args):
            return tree_map(lambda x: ShapedArray(x.shape, dtype_t2j(x.dtype)), out)

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
        return torch_prim.bind(*args)

    def torch_call_batching(args, axes):
        if use_torch_vmap:

            def torch_fn_vmap(*args):
                return torch.vmap(fn, in_dims=axes)(*args)

            assert any(ax is not None for ax in axes)
            batch_size = [arg.shape[ax] for arg, ax in zip(args, axes) if ax is not None][0]
            assert output_shapes is not None
            output_shapes_ = _torch_call_abstract(*args)
            output_shapes_vmap = [
                ShapedArray((batch_size,) + tuple(shape.shape), shape.dtype)
                for shape in output_shapes_
            ]
            outaxes = (0 for _ in output_shapes_vmap)
            return (
                torch2jax_flat(torch_fn_vmap, args, output_shapes=output_shapes_vmap)(*args),
                outaxes,
            )
        else:
            warn(
                "You are NOT using PyTorch's functional vmap. "
                + "This is highly experimental and may be slower."
            )
            assert all(axis is None or axis == 0 for axis in axes)
            if all(axis is None for axis in axes):
                return wrapped_fn(*args)
            n = 0
            for i, axis in enumerate(axes):
                if axis is not None:
                    n = args[i].shape[axis]
                    break
            output_lists, output_struct = None, None
            for i in range(n):
                args_ = [arg if axis is None else arg[i] for arg, axis in zip(args, axes)]
                outputs = wrapped_fn(*args_)
                output_flat, output_struct = tree_flatten(outputs)
                if output_lists is None:
                    output_lists = [[] for _ in output_flat]
                for output_list, output in zip(output_lists, output_flat):
                    output_list.append(output)
            outputs = tuple([jnp.stack(output_list, 0) for output_list in output_lists])
            outputs = tree_unflatten(output_struct, outputs)
            return outputs, tree_unflatten(output_struct, (0 for _ in outputs))

    batching.primitive_batchers[torch_prim] = torch_call_batching

    return wrapped_fn


####################################################################################################


def torch2jax(
    fn: Callable,
    *example_args: Any,
    example_kw: Any | None = None,
    example_kwargs: Any | None = None,
    output_shapes: Any = None,
    input_struct: PyTreeDef | None = None,
    use_torch_vmap: bool = True,
) -> Callable:
    """Define a jit-compatible JAX function that calls a PyTorch function.  Arbitrary nesting of
    arguments and outputs is supported.

    Args:
        fn (Callable): PyTorch function to wrap.
        *example_args (Any): Example arguments as tensors or torch-compatible args.
        example_kw (Any | None, optional): Example keyword arguments. Defaults to None.
        example_kwargs (Any | None, optional): Example keyword arguments. Defaults to None.
        output_shapes (Any, optional): Output shapes or shapes + dtype struct. Defaults to None.
        input_struct (PyTreeDef | None, optional): Input structure, which can be inferred from
                                                   example arguments and keywords. Defaults to None.
        use_torch_vmap (bool, optional): Whether to batch using torch.vmap or a dumb loop. Defaults to
                                         True.
    Returns:
        Callable: JIT-compatible JAX function.

    Examples:
        >>> import torch, jax
        >>> from torch2jax import torch2jax_with_vjp, tree_t2j
        >>> # let's define the torch function and create some example arguments
        >>> torch_fn = lambda x, y: torch.nn.CrossEntropyLoss()(x, y)
        >>> xt, yt = torch.randn(10, 5), torch.randint(0, 5, (10,))
        >>> # we can not convert the function to jax using the torch fn and example args
        >>> jax_fn = torch2jax_with_vjp(torch_fn, xt, yt)
        >>> jax_fn = jax.jit(jax_fn) # we can jit it too
        >>> # let's convert the arguments to JAX arrays and call the function
        >>> x, y = tree_t2j((xt, yt))
        >>> jax_fn(x, y)
        >>> # it works!
    """

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
        with torch.no_grad():
            output = fn(*example_args, **example_kw) if has_kw else fn(*example_args)
        output_shapes, output_struct = tree_flatten(
            tree_map(lambda x: ShapeDtypeStruct(x.shape, dtype_t2j(x.dtype)), output)
        )

    else:
        output_shapes, output_struct = tree_flatten(output_shapes)
        msg = "Please provide all shapes as torch.Size or jax.ShapeDtypeStruct."
        assert all(
            isinstance(x, (torch.Size, ShapedArray, ShapeDtypeStruct)) or hasattr(x, "shape")
            for x in output_shapes
        ), msg

    # define flattened version of the function (flat arguments and outputs)
    def flat_fn(*args_flat):
        nonlocal output_shapes, example_args
        if has_kw:
            args, kw = tree_unflatten(input_struct, args_flat)
            ret = fn(*args, **kw)
        else:
            args = tree_unflatten(input_struct, args_flat)
            ret = fn(*args)
        return tree_flatten(ret)[0]

    # define the wrapped function using flat interface
    wrapped_fn_flat = torch2jax_flat(
        flat_fn, output_shapes=output_shapes, use_torch_vmap=use_torch_vmap
    )

    if has_kw:

        def wrapped_fn(*args, **kw):
            ret = wrapped_fn_flat(*tree_flatten((args, kw))[0])
            return tree_unflatten(output_struct, ret)

    else:

        def wrapped_fn(*args):
            ret = wrapped_fn_flat(*tree_flatten(args)[0])
            return tree_unflatten(output_struct, ret)

    return wrapped_fn
