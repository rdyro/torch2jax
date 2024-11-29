from __future__ import annotations

from typing import Callable, Any
from inspect import signature

import torch
from torch import Tensor

import jax
from jax import ShapeDtypeStruct
from jax.extend import ffi

# from jax.abstract_arrays import ShapedArray
from jax.tree_util import PyTreeDef

from .compile import compile_and_import_module
from .utils import find_unique_id, dtype_t2j, normalize_shapes


def torch2jax_flat(
    fn: Callable,
    example_args: list[Tensor] | tuple[Tensor] | None = None,
    output_shapes: list[tuple[int]] | tuple[tuple[int]] | None = None,
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
    assert example_args is not None or output_shapes is not None
    _ = compile_and_import_module()
    id = find_unique_id()

    def torch_call_fn_(args: list[torch.Tensor]):
        out = fn(*args)
        return (out,) if isinstance(out, Tensor) else tuple(out)

    setattr(torch, f"_torch2jax_fn_{id:d}", torch_call_fn_)

    @jax.jit
    def wrapped_flat_fn(*args_flat):
        if example_args is not None:
            with torch.no_grad():
                out = fn(*example_args)
            assert isinstance(out, (tuple, list, Tensor))
            out = (out,) if isinstance(out, Tensor) else tuple(out)
            outshapes = jax.tree.map(lambda x: ShapeDtypeStruct(x.shape, dtype_t2j(x.dtype)), out)
        else:
            outshapes = normalize_shapes(output_shapes, args_flat)
        if signature(ffi.ffi_call).return_annotation.startswith("Callable"):
            fn_ = ffi.ffi_call("torch_call", outshapes, vmap_method="sequential")
            return fn_(*args_flat, fn_id=f"{id:d}")
        else:
            return ffi.ffi_call("torch_call", outshapes, *args_flat, vectorized=False, fn_id=f"{id:d}")

    return wrapped_flat_fn


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
            input_struct = jax.tree.structure((example_args, example_kw))
        else:
            input_struct = jax.tree.structure(example_args)

    if output_shapes is None:
        with torch.no_grad():
            output = fn(*example_args, **example_kw) if has_kw else fn(*example_args)
        output_shapes, output_struct = jax.tree.flatten(
            jax.tree.map(lambda x: ShapeDtypeStruct(x.shape, dtype_t2j(x.dtype)), output)
        )

    else:
        output_shapes, output_struct = jax.tree.flatten(output_shapes)
        msg = "Please provide all shapes as torch.Size or jax.ShapeDtypeStruct."
        assert all(isinstance(x, (torch.Size, ShapeDtypeStruct)) or hasattr(x, "shape") for x in output_shapes), msg

    # define flattened version of the function (flat arguments and outputs)
    def flat_fn(*args_flat):
        nonlocal output_shapes, example_args
        if has_kw:
            args, kw = jax.tree.unflatten(input_struct, args_flat)
            ret = fn(*args, **kw)
        else:
            args = jax.tree.unflatten(input_struct, args_flat)
            ret = fn(*args)
        return jax.tree.flatten(ret)[0]

    # define the wrapped function using flat interface
    wrapped_fn_flat = torch2jax_flat(flat_fn, output_shapes=output_shapes, use_torch_vmap=use_torch_vmap)

    if has_kw:

        def wrapped_fn(*args, **kw):
            ret = wrapped_fn_flat(*jax.tree.flatten((args, kw))[0])
            return jax.tree.unflatten(output_struct, ret)

    else:

        def wrapped_fn(*args):
            ret = wrapped_fn_flat(*jax.tree.flatten(args)[0])
            return jax.tree.unflatten(output_struct, ret)

    return wrapped_fn
