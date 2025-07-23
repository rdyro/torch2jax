from __future__ import annotations

import math
import functools
from typing import Callable, Any
from inspect import signature

import torch
from torch import Tensor
import jax
from jax import ShapeDtypeStruct
try:
    from jax.util import safe_zip
except ImportError:
    safe_zip = zip

jax.config.update('jax_use_shardy_partitioner', False)  # TODO: temporary workaround for JAX 0.7.0

# jax version-friendly way of importing the ffi module in jax
try:
    from jax import ffi
except ImportError:
    from jax.extend import ffi

from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec, Mesh

from .compile import compile_and_import_module
from .utils import find_unique_id, dtype_t2j, normalize_shapes, warn_once


def _gen_ffi_call(outshapes, vmap_method: str):
    if signature(ffi.ffi_call).return_annotation.startswith("Callable"):
        fn_ = ffi.ffi_call("torch_call", outshapes, vmap_method=vmap_method)
    else:
        if vmap_method != "sequential":
            raise ValueError(
                f"You specificed {vmap_method=}, but your jax version {jax.__version__} does not support new style of"
                " `vmap_method=` specification. Please upgrade your JAX version to use this features"
            )
        fn_ = lambda *args_flat, fn_id: ffi.ffi_call("torch_call", outshapes, *args_flat, vectorized=False, fn_id=fn_id)
    return fn_


def _torch2jax_flat(
    fn: Callable,
    input_shapes: list[jax.Array | Tensor | ShapeDtypeStruct] = None,
    output_shapes: list[jax.Array | Tensor | ShapeDtypeStruct] = None,
    output_sharding_spec: PartitionSpec | None = None,
    vmap_method: str = "sequential",
) -> Callable:
    """Define a jit-compatible JAX function that calls a PyTorch function. Flat
    arguments and outputs.

    Args:
        fn (Callable): PyTorch function.
        example_args: Example arguments. Defaults to None.
        output_shapes: Output shapes (or shapes with dtype). Defaults to None.
        output_sharding_spec: jax.sharding.PartitionSpec specifying the sharding spec of the output, uses input mesh.
    Returns:
        Callable: Wrapped jit-compatible jax function.
    """
    # assert example_args is not None or output_shapes is not None or output_shapes_fn is not None
    _ = compile_and_import_module()
    id = find_unique_id()

    def torch_call_fn_(args: list[torch.Tensor]):
        nonlocal output_shapes
        out = fn(*args)
        return (out,) if isinstance(out, Tensor) else tuple(out)

    setattr(torch, f"_torch2jax_fn_{id:d}", torch_call_fn_)

    inshapes = None
    if input_shapes is not None:
        inshapes = jax.tree.map(lambda x: ShapeDtypeStruct(x.shape, dtype_t2j(x.dtype)), input_shapes)
    assert output_shapes is not None, "`output_shapes` cannot be None"
    outshapes = jax.tree.map(lambda x: ShapeDtypeStruct(x.shape, dtype_t2j(x.dtype)), output_shapes)

    @jax.jit
    def wrapped_flat_fn(*args_flat):
        nonlocal inshapes, outshapes
        fn_ = _gen_ffi_call(outshapes, vmap_method=vmap_method)

        if output_sharding_spec is None:
            fn_id = f"{id:d}"
            return fn_(*args_flat, fn_id=fn_id)

        @functools.partial(custom_partitioning, static_argnums=(0,))
        def partitioned_f(fn_id: str, *args_flat):
            assert fn_id is not None
            return fn_(*args_flat, fn_id=fn_id)

        def infer_sharding(fn_id, mesh, args_info, result_info):
            del fn_id
            assert len(args_info) > 0
            result_sharding = jax.tree.map(lambda r, spec: NamedSharding(mesh, spec), result_info, output_sharding_spec)
            return result_sharding

        def fn_partition(fn_id, mesh: Mesh, args_info, result_info):
            args_sharding = jax.tree.map(lambda x: x.sharding, args_info)
            result_sharding = infer_sharding(fn_id, mesh, args_info, result_info)

            def _partitioned_fn_(*args_flat, fn_id=fn_id):
                axis_sizes = dict(safe_zip(mesh.axis_names, mesh.device_ids.shape))
                for arg_info, arg in safe_zip(jax.tree.leaves(args_info), jax.tree.leaves(args_flat)):
                    for s_all, s_part, axis in safe_zip(arg_info.shape, arg.shape, arg_info.sharding.spec):
                        if axis is None:
                            continue
                        axes = axis if isinstance(axis, (list, tuple)) else [axis]
                        div = math.prod(axis_sizes[ax] for ax in axes)
                        assert s_part * div == s_all

                def _map_outshape(outshape: jax.ShapeDtypeStruct, result_info, result_sharding):
                    new_outshape = []
                    spec = tuple(result_sharding.spec)
                    assert len(spec) == len(outshape.shape)
                    for s, axis in safe_zip(outshape.shape, spec):
                        if axis is None:
                            new_outshape.append(s)
                        else:
                            axes = axis if isinstance(axis, (list, tuple)) else [axis]
                            div = math.prod(axis_sizes[ax] for ax in axes)
                            new_outshape.append(s // div)
                    return jax.ShapeDtypeStruct(new_outshape, dtype=outshape.dtype)

                new_outshapes = jax.tree.map(_map_outshape, outshapes, result_info, result_sharding)
                fn_part_ = _gen_ffi_call(new_outshapes, vmap_method=vmap_method)
                return fn_part_(*args_flat, fn_id=fn_id)

            return mesh, _partitioned_fn_, result_sharding, args_sharding

        fn_id = f"{id:d}"

        partitioned_f.def_partition(infer_sharding_from_operands=infer_sharding, partition=fn_partition)
        return partitioned_f(fn_id, *args_flat)

    return wrapped_flat_fn


def torch2jax(
    fn: Callable,
    *example_args: Any,
    example_kw: Any | None = None,
    output_shapes: Any = None,
    output_sharding_spec: PartitionSpec | None = None,
    vmap_method: str = "sequential",
) -> Callable:
    """Define a jit-compatible JAX function that calls a PyTorch function.  Arbitrary nesting of
    arguments and outputs is supported.

    Args:
        fn (Callable): PyTorch function to wrap.
        *example_args (Any): Example arguments as tensors or torch-compatible args.
        example_kw: Example keyword arguments. Defaults to None.
        output_shapes: Output shapes or shapes + dtype struct. Defaults to None.
        output_sharding_spec: jax.sharding.PartitionSpec specifying the sharding spec of the output, uses input mesh.
        vmap_method: batching method, see
            [https://docs.jax.dev/en/latest/ffi.html#batching-with-vmap](https://docs.jax.dev/en/latest/ffi.html#batching-with-vmap)

            NOTE: only vmap_method="sequntial" is supported non-experimentally

            NOTE: try "expand_dims", "broadcast_all" if you want to experiment with pytorch-side batching
    Returns:
        Callable: JIT-compatible JAX function.

    Examples:
        >>> import torch, jax
        >>> from torch2jax import torch2jax_with_vjp, tree_t2j
        >>> # let's define the torch function and create some example arguments
        >>> torch_fn = lambda x, y: torch.nn.CrossEntropyLoss()(x, y)
        >>> xt, yt = torch.randn(10, 5), torch.randint(0, 5, (10,))
        >>> # we can now convert the function to jax using the torch fn and example args
        >>> jax_fn = torch2jax_with_vjp(torch_fn, xt, yt)
        >>> jax_fn = jax.jit(jax_fn) # we can jit it too
        >>> # let's convert the arguments to JAX arrays and call the function
        >>> x, y = tree_t2j((xt, yt))
        >>> jax_fn(x, y)
        >>> # it works!
    """

    # check for presence of example_args and example_kw
    has_kw = example_kw is not None

    # find the input structure
    if has_kw:
        input_struct = jax.tree.structure((example_args, example_kw))
    else:
        input_struct = jax.tree.structure(example_args)

    # define flattened version of the function (flat arguments and outputs)
    def flat_fn(*args_flat):
        if has_kw:
            args, kw = jax.tree.unflatten(input_struct, args_flat)
            ret = fn(*args, **kw)
        else:
            args = jax.tree.unflatten(input_struct, args_flat)
            ret = fn(*args)
        return jax.tree.leaves(ret)

    example_inputs = (example_args, example_kw) if has_kw else example_args
    input_shapes = jax.tree.map(lambda x: ShapeDtypeStruct(x.shape, dtype_t2j(x.dtype)), example_inputs)

    # find the output structure
    if output_shapes is None:
        with torch.no_grad():
            output = fn(*example_args, **example_kw) if has_kw else fn(*example_args)
        output_shapes, output_struct = jax.tree.flatten(
            jax.tree.map(lambda x: ShapeDtypeStruct(x.shape, dtype_t2j(x.dtype)), output)
        )
    else:
        if not all(
            isinstance(x, (torch.Size, ShapeDtypeStruct, jax.Array, torch.Tensor)) or hasattr(x, "shape")
            for x in jax.tree.leaves(output_shapes)
        ):
            warn_once(
                "Please provide all shapes as torch.Size or jax.ShapeDtypeStruct. We'll attempt to guess all"
                " containers with only integer entries are shapes (for compatibility), but this is very error-prone.",
                fn,
            )
        output_shapes = normalize_shapes(output_shapes, extra_args=input_shapes)
        output_shapes, output_struct = jax.tree.flatten(output_shapes)
    if output_sharding_spec is not None:
        output_sharding_spec_flat, output_sharding_struct = jax.tree.flatten(output_sharding_spec)
        msg = (
            "When providing `output_shading_spec` its structure must match the structure of `output_shapes`."
            f"\nExpected: {output_struct}\n Actual:   {output_sharding_struct}"
        )
        assert jax.tree.structure(output_sharding_spec) == output_struct, msg
    else:
        output_sharding_spec_flat, output_sharding_struct = None, None

    # define the wrapped function using flat interface
    wrapped_fn_flat = _torch2jax_flat(
        flat_fn,
        input_shapes=None,
        output_shapes=output_shapes,
        output_sharding_spec=output_sharding_spec_flat,
        vmap_method=vmap_method,
    )

    # define the actual wrapper function
    def wrapped_fn(*args, **kw):
        nonlocal fn, input_shapes, output_shapes
        if not has_kw and len(kw) > 0:
            raise RuntimeError("Keyword arguments not expected!")
        if has_kw:
            args = (args, kw)
            mismatch_args_msg = (
                "Provided (args, kw) =\n{} do not match the torch2jax function's expected input structure =\n{}"
            )
        else:
            mismatch_args_msg = (
                "Provided args =\n{} do not match the torch2jax function's expected input structure =\n{}"
            )
        if jax.tree.structure(args) != input_struct:
            raise RuntimeError(mismatch_args_msg.format(args, input_struct))

        common_mismatch_input_msg = (
            f"\nActual = {args}\nExpected = {input_shapes}"
            f"\nAre you perhaps using a JAX transformation like `shard_map`, `vmap` or `pmap`?"
            " You can try defining torch2jax eagerly inside `shard_map` or defining an un-batched version for `pmap`."
            " However, torch2jax is currently NOT WORKING with `pmap`, please use `shard_map`"
            " or the experimental `auto_partitioning=True`"
        )
        if output_sharding_spec:
            if not jax.tree.all(jax.tree.map(lambda x, y: getattr(x, "ndim", -1) == y.ndim, args, input_shapes)):
                msg = (
                    "Not all inputs to your torch2jax function match the dimensions of the expected input."
                    + common_mismatch_input_msg
                )
                raise RuntimeError(msg)
        else:
            if not jax.tree.all(jax.tree.map(lambda x, y: getattr(x, "shape", [-1]) == y.shape, args, input_shapes)):
                msg = (
                    "Not all inputs to your torch2jax function match the shapes of the expected input."
                    + common_mismatch_input_msg
                )
                raise RuntimeError(msg)
        ret = wrapped_fn_flat(*jax.tree.leaves(args))
        return jax.tree.unflatten(output_struct, ret)

    return wrapped_fn
