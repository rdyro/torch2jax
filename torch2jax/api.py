import functools
from typing import Callable, Any

import torch
from torch import Tensor
import jax
from jax import ShapeDtypeStruct
import numpy as np

from jax import ffi

from jax.sharding import PartitionSpec

from .compile import compile_and_import_module
from .utils import find_unique_id, dtype_t2j, dtype_j2t, normalize_shapes, warn_once, warn_always
from .utils import canonical_dtype, shape_key, placeholder_like, infer_outputs
from .sharding import shard_call, match_vma, union_vma

zip_ = zip
zip = functools.partial(zip_, strict=True)

_SHAPE_CHANGE_WARN_EXPLICIT = (
    "torch2jax: input shapes changed, but `output_shapes` was explicitly provided. Output shapes for the new inputs"
    " will be inferred by re-running the torch function on placeholder (meta) tensors.\nExpecting: {}\nActual:    {}"
)
_SHAPE_CHANGE_WARN_CONCRETE = (
    "torch2jax: input shapes changed. The torch function will be re-run on placeholder (meta) tensors"
    " (not the original concrete inputs) to infer output shapes for the new input shapes.\nExpecting: {}\nActual:    {}"
)
_WARN_OUTPUT_SHAPES_FORMAT = (
    "Please provide all shapes as torch.Size or jax.ShapeDtypeStruct. We'll attempt to guess all"
    " containers with only integer entries are shapes (for compatibility), but this is very error-prone."
)
_MISMATCH_ARGS_KW_MSG = "Provided (args, kw) =\n{} do not match the torch2jax function's expected input structure =\n{}"
_WARN_OUTPUT_SHARDING_SPEC_DEPRECATED = "`output_sharding_spec` is deprecated, use `out_specs` instead."
_MISMATCH_ARGS_MSG = "Provided args =\n{} do not match the torch2jax function's expected input structure =\n{}"


def _torch2jax_flat(
    fn: Callable,
    output_shapes: list[jax.Array | Tensor | ShapeDtypeStruct] = None,
    vmap_method: str = "sequential",
) -> Callable:
    """Define a jit-compatible JAX function that calls a PyTorch function. Flat
    arguments and outputs.

    Args:
        fn (Callable): PyTorch function.
        output_shapes: Output shapes (or shapes with dtype). Defaults to None.
    Returns:
        Callable: Wrapped jit-compatible jax function.
    """
    _ = compile_and_import_module()
    id = find_unique_id()

    assert output_shapes is not None, "`output_shapes` cannot be None"
    outshapes = jax.tree.map(lambda x: ShapeDtypeStruct(x.shape, canonical_dtype(x.dtype)), output_shapes)
    out_dtypes = [dtype_j2t(x.dtype) for x in jax.tree.leaves(outshapes)]

    def torch_call_fn_(args: list[torch.Tensor]):
        out = fn(*args)
        out = (out,) if isinstance(out, Tensor) else tuple(out)
        if len(out) != len(out_dtypes):
            return out  # reported as an error by the FFI call
        # cast only dtypes JAX treats as equivalent (e.g., int64 -> int32 when x64 is disabled), the rest is validated
        return tuple(
            o.to(dt) if isinstance(o, Tensor) and o.dtype != dt and canonical_dtype(o.dtype) == dt_j else o
            for o, dt, dt_j in zip_(out, out_dtypes, map(dtype_t2j, out_dtypes))
        )

    setattr(torch, f"_torch2jax_fn_{id:d}", torch_call_fn_)

    @jax.jit
    def wrapped_flat_fn(*args_flat):
        return ffi.ffi_call("torch_call", outshapes, vmap_method=vmap_method)(*args_flat, fn_id=f"{id:d}")

    return wrapped_flat_fn


def _torch2jax(
    fn: Callable,
    *example_args: Any,
    example_kw: Any | None = None,
    output_shapes: Any = None,
    out_specs: Any | None = None,
    vmap_method: str = "sequential",
    output_sharding_spec: PartitionSpec | None = None,
) -> Callable:
    """Define a jit-compatible JAX function that calls a PyTorch function.  Arbitrary nesting of
    arguments and outputs is supported.

    Args:
        fn (Callable): PyTorch function to wrap.
        *example_args (Any): Example arguments as tensors or torch-compatible args.
        example_kw: Example keyword arguments. Defaults to None.
        output_shapes: Output shapes or shapes + dtype struct. Defaults to None.
        out_specs: Output PartitionSpec(s) (a prefix of the output tree) for inputs sharded along explicit mesh axes.
            The torch function is then called per-shard inside `jax.shard_map` over those axes, with in_specs taken
            from the input shardings. Without `out_specs`, inputs sharded along explicit axes raise an error.
        vmap_method: batching method, see
            [https://docs.jax.dev/en/latest/ffi.html#batching-with-vmap](https://docs.jax.dev/en/latest/ffi.html#batching-with-vmap)

            NOTE: only vmap_method="sequential" is supported non-experimentally

            NOTE: try "expand_dims", "broadcast_all" if you want to experiment with pytorch-side batching
        output_sharding_spec: Deprecated alias for `out_specs`.
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

    if output_sharding_spec is not None:
        warn_once(_WARN_OUTPUT_SHARDING_SPEC_DEPRECATED, fn)
        out_specs = output_sharding_spec if out_specs is None else out_specs
    # check for presence of example_args and example_kw
    _had_output_shapes = output_shapes is not None
    has_kw = example_kw is not None

    # find the input structure
    if has_kw:
        input_struct = jax.tree.structure((example_args, example_kw))
    else:
        input_struct = jax.tree.structure(example_args)

    example_inputs = (example_args, example_kw) if has_kw else example_args
    # torch dtypes of example tensors, restored at call time (e.g., int64 args arrive as int32 when x64 is disabled)
    torch_dtypes = [x.dtype if isinstance(x, Tensor) else None for x in jax.tree.leaves(example_inputs)]

    # define flattened version of the function (flat arguments and outputs)
    def flat_fn(*args_flat):
        args_flat = [a if dt is None or a.dtype == dt else a.to(dt) for a, dt in zip(args_flat, torch_dtypes)]
        if has_kw:
            args, kw = jax.tree.unflatten(input_struct, args_flat)
            ret = fn(*args, **kw)
        else:
            args = jax.tree.unflatten(input_struct, args_flat)
            ret = fn(*args)
        return jax.tree.leaves(ret)

    input_shapes = jax.tree.map(lambda x: ShapeDtypeStruct(x.shape, dtype_t2j(x.dtype)), example_inputs)

    # find the output structure
    if output_shapes is None:
        output = infer_outputs(fn, example_args, example_kw)
        output_shapes, output_struct = jax.tree.flatten(
            jax.tree.map(lambda x: ShapeDtypeStruct(x.shape, dtype_t2j(x.dtype)), output)
        )
    else:
        if not all(
            isinstance(x, (torch.Size, ShapeDtypeStruct, jax.Array, torch.Tensor)) or hasattr(x, "shape")
            for x in jax.tree.leaves(output_shapes)
        ):
            warn_once(_WARN_OUTPUT_SHAPES_FORMAT, fn)
        output_shapes = normalize_shapes(output_shapes, extra_args=input_shapes)
        output_shapes, output_struct = jax.tree.flatten(output_shapes)

    # define the wrapped function using flat interface
    wrapped_fn_flat = _torch2jax_flat(
        flat_fn,
        output_shapes=output_shapes,
        vmap_method=vmap_method,
    )

    # shape-aware cache for automatic re-wrapping on shape changes
    _cache = {}
    _original_shape_key = shape_key(example_inputs)
    format_key = lambda key: ", ".join([f"{np.dtype(k[1]).name}{list(k[0])}" for k in key])

    def local_call(args, to_local: Callable | None):
        key = shape_key(args)
        if key != _original_shape_key:
            if key not in _cache:
                if to_local is None:  # per-shard shapes are expected to differ from the global example shapes
                    msg = _SHAPE_CHANGE_WARN_EXPLICIT if _had_output_shapes else _SHAPE_CHANGE_WARN_CONCRETE
                    warn_always(msg.format(format_key(_original_shape_key), format_key(key)))
                dummy_flat = [placeholder_like(a, dt) for a, dt in zip(jax.tree.leaves(args), torch_dtypes)]
                dummy_tree = jax.tree.unflatten(input_struct, dummy_flat)
                dummy_args, dummy_kw = dummy_tree if has_kw else (dummy_tree, None)
                opts = dict(example_kw=dummy_kw, vmap_method=vmap_method)
                if _had_output_shapes and to_local is not None:  # split the global output shapes, don't run fn
                    opts["output_shapes"] = to_local(jax.tree.unflatten(output_struct, output_shapes))
                _cache[key] = _torch2jax(fn, *dummy_args, **opts)
            return _cache[key](*args[0], **args[1]) if has_kw else _cache[key](*args)
        vma = union_vma(args)  # inside shard_map, the FFI call needs inputs and outputs varying along the same axes
        ret = wrapped_fn_flat(*match_vma(jax.tree.leaves(args), vma))
        return jax.tree.unflatten(output_struct, match_vma(ret, vma))

    # define the actual wrapper function
    def wrapped_fn(*args, **kw):
        if not has_kw and len(kw) > 0:
            raise RuntimeError("Keyword arguments not expected!")
        if has_kw:
            args = (args, kw)
        if jax.tree.structure(args) != input_struct:
            msg = (_MISMATCH_ARGS_KW_MSG if has_kw else _MISMATCH_ARGS_MSG).format(args, input_struct)
            raise RuntimeError(msg)
        return shard_call(local_call, args, out_specs)

    return wrapped_fn


torch2jax_without_vjp = _torch2jax
