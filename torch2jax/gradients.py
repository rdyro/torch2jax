import traceback
from typing import Callable, Any
from functools import partial

import torch
import jax
from jax import ShapeDtypeStruct
import numpy as np

from .api import _torch2jax, _SHAPE_CHANGE_WARN_CONCRETE, _SHAPE_CHANGE_WARN_EXPLICIT
from .sharding import shard_call, match_vma
from .utils import (
    _is_floating,
    dtype_t2j,
    normalize_shapes,
    warn_once,
    warn_always,
    shape_key,
    placeholder_like,
    infer_outputs,
)

_WARN_VJP_FALLBACK = (
    "`torch.func.vjp` failed on your PyTorch function, e.g., because a custom backward function is defined in the"
    ' old way (see "https://pytorch.org/docs/stable/notes/extending.html") or the function accesses tensor data'
    " (`.numpy()`, `.data_ptr()`). We used a fallback based on `torch.autograd.grad` instead. Please pass"
    " `use_torch_vjp=False` to `torch2jax` if you wish to use this fallback explicitly. Original error message:\n{}"
)
_WARN_EXPERIMENTAL_VJP = "You are NOT using PyTorch's functional VJP. This is highly experimental."
_WARN_TORCH2JAX_WITH_VJP_DEPRECATED = "`torch2jax_with_vjp` is deprecated, use `torch2jax(..., depth=2)` instead."


####################################################################################################


def torch2jax(
    torch_fn: Callable,
    *example_args: Any,
    example_kw: Any | None = None,
    depth: int = 2,
    nondiff_argnums: list | tuple | None = None,
    nondiff_mask: Any | None = None,
    output_shapes: Any | None = None,
    out_specs: Any | None = None,
    use_zeros: bool = True,
    use_torch_vjp: bool = True,
    vmap_method: str = "sequential",
) -> Callable:
    """Define a jit-compatible JAX function that calls a PyTorch function, optionally with custom VJP rules.

    Sharding: inside ``jax.shard_map`` the function works as-is (per-shard). For inputs sharded along explicit mesh
    axes, pass ``out_specs`` to call the torch function per-shard; sharded inputs are never implicitly all-gathered.

    Args:
        torch_fn (Callable): Torch function to convert.
        *example_args (Any): Example arguments as tensors or torch-compatible args.
        example_kw: Example keyword arguments. Defaults to None. Only supported with depth=0.
        depth (int, optional): Max allowed differentiation depth. 0 = no VJP. Defaults to 2.
        nondiff_argnums (list | tuple | None, optional): Which (whole) args to not differentiate. Defaults to None.
        nondiff_mask (Any | None, optional): Full arg matching mask. Defaults to None.
        output_shapes (Any | None, optional): Output shapes out of the function, if provided, we never call torch
            function to infer them. Defaults to None.
        out_specs: Output PartitionSpec(s) for inputs sharded along explicit mesh axes: the torch function (and its
            VJP) is then called per-shard inside `jax.shard_map` over those axes. Defaults to None.
        use_zeros (bool, optional): Whether to set gradients of non-diff args to zeros or None. Defaults to True.
        use_torch_vjp (bool, optional): Whether to use torch.func.vjp or fallback to torch.autograd.grad.
            Defaults to True.
        vmap_method: batching method, see
            `jax ffi docs <https://docs.jax.dev/en/latest/ffi.html#batching-with-vmap>`_.

            NOTE: only vmap_method="sequential" is supported non-experimentally

            NOTE: try "expand_dims", "broadcast_all" if you want to experiment with pytorch-side batching
    Returns:
        Callable: JIT-compatible JAX version of the torch function (VJP defined up to depth `depth`).

    Examples:
        >>> import torch, jax
        >>> from torch2jax import torch2jax, tree_t2j
        >>> torch_fn = lambda x, y: torch.nn.CrossEntropyLoss()(x, y)
        >>> xt, yt = torch.randn(10, 5), torch.randint(0, 5, (10,))
        >>> jax_fn = torch2jax(torch_fn, xt, yt)
        >>> x, y = tree_t2j((xt, yt))
        >>> jax_fn(x, y)

        >>> # with gradients (depth=2 is the default)
        >>> jax.grad(lambda x, y: jax_fn(x, y).sum(), argnums=0)(x, y).shape
        (10, 5)
    """
    if depth > 0 and example_kw is not None:
        raise RuntimeError("`example_kw` is not supported with `depth > 0` (VJP path does not support kwargs yet).")
    _had_output_shapes = output_shapes is not None

    if output_shapes is None and depth > 0:
        outputs = infer_outputs(torch_fn, example_args)
        output_shapes = jax.tree.map(lambda x: ShapeDtypeStruct(dtype=dtype_t2j(x.dtype), shape=x.shape), outputs)
    fn = _torch2jax(
        torch_fn,
        *example_args,
        example_kw=example_kw,
        output_shapes=output_shapes,
        out_specs=out_specs if depth <= 0 else None,  # for depth > 0, sharding is handled around the custom_vjp
        vmap_method=vmap_method,
    )

    # if this we've reached the requested differentiation depth, refrain from defining a vjp rule ##
    if depth <= 0:
        return fn

    # begin defining custom vjp ####################################################################
    fn = jax.custom_vjp(fn)
    example_args_flat, args_struct = jax.tree.flatten(example_args)

    # define forward function
    def fwd_fn(*args):
        return fn(*args), args

    # handle determining which arguments are nondifferentiable #####################################
    if nondiff_argnums is not None:
        # assume the user means the entire e.g., 2nd arg if they pass argnums=(2,)
        nondiff_argnums = (nondiff_argnums,) if isinstance(nondiff_argnums, int) else tuple(nondiff_argnums)
        nondiff_mask = [
            jax.tree.map(lambda _: True, arg) if (i in nondiff_argnums) else jax.tree.map(lambda _: False, arg)
            for (i, arg) in enumerate(example_args)
        ]
    if nondiff_mask is not None:
        nondiff_mask_flat = jax.tree.flatten(nondiff_mask)[0]
        assert len(nondiff_mask_flat) == len(example_args_flat), "`nondiff_mask` must match `args`"
        nondiff_mask_flat = [(m or (not _is_floating(arg))) for m, arg in zip(nondiff_mask_flat, example_args_flat)]
    else:
        nondiff_mask_flat = [not _is_floating(arg) for i, arg in enumerate(example_args_flat)]

    # define two torch helper functions for computing the VJP ######################################
    def _torch_fn_diff_flat(*diff_args_flat, all_args_flat=None):
        args_collected_flat, diff_args_flat = [], list(diff_args_flat)
        for arg, m in zip(all_args_flat, nondiff_mask_flat):
            args_collected_flat.append(arg if m else diff_args_flat.pop(0))
        args_collected = jax.tree.unflatten(args_struct, args_collected_flat)
        return jax.tree.flatten(torch_fn(*args_collected))[0]

    # define the actual torch VJP function #########################################################
    def bwd_fn_torch(args, gs):
        args_flat = jax.tree.flatten(args)[0]
        diff_args_flat = [arg for (arg, m) in zip(args_flat, nondiff_mask_flat) if not m]
        gs_flat = jax.tree.flatten(gs)[0]

        # use either torch's vjp or our custom vjp only wrt differentiable arguments ###############
        diff_vjp_vals_flat, vjp_error = None, None
        if use_torch_vjp:
            try:
                diff_vjp_vals_flat = list(
                    torch.func.vjp(partial(_torch_fn_diff_flat, all_args_flat=args_flat), *diff_args_flat)[1](gs_flat)
                )
            except RuntimeError as e:
                vjp_error = e
        else:
            warn_once(_WARN_EXPERIMENTAL_VJP, torch_fn)
        if diff_vjp_vals_flat is None:
            try:
                [diff_arg_flat.requires_grad_(True) for diff_arg_flat in diff_args_flat]
                ret = sum(
                    torch.sum(g * r)
                    for (g, r) in zip(gs_flat, _torch_fn_diff_flat(*diff_args_flat, all_args_flat=args_flat))
                )
                diff_vjp_vals_flat = list(torch.autograd.grad(ret, diff_args_flat, create_graph=True))
            except Exception:
                if vjp_error is None:
                    raise
                raise vjp_error  # the fallback failed too, report the torch.func error (fallback error as context)
            if vjp_error is not None:
                warn_once(_WARN_VJP_FALLBACK.format("".join(traceback.format_exception(vjp_error))), torch_fn)

        # reconstruct the full vjp including for nondiff arguments #################################
        vjp_vals_flat = []
        for arg, m in zip(args_flat, nondiff_mask_flat):
            vjp_vals_flat.append((None if not use_zeros else 0 * arg) if m else diff_vjp_vals_flat.pop(0))
        return jax.tree.unflatten(args_struct, vjp_vals_flat)

    # construct example outputs out of the bwd_fn (sensitivty wrt args) ############################
    # and next shapes (args, outputs) ##############################################################
    example_outputs = normalize_shapes(output_shapes, example_args)
    next_output_shapes = jax.tree.unflatten(
        args_struct,
        [
            ShapeDtypeStruct(dtype=dtype_t2j(x.dtype), shape=x.shape) if (not m or use_zeros) else None
            for (x, m) in zip(example_args_flat, nondiff_mask_flat)
        ],
    )
    bwd_fn = torch2jax(
        bwd_fn_torch,
        example_args,
        example_outputs,
        output_shapes=next_output_shapes,
        depth=depth - 1,
        use_torch_vjp=use_torch_vjp,
        vmap_method=vmap_method,
    )
    # define the custom vjp using the fwd_fn and bwd_fn ############################################
    fn.defvjp(fwd_fn, bwd_fn)

    # shape-aware cache for automatic re-wrapping on shape changes
    _vjp_cache = {}
    _original_vjp_key = shape_key(example_args)
    format_key = lambda key: ", ".join([f"{np.dtype(k[1]).name}{list(k[0])}" for k in key])
    torch_dtypes = [x.dtype if isinstance(x, torch.Tensor) else None for x in example_args_flat]

    def local_call(args, to_local: Callable | None):
        # common varying manual axes before custom_vjp: AD transposes this cast into a psum for replicated inputs
        args = match_vma(args)
        key = shape_key(args)
        if key == _original_vjp_key:
            return fn(*args)
        if key not in _vjp_cache:
            if to_local is None:  # per-shard shapes are expected to differ from the global example shapes
                msg = _SHAPE_CHANGE_WARN_EXPLICIT if _had_output_shapes else _SHAPE_CHANGE_WARN_CONCRETE
                warn_always(msg.format(format_key(_original_vjp_key), format_key(key)))
            dummy_flat = [placeholder_like(a, dt) for a, dt in zip(jax.tree.leaves(args), torch_dtypes)]
            dummy_args = jax.tree.unflatten(jax.tree.structure(args), dummy_flat)
            split = _had_output_shapes and to_local is not None  # split the global output shapes, don't run fn
            _vjp_cache[key] = torch2jax(
                torch_fn,
                *dummy_args,
                output_shapes=to_local(example_outputs) if split else None,
                depth=depth,
                nondiff_argnums=nondiff_argnums,
                nondiff_mask=nondiff_mask,
                use_zeros=use_zeros,
                use_torch_vjp=use_torch_vjp,
                vmap_method=vmap_method,
            )
        return _vjp_cache[key](*args)

    return lambda *args: shard_call(local_call, args, out_specs)


def torch2jax_with_vjp(*args, depth=2, **kw):
    """Deprecated: use ``torch2jax(..., depth=2)`` instead."""
    warn_once(_WARN_TORCH2JAX_WITH_VJP_DEPRECATED, torch2jax_with_vjp)
    return torch2jax(*args, depth=depth, **kw)
