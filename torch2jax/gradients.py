from __future__ import annotations

import traceback
from typing import Callable, Any
from functools import partial

import torch
import jax
from jax import ShapeDtypeStruct
from jax.tree_util import tree_map, tree_flatten, tree_unflatten

from .api import _torch2jax, _SHAPE_CHANGE_WARN_CONCRETE, _SHAPE_CHANGE_WARN_EXPLICIT
from .utils import _is_floating, dtype_t2j, dtype_j2t, normalize_shapes, warn_once

_ERR_SHARDING_SPEC_UNSUPPORTED = (
    "`output_sharding_spec` not supported in `torch2jax(depth>0)`, it's somewhat difficult to automatically"
    " define sharding spec for automatically defined vjp functions. As a work-around, please use this function"
    " inside `shard_map` without specifying `output_sharding_spec` - you don't need to specify the specs there."
)
_WARN_OLD_BACKWARD_FN = (
    "Somewhere in your PyTorch computation graph, a custom backward function is defined in the old way"
    ' (see "https://pytorch.org/docs/stable/notes/extending.html"). This is only experimentally'
    " supported in torch2jax. We will use a fallback based on `torch.autograd.grad` instead. Please"
    " pass `use_torch_vjp=False` to `torch2jax` if you wish to use this fallback explicitly."
    " Original error message:\n{}"
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
    use_zeros: bool = True,
    use_torch_vjp: bool = True,
    vmap_method: str = "sequential",
) -> Callable:
    """Define a jit-compatible JAX function that calls a PyTorch function, optionally with custom VJP rules.

    For sharding support, use ``torch2jax_without_vjp`` or wrap this function inside ``shard_map``.

    Args:
        torch_fn (Callable): Torch function to convert.
        *example_args (Any): Example arguments as tensors or torch-compatible args.
        example_kw: Example keyword arguments. Defaults to None. Only supported with depth=0.
        depth (int, optional): Max allowed differentiation depth. 0 = no VJP. Defaults to 2.
        nondiff_argnums (list | tuple | None, optional): Which (whole) args to not differentiate. Defaults to None.
        nondiff_mask (Any | None, optional): Full arg matching mask. Defaults to None.
        output_shapes (Any | None, optional): Output shapes out of the function, if provided, we never call torch
            function to infer them. Defaults to None.
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
        outputs = torch_fn(*example_args)
        output_shapes = tree_map(lambda x: ShapeDtypeStruct(dtype=dtype_t2j(x.dtype), shape=x.shape), outputs)
    fn = _torch2jax(
        torch_fn,
        *example_args,
        example_kw=example_kw,
        output_shapes=output_shapes,
        vmap_method=vmap_method,
    )

    # if this we've reached the requested differentiation depth, refrain from defining a vjp rule ##
    if depth <= 0:
        return fn

    # begin defining custom vjp ####################################################################
    fn = jax.custom_vjp(fn)
    example_args_flat, args_struct = tree_flatten(example_args)

    # define forward function
    def fwd_fn(*args):
        return fn(*args), args

    # handle determining which arguments are nondifferentiable #####################################
    if nondiff_argnums is not None:
        # assume the user means the entire e.g., 2nd arg if they pass argnums=(2,)
        nondiff_argnums = (nondiff_argnums,) if isinstance(nondiff_argnums, int) else tuple(nondiff_argnums)
        nondiff_mask = [
            tree_map(lambda _: True, arg) if (i in nondiff_argnums) else tree_map(lambda _: False, arg)
            for (i, arg) in enumerate(example_args)
        ]
    if nondiff_mask is not None:
        nondiff_mask_flat = tree_flatten(nondiff_mask)[0]
        assert len(nondiff_mask_flat) == len(example_args_flat), "`nondiff_mask` must match `args`"
        nondiff_mask_flat = [(m or (not _is_floating(arg))) for m, arg in zip(nondiff_mask_flat, example_args_flat)]
    else:
        nondiff_mask_flat = [not _is_floating(arg) for i, arg in enumerate(example_args_flat)]

    # define two torch helper functions for computing the VJP ######################################
    def _torch_fn_diff_flat(*diff_args_flat, all_args_flat=None):
        args_collected_flat, diff_args_flat = [], list(diff_args_flat)
        for arg, m in zip(all_args_flat, nondiff_mask_flat):
            args_collected_flat.append(arg if m else diff_args_flat.pop(0))
        args_collected = tree_unflatten(args_struct, args_collected_flat)
        return tree_flatten(torch_fn(*args_collected))[0]

    # define the actual torch VJP function #########################################################
    def bwd_fn_torch(args, gs):
        args_flat = tree_flatten(args)[0]
        diff_args_flat = [arg for (arg, m) in zip(args_flat, nondiff_mask_flat) if not m]
        gs_flat = tree_flatten(gs)[0]

        # use either torch's vjp or our custom vjp only wrt differentiable arguments ###############
        grads_computed = False
        if use_torch_vjp:
            try:
                diff_vjp_vals_flat = list(
                    torch.func.vjp(partial(_torch_fn_diff_flat, all_args_flat=args_flat), *diff_args_flat)[1](gs_flat)
                )
                grads_computed = True
            except RuntimeError:
                warn_once(_WARN_OLD_BACKWARD_FN.format(traceback.format_exc()), torch_fn)
                grads_computed = False
        if not grads_computed:
            if not use_torch_vjp:
                warn_once(_WARN_EXPERIMENTAL_VJP, torch_fn)
            [diff_arg_flat.requires_grad_(True) for diff_arg_flat in diff_args_flat]
            ret = sum(
                torch.sum(g * r)
                for (g, r) in zip(gs_flat, _torch_fn_diff_flat(*diff_args_flat, all_args_flat=args_flat))
            )
            diff_vjp_vals_flat = list(torch.autograd.grad(ret, diff_args_flat, create_graph=True))

        # reconstruct the full vjp including for nondiff arguments #################################
        vjp_vals_flat = []
        for arg, m in zip(args_flat, nondiff_mask_flat):
            vjp_vals_flat.append((None if not use_zeros else 0 * arg) if m else diff_vjp_vals_flat.pop(0))
        return tree_unflatten(args_struct, vjp_vals_flat)

    # construct example outputs out of the bwd_fn (sensitivty wrt args) ############################
    # and next shapes (args, outputs) ##############################################################
    example_outputs = normalize_shapes(output_shapes, example_args)
    next_output_shapes = tree_unflatten(
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
    _original_vjp_key = tuple((tuple(a.shape), dtype_t2j(a.dtype)) for a in tree_flatten(example_args)[0])

    def _cached_fn(*args):
        key = tuple((tuple(a.shape), a.dtype) for a in tree_flatten(args)[0])
        if key == _original_vjp_key:
            return fn(*args)
        if key not in _vjp_cache:
            warn_once(_SHAPE_CHANGE_WARN_EXPLICIT if _had_output_shapes else _SHAPE_CHANGE_WARN_CONCRETE, torch_fn)
            dummy_flat = [torch.zeros(a.shape, dtype=dtype_j2t(a.dtype)) for a in tree_flatten(args)[0]]
            dummy_args = tree_unflatten(tree_flatten(example_args)[1], dummy_flat)
            _vjp_cache[key] = torch2jax(
                torch_fn,
                *dummy_args,
                depth=depth,
                nondiff_argnums=nondiff_argnums,
                nondiff_mask=nondiff_mask,
                use_zeros=use_zeros,
                use_torch_vjp=use_torch_vjp,
                vmap_method=vmap_method,
            )
        return _vjp_cache[key](*args)

    return _cached_fn


def torch2jax_with_vjp(*args, depth=2, **kw):
    """Deprecated: use ``torch2jax(..., depth=2)`` instead."""
    warn_once(_WARN_TORCH2JAX_WITH_VJP_DEPRECATED, torch2jax_with_vjp)
    return torch2jax(*args, depth=depth, **kw)
