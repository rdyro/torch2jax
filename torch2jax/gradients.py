from __future__ import annotations

import traceback
from typing import Callable, Any
from functools import partial

import torch
import jax
from jax import ShapeDtypeStruct
from jax.tree_util import tree_map, tree_flatten, tree_unflatten
from jax.sharding import PartitionSpec as P

from .api import torch2jax
from .utils import _is_floating, dtype_t2j, normalize_shapes, warn_once


####################################################################################################


def torch2jax_with_vjp(
    torch_fn: Callable,
    *example_args: Any,
    depth: int = 2,
    nondiff_argnums: list | tuple | None = None,
    nondiff_mask: Any | None = None,
    output_shapes: Any | None = None,
    use_zeros: bool = True,
    use_torch_vjp: bool = True,
    output_sharding_spec: P | None = None,
    vmap_method: str = "sequential",
) -> Callable:
    """Convert a torch function to a jax function and define a custom vjp rule for it up to `depth` recursively deep.

    Args:
        torch_fn (Callable): Torch function to convert.
        *example_args (Any): Example arguments as tensors or torch-compatible args.
        depth (int, optional): Max allowed differentiation depth, this is cheap. Defaults to 1.
        nondiff_argnums (list | tuple | None, optional): Which (whole) args to not differentiate. Defaults to None.
        nondiff_mask (Any | None, optional): Full arg matching mask. Defaults to None.
        output_shapes (Any | None, optional): Output shapes out of the function, if provided, we never call torch
            function to infer them. Defaults to None.
        use_zeros (bool, optional): Whether to set gradients of non-diff args to zeros or None. None does not appear to
            work with JAX currently. Defaults to True.
        use_torch_vjp (bool, optional): (Not supported, please use inside `shard_map`) Whether to use custom vjp or the
            one from torch. False means fallback to `torch.autograd.grad` for more compatibility. Some older external
            library PyTorch code may need this fallback. Defaults to True (i.e., do not use fallback).
        output_sharding_spec: (not supported) sharding spec of the output, use shard_map instead for a device-local
            version of this function
        vmap_method: batching method, see
            [https://docs.jax.dev/en/latest/ffi.html#batching-with-vmap](https://docs.jax.dev/en/latest/ffi.html#batching-with-vmap)

            NOTE: only vmap_method="sequntial" is supported non-experimentally

            NOTE: try "expand_dims", "broadcast_all" if you want to experiment with pytorch-side batching
    Returns:
        Callable: JIT-compatible JAX version of the torch function (VJP defined up to depth `depth`).

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

        >>> # taking gradients is easy too
        >>> g_fn = jax.grad(jax_fn, argnums=0)
        >>> g_fn(x, y).shape
        (10, 5)

        >>> # creating a more complicated computational graph is of course possible
        >>> lin_model = lambda z, W, b: z @ W + b
        >>> z, W, b = tree_t2j([torch.randn((10, 20)), torch.randn(20, 5), torch.randn(5)])
        >>> gz_fn = jax.grad(lambda z, W, b: jax_fn(lin_model(z, W, b), y), argnums=(1, 2))
        >>> dW, db = gz_fn(z, W, b)
        >>> dW.shape, db.shape
        ((20, 5), (5,))
    """
    if output_sharding_spec is not None:
        raise RuntimeError(
            "`output_sharding_spec` not supported in `torch2jax_with_vjp`, it's somewhat difficult to automatically"
            " define sharding spec for automatically defined vjp functions. As a work-around, please use this function"
            " inside `shard_map` without specifying `output_sharding_spec` - you don't need to specify the specs there."
        )

    if output_shapes is None:
        outputs = torch_fn(*example_args)
        output_shapes = tree_map(lambda x: ShapeDtypeStruct(dtype=dtype_t2j(x.dtype), shape=x.shape), outputs)
    fn = torch2jax(
        torch_fn,
        *example_args,
        output_shapes=output_shapes,
        output_sharding_spec=output_sharding_spec,
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
                tb = traceback.format_exc()
                msg = (
                    "Somewhere in your PyTorch computation graph, a custom backward function is defined in the old way"
                    ' (see "https://pytorch.org/docs/stable/notes/extending.html"). This is only experimentally'
                    " supported in torch2jax. We will use a fallback based on `torch.autograd.grad` instead. Please"
                    " pass `use_torch_vjp=False` to `torch2jax_with_vjp` if you wish to use this fallback explicitly."
                    f" Original error message:\n{tb}"
                )
                warn_once(msg, torch_fn)
                grads_computed = False
        if not grads_computed:
            if not use_torch_vjp:
                warn_once("You are NOT using PyTorch's functional VJP. This is highly experimental.", torch_fn)
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
    bwd_fn = torch2jax_with_vjp(
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

    return fn
