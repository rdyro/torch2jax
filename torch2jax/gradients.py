from __future__ import annotations

from typing import Callable, Any
from functools import partial

import torch
import jax
from jax import ShapeDtypeStruct
from jax.tree_util import tree_map, tree_flatten, tree_unflatten

from .api import torch2jax
from .utils import _is_floating_point, dtype_t2j, normalize_shapes


####################################################################################################


def torch2jax_with_vjp(
    torch_fn: Callable,
    *example_args: Any,
    depth: int = 2,
    nondiff_argnums: list | tuple | None = None,
    nondiff_mask: Any | None = None,
    output_shapes: Any | None = None,
    use_zeros: bool = True,
    _use_torch_vjp: bool = True,
) -> Callable:
    """Convert a torch function to a jax function and define a custom vjp rule
    for it up to `depth` recursively deep.

    Args:
        torch_fn (Callable): Torch function to convert.
        *example_args (Any): Example arguments as tensors or torch-compatible args.
        depth (int, optional): Max allowed differentiation depth, this is cheap. Defaults to 1.
        nondiff_argnums (list | tuple | None, optional): Which (whole) args to
                                                         not differentiate. Defaults to None.
        nondiff_mask (Any | None, optional): Full arg matching mask. Defaults to None.
        output_shapes (Any | None, optional): Output shapes out of the function,
                                              if provided, we never call torch
                                              function to infer them. Defaults
                                              to None.
        use_zeros (bool, optional): Whether to set gradients of non-diff args to
                                    zeros or None. None does not appear to work
                                    with JAX currently. Defaults to True.
        _use_torch_vjp (bool, optional): Whether to use custom vjp or the one
                                         from torch. Defaults to True.

    Returns:
        Callable: JIT-compatible JAX version of the torch function (VJP defined up to depth `depth`).


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
    if output_shapes is None:
        with torch.no_grad():
            outputs = torch_fn(*example_args)
        output_shapes = tree_map(
            lambda x: ShapeDtypeStruct(dtype=dtype_t2j(x.dtype), shape=x.shape), outputs
        )
    fn = torch2jax(torch_fn, *example_args, output_shapes=output_shapes)

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
        nondiff_argnums = (
            (nondiff_argnums,) if isinstance(nondiff_argnums, int) else tuple(nondiff_argnums)
        )
        nondiff_mask = [
            tree_map(lambda _: True, arg)
            if (i in nondiff_argnums)
            else tree_map(lambda _: False, arg)
            for (i, arg) in enumerate(example_args)
        ]
    if nondiff_mask is not None:
        nondiff_mask_flat = tree_flatten(nondiff_mask)[0]
        assert len(nondiff_mask_flat) == len(example_args_flat), "`nondiff_mask` must match `args`"
        nondiff_mask_flat = [
            (m or (not _is_floating_point(arg)))
            for m, arg in zip(nondiff_mask_flat, example_args_flat)
        ]
    else:
        nondiff_mask_flat = [not _is_floating_point(arg) for i, arg in enumerate(example_args_flat)]

    # define two torch helper functions for computing the VJP ######################################
    def _torch_fn_diff_flat(*diff_args_flat, all_args_flat=None):
        args_collected_flat, diff_args_flat = [], list(diff_args_flat)
        for arg, m in zip(all_args_flat, nondiff_mask_flat):
            args_collected_flat.append(arg if m else diff_args_flat.pop(0))
        args_collected = tree_unflatten(args_struct, args_collected_flat)
        return tree_flatten(torch_fn(*args_collected))[0]

    @partial(torch.func.grad, argnums=tuple(range(len(example_args_flat) - sum(nondiff_mask_flat))))
    def _custom_grad_fn(*diff_args_flat, all_args_flat=None, gs_flat=None):
        return sum(
            torch.sum(g * o)
            for (g, o) in zip(
                gs_flat, _torch_fn_diff_flat(*diff_args_flat, all_args_flat=all_args_flat)
            )
        )

    # define the actual torch VJP function #########################################################
    def bwd_fn_torch(args, gs):
        if all(not m for m in nondiff_mask_flat):
            _, vjp_fn = torch.func.vjp(
                torch_fn,
                *args,
            )
            return vjp_fn(gs)

        args_flat = tree_flatten(args)[0]
        diff_args_flat = [arg for (arg, m) in zip(args_flat, nondiff_mask_flat) if not m]
        gs_flat = tree_flatten(gs)[0]

        # use either torch's vjp or our custom vjp only wrt differentiable arguments ###############
        if _use_torch_vjp:
            diff_vjp_vals_flat = list(
                torch.func.vjp(
                    partial(_torch_fn_diff_flat, all_args_flat=args_flat), *diff_args_flat
                )[1](gs_flat)
            )
        else:
            diff_vjp_vals_flat = list(
                _custom_grad_fn(*diff_args_flat, all_args_flat=args_flat, gs_flat=gs_flat)
            )

        # reconstruct the full vjp including for nondiff arguments #################################
        vjp_vals_flat = []
        for arg, m in zip(args_flat, nondiff_mask_flat):
            vjp_vals_flat.append(
                (None if not use_zeros else 0 * arg) if m else diff_vjp_vals_flat.pop(0)
            )
        return tree_unflatten(args_struct, vjp_vals_flat)

    # construct example outputs out of the bwd_fn (sensitivty wrt args) ############################
    # and next shapes (args, outputs) ##############################################################
    example_outputs = normalize_shapes(output_shapes, example_args)
    next_output_shapes = tree_unflatten(
        args_struct,
        [
            ShapeDtypeStruct(dtype=dtype_t2j(x.dtype), shape=x.shape)
            if (not m or use_zeros)
            else None
            for (x, m) in zip(example_args_flat, nondiff_mask_flat)
        ],
    )
    bwd_fn = torch2jax_with_vjp(
        bwd_fn_torch,
        example_args,
        example_outputs,
        output_shapes=next_output_shapes,
        depth=depth - 1,
    )
    # define the custom vjp using the fwd_fn and bwd_fn ############################################
    fn.defvjp(fwd_fn, bwd_fn)

    return fn
