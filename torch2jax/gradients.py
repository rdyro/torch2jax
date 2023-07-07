from __future__ import annotations

from typing import Callable, Any

import torch
from torch import Size, Tensor
import jax
from jax import Array, numpy as jnp
from jax.tree_util import tree_map, tree_flatten, tree_structure, tree_unflatten

from .api import torch2jax


def _is_floating_point(x: Tensor | Array) -> bool:
    return x.dtype in (
        torch.float16,
        torch.float32,
        torch.float64,
        jnp.float16,
        jnp.float32,
        jnp.float64,
    )


####################################################################################################

def torch2jax_with_vjp(
    torch_fn: Callable,
    *example_args,
    depth: int = 1,
    nondiff_argnums: list | tuple | None = None,
    # has_aux: bool = False,
) -> Callable:
    outputs = torch_fn(*example_args)
    output_shape = tree_map(lambda x: Size(x.shape), outputs)
    fn = torch2jax(torch_fn, *example_args, output_shapes=output_shape)

    if depth <= 0:
        return fn

    fn = jax.custom_vjp(fn)

    def fwd_fn(*args):
        return fn(*args), args

    args_flat = tree_flatten(example_args)[0]
    if nondiff_argnums is not None:
        nondiff_mask_flat = [
            i in nondiff_argnums or not _is_floating_point(arg) for i, arg in enumerate(args_flat)
        ]
    else:
        nondiff_mask_flat = [not _is_floating_point(arg) for i, arg in enumerate(args_flat)]

    def bwd_fn_torch(args, gs):
        if all(not m for m in nondiff_mask_flat):
            _, vjp_fn = torch.func.vjp(
                torch_fn,
                *args,
            )
            return vjp_fn(gs)

        args_flat, args_struct = tree_flatten(args)

        def torch_fn_(*diff_args_flat):
            all_args_flat, diff_args_flat = [], list(diff_args_flat)
            for arg, m in zip(args_flat, nondiff_mask_flat):
                all_args_flat.append(arg if m else diff_args_flat.pop(0))
            all_args = tree_unflatten(args_struct, all_args_flat)
            return tree_flatten(torch_fn(*all_args))[0]

        diff_args_flat = [arg for (arg, m) in zip(args_flat, nondiff_mask_flat) if not m]
        _, vjp_fn = torch.func.vjp(torch_fn_, *diff_args_flat)
        gs_flat = tree_flatten(gs)[0]
        diff_vjp_vals_flat = list(vjp_fn(gs_flat))
        vjp_vals_flat = []
        for m in nondiff_mask_flat:
            vjp_vals_flat.append(None if m else diff_vjp_vals_flat.pop(0))
        return tree_unflatten(args_struct, vjp_vals_flat)

    if False:  # create_jvp:
        raise NotImplementedError("JVP is not implemented yet.")
        bwd_fn = create_custom_jvp(bwd_fn_torch, args, outputs, dtype=dtype, device=device, depth=1)
    else:
        bwd_fn = torch2jax_with_vjp(bwd_fn_torch, example_args, outputs, depth=depth - 1)
        fn.defvjp(fwd_fn, bwd_fn)

    return fn
