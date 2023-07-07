from __future__ import annotations

from typing import Callable, Any

import torch
import jax
from jax import ShapeDtypeStruct
from jax.tree_util import tree_map, tree_flatten, tree_unflatten

from .api import torch2jax
from .utils import _is_floating_point, dtype_t2j, normalize_shapes


####################################################################################################


def torch2jax_with_vjp(
    torch_fn: Callable,
    *example_args,
    depth: int = 1,
    nondiff_argnums: list | tuple | None = None,
    output_shapes: Any | None = None,
    # has_aux: bool = False, # not currently supported
) -> Callable:
    if output_shapes is None:
        with torch.no_grad():
            outputs = torch_fn(*example_args)
        output_shapes = tree_map(
            lambda x: ShapeDtypeStruct(dtype=dtype_t2j(x.dtype), shape=x.shape), outputs
        )
    fn = torch2jax(torch_fn, *example_args, output_shapes=output_shapes)

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
        # bwd_fn = torch2jax_with_vjp(bwd_fn_torch, example_args, outputs, depth=depth - 1)
        example_outputs = normalize_shapes(output_shapes, example_args)
        args_flat, args_struct = tree_flatten(example_args)
        next_output_shapes = tree_unflatten(
            args_struct,
            [
                ShapeDtypeStruct(dtype=dtype_t2j(x.dtype), shape=x.shape) if not m else None
                for (x, m) in zip(args_flat, nondiff_mask_flat)
            ],
        )
        bwd_fn = torch2jax_with_vjp(
            bwd_fn_torch,
            example_args,
            example_outputs,
            output_shapes=next_output_shapes,
            depth=depth - 1,
        )
        fn.defvjp(fwd_fn, bwd_fn)

    return fn
