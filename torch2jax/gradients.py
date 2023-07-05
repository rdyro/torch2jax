from typing import Callable, Any

import torch
from torch import Size
import jax
from jax.tree_util import tree_map

from .api import torch2jax


def torch2jax_with_vjp(
    torch_fn: Callable,
    *example_args,
    depth: int = 1,
    #create_jvp: bool = False,
    #nondiff_argnums: tuple | None = None,
    #has_aux: bool = False,
) -> Callable:
    outputs = torch_fn(*example_args)
    output_shape = tree_map(lambda x: Size(x.shape), outputs)
    fn = torch2jax(torch_fn, *example_args, output_shapes=output_shape)

    if depth <= 0:
        return fn

    fn = jax.custom_vjp(fn)

    def fwd_fn(*args):
        return fn(*args), args

    def bwd_fn_torch(args, gs):
        if True: # nondiff_argnums is None:
            _, vjp_fn = torch.func.vjp(
                torch_fn,
                *args,
            )
            return vjp_fn(gs)
        raise NotImplementedError("Nondiff arguments are not implemented yet.")

        def torch_fn_(*diff_args):
            all_args, k = [], 0
            for i, arg in enumerate(args):
                if i in nondiff_argnums:
                    all_args.append(arg)
                else:
                    all_args.append(diff_args[k])
                    k += 1
            return torch_fn(*all_args)

        diff_args = [arg for (i, arg) in enumerate(args) if i not in nondiff_argnums]
        _, vjp_fn = torch.func.vjp(torch_fn_, *args)
        return vjp_fn(gs, *diff_args)

    if False: # create_jvp:
        raise NotImplementedError("JVP is not implemented yet.")
        bwd_fn = create_custom_jvp(bwd_fn_torch, args, outputs, dtype=dtype, device=device, depth=1)
    else:
        bwd_fn = torch2jax_with_vjp(bwd_fn_torch, example_args, outputs, depth=depth - 1)
        fn.defvjp(fwd_fn, bwd_fn)

    return fn