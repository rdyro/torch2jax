from __future__ import annotations

from typing import Callable, Any

import jax
from jax import ShapeDtypeStruct
import torch
from torch import Tensor
from jax.tree_util import tree_map

from torch2jax.utils import dtype_t2j


def wrap_torch_fn(
    fn,
    output_shapes: Any,
    device: str = "cpu",
) -> Callable:
    def numpy_fn(*args):
        args = tree_map(lambda x: torch.as_tensor(x, device=device), args)
        out = fn(*args)
        out = (out,) if isinstance(out, Tensor) else tuple(out)
        out = [z.detach().cpu().numpy() for z in out]
        return out

    jax_output_shapes = tree_map(lambda x: ShapeDtypeStruct(x.shape, dtype_t2j(x.dtype)), output_shapes)

    def wrapped_fn(*args):
        return jax.pure_callback(numpy_fn, jax_output_shapes, *args)

    return wrapped_fn
