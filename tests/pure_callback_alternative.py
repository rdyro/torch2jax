from __future__ import annotations

import sys
from typing import Callable, Any
from pathlib import Path

import jax
from jax import numpy as jnp
import torch
from torch import Tensor
from jax.core import ShapedArray
from jax.tree_util import tree_map

root_path = Path(__file__).absolute().parents[1]
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

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

    jax_output_shapes = tree_map(lambda x: ShapedArray(x.shape, dtype_t2j(x.dtype)), output_shapes)

    def wrapped_fn(*args):
        return jax.pure_callback(numpy_fn, jax_output_shapes, *args)

    return wrapped_fn
