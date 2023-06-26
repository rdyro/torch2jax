from __future__ import annotations

from typing import Callable

import jax
import torch
from torch import Tensor
from jax import ShapedArray


def wrap_torch_fn(
    fn,
    output_shapes: list[list[int]] | tuple[tuple[int]],
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> Callable:
    assert device in ["cpu", "gpu"]
    assert dtype in [torch.float32, torch.float64]
    assert (
        isinstance(output_shapes, (list, tuple))
        and len(output_shapes) > 0
        and isinstance(output_shapes, (list, tuple))
    )

    def numpy_fn(*args):
        args = [torch.as_tensor(arg, dtype=dtype, device=device) for arg in args]
        out = fn(*args)
        out = (out,) if isinstance(out, Tensor) else tuple(out)
        out = [torch.as_tensor(z.detach().cpu().numpy()) for z in out]
        return out

    dtype_map = {torch.float32: jax.float32, torch.float64: jax.float64}
    jax_output_shapes = [ShapedArray(shape, dtype_map[dtype]) for shape in output_shapes]

    def wrapped_fn(*args):
        return jax.pure_callback(numpy_fn, output_shapes=jax_output_shapes)

    return wrapped_fn
