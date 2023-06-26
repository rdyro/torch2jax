import sys
from pathlib import Path

import torch
from torch import Tensor
import jax
from jax import numpy as jnp
from jax import Array
import numpy as np

paths = [Path(__file__).absolute().parents[1], Path(__file__).absolute().parent]
for path in paths:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from utils import jax_randn
from pure_callback_alternative import wrap_torch_fn
from torch2jax import torch2jax

def test_inplace_memory():
    # we're going to test if we can write in memory inplace

    def torch_fn(x):
        y = torch.randn_like(x)
        x[:5].add_(17.0)
        return y


    for device in ["cpu", "cuda"]:
        for dtype in [jnp.float32, jnp.float64]:
            x = jax_randn((50,), device=device, dtype=dtype) * 0
            jax_fn = torch2jax(torch_fn, output_shapes=[x.shape])
            out = jax_fn(x)[0]
            expected = jnp.concatenate([jnp.ones(5) * 17, jnp.zeros(45)])
            jax_device = jax.devices(device)[0]
            expected = jax.device_put(expected, jax_device).astype(dtype)
            err = jnp.linalg.norm(x - expected) / jnp.linalg.norm(expected)
            assert err < 1e-5
