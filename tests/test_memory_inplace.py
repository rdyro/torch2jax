import sys
from pathlib import Path

import torch
from torch import Tensor, Size
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
from torch2jax.dlpack_passing import j2t
from torch2jax.compat import torch2jax as torch2jax_v1

def test_memory_inplace_v1():
    # we're going to test if we can write in memory inplace

    def torch_fn(x):
        y = torch.randn_like(x)
        x[:5].add_(17.0)
        return y

    device_list = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

    for device in device_list:
        for dtype in [jnp.float32, jnp.float64]:
            x = jax_randn((50,), device=device, dtype=dtype) * 0
            jax_fn = torch2jax_v1(torch_fn, output_shapes=[x.shape])
            out = jax_fn(x)[0]
            expected = jnp.concatenate([jnp.ones(5) * 17, jnp.zeros(45)])
            jax_device = jax.devices(device)[0]
            expected = jax.device_put(expected, jax_device).astype(dtype)
            err = jnp.linalg.norm(x - expected) / jnp.linalg.norm(expected)
            assert err < 1e-5

####################################################################################################

def test_memory_inplace():
    # we're going to test if we can write in memory inplace

    def torch_fn(x):
        y = torch.randn_like(x)
        x[:5].add_(17.0)
        return y

    device_list = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

    for device in device_list:
        for dtype in [jnp.float32, jnp.float64]:
            x = jax_randn((50,), device=device, dtype=dtype) * 0
            jax_fn = torch2jax(torch_fn, x, output_shapes=Size(x.shape))
            out = jax_fn(x)
            expected = jnp.concatenate([jnp.ones(5) * 17, jnp.zeros(45)])
            jax_device = jax.devices(device)[0]
            expected = jax.device_put(expected, jax_device).astype(dtype)
            err = jnp.linalg.norm(x - expected) / jnp.linalg.norm(expected)
            assert err < 1e-5

if __name__ == "__main__":
    test_memory_inplace()