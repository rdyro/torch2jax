import sys
from pathlib import Path

import torch
from torch import Size, Tensor
import jax
from jax import numpy as jnp

paths = [Path(__file__).absolute().parents[1], Path(__file__).absolute().parent]
for path in paths:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from torch2jax import torch2jax  # noqa: E402


def test_multi_gpu_call():
    if torch.cuda.device_count() < 2:
        return

    def torch_device_11_fn(x, y):
        assert x.device.index == 1 == y.device.index, "Device must be index 1"
        z = x + y
        return z

    def torch_device_00_fn(x: Tensor, y: Tensor):
        assert x.device.index == 0 == y.device.index, "Device must be index 0"
        z = x + y
        return z

    def torch_device_xx_fn(x: Tensor, y: Tensor):
        z = x + y
        return z

    device0, device1 = jax.devices("cuda")[0], jax.devices("cuda")[1]

    x = jax.device_put(jnp.zeros(10), device1)
    y = jax.device_put(jnp.zeros(10), device1)

    torchfn = torch2jax(torch_device_11_fn, x, y, output_shapes=Size(x.shape))
    z = torchfn(x, y)
    assert len(z.devices()) == 1 and list(z.devices())[0] == device1
    assert jnp.linalg.norm(z - (x + y)) < 1e-6

    ################################################################################################
    x = jax.device_put(jnp.zeros(10), device0)
    y = jax.device_put(jnp.zeros(10), device0)

    torchfn = torch2jax(torch_device_00_fn, x, y, output_shapes=Size(x.shape))
    z = torchfn(x, y)
    assert len(z.devices()) == 1 and list(z.devices())[0] == device0
    assert jnp.linalg.norm(z - (x + y)) < 1e-6

    ################################################################################################
    x = jax.device_put(jnp.zeros(10), device0)
    y = jax.device_put(jnp.zeros(10), device0)

    torchfn = torch2jax(torch_device_xx_fn, x, y, output_shapes=Size(x.shape))
    z = torchfn(x, y)
    assert len(z.devices()) == 1 and list(z.devices())[0] == device0
    assert jnp.linalg.norm(z - (x + y)) < 1e-6

    x = jax.device_put(jnp.zeros(10), device1)
    y = jax.device_put(jnp.zeros(10), device1)

    z = torchfn(x, y)
    assert len(z.devices()) == 1 and list(z.devices())[0] == device1
    assert jnp.linalg.norm(z - (x + y)) < 1e-6

if __name__ == "__main__":
    test_multi_gpu_call()