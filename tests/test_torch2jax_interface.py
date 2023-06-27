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


def test_single_output_fn():
    shape = (10, 2)

    def torch_fn(x, y):
        return (x + 1 - y.reshape(x.shape)) / torch.norm(y)

    device_list = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    dtype_list = [jnp.float32, jnp.float64]

    for device in device_list:
        for dtype in dtype_list:
            for method in ["output_shapes", "output_shapes_fn", "example_args"]:
                if method == "output_shapes":
                    jax_fn = torch2jax(torch_fn, output_shapes=[shape])
                elif method == "output_shapes_fn":
                    jax_fn = torch2jax(torch_fn, output_shapes_fn=lambda x, y: [x.shape])
                elif method == "example_args":
                    example_args = (torch.randn(shape), torch.randn(shape).reshape(-1))
                    jax_fn = torch2jax(torch_fn, example_args=example_args)

                x = jax_randn(shape, device=device, dtype=dtype)
                y = jax_randn(shape, device=device, dtype=dtype).reshape(-1)

                # non-jit version
                out = jax_fn(x, y)
                assert isinstance(out, (list, tuple)) and len(out) == 1
                out1 = out[0]
                assert isinstance(out1, Array)
                expected = (x + 1 - y.reshape(x.shape)) / jnp.linalg.norm(y)
                err = jnp.linalg.norm(out1 - expected) / jnp.linalg.norm(expected)
                assert err < 1e-5

                # jit version
                @jax.jit
                def complication_fn(x, y):
                    a = jax_fn(x, y)[0]
                    y2 = y.reshape(x.shape)
                    b, c = x - y2 + 1, x + y2 + 1
                    d = jnp.linalg.norm(x) - jnp.linalg.norm(y)
                    return a, b, c, d

                out = complication_fn(x, y)
                assert isinstance(out, (list, tuple)) and len(out) == 4
                out1 = out[0]
                assert isinstance(out1, Array)
                expected = (x + 1 - y.reshape(x.shape)) / jnp.linalg.norm(y)
                err = jnp.linalg.norm(out1 - expected) / jnp.linalg.norm(expected)
                assert err < 1e-5


def test_multi_output_fn():
    shape = (10, 2)

    def torch_fn(x, y):
        a = (x + 1 - y.reshape(x.shape)) / torch.norm(y)
        b = (x - y.reshape(x.shape)).reshape(-1)[:5]
        return a, b

    device_list = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    dtype_list = [jnp.float32, jnp.float64]

    for device in device_list:
        for dtype in dtype_list:
            for method in ["output_shapes", "output_shapes_fn", "example_args"]:
                if method == "output_shapes":
                    jax_fn = torch2jax(torch_fn, output_shapes=[shape, (5,)])
                elif method == "output_shapes_fn":
                    jax_fn = torch2jax(torch_fn, output_shapes_fn=lambda x, y: [x.shape, (5,)])
                elif method == "example_args":
                    example_args = (torch.randn(shape), torch.randn(shape).reshape(-1))
                    jax_fn = torch2jax(torch_fn, example_args=example_args)

                x = jax_randn(shape, device=device, dtype=dtype)
                y = jax_randn(shape, device=device, dtype=dtype).reshape(-1)

                # non-jit version
                out = jax_fn(x, y)
                assert isinstance(out, (list, tuple)) and len(out) == 2
                assert all(isinstance(z, Array) for z in out)
                expected1 = (x + 1 - y.reshape(x.shape)) / jnp.linalg.norm(y)
                expected2 = (x - y.reshape(x.shape)).reshape(-1)[:5]
                err1 = jnp.linalg.norm(out[0] - expected1) / jnp.linalg.norm(expected1)
                err2 = jnp.linalg.norm(out[1] - expected2) / jnp.linalg.norm(expected2)
                assert err1 < 1e-5 and err2 < 1e-5

                # jit version
                @jax.jit
                def complication_fn(x, y):
                    a = jax_fn(x, y)
                    y2 = y.reshape(x.shape)
                    b, c = x - y2 + 1, x + y2 + 1
                    d = jnp.linalg.norm(x) - jnp.linalg.norm(y)
                    return a, b, c, d

                out = complication_fn(x, y)
                assert isinstance(out, (list, tuple)) and len(out) == 4
                assert all(isinstance(z, Array) for z in out[0])
                out1 = out[0]
                expected1 = (x + 1 - y.reshape(x.shape)) / jnp.linalg.norm(y)
                expected2 = (x - y.reshape(x.shape)).reshape(-1)[:5]
                err1 = jnp.linalg.norm(out1[0] - expected1) / jnp.linalg.norm(expected2)
                err2 = jnp.linalg.norm(out1[1] - expected2) / jnp.linalg.norm(expected2)
                assert err1 < 1e-5 and err2 < 1e-5
