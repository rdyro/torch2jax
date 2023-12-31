import sys
from pathlib import Path
from functools import partial

import torch
import jax
from jax import numpy as jnp
from jax import ShapeDtypeStruct

paths = [Path(__file__).absolute().parents[1], Path(__file__).absolute().parent]
for path in paths:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from torch2jax import torch2jax, torch2jax_with_vjp  # noqa: E402
from utils import jax_randn  # noqa: E402


def torch_fn(a, b):
    return torch.sin(a + b), torch.mean(torch.cos(a - b))


def compute(a, b, c, with_grad: bool = False):
    output_shapes = (
        ShapeDtypeStruct(a.shape, a.dtype),
        ShapeDtypeStruct((), b.dtype),
    )
    transform = torch2jax_with_vjp if with_grad else torch2jax
    ret = transform(
        torch_fn,
        ShapeDtypeStruct(a.shape, a.dtype),
        ShapeDtypeStruct(b.shape, b.dtype),
        output_shapes=output_shapes,
    )(a, b)
    return ret + (a - b + c,)


def expected_fn(a, b, c):
    return jnp.sin(a + b), jnp.mean(jnp.cos(a - b)), a - b + c


####################################################################################################


def test_with_jit():
    device_list = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    shapes = [(2, 3), (5, 10), (7,)]
    for shape in shapes:
        for dtype in [jnp.float32, jnp.float64]:
            for device in device_list:
                a = jax_randn(shape, dtype=dtype, device=device)
                b = jax_randn(shape, dtype=dtype, device=device)
                c = jax_randn(shape, dtype=dtype, device=device)

                ret = jax.jit(partial(compute, with_grad=False))(a, b, c)
                expected = expected_fn(a, b, c)
                err = sum([jnp.linalg.norm(v1 - v2) for (v1, v2) in zip(ret, expected)])
                # print(f"dtype = {dtype}, device = {device}, err = {err}")
                assert err < 1e-6


def test_without_jit():
    device_list = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    shapes = [(2, 3), (5, 10), (7,)]
    for shape in shapes:
        for dtype in [jnp.float32, jnp.float64]:
            for device in device_list:
                a = jax_randn(shape, dtype=dtype, device=device)
                b = jax_randn(shape, dtype=dtype, device=device)
                c = jax_randn(shape, dtype=dtype, device=device)

                ret = partial(compute, with_grad=False)(a, b, c)
                expected = expected_fn(a, b, c)
                err = sum([jnp.linalg.norm(v1 - v2) for (v1, v2) in zip(ret, expected)])
                # print(f"dtype = {dtype}, device = {device}, err = {err}")
                assert err < 1e-6


def test_grads_without_jit():
    device_list = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    shapes = [(2, 3), (5, 10), (7,)]
    for shape in shapes:
        for dtype in [jnp.float32, jnp.float64]:
            for device in device_list:
                a = jax_randn(shape, dtype=dtype, device=device)
                b = jax_randn(shape, dtype=dtype, device=device)
                c = jax_randn(shape, dtype=dtype, device=device)

                ret = jax.grad(
                    lambda a, b, c: partial(compute, with_grad=True)(a, b, c)[1], (0, 1, 2)
                )(a, b, c)
                expected = jax.grad(lambda a, b, c: expected_fn(a, b, c)[1], (0, 1, 2))(a, b, c)
                err = sum([jnp.linalg.norm(v1 - v2) for (v1, v2) in zip(ret, expected)])
                # print(f"dtype = {dtype}, device = {device}, err = {err}")
                assert err < 1e-6


def test_grads_with_jit():
    device_list = ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"]
    shapes = [(2, 3), (5, 10), (7,)]
    for shape in shapes:
        for dtype in [jnp.float32, jnp.float64]:
            for device in device_list:
                a = jax_randn(shape, dtype=dtype, device=device)
                b = jax_randn(shape, dtype=dtype, device=device)
                c = jax_randn(shape, dtype=dtype, device=device)

                ret = jax.jit(
                    jax.grad(
                        lambda a, b, c: partial(compute, with_grad=True)(a, b, c)[1], (0, 1, 2)
                    )
                )(a, b, c)
                expected = jax.grad(lambda a, b, c: expected_fn(a, b, c)[1], (0, 1, 2))(a, b, c)
                err = sum([jnp.linalg.norm(v1 - v2) for (v1, v2) in zip(ret, expected)])
                # print(f"dtype = {dtype}, device = {device}, err = {err}")
                assert err < 1e-6


####################################################################################################

if __name__ == "__main__":
    test_with_jit()
    test_without_jit()
    test_grads_with_jit()
    test_grads_without_jit()