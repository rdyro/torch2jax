import sys
from pathlib import Path
from functools import partial

from absl.testing import parameterized, absltest
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


def _compute(a, b, c, with_grad: bool = False):
    output_shapes = (ShapeDtypeStruct(a.shape, a.dtype), ShapeDtypeStruct((), b.dtype))
    transform = torch2jax_with_vjp if with_grad else torch2jax
    ret = transform(
        torch_fn, ShapeDtypeStruct(a.shape, a.dtype), ShapeDtypeStruct(b.shape, b.dtype), output_shapes=output_shapes
    )(a, b)
    return ret + (a - b + c,)


def _expected_fn(a, b, c):
    return jnp.sin(a + b), jnp.mean(jnp.cos(a - b)), a - b + c


class TestTransformUnderJIT(parameterized.TestCase):
    @parameterized.product(
        device=["cpu", "cuda"],
        dtype=[jnp.float32, jnp.float64],
        shape=[(2, 3), (5, 10), (7,)],
    )
    def test_with_jit(self, device, dtype, shape):
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("Skipping CUDA tests when CUDA is not available")
        a = jax_randn(shape, dtype=dtype, device=device)
        b = jax_randn(shape, dtype=dtype, device=device)
        c = jax_randn(shape, dtype=dtype, device=device)

        ret = jax.jit(partial(_compute, with_grad=False))(a, b, c)
        expected = _expected_fn(a, b, c)
        err = sum([jnp.linalg.norm(v1 - v2) for (v1, v2) in zip(ret, expected)])
        assert err < 1e-6

    @parameterized.product(
        device=["cpu", "cuda"],
        dtype=[jnp.float32, jnp.float64],
        shape=[(2, 3), (5, 10), (7,)],
    )
    def test_without_jit(self, device, dtype, shape):
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("Skipping CUDA tests when CUDA is not available")
        a = jax_randn(shape, dtype=dtype, device=device)
        b = jax_randn(shape, dtype=dtype, device=device)
        c = jax_randn(shape, dtype=dtype, device=device)

        ret = partial(_compute, with_grad=False)(a, b, c)
        expected = _expected_fn(a, b, c)
        err = sum([jnp.linalg.norm(v1 - v2) for (v1, v2) in zip(ret, expected)])
        assert err < 1e-6

    @parameterized.product(
        device=["cpu"],
        dtype=[jnp.float32, jnp.float64],
        shape=[(2, 3), (5, 10), (7,)],
    )
    def test_grads_without_jit(self, device, dtype, shape):
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("Skipping CUDA tests when CUDA is not available")
        a = jax_randn(shape, dtype=dtype, device=device)
        b = jax_randn(shape, dtype=dtype, device=device)
        c = jax_randn(shape, dtype=dtype, device=device)

        ret = jax.grad(lambda a, b, c: partial(_compute, with_grad=True)(a, b, c)[1], (0, 1, 2))(a, b, c)
        expected = jax.grad(lambda a, b, c: _expected_fn(a, b, c)[1], (0, 1, 2))(a, b, c)
        err = sum([jnp.linalg.norm(v1 - v2) for (v1, v2) in zip(ret, expected)])
        assert err < 1e-6

    @parameterized.product(
        device=["cpu", "cuda"],
        dtype=[jnp.float32, jnp.float64],
        shape=[(2, 3), (5, 10), (7,)],
    )
    def test_grads_with_jit(self, device, dtype, shape):
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("Skipping CUDA tests when CUDA is not available")
        a = jax_randn(shape, dtype=dtype, device=device)
        b = jax_randn(shape, dtype=dtype, device=device)
        c = jax_randn(shape, dtype=dtype, device=device)

        ret = jax.jit(jax.grad(lambda a, b, c: partial(_compute, with_grad=True)(a, b, c)[1], (0, 1, 2)))(a, b, c)
        expected = jax.grad(lambda a, b, c: _expected_fn(a, b, c)[1], (0, 1, 2))(a, b, c)
        err = sum([jnp.linalg.norm(v1 - v2) for (v1, v2) in zip(ret, expected)])
        assert err < 1e-6


if __name__ == "__main__":
    absltest.main()
