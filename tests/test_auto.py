import itertools

from absl.testing import absltest
from absl.testing import parameterized
import torch
import jax
from jax import numpy as jnp
from jax import Array

from torch2jax import torch2jax_auto

####################################################################################################

randn_keys = None


def jax_randn(shape, device, dtype):
    global randn_keys
    if randn_keys is None:
        randn_keys = itertools.cycle(jax.random.split(jax.random.key(0), 1024))
    device = device if not isinstance(device, str) else jax.devices(device)[0]
    return jax.device_put(jax.random.normal(next(randn_keys), shape, dtype=dtype), device)


class AutoTesting(parameterized.TestCase):
    @parameterized.product(shape=[(10, 2), (10,)], device=["cpu", "cuda"], dtype=[jnp.float32, jnp.float64])
    def test_auto_basic(self, shape, device, dtype):
        if not torch.cuda.is_available() and device == "cuda":
            self.skipTest("CUDA not available, skipping CUDA test")
        def torch_fn(x, y):
            return x + y

        auto_fn = torch2jax_auto(torch_fn)

        x = jax_randn(shape, device=device, dtype=dtype)
        y = jax_randn(shape, device=device, dtype=dtype)

        out = auto_fn(x, y)
        assert isinstance(out, Array)
        assert out.shape == shape

        expected = x + y
        err = jnp.linalg.norm(out - expected) / jnp.linalg.norm(expected + 1e-6)
        assert err < 1e-5

        # Test with different shape to trigger re-compilation
        shape2 = (shape[0] + 1,) + shape[1:]
        x2 = jax_randn(shape2, device=device, dtype=dtype)
        y2 = jax_randn(shape2, device=device, dtype=dtype)
        out2 = auto_fn(x2, y2)
        assert out2.shape == shape2

    @parameterized.product(shape=[(10, 2), (10,)], device=["cpu", "cuda"], dtype=[jnp.float32, jnp.float64])
    def test_auto_vjp(self, shape, device, dtype):
        if not torch.cuda.is_available() and device == "cuda":
            self.skipTest("CUDA not available, skipping CUDA test")
        def torch_fn(x, y):
            return torch.sum(x * y)

        auto_fn = torch2jax_auto(torch_fn)
        grad_fn = jax.grad(auto_fn, argnums=0)

        x = jax_randn(shape, device=device, dtype=dtype)
        y = jax_randn(shape, device=device, dtype=dtype)

        grad_out = grad_fn(x, y)
        assert isinstance(grad_out, Array)
        assert grad_out.shape == shape

        expected = y
        err = jnp.linalg.norm(grad_out - expected) / jnp.linalg.norm(expected + 1e-6)
        assert err < 1e-5

    @parameterized.product(shape=[(10, 2), (10,)], device=["cpu", "cuda"], dtype=[jnp.float32, jnp.float64])
    def test_auto_kwargs(self, shape, device, dtype):
        if not torch.cuda.is_available() and device == "cuda":
            self.skipTest("CUDA not available, skipping CUDA test")
        def torch_fn(x, alpha):
            return x * alpha

        auto_fn = torch2jax_auto(torch_fn, depth=0)

        x = jax_randn(shape, device=device, dtype=dtype)
        out = auto_fn(x, 2.0)

        assert isinstance(out, Array)
        assert out.shape == shape

        expected = x * 2.0
        err = jnp.linalg.norm(out - expected) / jnp.linalg.norm(expected + 1e-6)
        assert err < 1e-5

    @parameterized.product(shape=[(10, 2), (10,)], device=["cpu", "cuda"], dtype=[jnp.float32, jnp.float64])
    def test_auto_jit(self, shape, device, dtype):
        if not torch.cuda.is_available() and device == "cuda":
            self.skipTest("CUDA not available, skipping CUDA test")
        if len(shape) != 2:
            self.skipTest("Requires 2D shape for matrix multiplication")

        def torch_fn(x):
            return x @ x.T

        auto_fn = jax.jit(torch2jax_auto(torch_fn))

        x = jax_randn(shape, device=device, dtype=dtype)
        out = auto_fn(x)

        assert isinstance(out, Array)
        assert out.shape == (shape[0], shape[0])

        expected = x @ x.T
        err = jnp.linalg.norm(out - expected) / jnp.linalg.norm(expected + 1e-6)
        assert err < 1e-5


####################################################################################################

if __name__ == "__main__":
    absltest.main()
