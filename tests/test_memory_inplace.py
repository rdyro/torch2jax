import itertools

from absl.testing import parameterized, absltest
import torch
from torch import Size
import jax
from jax import numpy as jnp

from torch2jax import torch2jax  # noqa: E402

CUDA_AVAILABLE = torch.cuda.is_available() and jax.default_backend() == "gpu"


randn_keys = None


def jax_randn(shape, device, dtype):
    global randn_keys
    if randn_keys is None:
        randn_keys = itertools.cycle(jax.random.split(jax.random.key(0), 1024))
    device = device if not isinstance(device, str) else jax.devices(device)[0]
    return jax.device_put(jax.random.normal(next(randn_keys), shape, dtype=dtype), device)


class TestMemoryInPlace(parameterized.TestCase):
    @parameterized.product(device=["cpu", "cuda"], dtype=[jnp.float32, jnp.float64])
    def test_memory_inplace(self, device, dtype):
        if device == "cuda" and not CUDA_AVAILABLE:
            self.skipTest("Skipping CUDA tests when CUDA is not available")

        # we're going to test if we can write in memory inplace
        def torch_fn(x):
            y = torch.randn_like(x)
            x[:5].add_(17.0)
            return y

        x = jax_randn((50,), device=device, dtype=dtype) * 0
        jax_fn = torch2jax(torch_fn, x, output_shapes=Size(x.shape))
        _ = jax_fn(x)
        expected = jnp.concatenate([jnp.ones(5) * 17, jnp.zeros(45)])
        jax_device = jax.devices(device)[0]
        expected = jax.device_put(expected, jax_device).astype(dtype)
        err = jnp.linalg.norm(x - expected) / jnp.linalg.norm(expected)
        self.assertLess(err, 1e-5)


if __name__ == "__main__":
    absltest.main()
