import sys
from pathlib import Path

from absl.testing import parameterized, absltest
import torch
from torch import Size
import jax
from jax import numpy as jnp

paths = [Path(__file__).absolute().parents[1], Path(__file__).absolute().parent]
for path in paths:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from utils import jax_randn  # noqa: E402
from torch2jax import torch2jax  # noqa: E402
from torch2jax.compat import torch2jax as torch2jax_flat  # noqa: E402


class TestMemoryInPlace(parameterized.TestCase):
    @parameterized.product(device=["cpu", "cuda"], dtype=[jnp.float32, jnp.float64])
    def test_memory_inplace(self, device, dtype):
        if device == "cuda" and not torch.cuda.is_available():
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
