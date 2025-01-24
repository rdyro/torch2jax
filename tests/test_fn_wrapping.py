import sys
from pathlib import Path

from absl.testing import parameterized, absltest
import torch
from torch import Size
import jax
from jax import numpy as jnp
from jax import Array


paths = [Path(__file__).absolute().parents[1], Path(__file__).absolute().parent]
for path in paths:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from utils import jax_randn  # noqa: E402
from torch2jax import torch2jax  # noqa: E402

####################################################################################################


class TestFnWrapping(parameterized.TestCase):
    @parameterized.product(device=["cpu", "cuda"], dtype=[jnp.float32, jnp.float64])
    def test_single_output_fn(self, device, dtype):
        if not torch.cuda.is_available() and device == "cuda":
            self.skipTest("Skipping CUDA test when CUDA is not available.")
        shape = (10, 2)

        def torch_fn(x, y):
            return (x + 1 - y.reshape(x.shape)) / torch.norm(y)

        device_list = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        dtype_list = [jnp.float32, jnp.float64]

        x = jax_randn(shape, device="cpu", dtype=jnp.float64)
        y = jax_randn(shape, device="cpu", dtype=jnp.float64).reshape(-1)
        jax_fn = torch2jax(torch_fn, x, y, output_shapes=Size(shape))

        x = jax_randn(shape, device=device, dtype=dtype)
        y = jax_randn(shape, device=device, dtype=dtype).reshape(-1)

        # non-jit version
        out = jax_fn(x, y)
        assert isinstance(out, Array)
        expected = (x + 1 - y.reshape(x.shape)) / jnp.linalg.norm(y)
        err = jnp.linalg.norm(out - expected) / jnp.linalg.norm(expected)
        assert err < 1e-5

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
        out1 = out[0]
        assert isinstance(out1, Array)
        expected = (x + 1 - y.reshape(x.shape)) / jnp.linalg.norm(y)
        err = jnp.linalg.norm(out1 - expected) / jnp.linalg.norm(expected)
        assert err < 1e-5, f"Error is quite high: {err:.4e}"

    @parameterized.product(device=["cpu", "cuda"], dtype=[jnp.float32, jnp.float64])
    def test_multi_output_fn(self, device, dtype):
        if not torch.cuda.is_available() and device == "cuda":
            self.skipTest("Skipping CUDA test when CUDA is not available.")

        shape = (10, 2)

        def torch_fn(x, y):
            a = (x + 1 - y.reshape(x.shape)) / torch.norm(y)
            b = (x - y.reshape(x.shape)).reshape(-1)[:5]
            return a, b

        x = jax_randn(shape, device="cpu", dtype=jnp.float64)
        y = jax_randn(shape, device="cpu", dtype=jnp.float64).reshape(-1)
        jax_fn = torch2jax(torch_fn, x, y, output_shapes=(Size(shape), Size((5,))))

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


####################################################################################################

if __name__ == "__main__":
    absltest.main()
