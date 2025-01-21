import sys
from pathlib import Path

from absl.testing import absltest
from absl.testing import parameterized
import torch
from torch import Tensor
import jax
from jax import numpy as jnp
from jax import Array, jit
import numpy as np


paths = [Path(__file__).absolute().parents[1], Path(__file__).absolute().parent]
for path in paths:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from utils import jax_randn  # noqa: E402
from torch2jax import j2t, t2j, tree_j2t, tree_t2j  # noqa: E402

DTYPE_MAP = {torch.float32: jnp.float32, torch.float64: jnp.float64}
DEVICE_MAP = {"gpu": "cuda", "cpu": "cpu", "cuda": "cuda"}


class TestDlpackTransfers(parameterized.TestCase, absltest.TestCase):
    @parameterized.product(device=["cpu", "cuda"], dtype=[torch.float32, torch.float64], via=["host", "dlpack"])
    def test_dlpack_transfer(self, device, dtype, via):
        if not torch.cuda.is_available() and device == "cuda":
            self.skipTest("skipping CUDA test because CUDA is not available")
        old_jax_enable_x64 = jax.config.jax_enable_x64
        try:
            jax.config.update("jax_enable_x64", True)
            if device == "cuda" and not torch.cuda.is_available():
                self.skipTest("Skipping CUDA tests when no CUDA available")
            shape = (2, 3, 5, 1)

            # jax -> torch
            x1 = jax_randn(shape, device, DTYPE_MAP[dtype])
            x2 = j2t(x1, via=via)
            assert DEVICE_MAP[x2.device.type] == DEVICE_MAP[device] and x2.dtype == dtype
            err = np.linalg.norm(x2.cpu().numpy() - np.array(x1)) / np.linalg.norm(np.array(x1))
            assert err < 1e-5

            # torch -> jax
            x1 = torch.randn(shape, device=device, dtype=dtype)
            x2 = t2j(x1, via=via)
            devices = x2.devices()
            self.assertEqual(len(devices), 1)
            assert DEVICE_MAP[list(devices)[0].platform] == DEVICE_MAP[device] and x2.dtype == DTYPE_MAP[dtype]
            err = np.linalg.norm(x1.cpu().numpy() - np.array(x2)) / np.linalg.norm(x1.cpu().numpy())
            self.assertLess(err, 1e-5)
        finally:
            jax.config.update("jax_enable_x64", old_jax_enable_x64)

    @parameterized.product(device=["cpu", "cuda"])
    def test_tree_dlpack_transfer(self, device):
        if not torch.cuda.is_available() and device == "cuda":
            self.skipTest("skipping CUDA test because CUDA is not available")

        args = dict(
            a=1,
            b=jax_randn((10, 2), dtype=jnp.float32, device=device),
            c="hello",
            d=(1, jax_randn((10, 2), dtype=jnp.float32, device=device), torch.randn((10, 2), device=device)),
        )
        args2 = tree_j2t(args)
        assert isinstance(args2["a"], int)
        assert DEVICE_MAP[args2["b"].device.type] == device
        assert isinstance(args2["c"], str)
        assert isinstance(args2["d"][1], Tensor)
        assert DEVICE_MAP[args2["d"][1].device.type] == device
        assert isinstance(args2["d"][2], Tensor)

        args = dict(
            a=1,
            b=torch.randn((10, 2), device=device),
            c="hello",
            d=(1, torch.randn((10, 2), device=device), jax_randn((10, 2), dtype=jnp.float32, device=device)),
        )
        args2 = tree_t2j(args)
        assert isinstance(args2["a"], int)
        devices = args2["b"].devices()
        assert len(devices) == 1
        assert DEVICE_MAP[list(devices)[0].platform] == device
        assert isinstance(args2["c"], str)
        assert isinstance(args2["d"][1], Array)
        devices = args2["d"][1].devices()
        assert len(devices) == 1
        assert DEVICE_MAP[list(devices)[0].platform] == device
        assert isinstance(args2["d"][2], Array)

    def test_runtime_error(self):
        x = jax_randn((10, 2), dtype=jnp.float32, device="cpu")

        @jit
        def fn(x):
            x2 = j2t(x)
            x2 = 2 * x2
            return x * 2

        try:
            fn(x)
            assert False
        except RuntimeError:
            pass


if __name__ == "__main__":
    absltest.main()
