from __future__ import annotations

import sys
from pathlib import Path

from absl.testing import parameterized, absltest
import torch
from torch.autograd import Function
import jax
from jax import numpy as jnp, Array

paths = [Path(__file__).absolute().parents[1], Path(__file__).absolute().parent]
for path in paths:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from torch2jax import torch2jax_with_vjp  # noqa: E402
from utils import jax_randn  # noqa: E402

####################################################################################################


class TestGradFallback(parameterized.TestCase):
    @parameterized.product(device=["cpu", "cuda"], dtype=[jnp.float32, jnp.float64])
    def test_torch2jax_with_vjp_vjp_fallback(self, device, dtype):
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("Skipping CUDA tests when CUDA is not available")
        shape = (5, 7)

        class OldInterfaceFunction(Function):
            @staticmethod
            def forward(ctx, x, y):
                ctx.save_for_backward(x, y)
                return x**2 + y**2

            @staticmethod
            def backward(ctx, grad_output):
                x, y = ctx.saved_tensors
                return 2 * x * grad_output, 2 * y * grad_output

        def torch_fn(x, y):
            return OldInterfaceFunction.apply(x, y)

        def expected_f_fn(x, y):
            return x**2 + y**2

        expected_g_fn = jax.grad(lambda *args: jnp.sum(expected_f_fn(*args)), argnums=(0, 1))
        expected_h_fn = jax.grad(
            lambda *args: jnp.sum(expected_g_fn(*args)[0] + expected_g_fn(*args)[1]), argnums=(0, 1)
        )

        xt, yt = torch.randn(shape), torch.randn(shape)

        wrap_jax_f_fn = torch2jax_with_vjp(torch_fn, xt, yt, depth=2)
        wrap_jax_g_fn = jax.grad(lambda x, y: jnp.sum(wrap_jax_f_fn(x, y)), argnums=(0, 1))
        wrap_jax_h_fn = jax.grad(lambda x, y: jnp.sum(wrap_jax_g_fn(x, y)[0] + wrap_jax_g_fn(x, y)[1]), argnums=(0, 1))

        x = jax_randn(shape, dtype=dtype, device=device)
        y = jax_randn(shape, dtype=dtype, device=device)
        f = wrap_jax_f_fn(x, y)
        g = wrap_jax_g_fn(x, y)
        h = wrap_jax_h_fn(x, y)

        f_expected = expected_f_fn(x, y)
        g_expected = expected_g_fn(x, y)
        h_expected = expected_h_fn(x, y)

        # test output structure #############################
        self.assertIsInstance(f, Array)
        assert isinstance(g, (tuple, list)) and len(g) == 2 and isinstance(g[0], Array) and isinstance(g[1], Array)
        assert isinstance(h, (tuple, list)) and len(h) == 2 and isinstance(h[0], Array) and isinstance(h[1], Array)

        # test values not under JIT #########################
        err_f = jnp.linalg.norm(f - f_expected)
        err_g = jnp.linalg.norm(g[0] - g_expected[0]) + jnp.linalg.norm(g[1] - g_expected[1])
        err_h = jnp.linalg.norm(h[0] - h_expected[0]) + jnp.linalg.norm(h[1] - h_expected[1])

        self.assertLess(err_f, 1e-5)
        self.assertLess(err_g, 1e-5)
        self.assertLess(err_h, 1e-5)

        # test values when under JIT ########################
        f = jax.jit(wrap_jax_f_fn)(x, y)
        g = jax.jit(wrap_jax_g_fn)(x, y)
        h = jax.jit(wrap_jax_h_fn)(x, y)

        err_f = jnp.linalg.norm(f - f_expected)
        err_g = jnp.linalg.norm(g[0] - g_expected[0]) + jnp.linalg.norm(g[1] - g_expected[1])
        err_h = jnp.linalg.norm(h[0] - h_expected[0]) + jnp.linalg.norm(h[1] - h_expected[1])

        self.assertLess(err_f, 1e-5)
        self.assertLess(err_g, 1e-5)
        self.assertLess(err_h, 1e-5)


if __name__ == "__main__":
    absltest.main()
