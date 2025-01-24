from __future__ import annotations

import sys
from pathlib import Path

from absl.testing import parameterized, absltest
import torch
import jax
from jax import numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve

paths = [Path(__file__).absolute().parents[1], Path(__file__).absolute().parent]
for path in paths:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from torch2jax import torch2jax_with_vjp  # noqa: E402
from utils import jax_randn  # noqa: E402

####################################################################################################


class TestVmap(parameterized.TestCase):
    @parameterized.product(device=["cuda", "cpu"], dtype=[jnp.float32, jnp.float64])
    def test_simple_vmap(self, device, dtype):
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("Skipping CUDA tests when CUDA is not available")

        def torch_fn(A, x):
            return torch.linalg.solve(A, x)

        def expected_fn(A, x):
            return cho_solve(cho_factor(A), x)

        A = jax_randn((10, 5, 5), dtype=dtype, device=device)
        A = jnp.swapaxes(A, -1, -2) @ A + 1e-2 * jnp.tile(
            jnp.diag(1.0 + 0.0 * jax_randn((A.shape[-1],), dtype=dtype, device=device)),
            (A.shape[0], 1, 1),
        )
        x = jax_randn((5, 7), dtype=dtype, device=device)
        jax_fn = torch2jax_with_vjp(torch_fn, A[0, ...], x, output_shapes=x, depth=2)

        sol = jax.jit(jax.vmap(jax_fn, in_axes=(0, None)))(A, x)
        sol_expected = jax.jit(jax.vmap(expected_fn, in_axes=(0, None)))(A, x)
        err = jnp.linalg.norm(sol - sol_expected) / jnp.linalg.norm(sol_expected)
        assert err < 1e-3


if __name__ == "__main__":
    absltest.main()
